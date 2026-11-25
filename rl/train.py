"""Synchronous colocated RL for the chess puzzle teams (and the single-agent baseline).

    torchrun --nproc-per-node 8 -m rl.train --preset team4 --out runs/rl-team4

Each step (all 8 GPUs take turns):
  1. rank 0 runs `puzzles_per_step x groups` episodes on the vLLM servers with the
     current adapter, recording sampled tokens and sampler logprobs;
  2. rewards are centred within each puzzle's rollouts (Dr. GRPO) and groups with
     no reward variance are dropped (DAPO);
  3. the servers sleep (weights to CPU), the trainer moves its shards to the GPUs
     and takes one AdamW step on the LoRA weights, then moves them back;
  4. the adapter is saved and hot-loaded into the woken servers.
Rollouts are never stale. The loss is token-level REINFORCE with a truncated
importance weight min(exp(logp_train - logp_sampler), C) correcting for the
FP8 sampler vs BF16 trainer mismatch, normalised by a constant (Dr. GRPO).
"""
import argparse
import asyncio
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist

from chess_env.env import preset, run_episode
from chess_env.rewards import group_advantages
from chess_env.run import summarize
from .rollout import Sequence, Servers, agent_sequences

ROOT = Path(__file__).resolve().parents[1]
BF16_MODEL = os.environ.get('BF16_MODEL_DIR', str(ROOT / 'models/Qwen3.8-Flash-Next'))
# Dense QSA masks make memory quadratic in length; 32K fits on an H100 next to sleeping vLLM servers.
MAX_TRAIN_TOKENS = 32_768


# One persistent loop: the HTTP clients are bound to the loop they were first used on.
LOOP = asyncio.new_event_loop()


def log(*args):
    print(f'[{time.strftime("%H:%M:%S")}]', *args, flush=True)


# ------------------------------------------------------------------ rollouts
async def collect(servers, puzzles, groups, config, concurrency):
    gate = asyncio.Semaphore(concurrency)

    async def one(puzzle, g):
        async with gate:
            return await run_episode(puzzle, servers.next_policy(), config)

    return await asyncio.gather(*(one(p, g) for p in puzzles for g in range(groups)))


def build_sequences(episodes):
    seqs, broken, missing, truncated = [], 0, 0, 0
    for i, j, adv in group_advantages(episodes):
        agent = episodes[i]['agents'][j]
        trace = agent.get('trace') or []
        if not trace:
            missing += 1
            continue
        parts, b = agent_sequences(trace)
        broken += b
        for ids, targets, lps in parts:
            if len(ids) > MAX_TRAIN_TOKENS:
                # Causal model: logprobs of the kept prefix are exact, so truncation drops tokens without bias.
                keep = [k for k, t in enumerate(targets) if t < MAX_TRAIN_TOKENS]
                ids, targets, lps = ids[:MAX_TRAIN_TOKENS], [targets[k] for k in keep], [lps[k] for k in keep]
                truncated += 1
            if targets:
                seqs.append(Sequence(ids, targets, lps, adv, (i, j)))
    return seqs, {'merge_breaks': broken, 'agents_without_tokens': missing, 'truncated': truncated}


def strip_tokens(episode):
    for agent in episode['agents']:
        agent.pop('trace', None)
    return episode


def partition(seqs, world):
    """Longest-processing-time assignment by approximate cost; equal counts per rank via padding."""
    order = sorted(range(len(seqs)), key=lambda k: -len(seqs[k].ids))
    load, bins = [0.0] * world, [[] for _ in range(world)]
    for k in order:
        r = min(range(world), key=lambda r: load[r])
        bins[r].append(k)
        load[r] += len(seqs[k].ids) * (1 + len(seqs[k].ids) / 32768)
    n = max(len(b) for b in bins)
    return [b + [None] * (n - len(b)) for b in bins], load


# ------------------------------------------------------------------ training
def train_step(policy, optimizer, seqs, mine, n_rollouts, args, world):
    torch.cuda.reset_peak_memory_stats()
    stats = {'tokens': 0, 'diff_sum': 0.0, 'absdiff_sum': 0.0, 'capped': 0, 'loss': 0.0}
    scale = world / (n_rollouts * args.norm_tokens)
    for k in mine:
        if k is None:  # padding keeps FSDP collectives in step across ranks
            lp = policy.token_logprobs([0] * 8, [1])
            (lp.sum() * 0).backward()
            continue
        s = seqs[k]
        lp = policy.token_logprobs(s.ids, s.targets)
        q = torch.tensor(s.sampler_logprobs, device=lp.device, dtype=torch.float32)
        diff = lp.detach() - q
        w = diff.exp().clamp(max=args.tis_cap)
        loss = -(s.advantage * w * lp).sum() * scale
        loss.backward()
        stats['tokens'] += lp.numel()
        stats['diff_sum'] += diff.sum().item()
        stats['absdiff_sum'] += diff.abs().sum().item()
        stats['capped'] += (diff.exp() > args.tis_cap).sum().item()
        stats['loss'] += loss.item() / world
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.lora_params, args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.tensor([stats['tokens'], stats['diff_sum'], stats['absdiff_sum'], stats['capped'], stats['loss']],
                          device='cuda', dtype=torch.float64)
    dist.all_reduce(totals)
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 2 ** 30], device='cuda')
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    t = totals.tolist()
    return {'train_tokens': int(t[0]), 'logprob_diff_mean': t[1] / max(t[0], 1),
            'logprob_absdiff_mean': t[2] / max(t[0], 1), 'tis_capped_frac': t[3] / max(t[0], 1),
            'loss': t[4], 'grad_norm': float(grad_norm.full_tensor() if hasattr(grad_norm, 'full_tensor') else grad_norm),
            'peak_mem_gb': round(peak.item(), 1)}


def save_ckpt(out, step, policy, optimizer, rank):
    d = out / 'ckpt'
    d.mkdir(exist_ok=True)
    torch.save({'step': step, 'lora': policy.lora_state(), 'optim': optimizer.state_dict()}, d / f'rank{rank}.pt')


def load_ckpt(out, policy, optimizer, rank):
    path = out / 'ckpt' / f'rank{rank}.pt'
    if not path.exists():
        return 0
    state = torch.load(path, map_location='cpu', weights_only=False)
    policy.load_lora_state(state['lora'])
    optimizer.load_state_dict(state['optim'])
    return state['step']


# ---------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preset', default='team4')
    p.add_argument('--config', default='{}', help='JSON overrides of the preset Config')
    p.add_argument('--steps', type=int, default=115)
    p.add_argument('--puzzles-per-step', type=int, default=16)
    p.add_argument('--groups', type=int, default=4, help='Episodes (teams) per puzzle')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--tis-cap', type=float, default=2.0)
    p.add_argument('--norm-tokens', type=int, default=8192)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--eval-limit', type=int, default=100)
    p.add_argument('--eval-samples', type=int, default=1, help='Episodes per held-out puzzle (4 gives single-agent pass@4)')
    p.add_argument('--ckpt-every', type=int, default=5)
    p.add_argument('--servers', default='http://127.0.0.1:8000,http://127.0.0.1:8001')
    p.add_argument('--served-model', default='Qwen3.8-Flash-Next')
    p.add_argument('--concurrency', type=int, default=128, help='Episodes in flight')
    p.add_argument('--model-dir', default=BF16_MODEL)
    p.add_argument('--train-data', default=str(ROOT / 'data/train.jsonl'))
    p.add_argument('--eval-data', default=str(ROOT / 'data/heldout.jsonl'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    dist.init_process_group('nccl', timeout=timedelta(hours=6))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
    cpu_group = dist.new_group(backend='gloo', timeout=timedelta(hours=6))
    out = Path(args.out)
    config = preset(args.preset, **json.loads(args.config))
    servers = None
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'settings.json').write_text(json.dumps(dict(vars(args), config=asdict(config)), indent=2))
        servers = Servers(args.servers.split(','), args.served_model)
        LOOP.run_until_complete(servers.ready())
        LOOP.run_until_complete(servers.sleep())
        log('servers asleep; loading trainer')
    dist.barrier(group=cpu_group)

    from .model import Policy
    t0 = time.time()
    policy = Policy(args.model_dir, rank, world)
    optimizer = torch.optim.AdamW(policy.lora_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    start = load_ckpt(out, policy, optimizer, rank)
    policy.offload()
    dist.barrier(group=cpu_group)
    if rank == 0:
        log(f'trainer loaded {policy.loaded} tensors in {time.time() - t0:.0f}s; resuming at step {start}')
    adapter = None
    if start > 0 and rank == 0:
        adapter = f'step{start}'
    if rank == 0:
        LOOP.run_until_complete(servers.wake())
        if adapter:
            LOOP.run_until_complete(servers.load_adapter(adapter, out / 'adapters' / adapter))
            servers.set_model(adapter)

    train = [json.loads(l) for l in Path(args.train_data).read_text().splitlines() if l.strip()]
    heldout = [json.loads(l) for l in Path(args.eval_data).read_text().splitlines() if l.strip()][:args.eval_limit]
    rng = random.Random(args.seed)
    order = []
    for _ in range(args.steps * args.puzzles_per_step // len(train) + 2):
        epoch = list(range(len(train)))
        rng.shuffle(epoch)
        order += epoch

    for step in range(start, args.steps + 1):
        record = {'step': step}
        if rank == 0:
            if step % args.eval_every == 0 or step == args.steps:
                t = time.time()
                eval_eps = LOOP.run_until_complete(collect(servers, heldout, args.eval_samples, config, args.concurrency))
                summary = summarize(eval_eps)
                with (out / 'eval.jsonl').open('a') as f:
                    f.write(json.dumps({'step': step, 'adapter': servers.model, 'seconds': round(time.time() - t),
                                        **summary}) + '\n')
                with (out / f'eval-step{step}.jsonl').open('w') as f:
                    for e in eval_eps:
                        f.write(json.dumps(strip_tokens(e)) + '\n')
                log(f'eval step {step}: first_answer_acc={summary["first_answer_acc"]} '
                    f'pass@{config.n_agents}={summary.get(f"team_pass@{config.n_agents}")} ({time.time() - t:.0f}s)')
        if step == args.steps:
            break
        payload = [None]
        if rank == 0:
            t = time.time()
            puzzles = [train[k] for k in order[step * args.puzzles_per_step:(step + 1) * args.puzzles_per_step]]
            episodes = LOOP.run_until_complete(collect(servers, puzzles, args.groups, config, args.concurrency))
            record['rollout_seconds'] = round(time.time() - t, 1)
            seqs, extra = build_sequences(episodes)
            record.update(extra)
            record['rollout'] = summarize(episodes)
            record['sequences'] = len(seqs)
            record['groups_kept'] = len({episodes[s.agent_key[0]]['puzzle']['id'] for s in seqs})
            with (out / 'rollouts.jsonl').open('a') as f:
                for e in episodes:
                    f.write(json.dumps({'step': step, **strip_tokens(e)}) + '\n')
            t = time.time()
            LOOP.run_until_complete(servers.sleep())
            record['sleep_seconds'] = round(time.time() - t, 1)
            n_rollouts = sum(len(e['agents']) for e in episodes)
            bins, load = partition(seqs, world)
            payload = [(seqs, bins, n_rollouts, record)]
        dist.broadcast_object_list(payload, src=0, group=cpu_group)
        seqs, bins, n_rollouts, record = payload[0]

        t = time.time()
        policy.onload()
        record['onload_seconds'] = round(time.time() - t, 1)
        t = time.time()
        if seqs:
            record.update(train_step(policy, optimizer, seqs, bins[rank], n_rollouts, args, world))
        record['train_seconds'] = round(time.time() - t, 1)
        name = f'step{step + 1}'
        t = time.time()
        policy.save_adapter(out / 'adapters' / name)
        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(out, step + 1, policy, optimizer, rank)
        policy.offload()
        record['save_offload_seconds'] = round(time.time() - t, 1)
        dist.barrier(group=cpu_group)

        if rank == 0:
            t = time.time()
            LOOP.run_until_complete(servers.wake())
            LOOP.run_until_complete(servers.load_adapter(name, out / 'adapters' / name))
            previous = servers.model
            servers.set_model(name)
            if previous.startswith('step'):
                LOOP.run_until_complete(servers.unload_adapter(previous))
            record['wake_seconds'] = round(time.time() - t, 1)
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            r = record.get('rollout', {})
            log(f'step {step + 1}: reward={r.get("mean_reward")} first_acc={r.get("first_answer_acc")} '
                f'seqs={record.get("sequences")} tokens={record.get("train_tokens")} '
                f'diff={record.get("logprob_diff_mean", 0):.4f} gnorm={record.get("grad_norm", 0):.3f} '
                f'rollout={record.get("rollout_seconds")}s train={record["train_seconds"]}s')
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
