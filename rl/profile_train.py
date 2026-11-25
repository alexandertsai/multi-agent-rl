"""Profile one forward+backward of the real trainer on a sampled-length sequence (servers must be asleep).

    torchrun --nproc-per-node 8 -m rl.profile_train --tokens 10000
"""
import argparse
from datetime import timedelta
import os
import time

import torch
import torch.distributed as dist

from .model import Policy
from .train import BF16_MODEL


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, default=10000)
    p.add_argument('--targets', type=int, default=9000)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    dist.init_process_group('nccl', timeout=timedelta(hours=1))
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
    policy = Policy(BF16_MODEL, rank, dist.get_world_size())
    g = torch.Generator().manual_seed(rank)
    ids = torch.randint(1000, 200000, (args.tokens,), generator=g).tolist()
    targets = list(range(args.tokens - args.targets, args.tokens))

    def once():
        lp = policy.token_logprobs(ids, targets)
        lp.sum().backward()
        torch.cuda.synchronize()

    once()
    for _ in range(2):
        t = time.time(); once(); dist.barrier()
        if rank == 0:
            print(f'fwd+bwd {args.tokens} tokens: {time.time() - t:.2f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB', flush=True)
    t = time.time()
    with torch.no_grad():
        policy.token_logprobs(ids, targets)
    torch.cuda.synchronize(); dist.barrier()
    if rank == 0:
        print(f'fwd only: {time.time() - t:.2f}s', flush=True)
    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            once()
        if rank == 0:
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=70), flush=True)
            print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=12, max_name_column_width=70), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
