"""Run chess puzzle episodes against an OpenAI-compatible server and summarise them.

    python -m chess_env.run --preset team4 --samples 1
    python -m chess_env.run --preset single --samples 4
"""
import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import json
from math import comb
from pathlib import Path
import random

from .env import PRESETS, preset, run_episode
from .policy import OpenAIPolicy

ROOT = Path(__file__).resolve().parents[1]


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def mean(values):
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def summarize(episodes):
    agents = [a for e in episodes for a in e['agents']]
    n = episodes[0]['config']['n_agents']
    summary = {
        'episodes': len(episodes), 'puzzles': len({e['puzzle']['id'] for e in episodes}),
        'first_answer_acc': mean(e['team_correct'] for e in episodes),
        f'team_pass@{n}': mean(e['any_correct'] for e in episodes),
        # The team's first answer was wrong while a teammate had the right move.
        'rescuable': mean(e['any_correct'] and not e['team_correct'] for e in episodes),
        'agent_acc': mean(a['correct'] for a in agents),
        'no_answer': mean(a['answer'] is None for a in agents),
        'never_messaged': mean(a['messages_sent'] == 0 for a in agents),
        'messages_per_agent': mean(a['messages_sent'] for a in agents),
        'inbox_checks_per_agent': mean(a['inbox_checks'] for a in agents),
        'turns_per_agent': mean(a['turns'] for a in agents),
        'tokens_per_agent': mean(a['completion_tokens'] for a in agents),
        'tokens_per_episode': mean(sum(a['completion_tokens'] for a in e['agents']) for e in episodes),
        'mean_reward': mean(a['reward'] for a in agents),
        'stops': {s: sum(a['stop'] == s for a in agents) for s in sorted({a['stop'] for a in agents})},
    }
    # Pass@k over independent episodes of the same puzzle, e.g. four single agents vs one team.
    by_puzzle = defaultdict(list)
    for e in episodes:
        by_puzzle[e['puzzle']['id']].append(e['team_correct'])
    samples = min(len(v) for v in by_puzzle.values())
    for k in sorted({1, 4, samples}):
        if k <= samples:
            summary[f'episode_pass@{k}'] = mean(pass_at_k(len(v), sum(v), k) for v in by_puzzle.values())
    bands = defaultdict(list)
    for e in episodes:
        bands[e['puzzle'].get('band')].append(e['team_correct'])
    summary['first_answer_acc_by_band'] = {b: mean(v) for b, v in sorted(bands.items(), key=str)}
    return summary


async def run(args):
    config = preset(args.preset, **json.loads(args.config))
    puzzles = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if args.limit:
        puzzles = puzzles[:args.limit]
    # Comma-separated base URLs spread episodes round-robin across servers.
    policies = [OpenAIPolicy(url, args.model, reasoning_effort=args.reasoning_effort,
                             preserve_thinking=not args.no_preserve_thinking) for url in args.base_url.split(',')]
    out = Path(args.out or ROOT / 'runs' / f"{args.preset}-{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    (out / 'settings.json').write_text(json.dumps(dict(vars(args), config=asdict(config)), indent=2))
    jobs = [(p, s) for p in puzzles for s in range(args.samples)]
    random.Random(0).shuffle(jobs)
    gate, episodes = asyncio.Semaphore(args.concurrency), []

    async def one(n, puzzle, sample):
        async with gate:
            episode = await run_episode(puzzle, policies[n % len(policies)], config)
        episode['sample'] = sample
        episodes.append(episode)
        with (out / 'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(episode) + '\n')
        print(f"[{len(episodes)}/{len(jobs)}] {puzzle['id']} team={episode['team_answer']} "
              f"solution={puzzle['solution_san']} correct={episode['team_correct']}", flush=True)

    await asyncio.gather(*(one(n, p, s) for n, (p, s) in enumerate(jobs)))
    summary = summarize(episodes)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preset', choices=sorted(PRESETS), default='team4')
    p.add_argument('--config', default='{}', help='JSON overrides of the preset Config fields')
    p.add_argument('--data', default=str(ROOT / 'data/heldout.jsonl'))
    p.add_argument('--samples', type=int, default=1, help='Episodes per puzzle')
    p.add_argument('--limit', type=int)
    p.add_argument('--concurrency', type=int, default=16, help='Episodes in flight')
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1', help='Comma-separated for several servers')
    p.add_argument('--model', default='Qwen3.8-Flash-Next')
    p.add_argument('--reasoning-effort', default='low', choices=['low', 'medium', 'xhigh'])
    p.add_argument('--no-preserve-thinking', action='store_true')
    p.add_argument('--out')
    asyncio.run(run(p.parse_args()))


if __name__ == '__main__':
    main()
