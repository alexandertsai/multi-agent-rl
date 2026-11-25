"""Closed-loop generation throughput benchmark with realistic chess prompts.

    python -m rl.bench --base-url http://127.0.0.1:8001/v1 --concurrency 128 --seconds 240
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

from openai import AsyncOpenAI

from chess_env.env import preset, prompt

ROOT = Path(__file__).resolve().parents[1]


async def main(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key='EMPTY', timeout=3600)
    puzzles = [json.loads(l) for l in (ROOT / 'data/train.jsonl').read_text().splitlines()[:512]]
    config = preset('single')
    done, tokens, start = [0], [0], time.monotonic()
    deadline = start + args.seconds

    async def worker(w):
        k = w
        while time.monotonic() < deadline:
            r = await client.chat.completions.create(
                model=args.model, messages=[{'role': 'user', 'content': prompt(puzzles[k % len(puzzles)], 1, config)}],
                max_tokens=args.max_tokens, reasoning_effort='low', temperature=1.0, top_p=0.95,
                extra_body={'top_k': 20, 'chat_template_kwargs': {'enable_thinking': True}})
            if time.monotonic() <= deadline:
                done[0] += 1
                tokens[0] += r.usage.completion_tokens
            k += args.concurrency

    await asyncio.gather(*(worker(w) for w in range(args.concurrency)))
    elapsed = time.monotonic() - start
    print(json.dumps({'concurrency': args.concurrency, 'requests': done[0], 'completion_tokens': tokens[0],
                      'tokens_per_s': round(tokens[0] / elapsed, 1), 'seconds': round(elapsed)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', default='Qwen3.8-Flash-Next')
    p.add_argument('--concurrency', type=int, default=128)
    p.add_argument('--max-tokens', type=int, default=3000)
    p.add_argument('--seconds', type=int, default=240)
    asyncio.run(main(p.parse_args()))
