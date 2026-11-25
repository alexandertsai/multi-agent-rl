"""Build disjoint train and held-out splits from the Lichess puzzle parquet shards.

Puzzles rated 2000-2800 are sampled evenly across four 200-point bands, as in
"Simple multi-agent RL turns pass@k into pass@1". Held-out and train puzzles
come from disjoint source games.
"""
import argparse
import json
from pathlib import Path
import random

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .puzzles import from_lichess

BANDS = [(2000, 2200), (2200, 2400), (2400, 2600), (2600, 2801)]
COLUMNS = ['PuzzleId', 'GameId', 'FEN', 'Moves', 'Rating', 'Themes']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parquet', nargs='+')
    p.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'data'))
    p.add_argument('--heldout-per-band', type=int, default=25)
    p.add_argument('--train-per-band', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    rows = []
    for path in args.parquet:
        table = pq.read_table(path, columns=COLUMNS)
        rating = table['Rating']
        rows += table.filter(pc.and_(pc.greater_equal(rating, BANDS[0][0]),
                                     pc.less(rating, BANDS[-1][1]))).to_pylist()
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    splits, games = {'heldout': [], 'train': []}, set()
    for name, per_band in [('heldout', args.heldout_per_band), ('train', args.train_per_band)]:
        for low, high in BANDS:
            band = [r for r in rows if low <= r['Rating'] < high and r['GameId'] not in games]
            picked = band[:per_band]
            if len(picked) < per_band:
                raise ValueError(f'Only {len(picked)} puzzles in band {low}-{high}')
            games.update(r['GameId'] for r in picked)
            splits[name] += [dict(from_lichess(r), band=f'{low}-{min(high, 2800)}') for r in picked]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, puzzles in splits.items():
        with (out / f'{name}.jsonl').open('w') as stream:
            stream.writelines(json.dumps(x) + '\n' for x in puzzles)
        print(f'{name}: {len(puzzles)} puzzles -> {out / name}.jsonl')


if __name__ == '__main__':
    main()
