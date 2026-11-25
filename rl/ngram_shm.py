"""Build (once) and map the frozen 102 GB n-gram embedding in /dev/shm.

The checkpoint stores the table as `split_ngram_parts` row shards; they are
concatenated into one raw bf16 file so every trainer rank can memory-map the
same physical pages instead of holding its own copy.
"""
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

SHM = Path('/dev/shm/agent-rl')


def shard_keys(model_dir):
    index = json.loads((Path(model_dir) / 'model.safetensors.index.json').read_text())['weight_map']
    keys = [k for k in index if '.ngram_embedding.shard_' in k]
    keys.sort(key=lambda k: int(k.rsplit('shard_', 1)[1].split('.')[0]))
    return keys, index


def build(model_dir, workers=16):
    from concurrent.futures import ThreadPoolExecutor
    keys, index = shard_keys(model_dir)
    prefix = keys[0].split('.shard_')[0]
    shapes = []
    for key in keys:
        with safe_open(Path(model_dir) / index[key], 'pt') as f:
            shapes.append(f.get_slice(key).get_shape())
    width = shapes[0][1]
    rows = sum(s[0] for s in shapes)
    SHM.mkdir(parents=True, exist_ok=True)
    path, meta = SHM / 'ngram.bin', SHM / 'ngram.json'
    info = {'prefix': prefix, 'rows': rows, 'width': width, 'dtype': 'bfloat16', 'source': str(model_dir)}
    if meta.exists() and json.loads(meta.read_text()) == info and path.stat().st_size == rows * width * 2:
        return info
    meta.unlink(missing_ok=True)
    table = torch.from_file(str(path), shared=True, size=rows * width, dtype=torch.bfloat16).view(rows, width)
    offsets = [0]
    for s in shapes:
        offsets.append(offsets[-1] + s[0])

    def copy(i):
        with safe_open(Path(model_dir) / index[keys[i]], 'pt') as f:
            table[offsets[i]:offsets[i + 1]].copy_(f.get_tensor(keys[i]))
        return i

    with ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(copy, range(len(keys))), 1):
            if n % 16 == 0:
                print(f'[ngram] {n}/{len(keys)} shards', flush=True)
    os.sync()
    meta.write_text(json.dumps(info))
    return info


def load():
    info = json.loads((SHM / 'ngram.json').read_text())
    table = torch.from_file(str(SHM / 'ngram.bin'), shared=True, size=info['rows'] * info['width'],
                            dtype=torch.bfloat16).view(info['rows'], info['width'])
    return info, table


if __name__ == '__main__':
    import sys
    print(build(sys.argv[1]))
