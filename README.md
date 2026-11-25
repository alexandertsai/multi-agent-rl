# Multi-agent RL on chess puzzles

A replication of [Simple multi-agent RL turns pass@k into pass@1](https://check-before-you-submit.vercel.app/) with Qwen3.8-Flash-Next. The experiment compares a team of four communicating agents with independent single-agent attempts on the same 100 held-out Lichess puzzles.

| Setup | Step 0 pass@4 | Best observed pass@4 | Step 115 pass@4 |
|---|---:|---:|---:|
| Four communicating agents | 20% | 44% (step 90) | 42% |
| Four independent attempts | 26% | 29% (step 10) | 24% |

Team first-answer accuracy rose from 13% to 22% and peaked at 27%. Single-agent pass@1 went from 10% to 7.5%. These are results from one training run per setup. The training conditions differ in reward, grouping, and communication, so the comparison does not identify the causal effect of messaging.

- [Experiment notes, configurations, and metric definitions](results/README.md)
- [All checkpoint evaluations as CSV](results/evaluations.csv)
- [Full held-out and training transcripts](https://github.com/alexandertsai/multi-agent-rl/releases/tag/chess-rl-results-v1)

## Reproduce the experiment

Use a Linux machine with eight 80 GB H100s, a compatible CUDA driver, and sufficient host RAM and shared memory. The trainer maps a roughly 102 GB n-gram tensor in `/dev/shm`. Supply local copies of the two model checkpoints used in the runs.

| Checkpoint | Recorded revision | Purpose |
|---|---|---|
| Qwen3.8-Flash-Next-FP8 | `7fb48d708ca6` | vLLM rollout serving |
| Qwen3.8-Flash-Next (BF16) | `812d96e10a0d` | FSDP2 LoRA training |

```bash
source setup.sh
export MODEL_LOCAL_DIR=/path/to/Qwen3.8-Flash-Next-FP8
export BF16_MODEL_DIR=/path/to/Qwen3.8-Flash-Next
python -m rl.ngram_shm "$BF16_MODEL_DIR"
bash rl/serve_pair.sh
```

Once both servers report application startup complete in `runs/vllm-rl-800{0,1}.log`, run the following jobs sequentially.

```bash
bash rl/supervise.sh runs/rl-team4 --preset team4 --steps 115 --eval-every 10
bash rl/supervise.sh runs/rl-single --preset single --groups 16 --steps 115 \
  --eval-every 10 --eval-samples 4 --concurrency 256
```

The separate untrained baseline runs used these commands.

```bash
U=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1
python -m chess_env.run --preset team4 --samples 2 --concurrency 128 --base-url "$U" --out runs/base-team4
python -m chess_env.run --preset team4_nocomm --samples 1 --concurrency 100 --base-url "$U" --out runs/base-team4-nocomm
python -m chess_env.run --preset single --samples 4 --concurrency 400 --base-url "$U" --out runs/base-single
```

The exact splits are in `data/`. They contain 4,000 training puzzles and 100 held-out puzzles from disjoint source games, with equal counts across four rating bands from 2000 to 2800. [Lichess publishes the source puzzles under CC0](https://database.lichess.org/#puzzles). To rebuild a split from the source Parquet shards, run `python -m chess_env.build_data <shards>`.

The GPU-free environment tests run with `python -m unittest tests.test_chess_env` after installing the Python dependencies.
