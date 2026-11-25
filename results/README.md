# Experiment results

## Conditions

Both RL runs update a rank-16, alpha-32 LoRA on attention projections and shared experts. The model sees the position after the opponent's setup move and submits the first solution move. The scorer accepts SAN or UCI and any legal mate in one.

| Setting | Team | Single agent |
|---|---:|---:|
| Agents per episode | 4 | 1 |
| Puzzles per training update | 16 | 16 |
| Episodes per training puzzle | 4 | 16 |
| Agent trajectories per update before filtering | 256 | 256 |
| Evaluation episodes per puzzle | 1 | 4 |
| Episode concurrency | 128 | 256 |
| Communication tools | Send message, check inbox | Disabled |
| Reward | 0.8 × team correctness + 0.2 × own correctness − 0.2 × rush penalty | Own correctness |

Both used seed 0, 115 updates, a 12,000-token per-turn limit, a 48,000-token context limit, low reasoning effort, and evaluation every 10 steps plus the final checkpoint. The first submitted answer is the team answer. Four independent attempts use the same held-out puzzle with separate episodes.

The trainer uses token-level REINFORCE, within-puzzle reward centering, and zero-variance group filtering. AdamW uses a learning rate of 1e-5, betas (0.9, 0.95), and gradient clipping at 1.0. The sampler/trainer importance ratio is capped at 2, and the loss uses constant normalization of 8,192 tokens per sampled trajectory. Two TP4 vLLM servers generate rollouts with FP8 weights. They sleep during BF16 FSDP2 updates on the same GPUs and reload the new adapter before the next rollout. Prefix caching and MTP speculative decoding were disabled.

## Interpreting the results

- The team reached 27% first-answer accuracy and 44% team pass@4 at step 90. It finished at 22% and 42%, respectively. The share of team agents without a submitted answer fell from 68.25% to 2.0%.
- Single-agent pass@4 peaked at 29% at step 10, fell to 4% at step 70, and recovered to 24% at step 115. Average evaluation tokens per attempt fell from 8,414 at step 0 to 887 at step 115. The token decline coincided with the accuracy decline and faster late training steps. The run does not establish why behavior changed.
- The untrained no-communication team scored 24% team pass@4. There is no trained no-communication control, no token-matched RL comparison, and one training seed per setup.
- The team was evaluated three times each at steps 10 and 40 after restarts. For example, step 40 pass@4 ranged from 31% to 42% across attempts. The raw log and CSV retain all attempts. The transcript files retain the last saved attempt at those steps.

## Files

- [evaluations.csv](evaluations.csv) contains one row per recorded RL evaluation attempt. Metric columns ending in `accuracy`, `rate`, or `pass_at_4` contain fractions. For the team, pass@4 means any correct agent within a team. For the single-agent run, it means any correct attempt across four independent episodes.
- Each `rl-*` directory contains its run settings, full evaluation summaries, per-update training metrics, progress logs, and supervisor logs. Training metrics use zero-based update indices while checkpoint labels count completed updates. Retried updates remain in the logs. Machine-local paths in settings were replaced with portable equivalents.
- Each `base-*` directory contains the configuration and summary for a separate untrained baseline. RL step 0 is a separate evaluation. The untrained team used two episodes per puzzle, and its `team_pass@4` is a per-episode metric.
- [The transcript release](https://github.com/alexandertsai/multi-agent-rl/releases/tag/chess-rl-results-v1) contains the saved held-out transcripts at every checkpoint, all recorded training rollouts, and the three untrained baseline transcript sets. Each archive extracts under `runs/`. Verify downloads against [the SHA-256 checksums](transcript-sha256sums.txt). It includes transcripts saved by the experiment, which omit token IDs and sampler log probabilities.
