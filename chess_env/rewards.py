"""Per-agent rewards and group-centred advantages.

r_k = 0.8 T + 0.2 C_k - 0.2 R_k, where T is the team's first answer being
correct, C_k agent k's own answer being correct, and R_k = 1 only for the first
submitter when it is wrong and at least one teammate is correct. With a single
agent T = C_k and R_k = 0, so the reward is plain correctness.
"""
from collections import defaultdict


def agent_rewards(agents, first, config):
    team = first is not None and agents[first - 1]['correct']
    rewards = []
    for agent in agents:
        own = agent['correct']
        rush = (agent['agent'] == first and not own
                and any(a['correct'] for a in agents if a['agent'] != first))
        reward = config.team_weight * team + config.own_weight * own - config.rush_penalty * rush
        rewards.append({'team_term': team, 'rush': rush, 'reward': round(reward, 6)})
    return rewards


def group_advantages(episodes):
    """Centre rewards within each puzzle's group of rollouts (Dr. GRPO, no std
    scaling) and drop groups whose rollouts all got the same reward (DAPO).
    Returns (episode index, agent index, advantage) triples."""
    groups = defaultdict(list)
    for i, episode in enumerate(episodes):
        for j, agent in enumerate(episode['agents']):
            groups[episode['puzzle']['id']].append((i, j, agent['reward']))
    out = []
    for rollouts in groups.values():
        rewards = [r for *_, r in rollouts]
        if max(rewards) - min(rewards) < 1e-9:
            continue
        mean = sum(rewards) / len(rewards)
        out += [(i, j, r - mean) for i, j, r in rollouts]
    return out
