import asyncio
import json
import time
import unittest

from chess_env.env import TOOLS, preset, run_episode
from chess_env.policy import Completion
from chess_env.puzzles import from_lichess, grade
from chess_env.rewards import group_advantages
from chess_env.run import summarize

ROW = {'PuzzleId': '0000D', 'GameId': 'F8M8OS71#53', 'FEN': '5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27',
       'Moves': 'd3d6 f8d8 d6d8 f6d8', 'Rating': 1529, 'Themes': ['advantage']}
PUZZLE = from_lichess(ROW)  # Black to move after Qd6; the solution is Rd8.


def call(name, **args):
    return {'id': f'{name}-{time.monotonic_ns()}', 'name': name, 'arguments': json.dumps(args)}


def say(text, *calls, finish='stop', tokens=10):
    return Completion(text, 'thinking', list(calls), finish, 100, tokens)


class ScriptedPolicy:
    """Each agent's script is a list of steps; a step is a Completion, or a
    (delay seconds, Completion) pair to model generation time."""

    def __init__(self, scripts):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.seen = {k: [] for k in scripts}

    async def complete(self, messages, tools, max_tokens):
        k = int(messages[0]['content'].split('Agent ')[1].split()[0]) if tools else 1
        self.seen[k].append((list(messages), tools, max_tokens))
        step = self.scripts[k].pop(0)
        delay, completion = step if isinstance(step, tuple) else (0, step)
        await asyncio.sleep(delay)
        return completion

    def assistant_message(self, completion):
        return {'role': 'assistant', 'content': completion.content}


def episode(scripts, name='team4', **overrides):
    policy = ScriptedPolicy(scripts)
    return asyncio.run(run_episode(PUZZLE, policy, preset(name, **overrides))), policy


class PuzzleTests(unittest.TestCase):
    def test_conversion(self):
        self.assertEqual((PUZZLE['side'], PUZZLE['last_move_san'], PUZZLE['solution_san']), ('Black', 'Qd6', 'Rd8'))

    def test_grading(self):
        self.assertTrue(grade(PUZZLE, 'Rd8')['correct'])
        self.assertTrue(grade(PUZZLE, '27... Rd8!')['correct'])
        self.assertTrue(grade(PUZZLE, 'f8d8')['correct'])
        self.assertFalse(grade(PUZZLE, 'Qb5')['correct'])
        self.assertIsNone(grade(PUZZLE, 'Ke2')['move_uci'])  # Illegal.

    def test_any_mate_in_one_is_correct(self):
        mate = {'fen': '6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1', 'solution_uci': 'a1a8'}
        self.assertTrue(grade(mate, 'Ra8#')['correct'])
        self.assertFalse(grade(mate, 'Ra7')['correct'])


class TeamTests(unittest.TestCase):
    def test_messaging_first_answer_and_rush_penalty(self):
        result, policy = episode({
            1: [say('', call('send_message', to='all', content='I like Qb5')), (0.05, say('\\boxed{Qb5}'))],
            2: [say('', call('check_inbox', wait_seconds=5)), (0.2, say('\\boxed{Rd8}'))],
            3: [(0.3, say('\\boxed{Rd8}'))],
            4: [(0.4, say('no idea'))],
        })
        agents = result['agents']
        self.assertEqual(result['first_agent'], 1)
        self.assertFalse(result['team_correct'])
        self.assertTrue(result['any_correct'])
        self.assertIn('Message from Agent 1', policy.seen[2][1][0][-1]['content'])
        self.assertEqual([a['reward'] for a in agents], [-0.2, 0.2, 0.2, 0.0])
        self.assertEqual(agents[3]['stop'], 'no_answer')
        self.assertEqual(policy.seen[1][0][1], TOOLS)

    def test_messages_reach_recipients_on_any_tool_call(self):
        _, policy = episode({
            1: [say('', call('send_message', to=['2'], content='hi')), (0.2, say('\\boxed{Rd8}'))],
            2: [(0.05, say('', call('send_message', to=['3'], content='x'))), (0.2, say('\\boxed{Rd8}'))],
            3: [(0.3, say('\\boxed{Rd8}'))], 4: [(0.3, say('\\boxed{Rd8}'))],
        })
        self.assertIn('New messages:', policy.seen[2][1][0][-1]['content'])

    def test_correct_team_and_bad_tool_input(self):
        result, policy = episode({
            1: [say('', call('send_message', to=['1'], content='me'), call('nope')), say('\\boxed{Rd8}')],
            2: [(0.1, say('\\boxed{Qb5}'))], 3: [(0.1, say('\\boxed{Qb5}'))], 4: [(0.1, say('\\boxed{Qb5}'))],
        })
        tool_results = [m['content'] for m in policy.seen[1][1][0] if m['role'] == 'tool']
        self.assertTrue(all(r.startswith('Error') for r in tool_results))
        self.assertTrue(result['team_correct'])
        self.assertEqual([a['reward'] for a in result['agents']], [1.0, 0.8, 0.8, 0.8])

    def test_inbox_wait_ends_when_no_teammate_can_reply(self):
        start = time.monotonic()
        episode({1: [say('', call('check_inbox', wait_seconds=60)), say('\\boxed{Rd8}')],
                 2: [say('\\boxed{Rd8}')], 3: [say('\\boxed{Rd8}')], 4: [say('\\boxed{Rd8}')]})
        self.assertLess(time.monotonic() - start, 5)

    def test_wait_is_capped(self):
        result, _ = episode({1: [say('', call('check_inbox', wait_seconds=600)), say('\\boxed{Rd8}')],
                             2: [(1.5, say('\\boxed{Qb5}'))]}, n_agents=2, max_wait_seconds=0.2)
        self.assertLess(result['agents'][0]['submitted_at'], 1.0)
        self.assertEqual(result['first_agent'], 1)

    def test_truncated_turn_has_no_answer(self):
        result, _ = episode({1: [say('\\boxed{Rd8}', finish='length')]}, 'single')
        self.assertEqual((result['agents'][0]['stop'], result['agents'][0]['answer']), ('turn_tokens', None))

    def test_context_budget(self):
        result, policy = episode({1: [Completion('', None, [call('check_inbox', wait_seconds=0)], 'stop', 30_000, 15_000),
                                      say('\\boxed{Rd8}')], 2: [say('\\boxed{Qb5}')]},
                                 n_agents=2)
        self.assertEqual([s[2] for s in policy.seen[1]], [12_000, 3_000])
        self.assertTrue(result['agents'][0]['correct'])


class SingleTests(unittest.TestCase):
    def test_single_agent_reward_is_correctness(self):
        result, policy = episode({1: [say('\\boxed{Rd8}')]}, 'single')
        self.assertIsNone(policy.seen[1][0][1])
        self.assertNotIn('Agent', policy.seen[1][0][0][0]['content'])
        self.assertEqual(result['agents'][0]['reward'], 1.0)
        result, _ = episode({1: [say('\\boxed{Qb5}')]}, 'single')
        self.assertEqual(result['agents'][0]['reward'], 0.0)

    def test_budget_preset(self):
        _, policy = episode({1: [say('\\boxed{Rd8}')]}, 'single_budget')
        self.assertEqual(policy.seen[1][0][2], 48_000)


class TrainingTests(unittest.TestCase):
    def test_group_advantages_and_summary(self):
        a, _ = episode({1: [say('\\boxed{Rd8}')]}, 'single')
        b, _ = episode({1: [say('\\boxed{Qb5}')]}, 'single')
        same = dict(a, puzzle=dict(PUZZLE, id='other'))
        self.assertEqual(group_advantages([a, b, same]), [(0, 0, 0.5), (1, 0, -0.5)])
        summary = summarize([a, b])
        self.assertEqual((summary['first_answer_acc'], summary['episode_pass@1']), (0.5, 0.5))


if __name__ == '__main__':
    unittest.main()
