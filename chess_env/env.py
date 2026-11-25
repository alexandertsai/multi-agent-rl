"""Concurrent N-agent chess puzzle episodes with messaging tools.

Rules follow "Simple multi-agent RL turns pass@k into pass@1":
- every agent gets the same puzzle, prompt and tools; there is no leader;
- an agent submits by ending a turn without a tool call, answering in \\boxed{};
- submitting is final, and messages sent to a stopped agent are never read;
- the team's answer is the first submitted \\boxed{} answer;
- an agent that stops without a \\boxed{} answer has no answer.
With one agent and no tools this is the single-agent baseline.
"""
import asyncio
from dataclasses import asdict, dataclass, replace
import json
import time

from .puzzles import diagram, extract_boxed, grade
from .rewards import agent_rewards


@dataclass(frozen=True)
class Config:
    n_agents: int = 4
    tools: bool = True
    max_turns: int = 50
    max_turn_tokens: int = 12_000
    max_context: int = 48_000
    max_wait_seconds: float = 120.0
    team_weight: float = 0.8
    own_weight: float = 0.2
    rush_penalty: float = 0.2


PRESETS = {
    'team4': Config(),
    # Four isolated agents, team answer = first submission: the "no communication" ablation.
    'team4_nocomm': Config(tools=False),
    'single': Config(n_agents=1, tools=False),
    # One agent given a four-agent team's token budget, the baseline the article left open.
    'single_budget': Config(n_agents=1, tools=False, max_turn_tokens=48_000, max_context=192_000),
}

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'send_message',
        'description': 'Send a message to teammates. Each recipient sees it the next time it calls a tool.',
        'parameters': {'type': 'object', 'required': ['to', 'content'], 'properties': {
            'to': {'description': '"all", or a list of agent numbers such as ["2", "3"].',
                   'anyOf': [{'type': 'string', 'enum': ['all']},
                             {'type': 'array', 'items': {'type': 'string'}}]},
            'content': {'type': 'string', 'description': 'The message text.'}}}}},
    {'type': 'function', 'function': {
        'name': 'check_inbox',
        'description': 'Return unread messages. If there are none, wait up to wait_seconds and return '
                       'as soon as a message arrives.',
        'parameters': {'type': 'object', 'required': ['wait_seconds'], 'properties': {
            'wait_seconds': {'type': 'number', 'description': 'Maximum seconds to wait for a message.'}}}}},
]


def prompt(puzzle, k, config):
    board = (f"{puzzle['side']} to move. {'Black' if puzzle['side'] == 'White' else 'White'} just played "
             f"{puzzle['last_move_san']}.\n\nFEN: {puzzle['fen']}\n\n{diagram(puzzle['fen'])}\n\n"
             'Uppercase letters are White pieces and lowercase letters are Black pieces.')
    task = (f"Find the best move for {puzzle['side']} in this chess puzzle. Give only the first move of the "
            'solution, in SAN, inside \\boxed{}, for example \\boxed{Nf3}.')
    if not config.tools:
        return f'{task}\n\n{board}'
    return (f'You are Agent {k} on a team of {config.n_agents} agents (numbered 1 to {config.n_agents}). '
            'Every agent has the same puzzle and the same tools, and there is no leader.\n\n'
            f'{task}\n\n{board}\n\n'
            'You can talk to teammates with send_message and check_inbox. You submit by ending a turn '
            'without a tool call; your answer is the move inside \\boxed{} in that turn. Submitting is final: '
            'you stop, and messages sent to you afterwards are never read. The team\'s answer is the first '
            'answer any agent submits.')


@dataclass
class Message:
    sender: int
    content: str
    sent_at: float


class Team:
    def __init__(self, puzzle, policy, config):
        self.puzzle, self.policy, self.config = puzzle, policy, config
        self.inbox = {k: [] for k in range(1, config.n_agents + 1)}
        self.wake = {k: asyncio.Event() for k in self.inbox}
        self.active = set(self.inbox)
        self.submissions = []  # (agent, answer) in submission order
        self.log = []
        self.start = time.monotonic()

    def now(self):
        return round(time.monotonic() - self.start, 2)

    def deliver(self, k):
        if not self.inbox[k]:
            return ''
        text = '\n\n'.join(f'Message from Agent {m.sender} at {m.sent_at}s:\n{m.content}' for m in self.inbox[k])
        self.inbox[k].clear()
        self.wake[k].clear()
        return text

    def send_message(self, k, to, content):
        if isinstance(to, (int, str)) and str(to).strip().isdigit():
            to = [str(to)]
        if to == 'all':
            targets = [j for j in self.inbox if j != k]
        elif isinstance(to, list) and to and all(str(j).strip().isdigit() for j in to):
            targets = sorted({int(str(j).strip()) for j in to})
            if any(j not in self.inbox or j == k for j in targets):
                return f'Error: recipients must be teammates numbered 1 to {self.config.n_agents}, not yourself.'
        else:
            return 'Error: "to" must be "all" or a list of agent numbers such as ["2", "3"].'
        if not isinstance(content, str) or not content.strip():
            return 'Error: "content" must be non-empty text.'
        message = Message(k, content, self.now())
        self.log.append({'from': k, 'to': targets, 'content': content, 'at': message.sent_at})
        for j in targets:
            if j in self.active:  # Messages to stopped agents are never read.
                self.inbox[j].append(message)
                self.wake[j].set()
        return f"Message sent to {', '.join(f'Agent {j}' for j in targets)}."

    async def check_inbox(self, k, wait_seconds):
        try:
            wait = min(max(float(wait_seconds), 0.0), self.config.max_wait_seconds)
        except (TypeError, ValueError):
            return 'Error: "wait_seconds" must be a number.'
        if not self.inbox[k] and self.active - {k}:
            try:
                await asyncio.wait_for(self.wake[k].wait(), wait)
            except asyncio.TimeoutError:
                pass
        return self.deliver(k) or 'No new messages.'

    def finish(self, k, answer):
        self.active.discard(k)
        if answer is not None:
            self.submissions.append((k, answer))
        # Waiters whose last possible sender just stopped return immediately.
        for j in self.active:
            if not self.active - {j}:
                self.wake[j].set()

    async def call_tool(self, k, call):
        try:
            args = json.loads(call['arguments'] or '{}')
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            return 'Error: tool arguments must be a JSON object.'
        if call['name'] == 'send_message':
            result = self.send_message(k, args.get('to'), args.get('content'))
        elif call['name'] == 'check_inbox':
            return await self.check_inbox(k, args.get('wait_seconds', 0))
        else:
            result = f"Error: unknown tool {call['name']!r}."
        unread = self.deliver(k)
        return f'{result}\n\nNew messages:\n\n{unread}' if unread else result

    async def run_agent(self, k):
        config = self.config
        messages = [{'role': 'user', 'content': prompt(self.puzzle, k, config)}]
        stats = {'agent': k, 'turns': 0, 'messages_sent': 0, 'inbox_checks': 0, 'completion_tokens': 0,
                 'context_tokens': 0, 'answer': None, 'stop': 'max_turns', 'submitted_at': None}
        try:
            for _ in range(config.max_turns):
                budget = min(config.max_turn_tokens, config.max_context - stats['context_tokens'])
                if budget <= 0:
                    stats['stop'] = 'context'
                    break
                completion = await self.policy.complete(messages, TOOLS if config.tools else None, budget)
                stats['turns'] += 1
                stats['completion_tokens'] += completion.completion_tokens
                stats['context_tokens'] = completion.prompt_tokens + completion.completion_tokens
                messages.append(self.policy.assistant_message(completion))
                if completion.tokens is not None:
                    stats.setdefault('trace', []).append(completion.tokens)
                if completion.finish_reason == 'length':
                    stats['stop'] = 'turn_tokens'
                    break
                if not completion.tool_calls:
                    stats['answer'] = extract_boxed(completion.content)
                    stats['stop'] = 'submitted' if stats['answer'] is not None else 'no_answer'
                    stats['submitted_at'] = self.now()
                    break
                for call in completion.tool_calls:
                    stats['messages_sent'] += call['name'] == 'send_message'
                    stats['inbox_checks'] += call['name'] == 'check_inbox'
                    result = await self.call_tool(k, call)
                    messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': result})
        except Exception as error:  # A failed agent stops without an answer; teammates continue.
            stats['stop'], stats['error'] = 'error', repr(error)
        finally:
            self.finish(k, stats['answer'])
        return dict(stats, messages=messages)

    async def run(self):
        agents = await asyncio.gather(*(self.run_agent(k) for k in self.inbox))
        for agent in agents:
            agent.update(grade(self.puzzle, agent['answer']))
        first = self.submissions[0][0] if self.submissions else None
        rewards = agent_rewards(agents, first, self.config)
        for agent, reward in zip(agents, rewards):
            agent.update(reward)
        team_correct = first is not None and agents[first - 1]['correct']
        return {'puzzle': self.puzzle, 'config': asdict(self.config), 'first_agent': first,
                'team_answer': agents[first - 1]['answer'] if first else None, 'team_correct': team_correct,
                'any_correct': any(a['correct'] for a in agents), 'seconds': self.now(),
                'message_log': self.log, 'agents': agents}


async def run_episode(puzzle, policy, config):
    return await Team(puzzle, policy, config).run()


def preset(name, **overrides):
    return replace(PRESETS[name], **overrides)
