"""Rollouts that record the exact sampled tokens, and their conversion to training sequences.

Each turn keeps vLLM's prompt token ids, generated token ids and per-token
sampler logprobs. An agent's turns are merged into one sequence whenever the
next prompt starts with the previous prompt plus its completion, so tool
results become context tokens (not trained) and every sampled token is trained
exactly once.
"""
import asyncio
from dataclasses import dataclass
import itertools

import httpx

from chess_env.policy import Completion, OpenAIPolicy


class RecordingPolicy(OpenAIPolicy):
    """OpenAIPolicy that also returns token ids and sampler logprobs."""

    def __init__(self, base_url, model, **kwargs):
        super().__init__(base_url, model, **kwargs)

    async def complete(self, messages, tools, max_tokens):
        kwargs = {'tools': tools} if tools else {}
        extra = dict(self.extra, return_token_ids=True)
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, reasoning_effort=self.reasoning_effort,
            logprobs=True, top_logprobs=0, extra_body=extra, **self.sampling, **kwargs)
        data = response.model_dump()
        choice = data['choices'][0]
        message = response.choices[0].message
        reasoning = getattr(message, 'reasoning_content', None) or (message.model_extra or {}).get('reasoning')
        calls = [{'id': c.id, 'name': c.function.name, 'arguments': c.function.arguments}
                 for c in message.tool_calls or []]
        ids = choice.get('token_ids') or []
        logprobs = [t['logprob'] for t in (choice.get('logprobs') or {}).get('content') or []]
        tokens = None
        if ids and len(ids) == len(logprobs) and data.get('prompt_token_ids'):
            tokens = {'prompt_ids': data['prompt_token_ids'], 'completion_ids': ids, 'logprobs': logprobs}
        return Completion(message.content or '', reasoning, calls, choice['finish_reason'],
                          response.usage.prompt_tokens, response.usage.completion_tokens, tokens)


class Servers:
    """A pool of vLLM servers: one RecordingPolicy per server, plus sleep/wake/adapter control."""

    def __init__(self, urls, model, reasoning_effort='low'):
        self.urls, self.model, self.effort = list(urls), model, reasoning_effort
        self.http = [httpx.AsyncClient(base_url=u, timeout=1800) for u in self.urls]
        self.rotation = itertools.cycle(range(len(self.urls)))
        self.set_model(model)

    def set_model(self, model):
        self.model = model
        self.policies = [RecordingPolicy(f'{u}/v1', model, reasoning_effort=self.effort) for u in self.urls]

    def next_policy(self):
        return self.policies[next(self.rotation)]

    async def _all(self, method, path, **kwargs):
        responses = await asyncio.gather(*(getattr(h, method)(path, **kwargs) for h in self.http))
        for url, r in zip(self.urls, responses):
            if r.status_code >= 400:
                raise RuntimeError(f'{url}{path}: {r.status_code} {r.text[:300]}')
        return responses

    async def sleep(self):
        await self._all('post', '/sleep', params={'level': 1})

    async def wake(self):
        await self._all('post', '/wake_up')

    async def load_adapter(self, name, path):
        # A resumed run may find the adapter still loaded; unload it (errors ignored) so loading is idempotent.
        await asyncio.gather(*(h.post('/v1/unload_lora_adapter', json={'lora_name': name}) for h in self.http))
        await self._all('post', '/v1/load_lora_adapter', json={'lora_name': name, 'lora_path': str(path)})

    async def unload_adapter(self, name):
        await self._all('post', '/v1/unload_lora_adapter', json={'lora_name': name})

    async def ready(self, timeout=1800):
        for _ in range(timeout // 5):
            try:
                await self._all('get', '/v1/models')
                return
            except Exception:
                await asyncio.sleep(5)
        raise TimeoutError('vLLM servers did not come up')


@dataclass
class Sequence:
    ids: list            # full token ids
    targets: list        # positions (into ids) of sampled tokens
    sampler_logprobs: list
    advantage: float
    agent_key: tuple     # (episode index, agent index)


def agent_sequences(trace):
    """Merge one agent's turns into as few prefix-consistent sequences as possible.

    Returns [(ids, target_positions, sampler_logprobs)], and the number of merges that failed."""
    out, current, broken = [], None, 0
    for turn in trace:
        prompt, completion, lps = turn['prompt_ids'], turn['completion_ids'], turn['logprobs']
        if current is not None and prompt[:len(current[0])] == current[0]:
            ids = prompt
        else:
            if current is not None:
                out.append(current)
                broken += 1
            ids = list(prompt)
            current = [None, [], []]
        start = len(ids)
        current[0] = list(ids) + list(completion)
        current[1] += range(start, start + len(completion))
        current[2] += list(lps)
    if current is not None:
        out.append(current)
    return [tuple(s) for s in out], broken
