"""Chat-completions policy for an OpenAI-compatible server (vLLM or SGLang)."""
from dataclasses import dataclass, field

from openai import AsyncOpenAI


@dataclass
class Completion:
    content: str
    reasoning: str | None
    tool_calls: list = field(default_factory=list)  # [{'id', 'name', 'arguments'}]
    finish_reason: str = 'stop'
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Set only by recording policies (RL): {'prompt_ids', 'completion_ids', 'logprobs'}.
    tokens: dict | None = None


class OpenAIPolicy:
    def __init__(self, base_url, model, api_key='EMPTY', reasoning_effort='low', preserve_thinking=True,
                 sampling=None, timeout=1800):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model, self.reasoning_effort = model, reasoning_effort
        self.preserve_thinking = preserve_thinking
        # Qwen3.8-Flash-Next model-card defaults for thinking mode.
        self.sampling = sampling or {'temperature': 1.0, 'top_p': 0.95}
        self.extra = {'top_k': 20, 'min_p': 0.0,
                      'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': preserve_thinking}}

    async def complete(self, messages, tools, max_tokens):
        kwargs = {'tools': tools} if tools else {}
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, reasoning_effort=self.reasoning_effort,
            extra_body=self.extra, **self.sampling, **kwargs)
        choice = response.choices[0]
        message = choice.message
        reasoning = getattr(message, 'reasoning_content', None) or (message.model_extra or {}).get('reasoning')
        calls = [{'id': c.id, 'name': c.function.name, 'arguments': c.function.arguments}
                 for c in message.tool_calls or []]
        return Completion(message.content or '', reasoning, calls, choice.finish_reason,
                          response.usage.prompt_tokens, response.usage.completion_tokens)

    def assistant_message(self, completion):
        message = {'role': 'assistant', 'content': completion.content}
        if completion.tool_calls:
            message['tool_calls'] = [{'id': c['id'], 'type': 'function',
                                      'function': {'name': c['name'], 'arguments': c['arguments']}}
                                     for c in completion.tool_calls]
        if self.preserve_thinking and completion.reasoning:
            message['reasoning_content'] = completion.reasoning
        return message
