"""
OpenAI LLM connector using the async OpenAI client.
"""

from typing import AsyncGenerator
from openai import AsyncOpenAI

from app.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    OpenAI connector for generating chat completions asynchronously.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", base_url: str | None = None):
        """
        Initializes the OpenAI async client.

        Args:
            api_key: OpenAI API key (or compatible service key).
            model_name: Chat completion model name (defaults to 'gpt-4o-mini').
            base_url: Optional custom base URL for API compatibility.
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Generates complete answer for a prompt using chat completions.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        # Set temperature to 0.0 for medical precision and factual consistency
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    async def generate_stream(
        self, prompt: str, system_instruction: str | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams answer chunks for a prompt using chat completions.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response_stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            stream=True,
        )

        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
