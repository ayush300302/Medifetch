"""
Google Gemini LLM connector using the google-generativeai client library.
"""

import asyncio
from typing import AsyncGenerator
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    """
    Google Gemini connector for generating completions asynchronously.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initializes the Gemini API settings.

        Args:
            api_key: Gemini API key.
            model_name: Gemini model name (defaults to 'gemini-1.5-flash').
        """
        genai.configure(api_key=api_key)
        self.model_name = model_name

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Generates complete answer for a prompt using Gemini.
        """
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
        )
        
        # Use temperature 0.0 for deterministic factual responses
        config = GenerationConfig(temperature=0.0)
        
        # Use asyncio.to_thread to run blocking synchronous call in a separate thread
        # to prevent async library hanging on Windows / certain environments.
        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=config
        )
        return response.text or ""

    async def generate_stream(
        self, prompt: str, system_instruction: str | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams answer chunks for a prompt using Gemini.
        """
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction,
        )
        
        config = GenerationConfig(temperature=0.0)
        
        # Run blocking stream generation inside a separate thread
        def sync_stream():
            return model.generate_content(
                prompt,
                stream=True,
                generation_config=config
            )

        response = await asyncio.to_thread(sync_stream)
        for chunk in response:
            if chunk.text:
                yield chunk.text
