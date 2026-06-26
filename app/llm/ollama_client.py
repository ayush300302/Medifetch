"""
Ollama LLM connector — connects to a local Ollama server running offline.
"""

import json
from typing import AsyncGenerator
import httpx

from app.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """
    Ollama connector for local, offline models.
    """

    def __init__(self, host: str = "http://localhost:11434", model_name: str = "llama3"):
        """
        Initializes the Ollama client.

        Args:
            host: Host URL of the Ollama server (defaults to http://localhost:11434).
            model_name: Name of the model installed in Ollama (defaults to 'llama3').
        """
        self.host = host.rstrip("/")
        self.model_name = model_name

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Generates complete answer for a prompt using Ollama.
        """
        url = f"{self.host}/api/chat"
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {"temperature": 0.0},
            "stream": False
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")
            except Exception as exc:
                raise RuntimeError(f"Ollama generation failed: {exc}")

    async def generate_stream(
        self, prompt: str, system_instruction: str | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams answer chunks for a prompt using Ollama.
        """
        url = f"{self.host}/api/chat"
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {"temperature": 0.0},
            "stream": True
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        chunk_text = data.get("message", {}).get("content", "")
                        if chunk_text:
                            yield chunk_text
            except Exception as exc:
                raise RuntimeError(f"Ollama stream failed: {exc}")
