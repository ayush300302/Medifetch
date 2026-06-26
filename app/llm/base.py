"""
Abstract base class for LLM clients.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseLLM(ABC):
    """
    Abstract interface for LLM connectors.
    Provides standard async generation and streaming interfaces.
    """

    @abstractmethod
    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Generates a text completion for the given prompt.

        Args:
            prompt: User prompt content.
            system_instruction: Optional developer system instruction.

        Returns:
            The model's textual response.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, system_instruction: str | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams a text completion chunk by chunk.

        Args:
            prompt: User prompt content.
            system_instruction: Optional developer system instruction.

        Yields:
            Text chunks of the generated response.
        """
        yield ""
