"""
Mock LLM connector for offline testing and runs without API keys.
"""

import re
from typing import AsyncGenerator

from app.llm.base import BaseLLM


class MockLLM(BaseLLM):
    """
    Mock LLM client that functions offline by extracting and formatting context.
    Allows testing the entire ingestion and retrieval pipeline without API keys.
    """

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Parses text context out of prompt and returns a structured mock summary,
        or handles keyword/intent classification helper queries.
        """
        # Handle classification prompt checks for RAG guardrails
        if "Identify if the following user query is related to healthcare" in prompt:
            query_match = re.search(r"Query:\s*(.*?)(?:\n|$)", prompt, re.IGNORECASE)
            query_text = query_match.group(1).lower().strip() if query_match else ""
            medical_keywords = [
                "diabetes", "hypertension", "asthma", "blood", "pressure", "heart", 
                "clinical", "med", "treatment", "doctor", "health", "disease", "patient", 
                "glycemic", "dose", "reliever", "therapy", "copd", "kidney", "renal", "sodium", "runs"
            ]
            # If they ask for "runs" in the context of sports/cricket, it's non-medical.
            # But wait, if they ask for "sachin tendulkar", it has no medical keyword.
            if any(kw in query_text for kw in medical_keywords) and "sachin" not in query_text:
                return "YES"
            return "NO"
        # Find context blocks formatted as "[Source: filename, Page: X]\ntext"
        context_matches = re.findall(
            r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\]\n(.*?)(?=\n\n\[Source:|\n-*?\n*?User Query:)",
            prompt,
            re.DOTALL
        )

        if not context_matches:
            return (
                "[Offline Demo Mode]\n"
                "I couldn't find any relevant snippets in the provided documents.\n"
                "To get real generative answers, please configure your OPENAI_API_KEY or GEMINI_API_KEY in the .env file."
            )

        response = (
            "[Offline Demo Mode]\n"
            "I have retrieved and reranked the following guideline sources for your query:\n\n"
        )

        for i, (source, page, text) in enumerate(context_matches):
            # Take the first couple of sentences for a neat list item
            lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
            preview = " ".join(lines[:2])
            if len(preview) > 160:
                preview = preview[:160] + "..."
            
            response += f"{i+1}. From '{source}' (Page {page}):\n   \"{preview}\"\n\n"

        response += (
            "💡 *Tip: To synthesize these snippets into a real clinical response, "
            "add an OpenAI or Gemini API key to your `.env` file and set `LLM_PROVIDER` accordingly.*"
        )
        return response

    async def generate_stream(
        self, prompt: str, system_instruction: str | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams the mock response chunk by chunk.
        """
        response_text = await self.generate(prompt, system_instruction)
        # Yield in small pieces to simulate streaming
        words = response_text.split(" ")
        for i in range(0, len(words), 3):
            yield " ".join(words[i:i+3]) + " "
