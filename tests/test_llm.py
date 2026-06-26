"""
Smoke test to check connection to OpenAI and Gemini LLM providers.

Run:
    .\\venv\\Scripts\\python tests\\test_llm.py
"""

import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.config import settings
from app.llm import OpenAILLM, GeminiLLM


async def test_openai() -> None:
    if not settings.OPENAI_API_KEY:
        print("Skipping OpenAI test: OPENAI_API_KEY is not defined in .env")
        return

    print("Testing OpenAI Client ('gpt-4o-mini')...")
    llm = OpenAILLM(api_key=settings.OPENAI_API_KEY)
    
    prompt = "Reply with exactly one word: 'Success'"
    system_instruction = "You are a helpful assistant."
    
    print("  -> Generating text...")
    response = await llm.generate(prompt, system_instruction=system_instruction)
    print(f"  -> Response: '{response.strip()}'")

    print("  -> Generating stream...")
    print("  -> Stream response: ", end="", flush=True)
    async for chunk in llm.generate_stream(prompt, system_instruction=system_instruction):
        print(chunk, end="", flush=True)
    print("\nOpenAI test completed successfully!\n")


async def test_gemini() -> None:
    if not settings.GEMINI_API_KEY:
        print("Skipping Gemini test: GEMINI_API_KEY is not defined in .env")
        return

    print("Testing Gemini Client ('gemini-1.5-flash')...")
    llm = GeminiLLM(api_key=settings.GEMINI_API_KEY)
    
    prompt = "Reply with exactly one word: 'Success'"
    system_instruction = "You are a helpful assistant."
    
    print("  -> Generating text...")
    response = await llm.generate(prompt, system_instruction=system_instruction)
    print(f"  -> Response: '{response.strip()}'")

    print("  -> Generating stream...")
    print("  -> Stream response: ", end="", flush=True)
    async for chunk in llm.generate_stream(prompt, system_instruction=system_instruction):
        print(chunk, end="", flush=True)
    print("\nGemini test completed successfully!\n")


async def main() -> None:
    print(f"Configured LLM_PROVIDER: {settings.LLM_PROVIDER}")
    print("Starting LLM client checks...\n")
    
    if settings.LLM_PROVIDER == "openai":
        await test_openai()
    elif settings.LLM_PROVIDER == "gemini":
        await test_gemini()
    else:
        print(f"Unknown provider '{settings.LLM_PROVIDER}'")
        
    # Also run the other one if keys are present
    if settings.LLM_PROVIDER != "openai":
        await test_openai()
    if settings.LLM_PROVIDER != "gemini":
        await test_gemini()


if __name__ == "__main__":
    asyncio.run(main())
