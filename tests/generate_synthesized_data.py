"""
Uses an LLM (OpenAI or Gemini) to generate rich, synthesized medical guidelines
and compiles them into a multi-page PDF for testing.
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz  # PyMuPDF
from app.llm.openai_client import OpenAILLM
from app.llm.gemini_client import GeminiLLM
from app.utils.config import settings


async def generate_page_text(llm, topic_prompt: str) -> str:
    system_instruction = (
        "You are an expert clinical guideline writer. Generate highly detailed, "
        "evidence-based medical practice guidelines. Do not include markdown formatting "
        "like bold asterisks or blockquotes in your response. Output clean, plain text with "
        "clear, uppercase headers, list items, and tabular-looking text blocks."
    )
    
    print("Querying LLM to generate guidelines...")
    try:
        content = await llm.generate(topic_prompt, system_instruction=system_instruction)
        return content.strip()
    except Exception as exc:
        print(f"Error during LLM generation: {exc}")
        sys.exit(1)


def write_to_pdf(output_path: str, pages_content: list) -> None:
    doc = fitz.open()

    for idx, page_text in enumerate(pages_content):
        page = doc.new_page()
        # Insert text with a generous margin to prevent clipping
        page.insert_text((50, 60), page_text, fontsize=9.5, lineheight=13)
        print(f"Successfully wrote page {idx+1} to PDF.")

    doc.save(output_path)
    doc.close()
    print(f"Saved synthesized PDF to: {output_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthesized medical guidelines PDF using an LLM.")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default=None,
        help="LLM provider to use (overrides config/environment)."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API Key for the selected provider (overrides config/environment)."
    )
    parser.add_argument(
        "--output",
        default="data/sample_medical_guidelines.pdf",
        help="Path where the generated PDF will be saved."
    )
    args = parser.parse_args()

    # Determine LLM Provider
    provider = args.provider or settings.LLM_PROVIDER
    provider = provider.lower().strip()

    # Determine API Key
    api_key = args.api_key
    if not api_key:
        if provider == "openai":
            api_key = settings.OPENAI_API_KEY
        elif provider == "gemini":
            api_key = settings.GEMINI_API_KEY

    if not api_key:
        print(f"\n[ERROR] No API key found for provider '{provider}'.")
        print("Please provide it via one of the following methods:")
        print("  1. Pass it as a command line argument: --api-key YOUR_KEY")
        print("  2. Set the environment variable: OPENAI_API_KEY or GEMINI_API_KEY")
        print("  3. Create a .env file containing the key.")
        sys.exit(1)

    print(f"Initializing LLM client for provider: {provider}...")
    if provider == "openai":
        llm = OpenAILLM(api_key=api_key)
    elif provider == "gemini":
        llm = GeminiLLM(api_key=api_key)
    else:
        print(f"[ERROR] Unsupported provider: {provider}")
        sys.exit(1)

    # Prompt topics
    prompts = [
        # Page 1: Heart Failure
        (
            "Generate a clinical practice guideline for CHRONIC HEART FAILURE (CHF) in adults. "
            "It must look like a formal document page and contain sections on:\n"
            "1. DIAGNOSTIC CRITERIA (EF thresholds e.g., HFrEF <= 40%, HFpEF >= 50%, BNP/NT-proBNP ranges).\n"
            "2. PHARMACOLOGICAL MANAGEMENT (first-line therapy like ACE inhibitors/ARNI, beta-blockers, "
            "SGLT2 inhibitors such as Empagliflozin or Dapagliflozin, and Aldosterone antagonists with exact target doses).\n"
            "3. LIFESTYLE INTERVENTIONS (sodium limit <2g/day, fluid restriction <1.5-2L/day, and daily weight monitoring).\n\n"
            "Do NOT use markdown bolding (**) or italics (*). Make it dense, realistic, and around 300 words."
        ),
        # Page 2: Chronic Kidney Disease
        (
            "Generate a clinical practice guideline for CHRONIC KIDNEY DISEASE (CKD) AND DIABETIC NEPHROPATHY. "
            "It must look like a formal document page and contain sections on:\n"
            "1. STAGING AND CLASSIFICATION (eGFR categories G1 to G5, Albuminuria categories A1 to A3).\n"
            "2. NEPHROPROTECTIVE PHARMACOTHERAPY (first-line treatment using ACE inhibitors or ARBs, "
            "contraindications like renal artery stenosis, and cardiorenal benefits of SGLT2 inhibitors).\n"
            "3. GLYCEMIC AND BLOOD PRESSURE TARGETS (HbA1c target < 7.0%, target BP < 130/80 mmHg, "
            "and dosage adjustments for other diabetic meds based on eGFR ranges e.g. Metformin contraindication at eGFR < 30).\n\n"
            "Do NOT use markdown bolding (**) or italics (*). Make it dense, realistic, and around 300 words."
        ),
        # Page 3: COPD Management
        (
            "Generate a clinical practice guideline for CHRONIC OBSTRUCTIVE PULMONARY DISEASE (COPD). "
            "It must look like a formal document page and contain sections on:\n"
            "1. DIAGNOSIS AND ASSESSMENT (spirometry criteria FEV1/FVC < 0.70 post-bronchodilator, GOLD groups A-D classification).\n"
            "2. STEPWISE THERAPY (relievers, maintenance using LAMA/LABA long-acting bronchodilators, inhaled corticosteroids).\n"
            "3. EXACERBATION MANAGEMENT (indications for systemic corticosteroids like Prednisone 40 mg daily for 5 days, "
            "oxygen target saturation 88-92% for carbon dioxide retainers, and short-acting relievers like Albuterol/Ipratropium).\n\n"
            "Do NOT use markdown bolding (**) or italics (*). Make it dense, realistic, and around 300 words."
        )
    ]

    pages_content = []
    for idx, prompt in enumerate(prompts):
        print(f"\nGenerating content for Page {idx+1}...")
        text = await generate_page_text(llm, prompt)
        pages_content.append(text)

    # Output directory
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nWriting guidelines to PDF...")
    write_to_pdf(str(out_path), pages_content)
    print("\nGeneration complete!")


if __name__ == "__main__":
    asyncio.run(main())
