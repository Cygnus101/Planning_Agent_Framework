"""scripts/generate_representative.py
────────────────────────────────────────────────────────────────────────────
Simple fallback generator that does *not* rely on `langchain_experimental`.
For each agent we ask the LLM to produce N exemplar prompts in one shot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())

LLM = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)
NUM_EXAMPLES = 100
OUTPUT_PATH = Path("utils/representative_works.json")


TEMPLATES: Dict[str, str] = {
    "search_agent": (
        "You are an expert prompt writer. Generate {{n}} distinct, concise factual lookup questions suitable for a web search. "
        "Ensure broad coverage (e.g., geography, science, economics) and avoid repetitive topics such as capitals. "
        "Make sure no two questions share the same core fact or subject. "
        "Vary the style: use interrogative forms, wh-questions, and statements framed as questions. "
    ),
    "math_agent": (
        "List {{n}} unique math problems (arithmetic, algebra, or basic calculus) resolvable without external lookup. "
        "Use varied phrasing: some as word problems, some as symbolic expressions, some as real-world scenarios. "
        "For example: 'If 3x + 5 = 20, solve for x.' and 'A car travels 60 miles in 1.5 hours; what is its average speed?'"
        "Output each question on its own line without numbering."
    ),
    "code_agent": (
        "Generate {{n}} different prompts requesting small Python code snippets for calculations or data processing. "
        "Vary between function definitions, scripts, or inline expressions; include both standard library and pandas examples. "
        "For example: 'Write a Python function to check if a number is prime.' and 'Using pandas, read a CSV and display the first five rows.'"
        "Output each instruction on its own line without numbering."
    ),
    "commonsense_agent": (
        "Provide {{n}} highly varied commonsense questions with obvious answers, covering daily life, nature, and social conventions. "
        "Avoid similar structures or topics—no two questions should address the same scenario. "
        "Use different formats: direct questions, true/false statements, and hypothetical 'what if' prompts. "
        "For example: 'Why do we wear sunscreen at the beach?' and 'What happens if you leave ice out at room temperature?'"
        "Output each question on its own line without numbering."
    ),
}

def generate_for_agent(agent: str, template: str) -> List[str]:
    prompt = template.replace("{{n}}", str(NUM_EXAMPLES))
    resp = LLM.invoke(prompt).content.strip()
    # Split by newline and clean
    lines = [l.strip("- ").strip() for l in resp.splitlines() if l.strip()]
    return lines[:NUM_EXAMPLES]


def main():
    rep: Dict[str, List[str]] = {}
    for agent, tmpl in TEMPLATES.items():
        rep[agent] = generate_for_agent(agent, tmpl)
        print(f"{agent}: {len(rep[agent])} examples")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"Representative works saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
