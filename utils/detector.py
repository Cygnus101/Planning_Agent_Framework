from __future__ import annotations
import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.prompts import load_prompt

# ensure .env is loaded no matter the working directory
load_dotenv(find_dotenv())

"""
utils/detector.py
────────────────────────────────────────────────────────────────────────────
Evaluate a refined plan for COMPLETENESS and NON-REDUNDANCY.

Public helper
─────────────
run_plan_detector(query: str, plan: list[dict]) -> str
    • `query` – the ORIGINAL user task.
    • `plan`  – the evaluated / possibly refined list of sub-task dicts.

The function loads `prompts/plan_detector_prompt.txt`, inserts the
query and pretty-printed plan, and calls Gemini-2.0-Flash at T=0.

Return value
────────────
A plain-text verdict from the LLM, e.g.

    "The plan satisfies completeness and non-redundancy."

or a critique listing missing info / redundant sub-tasks with suggestions.
"""


"""
utils/detector.py
────────────────────────────────────────────────────────────────────────────
Evaluate a refined plan for COMPLETENESS and NON-REDUNDANCY.

Public helper
─────────────
run_plan_detector(query: str, plan: list[dict]) -> str
    • `query` – the ORIGINAL user task.
    • `plan`  – the evaluated / possibly refined list of sub-task dicts.

The function loads `prompts/plan_detector_prompt.txt`, inserts the
query and pretty-printed plan, and calls Gemini-2.0-Flash at T=0.

Return value
────────────
A plain-text verdict from the LLM, e.g.

    "The plan satisfies completeness and non-redundancy."

or a critique listing missing info / redundant sub-tasks with suggestions.
"""




_PROMPT_PATH = "prompts/plan_detector_prompt.txt"

_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("API_KEY"),
)


def run_plan_detector(query: str, plan: List[Dict[str, Any]]) -> str:
    """Return detector feedback for (query, plan)."""
    prompt_template = load_prompt(_PROMPT_PATH)
    filled_prompt = (
        f"{prompt_template}\n"
        f"Task: {query}\n"
        f"Plan: {json.dumps(plan, indent=2)}\n\n"
        "Output:"
    )
    response = _LLM.invoke(filled_prompt).content.strip()
    return response
