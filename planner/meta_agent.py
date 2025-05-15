"""planner/meta_agent.py
------------------------------------------------
Meta‑agent implementation reverted to **raw LLM.invoke()** (no LangChain Agent).
This avoids extra tool chatter and keeps parsing simple. Tavily search has been
removed for now; if you need web results you can add a dedicated search sub‑task
instead.

Public helpers:
• fast_decompose_and_allocate
• fast_replan
• fast_plan_in_detail
• fast_redescribe

Each builds a prompt, calls Gemini 2.5 with `LLM.invoke()`, and extracts JSON.
"""
from __future__ import annotations

import os
import json
import re
from typing import List, Any

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.prompts import load_prompt

load_dotenv(find_dotenv())  # ensure .env is loaded regardless of cwd

# ────────────────────────────────────────────────────────────
# LLM setup (shared) – expects GOOGLE_API_KEY in env
# ────────────────────────────────────────────────────────────
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# ────────────────────────────────────────────────────────────
# Utility regexes for robust JSON extraction
# ────────────────────────────────────────────────────────────
FENCE_RE = re.compile(r"```json(.*?)```", re.DOTALL)
BRACE_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> Any:
    """Return first valid JSON object found in text."""
    match = FENCE_RE.search(text) or BRACE_RE.search(text)
    if not match:
        raise ValueError("No JSON found in LLM output")
    segment = match.group(1) if FENCE_RE.search(text) else match.group(0)
    return json.loads(segment)


def _run_llm(prompt: str) -> Any:
    raw = LLM.invoke(prompt).content.strip()
    return _extract_json(raw)

# ────────────────────────────────────────────────────────────
# Prompt paths
# ────────────────────────────────────────────────────────────
DEFAULT_DECOMPOSE_PROMPT = "prompts/meta_agent_prompt.txt"
DEFAULT_REPLAN_PROMPT = "prompts/replan_prompt.txt"
DEFAULT_DETAIL_PROMPT = "prompts/plan_in_detail_prompt.txt"
DEFAULT_REDESCRIBE_PROMPT = "prompts/redescribe_prompt.txt"

# ────────────────────────────────────────────────────────────
# Public helpers
# ────────────────────────────────────────────────────────────

def fast_decompose_and_allocate(user_query: str, *, prompt_path: str = DEFAULT_DECOMPOSE_PROMPT) -> List[dict]:
    template = load_prompt(prompt_path)
    prompt = f"{template}\nUser query: {user_query}\n\nOutput:"
    try:
        return _run_llm(prompt)
    except Exception as e:
        print("[meta‑agent] Decompose failed:", e)
        return []


def fast_replan(subtask: str, *, prompt_path: str = DEFAULT_REPLAN_PROMPT) -> List[dict]:
    template = load_prompt(prompt_path)
    prompt = f"{template}\nTask: {subtask}\n\nOutput:"
    try:
        return _run_llm(prompt)
    except Exception as e:
        print("[meta‑agent] Replan failed:", e)
        return [{"task": subtask, "name": "TODO"}]


def fast_plan_in_detail(subtask: str, *, prompt_path: str = DEFAULT_DETAIL_PROMPT) -> List[dict]:
    template = load_prompt(prompt_path)
    prompt = f"{template}\nTask: {subtask}\n\nOutput:"
    try:
        return _run_llm(prompt)
    except Exception as e:
        print("[meta‑agent] Plan‑in‑detail failed:", e)
        return [{"task": subtask, "name": "TODO"}]


def fast_redescribe(subtask: str, *, prompt_path: str = DEFAULT_REDESCRIBE_PROMPT) -> str:
    template = load_prompt(prompt_path)
    prompt = f"{template}\nTask: {subtask}\n\nOutput:"
    try:
        out = _run_llm(prompt)
        return out if isinstance(out, str) else json.dumps(out)
    except Exception as e:
        print("[meta‑agent] Redescribe failed:", e)
        return subtask


