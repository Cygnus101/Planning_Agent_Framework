"""utils/reward_model.py
────────────────────────────────────────────────────────────────────────────
Combines two signals to judge whether a (sub‑task, agent) pairing is suitable:
1. **LLM reward score** on a 0‑5 scale (≥3 means the agent can in principle do it).
2. **Embedding similarity** against the agent's representative works (anchors in JSON file).

If either check fails for the originally assigned agent, we re‑score *all* agents
and pick the one with the highest combined metric.

Returned plan entries gain:
• similarity   – max cosine similarity to exemplar prompts
• llm_score    – 0‑5 scalar from the LLM
• action       – one of {accept, re-describe, plan-in-detail, replan}
• reassigned_from/name (optional) – updated agent if reassigned.
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.prompts import load_prompt

load_dotenv(find_dotenv())

# ────────────────────────────────────────────────────────────
# Embedding setup
# ────────────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

# Representative works loaded from JSON
REPS_PATH = "utils/representative_works.json"
with open(REPS_PATH, encoding="utf-8") as f:
    representative_works = json.load(f)

embedded_reps: Dict[str, np.ndarray] = {
    agent: model.encode(tasks)
    for agent, tasks in representative_works.items()
}

# ────────────────────────────────────────────────────────────
# Parameters / thresholds (Section 4.3)
# ────────────────────────────────────────────────────────────
SIM_TOO_LOW      = 0.1    # cosine‑sim < 0.1 → plan‑in‑detail
SIM_TOO_HIGH     = 0.95    # cosine‑sim > 0.95 → re‑describe
LLM_PASS_THRESHOLD = 3.0  # 0-5 scale; <3 → replan

# ────────────────────────────────────────────────────────────
# LLM scorer
# ────────────────────────────────────────────────────────────
_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("API_KEY"),
)

_DESC_CACHE: Dict[str, str] | None = None


def _load_agent_descriptions() -> Dict[str, str]:
    """Parse prompts/agent_desc.txt into a {agent: desc} map."""
    global _DESC_CACHE
    if _DESC_CACHE is not None:
        return _DESC_CACHE
    raw = load_prompt("utils/agent_desc.txt")
    mapping: Dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            mapping[key.strip()] = val.strip()
    _DESC_CACHE = mapping
    return mapping


def _llm_score(subtask: str, agent: str) -> float:
    """Ask the LLM to rate from 0–5 how well *agent* can handle *subtask*."""
    desc = _load_agent_descriptions().get(agent, "")
    prompt = (
        "You are a task-allocation evaluator.\n"
        "Rate from 0 (cannot solve) to 5 (perfectly suited) how well the agent can handle the sub-task.\n"
        f"Agent: {agent}\nDescription: {desc}\nSub-Task: {subtask}\n"
        "Respond with a single number."
    )
    resp = _llm.invoke(prompt).content.strip()
    try:
        return float(resp)
    except ValueError:
        return 0.0


def evaluate_tasks(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Augment each task with 'similarity', 'llm_score', and 'action'.

    Actions:
      - 'replan'         if llm_score < LLM_PASS_THRESHOLD
      - 'plan-in-detail' if llm_score ≥ threshold & sim < SIM_TOO_LOW
      - 're-describe'    if llm_score ≥ threshold & sim > SIM_TOO_HIGH
      - 'accept'         if llm_score ≥ threshold and SIM_TOO_LOW ≤ sim ≤ SIM_TOO_HIGH

    If initial agent is unsuitable, reassess all agents and assign the best.
    """
    for task in plan:
        text = task.get("task", "")
        assigned = task.get("name", "")

        # Compute metrics for a given agent
        def metrics(agent: str) -> Tuple[float, float]:
            emb_list = embedded_reps.get(agent)
            sim = 0.0
            if emb_list is not None:
                emb = model.encode([text])[0]
                sim = float(np.max(cosine_similarity([emb], emb_list)))
            return sim, _llm_score(text, agent)

        sim, score = metrics(assigned)

        # Determine initial action
        if score < LLM_PASS_THRESHOLD:
            action = "replan"
        elif sim < SIM_TOO_LOW:
            action = "plan-in-detail"
        elif sim > SIM_TOO_HIGH:
            action = "re-describe"
        else:
            action = "accept"

        # If not replan, allow reassignment
        if action != "replan":
            best_agent, best_sim, best_score = assigned, sim, score
            for agent in embedded_reps:
                if agent == assigned:
                    continue
                s_i, sc_i = metrics(agent)
                if sc_i < LLM_PASS_THRESHOLD:
                    continue
                # prefer agent with sim in [low,high] band or closer to mid-band
                if SIM_TOO_LOW <= s_i <= SIM_TOO_HIGH:
                    compare = abs(s_i - (SIM_TOO_LOW+SIM_TOO_HIGH)/2)
                    best_compare = abs(best_sim - (SIM_TOO_LOW+SIM_TOO_HIGH)/2)
                    if compare < best_compare:
                        best_agent, best_sim, best_score = agent, s_i, sc_i
            if best_agent != assigned:
                task["reassigned_from"] = assigned
                task["name"] = best_agent
                sim, score = best_sim, best_score
                # recalc action after reassignment
                if score < LLM_PASS_THRESHOLD:
                    action = "replan"
                elif sim < SIM_TOO_LOW:
                    action = "plan-in-detail"
                elif sim > SIM_TOO_HIGH:
                    action = "re-describe"
                else:
                    action = "accept"

        task.update({
            "similarity": sim,
            "llm_score": score,
            "action": action,
        })

    return plan
