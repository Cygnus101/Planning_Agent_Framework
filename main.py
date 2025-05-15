# main.py  (excerpt – replace your old main)
import json, os
from planner.meta_agent import (
    fast_decompose_and_allocate,
    fast_replan,
    fast_plan_in_detail,
    fast_redescribe,
    DEFAULT_DECOMPOSE_PROMPT,
    DEFAULT_DETAIL_PROMPT,
    DEFAULT_REPLAN_PROMPT,
    DEFAULT_REDESCRIBE_PROMPT
)
from utils.reward_model import evaluate_tasks
from utils.detector import run_plan_detector
from agents.search_agent import search
from agents.math_agent import math_agent
from agents.code_agent import code_agent
from agents.commonsense_agent import commonsense_agent


MAX_RM_PASSES = 3      # reward-model / refinement iterations
MAX_DETECT_PASSES = 3  # detector iterations


def main() -> None:
    user_query = input("Enter your task query: ")

    # ── 1. initial meta-agent decomposition ──────────────────────────────
    plan = fast_decompose_and_allocate(user_query)

    # ── 2. reward-model refinement loop ──────────────────────────────────
    for rm_round in range(MAX_RM_PASSES):
        plan = evaluate_tasks(plan)
        needs_fix = False

        i = 0
        while i < len(plan):
            action = plan[i].get("action", "accept")
            subtask = plan[i]["task"]

            if action == "replan":
                plan[i:i + 1] = fast_replan(subtask, prompt_path=DEFAULT_REPLAN_PROMPT)
                needs_fix = True
                continue

            if action == "plan-in-detail":
                plan[i:i + 1] = fast_plan_in_detail(
                    subtask, prompt_path=DEFAULT_DETAIL_PROMPT
                )
                needs_fix = True
                continue

            if action == "re-describe":
                plan[i]["task"] = fast_redescribe(
                    subtask, prompt_path=DEFAULT_REDESCRIBE_PROMPT
                )
                plan[i]["action"] = "accept"
                needs_fix = True

            i += 1

        if not needs_fix:
            break  # all tasks accepted
    
    with open("outputs/evaluated_plan.txt", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    # ── 3. detector loop ─────────────────────────────────────────────────
    for det_round in range(MAX_DETECT_PASSES):
        feedback = run_plan_detector(user_query, plan)
        if "satisfies completeness" in feedback.lower():
            print("\nDetector PASS")
            break

        print("\nDetector flagged issues – re-planning…")
        # Simple strategy: full re-plan using feedback as context
        plan = fast_replan(
            subtask=user_query + "\n\n# Detector feedback:\n" + feedback,
            prompt_path=DEFAULT_REPLAN_PROMPT,
        )
        # re-run reward model once to tag actions for new plan
        plan = evaluate_tasks(plan)
    else:
        print("Detector did not pass after retries; proceeding anyway.")

    # ── 4. execute accepted tasks ────────────────────────────────────────
    print("\n=== Executing Accepted Tasks ===")
    results = []

    with open("outputs/detected_plan.txt", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    for t in plan:
        if t.get("action") != "accept":
            continue
        agent, job = t["name"], t["task"]
        if agent == "search_agent":
            out = search(job)
        elif agent == "math_agent":
            out = math_agent(job)
        elif agent == "code_agent":
            out = code_agent(job)
        elif agent == "commonsense_agent":
            out = commonsense_agent(job)
        else:
            out = f"[no handler for {agent}]"
        print(f"[{agent}] {job} -> {out}")
        results.append({"agent": agent, "task": job, "output": out})

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to outputs/final_results.json")


if __name__ == "__main__":
    main()


