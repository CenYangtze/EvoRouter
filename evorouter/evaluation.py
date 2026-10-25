"""BudgetBench item loading and per-item evaluation."""

from __future__ import annotations
from copy import deepcopy
import time
from typing import Any
from evorouter.runtime_helpers import (
    budgetbench_query,
    has_invalid_structure,
    is_provider_exception,
    reset_cost_counters,
)
from MAR.Benchmarks.budgetbench import evaluate_budgetbench_episode
from MAR.Utils.globals import CompletionTokens, Cost, PromptTokens


def coordinator_step_record(step) -> dict[str, Any]:
    return {
        "request_type": step.request_type,
        "messages": deepcopy(step.messages),
        "response_text": step.response_text,
        "max_tokens": step.max_tokens,
        "temperature": step.temperature,
        "accepted": step.accepted,
        "rejection_reason": step.rejection_reason,
    }


def evaluate_item(
    position, item, policy, model_pool, args, router_factory
) -> dict[str, Any]:
    reset_cost_counters()
    started = time.monotonic()
    query = (
        budgetbench_query(item, ignore_budget=True)
        if args.ignore_budget
        else str(item.get("query") or "").strip()
    )
    router = router_factory(policy, model_pool, item, args)
    router.set_episode_assets(
        item.get("worker_assets")
        or (item.get("budgetbench_task") or {}).get("assets")
        or [],
        asset_root=item.get("budgetbench_asset_root") or item.get("budgetbench_root"),
    )
    all_events = []
    attempts = []
    result: dict[str, Any] = {}
    invalid_structure = True
    error = ""

    format_retries_used = 0
    provider_retries_used = 0
    provider_failure = False
    while True:
        attempt_index = len(attempts)
        try:
            result = router.run(query)
            attempt_events = deepcopy(list(getattr(router, "_events", []) or []))
            all_events.extend(attempt_events)
            quality = deepcopy(result.get("quality") or {})
            invalid_structure = has_invalid_structure(quality)
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "invalid_structure": invalid_structure,
                    "quality": quality,
                    "final_answer": str(result.get("final_answer") or ""),
                    "final_plan": deepcopy(result.get("final_plan") or {}),
                    "history": deepcopy(result.get("history") or []),
                    "coordinator_steps": [
                        coordinator_step_record(step)
                        for step in getattr(router, "coordinator_steps", [])
                    ],
                }
            )
        except Exception as exc:
            all_events.extend(deepcopy(list(getattr(router, "_events", []) or [])))
            provider_failure = is_provider_exception(exc)
            error = type(exc).__name__
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "invalid_structure": False,
                    "provider_failure": provider_failure,
                    "error": error,
                    "coordinator_steps": [
                        coordinator_step_record(step)
                        for step in getattr(router, "coordinator_steps", [])
                    ],
                }
            )
            if provider_failure and provider_retries_used < getattr(
                args, "provider_retries", 2
            ):
                provider_retries_used += 1
                error = ""
                provider_failure = False
                continue
            invalid_structure = False
            break
        if not invalid_structure or format_retries_used >= args.format_retries:
            break
        format_retries_used += 1

    if invalid_structure and not error and len(attempts) >= args.format_retries + 1:
        error = "invalid_structure_after_format_retries"
    prediction = str(result.get("final_answer") or "") if not error else ""
    evaluation = evaluate_budgetbench_episode(
        task=item["budgetbench_task"],
        condition=item["budgetbench_condition"],
        prediction=prediction,
        events=all_events,
        release_root=item["budgetbench_root"],
    )
    task_success = bool(
        evaluation["correct"]
        and evaluation["terminated"]
        and not invalid_structure
        and not error
    )
    budget_compliant = bool(evaluation["compliance_at_budget"])
    within_budget_success = bool(task_success and budget_compliant)
    final_plan = result.get("final_plan") if isinstance(result, dict) else {}
    final_plan = final_plan if isinstance(final_plan, dict) else {}
    return {
        "position": position,
        "worker_execution_profile": getattr(args, "worker_execution_profile", "fixed"),
        "id": item["id"],
        "task_id": item["task_id"],
        "condition_id": item["condition_id"],
        "budget_level": item["budget_level"],
        "task_type": item.get("task_type"),
        "query": query,
        "answer": item["answer"],
        "prediction": prediction,
        "task_success": task_success,
        "budget_compliant": budget_compliant,
        "within_budget_success": within_budget_success,
        "invalid_structure": invalid_structure,
        "format_retries_used": format_retries_used,
        "provider_failure": provider_failure,
        "provider_retries_used": provider_retries_used,
        "attempt_count": len(attempts),
        "error": error,
        "worker_cost_units": float(Cost.instance().value),
        "worker_prompt_tokens": int(PromptTokens.instance().value),
        "worker_completion_tokens": int(CompletionTokens.instance().value),
        "weighted_worker_tokens": float(evaluation["weighted_tokens"]),
        "weighted_token_limit": float(evaluation["weighted_token_limit"]),
        "worker_call_count": int(evaluation["worker_call_count"]),
        "collaboration": final_plan.get("collaboration", ""),
        "rounds": final_plan.get("rounds", 0),
        "planned_agent_count": len(final_plan.get("agents") or []),
        "model_pool_order": [entry["Name"] for entry in model_pool],
        "evaluation": evaluation,
        "attempts": attempts,
        "events": all_events,
        "elapsed_sec": time.monotonic() - started,
    }
