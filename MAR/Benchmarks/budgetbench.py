"""BudgetBench grading and Worker token accounting."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def grade_budgetbench_prediction(
    task: Mapping[str, Any],
    prediction: Any,
    release_root: str | Path,
) -> dict[str, Any]:
    root = Path(release_root).expanduser().resolve()
    package_root = root / "src"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from budgetbench.graders import grade_task

    return grade_task(
        task, prediction, allow_code_execution=False, asset_root=str(root)
    )


def worker_token_ledger(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Account for Worker inference only with the BudgetBench v3 FLOPs proxy."""

    calls: list[dict[str, Any]] = []
    weighted_tokens = 0.0
    for event in events:
        if str(event.get("type")) != "llm_call":
            continue
        payload = (
            event.get("payload") if isinstance(event.get("payload"), Mapping) else event
        )
        if str(payload.get("request_type")) != "agent":
            continue
        usage = (
            payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        )
        context = (
            payload.get("trace_context")
            if isinstance(payload.get("trace_context"), Mapping)
            else {}
        )
        model = str(context.get("configured_model") or payload.get("model") or "")
        input_tokens = max(0, int(usage.get("input_tokens", 0) or 0))
        reasoning_tokens = max(0, int(usage.get("reasoning_tokens", 0) or 0))
        output_tokens = max(0, int(usage.get("output_tokens", 0) or 0))
        raw_coefficient = context.get("cost_coefficient")
        if raw_coefficient is None:
            raise ValueError(
                f"Worker call for {model!r} is missing its FLOPs cost coefficient"
            )
        coefficient = float(raw_coefficient)
        accounting = str(context.get("cost_accounting") or "").strip().lower()
        if accounting != "flops":
            raise ValueError(
                f"BudgetBench v3 supports FLOPs accounting only; got {accounting!r} for {model!r}"
            )
        weighted = coefficient * (input_tokens + output_tokens) / 3.0
        formula = "(input_tokens + output_tokens) * coefficient / 3"
        weighted_tokens += weighted
        calls.append(
            {
                "agent_id": str(context.get("agent_id") or ""),
                "model": model,
                "coefficient": coefficient,
                "cost_accounting": accounting,
                "cost_formula": formula,
                "input_tokens": input_tokens,
                "reasoning_tokens": reasoning_tokens,
                "output_tokens": output_tokens,
                "weighted_tokens": weighted,
            }
        )
    return {
        "worker_calls": calls,
        "worker_call_count": len(calls),
        "cost_accounting": "flops" if calls else "none",
        "cost_accounting_modes": ["flops"] if calls else [],
        "weighted_tokens": weighted_tokens,
    }


def evaluate_budgetbench_episode(
    *,
    task: Mapping[str, Any],
    condition: Mapping[str, Any],
    prediction: Any,
    events: Sequence[Mapping[str, Any]],
    release_root: str | Path,
) -> dict[str, Any]:
    grade = grade_budgetbench_prediction(task, prediction, release_root)
    ledger = worker_token_ledger(events)
    used = float(ledger["weighted_tokens"])
    limit = float(condition["budget"]["weighted_tokens"])
    correct = bool(grade.get("success"))
    terminated = bool(str(prediction or "").strip())
    within_budget = used <= limit + 1e-12 * max(1.0, abs(limit))
    if correct and terminated:
        budgeted_quality = 1.0 if within_budget else max(0.3, limit / max(used, 1e-12))
    else:
        budgeted_quality = 0.0
    return {
        "correct": correct,
        "terminated": terminated,
        "compliance_at_budget": within_budget,
        "success_at_budget": bool(correct and terminated and within_budget),
        "budgeted_quality": float(budgeted_quality),
        "weighted_tokens": used,
        "weighted_token_limit": limit,
        "over_budget_tokens": max(0.0, used - limit),
        "grade": grade,
        **ledger,
    }
