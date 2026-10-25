"""Shared evaluation helpers for the Coordinator and fixed baselines."""

from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens


def reset_cost_counters():
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()


def budgetbench_query(item, *, ignore_budget=False):
    """Build the Coordinator query with the configured budget visibility."""

    if ignore_budget:
        task = item.get("budgetbench_task") if isinstance(item, dict) else None
        instruction = str((task or {}).get("instruction") or "").strip()
        if not instruction:
            raise ValueError(
                "--ignore_budget requires budgetbench_task.instruction in every dataset item"
            )
        return f"Question: {instruction}"
    return str((item or {}).get("query") or "").strip()


def is_timeout_exception(exc):
    visited = set()
    current = exc
    transient_markers = (
        "timeout",
        "timed out",
        "readtimeout",
        "apitimeouterror",
        "connection error",
        "connection refused",
        "connecterror",
        "apiconnectionerror",
    )
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        text = f"{type(current).__name__}: {current}".lower()
        if any(marker in text for marker in transient_markers):
            return True
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    return False


def is_provider_exception(exc):
    """Recognize external model-service failures without treating invalid plans as provider errors."""
    if is_timeout_exception(exc):
        return True
    visited = set()
    current = exc
    provider_markers = (
        "provider/api error",
        "apierror",
        "badrequesterror",
        "authenticationerror",
        "permissiondeniederror",
        "notfounderror",
        "conflicterror",
        "unprocessableentityerror",
        "ratelimiterror",
        "internalservererror",
        "service unavailable",
        "error code: 4",
        "error code: 5",
    )
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        text = f"{type(current).__module__}.{type(current).__name__}: {current}".lower()
        if any(marker in text for marker in provider_markers):
            return True
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    return False


def has_invalid_structure(quality):
    if not isinstance(quality, dict):
        return False
    failure_keys = (
        "format_errors",
        "json_parse_failures",
        "non_strict_json",
        "invalid_plan",
        "invalid_update",
        "problem_nodes",
    )
    return any(int(quality.get(key, 0) or 0) > 0 for key in failure_keys)
