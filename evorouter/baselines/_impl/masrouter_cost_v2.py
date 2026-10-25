from __future__ import annotations
from .paths import WEIGHTS
import contextvars, sys, time
from pathlib import Path
import torch
from . import masrouter as original_implementation
from .core import CALLS, COEFFICIENTS, FINAL, task_text
from .scoring import score, failure_score

OUT = WEIGHTS / "masrouter"
CAPTURE = contextvars.ContextVar("masrouter_capture_graphs", default=None)
MATH_DATASETS = {"gsm8k", "math", "apex_shortlist", "omni_math_2_filtered"}
SEED = 20260929


def task_category(item):
    if item.get("task_type") == "code" or item["dataset"] in {
        "humaneval",
        "mbpp",
        "livecodebench",
    }:
        return 2
    if item.get("task_type") == "math" or item["dataset"] in MATH_DATASETS:
        return 0
    return 1


def flops(calls):
    return sum(
        (
            float((c.get("usage") or {}).get("total_tokens") or 0)
            * COEFFICIENTS[c["model"]]
            / 3
            for c in calls
        )
    )


def reward(success, cost, penalty, scale):
    return float(success) - penalty * cost / scale


def setup(api):
    repo = str(Path(__file__).resolve().parents[2])
    if repo not in sys.path:
        sys.path.append(repo)
    router, tasks, llms, collabs, prompt = original_implementation.setup(api)
    from MAR.Graph.graph import Graph

    execute = Graph.run

    def capture_run(graph, *args, **kwargs):
        records = CAPTURE.get()
        if records is None:
            return execute(graph, *args, **kwargs)
        records.append((graph, args, kwargs))
        return (["__planning_only__"], torch.tensor(0.0))

    Graph.run = capture_run
    return (router, tasks, llms, collabs, prompt, execute)


def plan(items, router, tasks, llms, collabs, prompt):
    captured = []
    token = CAPTURE.set(captured)
    try:
        outputs = router.forward(
            [task_text(i) + "\n" + FINAL for i in items],
            tasks,
            llms,
            collabs,
            [task_category(i) for i in items],
            prompt_file=prompt,
        )
    finally:
        CAPTURE.reset(token)
    if len(captured) != len(items):
        raise RuntimeError("Official graph capture count mismatch")
    return (captured, outputs)


def execute_job(job, execute, penalty, scale):
    item, (graph, args, kwargs) = job
    calls = []
    token = CALLS.set(calls)
    it = original_implementation.CURRENT.set(item)
    failed = {"error": ""}
    ft = original_implementation.FAILED.set(failed)
    start = time.monotonic()
    error = ""
    prediction = ""
    try:
        with torch.no_grad():
            result = execute(graph, *args, **kwargs)
        prediction = str(result[0][0])
        error = failed["error"]
        grading = failure_score(item) if error else score(prediction, item)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:250]}"
        grading = failure_score(item)
    finally:
        CALLS.reset(token)
        original_implementation.CURRENT.reset(it)
        original_implementation.FAILED.reset(ft)
    cost = flops(calls)
    return dict(
        id=item["id"],
        dataset=item["dataset"],
        method="masrouter_cost_v2",
        prediction=prediction,
        **grading,
        error=error,
        calls=calls,
        worker_flops=cost,
        cost_penalty=penalty,
        normalized_cost=cost / scale,
        utility=reward(grading["correct"], cost, penalty, scale),
        plan={
            "category": task_category(item),
            "domain": graph.domain,
            "collaboration": graph.reasoning_name,
            "roles": graph.agent_names,
            "models": graph.llm_names,
            "final_model": graph.final_llm_name,
            "rounds": kwargs.get("num_rounds"),
        },
        elapsed_sec=time.monotonic() - start,
    )
