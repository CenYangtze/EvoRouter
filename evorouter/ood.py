"""Load and evaluate BBH logical deduction and LiveCodeBench items."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import string
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("bbh_logical_deduction", "livecodebench")
COEFFICIENTS = {
    "qwen3.5-35b-a3b": 3,
    "qwen3.5-122b-a10b": 10,
    "deepseek-v4-flash": 13,
    "qwen3.5-397b-a17b": 17,
    "glm-5": 40,
}


def load_items(data_dir, datasets=None, source_jsonl=None):
    data_dir = Path(data_dir).expanduser().resolve()
    selected = list(datasets or DATASETS)
    unknown = set(selected).difference(DATASETS)
    if unknown:
        raise ValueError(f"Unknown OOD datasets: {sorted(unknown)}")
    rows = []
    for dataset in selected:
        path = data_dir / f"{dataset}.jsonl"
        with path.open(encoding="utf-8") as stream:
            group = [json.loads(line) for line in stream if line.strip()]
        for row in group:
            row["budgetbench_root"] = str(data_dir)
            row["budgetbench_asset_root"] = str(data_dir)
            if dataset == "livecodebench":
                row["grader"]["source_path"] = str(
                    Path(source_jsonl or data_dir / "sources/livecodebench_test6.jsonl")
                    .expanduser()
                    .resolve()
                )
            for image in row.get("images") or []:
                path = Path(image["path"])
                if not path.is_absolute():
                    image["path"] = str(data_dir / path)
        rows.extend(group)
    return rows


def _extract(text):
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S).strip()
    tags = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if tags:
        return tags[-1].strip()
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxes:
        return boxes[-1].strip()
    answers = re.findall(r"(?:final answer|answer)\s*(?:is|:)\s*([^\n]+)", text, re.I)
    if answers:
        return answers[-1].strip().strip("*. ")
    return text.strip()


def _normalized(value):
    value = re.sub(r"[^\w\s]", "", str(value).lower())
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _grader_timeout(source_path, byte_offset, test_timeout):
    configured = os.getenv("LIVECODEBENCH_GRADER_TIMEOUT")
    if configured:
        return max(10, int(configured))
    try:
        from .graders.livecodebench import decode_cases, load_source_row

        row = load_source_row(Path(source_path), int(byte_offset))
        count = len(decode_cases(row["public_test_cases"])) + len(
            decode_cases(row["private_test_cases"])
        )
        return max(120, min(900, (count + 1) * test_timeout + 30))
    except Exception:
        return 300


def grade_code(prediction, item):
    reference = item.get("grader") or {}
    source_path = reference.get("source_path")
    offset = reference.get("byte_offset")
    if source_path is None or offset is None:
        raise ValueError(
            "LiveCodeBench item requires grader.source_path and grader.byte_offset"
        )
    asset_root = Path(os.getenv("EVOROUTER_ASSET_ROOT", REPO_ROOT / "assets"))
    official_repo = Path(
        os.getenv("LIVECODEBENCH_REPO", asset_root / "upstream/LiveCodeBench")
    ).expanduser()
    test_timeout = max(1, int(os.getenv("LIVECODEBENCH_TEST_TIMEOUT", "6")))
    command = [
        sys.executable,
        str(Path(__file__).with_name("graders") / "livecodebench.py"),
        "--source_jsonl",
        str(source_path),
        "--byte_offset",
        str(int(offset)),
        "--official_repo",
        str(official_repo),
        "--test_timeout",
        str(test_timeout),
    ]
    timeout = _grader_timeout(source_path, offset, test_timeout)
    retries = max(0, int(os.getenv("LIVECODEBENCH_GRADER_RETRIES", "2")))
    last_error = ""
    for _ in range(retries + 1):
        try:
            result = subprocess.run(
                command,
                input=json.dumps({"prediction": str(prediction or "")}),
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=REPO_ROOT,
            )
        except subprocess.TimeoutExpired:
            last_error = f"process timed out after {timeout}s"
            continue
        if result.returncode:
            last_error = f"process exited {result.returncode}: {result.stderr[-2000:]}"
            continue
        for line in reversed(result.stdout.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "passed" in value:
                return bool(value["passed"])
        last_error = "grader returned no result JSON"
    raise RuntimeError(
        f"LiveCodeBench grader failed after {retries + 1} attempts: {last_error}"
    )


def score(text, item, raw=False):
    if item.get("task_type") == "code":
        correct = grade_code(str(text), item)
        return dict(
            correct=correct,
            parsed_prediction="code",
            score_value=float(correct),
            metric_name="pass_at_1",
            grader={"passed": correct},
        )
    answer = (str(text).strip() if raw else _extract(str(text)).strip()).strip("\"' ")
    choices = item.get("choices") or {}
    if choices:
        match = re.match(r"^\(?([A-Z])\)?(?:[.:)]\s*|$)", answer)
        if match and match[1] in choices:
            answer = choices[match[1]]
    correct = _normalized(answer) == _normalized(item["answers"][0])
    return dict(
        correct=correct,
        parsed_prediction=answer,
        score_value=float(correct),
        metric_name="ood_deterministic_accuracy_v2",
    )


def prediction_key(text, item):
    def normalize(value):
        value = "".join(
            character
            for character in str(value).lower()
            if character not in string.punctuation
        )
        value = re.sub(r"\b(a|an|the)\b", " ", value)
        return " ".join(value.split())

    prediction = _extract(text)
    choices = item.get("choices") or {}
    if choices:
        clean = prediction.strip().strip("().* ")
        if clean in choices:
            return clean
        match = re.fullmatch(r"([A-Z0-9])(?:[.) :\-].*)?", clean, re.S)
        if match and match[1] in choices:
            return match[1]
        matched = [
            label
            for label, value in choices.items()
            if normalize(prediction) == normalize(value)
        ]
        if len(matched) == 1:
            return matched[0]
        answer = prediction.strip().strip("* ")
        match = re.fullmatch(
            r"(?i)(?:option|choice|answer)\s*[:\-]?\s*([A-Z0-9])[.)]?", answer
        )
        return match[1].upper() if match and match[1].upper() in choices else ""
    return normalize(prediction)


def failure_score(item):
    return dict(
        correct=False,
        parsed_prediction="",
        score_value=0.0,
        metric_name=(
            "pass_at_1"
            if item.get("task_type") == "code"
            else "ood_deterministic_accuracy_v2"
        ),
    )


def _catalogs():
    from MAR.Skills import select_skills_for_task
    from .coordinator import load_catalogs, skill_names

    catalogs = load_catalogs()
    task = {
        "family": "code",
        "capability_tags": [
            "programming",
            "code_generation",
            "test_driven_verification",
        ],
        "output_contract": {"format": "python"},
    }
    names = skill_names(task)
    available = {
        skill["Name"]: skill
        for catalog in catalogs.values()
        for skill in catalog["skill_pool"]
    }
    available.update({skill["Name"]: skill for skill in select_skills_for_task(task)})
    catalog = deepcopy(next(iter(catalogs.values())))
    catalog["skill_pool"] = [deepcopy(available[name]) for name in names]
    begin = "Available Skills (One per agent):\n"
    end = "Available Collaboration Modes (and parameter requirements):"
    before, tail = catalog["system_prompt"].split(begin, 1)
    _, after = tail.split(end, 1)
    catalog["system_prompt"] = (
        before
        + begin
        + "\n".join(
            "- " + skill["Name"] + ": " + skill["Description"]
            for skill in catalog["skill_pool"]
        )
        + "\n"
        + end
        + after
    )
    catalogs[names] = catalog
    return catalogs


def build_router(policy, item, catalogs):
    from .coordinator import router_class, skill_names

    catalog = catalogs[skill_names(item["budgetbench_task"])]
    code = item.get("task_type") == "code"
    cap = 4096 if code else 1024
    router = router_class()(
        coordinator_policy=policy,
        model_pool=deepcopy(catalog["model_pool"]),
        skill_pool=deepcopy(catalog["skill_pool"]),
        max_agent=0,
        min_agent=0,
        max_rounds=1,
        mas_update_freq=0,
        max_mas_revisions=0,
        coordinator_temperature=0.0,
        agent_temperature=1.0,
        terminal_temperature=1.0,
        coordinator_plan_max_tokens=3072,
        coordinator_final_max_tokens=cap,
        agent_max_tokens=cap,
        terminal_max_tokens=cap,
        state_format_retries=0,
        state_think_turn_limit=4,
        workflow_protocol=False,
        update_during_workflow=False,
        review_after_workflow=False,
        gemini_terminal_mode="nothinking",
        gemini_intermediate_mode="nothinking",
        deepseek_terminal_mode="nothinking",
        deepseek_intermediate_mode="nothinking",
        capture_events=True,
    )
    router._system_prompt = catalog["system_prompt"]
    router._code_verbatim_submission = code
    router.set_episode_assets(
        item["budgetbench_task"].get("assets") or [],
        asset_root=item.get("budgetbench_asset_root") or item.get("budgetbench_root"),
    )
    return router


def evaluate(item, policy, catalogs, method):
    from .runtime_helpers import (
        has_invalid_structure,
        is_provider_exception,
        reset_cost_counters,
    )
    from MAR.Utils.globals import CompletionTokens, Cost, PromptTokens

    reset_cost_counters()
    started = time.monotonic()
    attempts, events = [], []
    error, result, invalid, retries = "", {}, False, 0
    while True:
        router = build_router(policy, item, catalogs)
        try:
            result = router.run("Question: " + item["budgetbench_task"]["instruction"])
            events.extend(deepcopy(getattr(router, "_events", []) or []))
            invalid = has_invalid_structure(result.get("quality") or {})
            attempts.append(
                dict(
                    attempt=len(attempts) + 1,
                    invalid_structure=invalid,
                    final_answer=str(result.get("final_answer") or ""),
                    final_plan=deepcopy(result.get("final_plan") or {}),
                    history=deepcopy(result.get("history") or []),
                    coordinator_steps=_steps(router),
                )
            )
            if invalid:
                error = "invalid_structure"
            break
        except Exception as exc:
            events.extend(deepcopy(getattr(router, "_events", []) or []))
            provider = is_provider_exception(exc)
            attempts.append(
                dict(
                    attempt=len(attempts) + 1,
                    provider_failure=provider,
                    error=type(exc).__name__,
                    coordinator_steps=_steps(router),
                )
            )
            if provider and retries < 2:
                retries += 1
                continue
            error = type(exc).__name__
            break
    prediction = str(result.get("final_answer") or "") if not error else ""
    try:
        graded = score(prediction, item) if not error else failure_score(item)
    except Exception as exc:
        error = "grader: " + type(exc).__name__
        graded = failure_score(item)
    calls, weighted = [], 0.0
    for event in events:
        payload = event.get("payload") or {}
        if event.get("type") != "llm_call" or payload.get("request_type") != "agent":
            continue
        usage = payload.get("usage") or {}
        total = sum(
            int(usage.get(key) or 0)
            for key in ("input_tokens", "output_tokens", "reasoning_tokens")
        )
        model = payload.get("model")
        weighted += total * COEFFICIENTS[model] / 3.0
        calls.append(
            dict(
                model=model,
                max_tokens=payload.get("max_tokens"),
                temperature=payload.get("temperature"),
                messages=payload.get("messages"),
                response=payload.get("response"),
                usage=dict(
                    prompt_tokens=int(usage.get("input_tokens") or 0),
                    completion_tokens=int(usage.get("output_tokens") or 0)
                    + int(usage.get("reasoning_tokens") or 0),
                    total_tokens=total,
                ),
                extra_body=(
                    {"enable_thinking": False}
                    if str(model).startswith("qwen")
                    else {"thinking": {"type": "disabled"}}
                ),
            )
        )
    plan = result.get("final_plan") or {}
    return dict(
        id=item["id"],
        dataset=item["dataset"],
        task_type=item["task_type"],
        method=method,
        prediction=prediction,
        **graded,
        task_success=bool(graded["correct"] and not error and not invalid),
        error=error,
        invalid_structure=invalid,
        provider_retries_used=retries,
        worker_cost_units=float(Cost.instance().value),
        worker_prompt_tokens=int(PromptTokens.instance().value),
        worker_completion_tokens=int(CompletionTokens.instance().value),
        weighted_worker_tokens=weighted,
        worker_call_count=len(calls),
        calls=calls,
        collaboration=plan.get("collaboration", ""),
        rounds=plan.get("rounds", 0),
        planned_agent_count=len(plan.get("agents") or []),
        attempts=attempts,
        events=events,
        elapsed_sec=time.monotonic() - started,
    )


def _steps(router):
    return [
        dict(
            request_type=step.request_type,
            messages=deepcopy(step.messages),
            response_text=step.response_text,
            max_tokens=step.max_tokens,
            temperature=step.temperature,
            accepted=step.accepted,
            rejection_reason=step.rejection_reason,
        )
        for step in getattr(router, "coordinator_steps", [])
    ]


def run_method(method, items, config):
    from MAR.MasRouter.coordinator_runtime import create_policy
    from MAR.Utils.utils import fix_random_seed

    fix_random_seed(int(config.get("seed", 20260906)))
    asset_root = Path(
        config.get("asset_root")
        or os.getenv("EVOROUTER_ASSET_ROOT", REPO_ROOT / "assets")
    )
    model_path = Path(config.get("model_path") or asset_root / "weights/EvoRouter")
    policy = create_policy(
        model_path,
        device=config.get("device"),
        dtype=config.get("dtype", "bfloat16"),
    )
    policy.batch_all_coordinator_requests = True
    policy.configure_judging_batching(
        max_batch_size=int(config.get("batch_size", 8)), wait_ms=100.0
    )
    policy.model.config.use_cache = True
    policy.model.eval()
    catalogs = _catalogs()
    try:
        with ThreadPoolExecutor(int(config.get("concurrency", 8))) as executor:
            futures = [
                executor.submit(evaluate, item, policy, catalogs, method)
                for item in items
            ]
            for future in as_completed(futures):
                yield future.result()
    finally:
        policy.shutdown_judging_batching()
