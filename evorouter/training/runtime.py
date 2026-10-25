from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import torch

from MAR.Benchmarks.budgetbench import evaluate_budgetbench_episode
from MAR.MasRouter.coordinator_runtime import CoordinatorPolicy
from evorouter.coordinator import load_catalogs, router_class, skill_names
from evorouter.runtime_helpers import (
    budgetbench_query,
    has_invalid_structure,
    is_provider_exception,
    reset_cost_counters,
)


@dataclass(frozen=True)
class RolloutSettings:
    temperature: float = 1.0
    plan_max_tokens: int = 3072
    final_max_tokens: int = 1024
    workers: int = 16
    generation_batch: int = 8
    format_retries: int = 0
    provider_retries: int = 2
    max_agent: int = 0
    min_agent: int = 0
    cost_coefficient: float = 0.0
    cost_cap: float = 0.1
    cost_budget_file: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Rollout temperature must be finite and nonnegative")
        if min(self.plan_max_tokens, self.final_max_tokens, self.workers, self.generation_batch) < 1:
            raise ValueError("Token limits, workers, and generation batch size must be positive")
        if min(self.format_retries, self.provider_retries) < 0:
            raise ValueError("Retry counts must be nonnegative")
        if not math.isfinite(self.cost_coefficient) or self.cost_coefficient < 0:
            raise ValueError("Cost coefficient must be finite and nonnegative")
        if not math.isfinite(self.cost_cap) or not 0 <= self.cost_cap < 1:
            raise ValueError("Cost cap must be in [0, 1)")
        if self.cost_coefficient and not self.cost_budget_file:
            raise ValueError("A train-calibrated cost_budget_file is required when cost penalties are enabled")


def configure_environment():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or os.getenv("URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("KEY")
    if base_url:
        os.environ["URL"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    os.environ.update(
        OPENAI_TIMEOUT="180",
        OPENAI_MAX_RETRIES="0",
        LLM_RETRY_ATTEMPTS="3",
        LLM_PROVIDER_CONCURRENCY_PER_MODEL="0",
        TOKENIZERS_PARALLELISM="false",
    )


class TrainingPolicy(CoordinatorPolicy):
    def __init__(self, model, tokenizer, model_name, max_prompt_length=12288, generation_batch=8):
        if int(max_prompt_length) < 2:
            raise ValueError("max_prompt_length must be at least two tokens")
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = str(model_name)
        self.device = str(model.device)
        self.dtype = model.dtype
        self.max_prompt_length = int(max_prompt_length)
        self._generation_lock = threading.Lock()
        self._judging_batch_condition = threading.Condition()
        self._judging_batch_queue = []
        self._judging_batch_thread = None
        self._judging_batch_shutdown = False
        self._judging_batch_max_size = 1
        self._judging_batch_wait_seconds = 0.0
        self._judging_batch_requests = 0
        self._judging_generation_batches = 0
        self._judging_batch_size_counts = {}
        self._active_requests = 0
        self.batch_all_coordinator_requests = True
        self.supports_thinking_switch = (
            str(getattr(model.config, "model_type", "")).lower() == "qwen3"
            or "qwen3" in self.model_path.lower()
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError("The student tokenizer requires a padding or EOS token")
        self.model.config.use_cache = True
        self.model.eval()
        self.configure_judging_batching(max_batch_size=generation_batch, wait_ms=100.0)

    def generate_concurrent(self, *args, **kwargs):
        with self._judging_batch_condition:
            self._active_requests += 1
        try:
            return super().generate_concurrent(*args, **kwargs)
        finally:
            with self._judging_batch_condition:
                self._active_requests -= 1
                self._judging_batch_condition.notify_all()

    def assert_idle(self):
        with self._judging_batch_condition:
            if self._active_requests or self._judging_batch_queue:
                raise RuntimeError("Finish all rollout requests before updating the student")
        acquired = self._generation_lock.acquire(blocking=False)
        if not acquired:
            raise RuntimeError("Student generation is still active")
        self._generation_lock.release()

    def _prompt_tokens(self, messages, max_new_tokens, enable_thinking):
        text = self.format_messages(messages, enable_thinking=enable_thinking)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        limit = min(self.max_prompt_length, int(getattr(self.model.config, "max_position_embeddings", self.max_prompt_length)))
        budget = limit - int(max_new_tokens)
        if budget < 1:
            raise ValueError("The context limit must exceed the response token budget")
        if len(ids) > budget:
            original = ids
            retained = budget
            while retained > 0:
                head = max(1, retained // 4)
                tail = retained - head
                selected = original[:head] + (original[-tail:] if tail else [])
                text = self.tokenizer.decode(selected, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if len(ids) <= budget:
                    break
                retained -= max(1, len(ids) - budget)
            if len(ids) > budget:
                raise ValueError("Tokenizer cannot preserve the prompt within the context budget")
        if not ids:
            raise ValueError("Coordinator prompt is empty")
        return text, ids

    @torch.no_grad()
    def generate_batch(self, messages_batch, max_new_tokens, temperature, top_p=1.0, enable_thinking=False):
        if self.model.training:
            raise RuntimeError("Switch the student to evaluation mode before collecting rollouts")
        prepared = [self._prompt_tokens(messages, max_new_tokens, enable_thinking) for messages in messages_batch]
        if not prepared:
            return [], []
        width = max(len(ids) for _, ids in prepared)
        input_ids = torch.full((len(prepared), width), self.tokenizer.pad_token_id, dtype=torch.long, device=self.model.device)
        attention_mask = torch.zeros_like(input_ids)
        for row, (_, ids) in enumerate(prepared):
            input_ids[row, -len(ids) :] = torch.tensor(ids, dtype=torch.long, device=input_ids.device)
            attention_mask[row, -len(ids) :] = 1
        config = deepcopy(self.model.generation_config)
        config.do_sample = float(temperature) > 1e-6
        config.temperature = float(temperature) if config.do_sample else None
        config.top_p = float(top_p) if config.do_sample else None
        if not config.do_sample:
            config.top_k = None
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=config,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        responses = [self.tokenizer.decode(ids[width:], skip_special_tokens=True).strip() for ids in output]
        return [text for text, _ in prepared], responses

    def generate(self, messages, max_new_tokens, temperature, top_p=1.0, enable_thinking=False):
        prompts, responses = self.generate_batch([messages], max_new_tokens, temperature, top_p, enable_thinking)
        return prompts[0], responses[0]

    def generate_many(self, messages, count, max_new_tokens, temperature, top_p=1.0, enable_thinking=False):
        if int(count) < 1 or (int(count) > 1 and float(temperature) <= 1e-6):
            raise ValueError("Multiple rollouts require a positive count and sampling temperature")
        prompts, responses = self.generate_batch([messages] * int(count), max_new_tokens, temperature, top_p, enable_thinking)
        return prompts[0], responses


def build_training_router(policy, item, settings):
    catalog = load_catalogs()[skill_names(item["budgetbench_task"])]
    router = router_class()(
        coordinator_policy=policy,
        model_pool=deepcopy(catalog["model_pool"]),
        skill_pool=deepcopy(catalog["skill_pool"]),
        collaboration_pool=deepcopy(catalog["collaboration_pool"]),
        max_agent=settings.max_agent,
        min_agent=settings.min_agent,
        max_rounds=1,
        mas_update_freq=0,
        max_mas_revisions=0,
        coordinator_temperature=settings.temperature,
        agent_temperature=1.0,
        terminal_temperature=1.0,
        coordinator_plan_max_tokens=settings.plan_max_tokens,
        coordinator_final_max_tokens=settings.final_max_tokens,
        agent_max_tokens=1024,
        terminal_max_tokens=1024,
        state_format_retries=0,
        state_think_turn_limit=4,
        workflow_protocol=False,
        update_during_workflow=False,
        review_after_workflow=False,
        gemini_terminal_mode="nothinking",
        gemini_intermediate_mode="auto",
        deepseek_terminal_mode="nothinking",
        deepseek_intermediate_mode="auto",
        capture_events=True,
    )
    router._system_prompt = catalog["system_prompt"]
    router.set_episode_assets(
        item.get("worker_assets") or item["budgetbench_task"].get("assets") or [],
        asset_root=item.get("budgetbench_asset_root") or item.get("budgetbench_root"),
    )
    return router


@lru_cache(maxsize=8)
def _cost_budget_model(path):
    model = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if model.get("schema") != "budgetbench.worker-flops-budget.v1" or model.get("fit_split") != "train":
        raise ValueError("Cost budgets must use a train-calibrated Worker FLOPs budget model")
    return model


def _estimated_cost_budget(item, path):
    model = _cost_budget_model(str(Path(path).expanduser().resolve()))
    task = item["budgetbench_task"]
    instruction = str(task["instruction"])
    assets = task.get("assets") or item.get("worker_assets") or []
    images = 0
    characters = len(instruction)
    for asset in assets:
        if not isinstance(asset, dict):
            characters += len(str(asset))
        elif asset.get("kind") == "image" or str(asset.get("media_type", "")).startswith("image/"):
            images += 1
        else:
            content = asset.get("content", "")
            characters += len(content if isinstance(content, str) else json.dumps(content))
    family = str(task.get("family") or item.get("task_type") or "unknown").lower()
    contract = task.get("output_contract") or item.get("output_contract") or {}
    format_name = str(contract.get("format") or "text") if isinstance(contract, dict) else str(contract)
    if "Choices:" in instruction:
        format_name = "choice"
    length = sum(characters > threshold for threshold in (512, 2048, 8192))
    bucket_name = f"{family}|{format_name}|length{length}|images{min(images, 2)}"
    task_id = str(task.get("task_id") or item.get("task_id") or str(item.get("id", "")).split("::")[0])
    observed = model["observed_tasks"].get(task_id)
    if observed and observed.get("bucket") == bucket_name:
        budget = observed["budget"]
    else:
        budget = (model["buckets"].get(bucket_name) or model["families"].get(family) or {"budget": model["global_budget"]})["budget"]
    budget = float(budget)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("The calibrated Worker FLOPs budget must be positive and finite")
    return budget


def run_episode(policy, item, settings):
    reset_cost_counters()
    started = time.monotonic()
    query = budgetbench_query(item, ignore_budget=True)
    router = build_training_router(policy, item, settings)
    steps = []
    events = []
    result = {}
    attempts = []
    invalid_structure = False
    provider_failure = False
    format_retries_used = 0
    provider_retries_used = 0
    error = ""
    while True:
        try:
            result = router.run(query)
            invalid_structure = has_invalid_structure(result.get("quality") or {})
            attempts.append({"invalid_structure": invalid_structure, "provider_failure": False})
        except Exception as exc:
            if not is_provider_exception(exc):
                raise
            result = {}
            provider_failure = True
            error = type(exc).__name__
            attempts.append({"invalid_structure": False, "provider_failure": True, "error": error})
        finally:
            steps.extend(deepcopy(list(getattr(router, "coordinator_steps", []) or [])))
            events.extend(deepcopy(list(getattr(router, "_events", []) or [])))
        if provider_failure:
            if provider_retries_used >= settings.provider_retries:
                break
            provider_retries_used += 1
            provider_failure = False
            error = ""
            continue
        if not invalid_structure or format_retries_used >= settings.format_retries:
            break
        format_retries_used += 1
    if invalid_structure:
        error = "invalid_structure_after_format_retries"
    prediction = str(result.get("final_answer") or "") if not error else ""
    grader_path = Path(item["budgetbench_root"]) / "src"
    if str(grader_path) not in sys.path:
        sys.path.insert(0, str(grader_path))
    evaluation = evaluate_budgetbench_episode(
        task=item["budgetbench_task"],
        condition=item["budgetbench_condition"],
        prediction=prediction,
        events=events,
        release_root=item.get("budgetbench_asset_root") or item["budgetbench_root"],
    )
    answer_correct = bool(evaluation["correct"] and evaluation["terminated"] and not invalid_structure and not provider_failure)
    worker_flops = float(evaluation["weighted_tokens"])
    if not math.isfinite(worker_flops) or worker_flops < 0:
        raise ValueError("Worker FLOPs cost must be finite and nonnegative")
    budget = _estimated_cost_budget(item, settings.cost_budget_file) if settings.cost_coefficient else None
    raw_penalty = settings.cost_coefficient * math.log2(1.0 + worker_flops / budget) if budget is not None else 0.0
    penalty = min(settings.cost_cap, raw_penalty) if answer_correct else raw_penalty
    final_mas = result.get("final_plan") or {}
    return {
        "item_id": str(item["id"]),
        "task_id": str(item.get("task_id") or ""),
        "task_type": item.get("task_type"),
        "query": query,
        "steps": steps,
        "final_mas": deepcopy(final_mas),
        "final_plan": deepcopy(final_mas),
        "final_answer": prediction,
        "prediction": prediction,
        "answer_correct": answer_correct,
        "answer_is_correct": answer_correct,
        "terminated": bool(evaluation["terminated"]),
        "reward": float(answer_correct) - penalty,
        "reward_before_cost": float(answer_correct),
        "worker_flops": worker_flops,
        "estimated_flops_budget": budget,
        "cost_penalty": penalty,
        "result": result,
        "history": deepcopy(result.get("history") or []),
        "events": events,
        "evaluation": evaluation,
        "attempts": attempts,
        "provider_failure": provider_failure,
        "invalid_structure": invalid_structure,
        "provider_retries_used": provider_retries_used,
        "format_retries_used": format_retries_used,
        "error": error,
        "elapsed_seconds": time.monotonic() - started,
    }
