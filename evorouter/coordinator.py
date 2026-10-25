"""Run EvoRouter on the evaluation test split."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = Path(__file__).with_name("profiles") / "coordinator_catalogs.json"
POOL = (
    "glm-5",
    "qwen3.5-397b-a17b",
    "deepseek-v4-flash",
    "qwen3.5-122b-a10b",
    "qwen3.5-35b-a3b",
)
PROTOCOL_ID = "evorouter-fivepool-all-assets-v1"


def load_catalogs():
    return {
        tuple(value["selection_key"]): {
            key: value[key]
            for key in (
                "system_prompt",
                "skill_pool",
                "model_pool",
                "collaboration_pool",
            )
        }
        for value in json.loads(PROFILE_PATH.read_text())
    }


def skill_names(task):
    from MAR.Skills import select_skills_for_task

    names = [entry["Name"] for entry in select_skills_for_task(task)]
    if "OriginalQuestion" in names:
        names[names.index("OriginalQuestion")] = "CriticalReflection"
    elif "CriticalReflection" not in names:
        names.insert(names.index("AnswerSynthesis"), "CriticalReflection")
    return tuple(names)


def router_class():
    from MAR.MasRouter.coordinator_runtime import CoordinatorRouter
    from MAR.MasRouter.llm_coordinator_router import VISION_UNAVAILABLE_TEXT

    class EvoRouter(CoordinatorRouter):
        def _mas_static_definition(self):
            return self._system_prompt

        def _worker_max_tokens(self, model, *, is_terminal):
            return self.terminal_max_tokens if is_terminal else self.agent_max_tokens

        def _worker_skill_text(
            self, skill, *, supports_vision=True, original_question=""
        ):
            text = f"{skill['Name']}: {skill['Instruction']}"
            passages = self._format_episode_assets(images=False)
            if passages:
                text += "\n\nPassages available to every Worker:\n" + passages
            images = self._format_episode_assets(images=True)
            if images:
                if supports_vision:
                    text += "\n\nImages attached to this Worker:\n" + images
                else:
                    text += (
                        "\n\nImages represented as text for this text-only Worker:\n"
                        + images
                        + "\n"
                        + VISION_UNAVAILABLE_TEXT
                    )
            return text

        def _worker_user_content(self, user_prompt, skill, *, supports_vision=True):
            if not supports_vision:
                return user_prompt
            image_blocks = self._visual_asset_blocks()
            if not image_blocks:
                return user_prompt
            return [{"type": "text", "text": user_prompt}, *image_blocks]

    return EvoRouter


def build_router(policy, model_pool, item, args):
    catalog = load_catalogs()[skill_names(item["budgetbench_task"])]
    if model_pool != catalog["model_pool"]:
        raise ValueError(
            "Worker model pool differs from the released evaluation profile"
        )
    router = router_class()(
        coordinator_policy=policy,
        model_pool=model_pool,
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
        coordinator_final_max_tokens=1024,
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
    if router.collaboration_pool != catalog["collaboration_pool"]:
        raise ValueError(
            "Collaboration pool differs from the released evaluation profile"
        )
    router._system_prompt = catalog["system_prompt"]
    return router


def run_method(method, items, config):
    """Yield one evaluated record per item using a downloaded model."""
    os.environ.update(
        OPENAI_TIMEOUT="180",
        OPENAI_MAX_RETRIES="0",
        LLM_RETRY_ATTEMPTS="3",
        LLM_PROVIDER_CONCURRENCY_PER_MODEL="0",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(name, None)
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or os.getenv("URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("KEY")
    if base_url:
        os.environ["URL"] = base_url
    if api_key:
        os.environ["KEY"] = api_key
    from evorouter import evaluation
    from MAR.MasRouter.coordinator_runtime import create_policy
    from MAR.Utils.utils import fix_random_seed

    asset_root = Path(
        config.get("asset_root")
        or os.getenv("EVOROUTER_ASSET_ROOT", REPO_ROOT / "assets")
    )
    model_path = Path(config.get("model_path") or asset_root / "weights/EvoRouter")
    items = list(items)
    if not items:
        return
    model_pool = next(iter(load_catalogs().values()))["model_pool"]
    settings = SimpleNamespace(
        coordinator_checkpoint=str(model_path),
        agent_model_pool=",".join(POOL),
        coordinator_temperature=0.0,
        coordinator_plan_max_tokens=3072,
        coordinator_final_max_tokens=1024,
        agent_max_tokens=1024,
        terminal_max_tokens=1024,
        ignore_budget=True,
        format_retries=0,
        provider_retries=2,
        worker_execution_profile="fixed",
        concurrency=int(config.get("concurrency", 16)),
        per_task_type=int(config.get("per_domain", 0)),
    )
    fix_random_seed(int(config.get("seed", 20260906)))
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
    try:
        with ThreadPoolExecutor(settings.concurrency) as executor:
            futures = [
                executor.submit(
                    evaluation.evaluate_item,
                    index,
                    item,
                    policy,
                    model_pool,
                    settings,
                    build_router,
                )
                for index, item in enumerate(items, 1)
            ]
            for future in as_completed(futures):
                row = future.result()
                row.update(
                    method=method,
                    material_protocol=PROTOCOL_ID,
                    coordinator_checkpoint=str(model_path),
                    correct=bool(row["task_success"]),
                    score_value=float(row["task_success"]),
                    metric_name="accuracy",
                    worker_flops_units=row["weighted_worker_tokens"],
                )
                yield row
    finally:
        policy.shutdown_judging_batching()
