"""Evaluation entry points for the baseline implementations."""

from __future__ import annotations

import os
from pathlib import Path

METHODS = (
    "routellm",
    "graphrouter",
    "routerdc",
    "router_r1",
    "xrouter",
    "daao",
    "masrouter",
    "rcr_router",
    "agentrouter",
)
ALIASES = {
    "route-llm": "routellm",
    "graph-router": "graphrouter",
    "router-dc": "routerdc",
    "router-r1": "router_r1",
    "rcr-router": "rcr_router",
    "agent-router": "agentrouter",
}
ARTIFACT_REQUIREMENTS = {
    "routellm": [
        "weights/baselines/routellm/model",
        "upstream/RouteLLM/routellm/routers/routers.py",
    ],
    "graphrouter": [
        "weights/baselines/graphrouter/best.pt",
        "weights/baselines/graphrouter/encoder",
        "weights/baselines/graphrouter/calibration/train.jsonl",
        "weights/baselines/graphrouter/calibration/validation.jsonl",
        "upstream/GraphRouter/model/graph_nn.py",
    ],
    "routerdc": [
        "weights/baselines/routerdc/best.pt",
        "weights/baselines/routerdc/encoder",
        "upstream/RouterDC",
    ],
    "router_r1": [
        "weights/baselines/router_r1/model",
        "upstream/Router-R1/infer_vllm.py",
    ],
    "xrouter": ["weights/baselines/xrouter/model", "upstream/xRouter"],
    "masrouter": [
        "weights/baselines/masrouter/latest.pt",
        "weights/baselines/graphrouter/encoder",
        "upstream/MasRouter",
    ],
    "daao": [
        "weights/baselines/daao/latest.pt",
        "weights/baselines/graphrouter/encoder",
        "upstream/DAAO",
    ],
    "agentrouter": [
        "weights/baselines/agentrouter/best.pt",
        "weights/baselines/agentrouter/encoder",
        "weights/baselines/agentrouter/spacy_model",
        "upstream/AgentRouter",
    ],
    "rcr_router": [],
}

ARTIFACT_REQUIREMENTS["graphrouter"] += [
    f"weights/baselines/graphrouter/calibration/outcomes__{model}.jsonl"
    for model in (
        "qwen3.5-35b-a3b",
        "qwen3.5-122b-a10b",
        "deepseek-v4-flash",
        "qwen3.5-397b-a17b",
        "glm-5",
    )
]


def validate_assets(method, asset_root):
    method = ALIASES.get(method, method)
    missing = [
        str(Path(asset_root) / p)
        for p in ARTIFACT_REQUIREMENTS.get(method, [])
        if not (Path(asset_root) / p).exists()
    ]
    if missing:
        raise FileNotFoundError("Required assets are missing:\n" + "\n".join(missing))


def run_method(method, items, config):
    """Yield result records. Invoke each method in its own process."""
    method = ALIASES.get(method, method)
    os.environ["EVOROUTER_ASSET_ROOT"] = str(Path(config["asset_root"]).resolve())
    os.environ["EVOROUTER_OUTPUT_DIR"] = str(Path(config["output_dir"]).resolve())
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    validate_assets(method, config["asset_root"])
    from .runner import run

    for row in run(method, items, config):
        from ._impl.core import COEFFICIENTS

        calls = row.get("calls", [])
        cost = sum(
            (
                float((call.get("usage") or {}).get("prompt_tokens") or 0)
                + float((call.get("usage") or {}).get("completion_tokens") or 0)
            )
            * COEFFICIENTS[call["model"]]
            / 3
            for call in calls
        )
        row["weighted_worker_tokens"] = cost
        row["worker_flops"] = cost
        row["task_success"] = bool(row.get("correct"))
        row.setdefault("error", "")
        yield row
