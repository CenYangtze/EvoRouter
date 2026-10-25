"""Run methods in separate processes and save their evaluation records."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from .baselines import ALIASES, METHODS
from .data import (
    baseline_item,
    dataset_name,
    load_budgetbench,
    read_jsonl,
    select_items,
)
from .results import summarize, summarize_method, write_json

MODELS = (
    "qwen3.5-35b-a3b",
    "qwen3.5-122b-a10b",
    "deepseek-v4-flash",
    "qwen3.5-397b-a17b",
    "glm-5",
)
EVOROUTER = "evorouter"
ROUTING = list(METHODS)


def expand_methods(names):
    result = []
    for word in names:
        for name in word.split(","):
            name = ALIASES.get(name.lower(), name.lower())
            if name == "all":
                result.append(EVOROUTER)
                result.extend("vanilla__" + model for model in MODELS)
                result.extend(ROUTING)
            elif name == "vanilla":
                result.extend("vanilla__" + model for model in MODELS)
            else:
                result.append(name)
    allowed = (
        {EVOROUTER}
        | set(ROUTING)
        | {"vanilla__" + model for model in MODELS}
        | {"cot", "reflection", "chain", "debate", "fullconnected", "tree", "sc"}
    )
    unknown = set(result) - allowed
    if unknown:
        raise ValueError("Unknown method: " + ", ".join(sorted(unknown)))
    return list(dict.fromkeys(result))


def load_selected(args):
    from .ood import DATASETS as OOD_DATASETS, load_items

    groups = {}
    if args.suite in {"main", "all"}:
        groups["main"] = (
            load_budgetbench(args.asset_root / "data" / "budgetbench")
            if not args.datasets or set(args.datasets) - set(OOD_DATASETS)
            else []
        )
    if args.suite in {"ood", "all"}:
        selected = [
            name for name in OOD_DATASETS if not args.datasets or name in args.datasets
        ]
        groups["ood"] = (
            load_items(args.asset_root / "data" / "ood", datasets=selected)
            if selected
            else []
        )
    return {
        name: select_items(rows, args.datasets, args.limit, args.per_domain, args.seed)
        for name, rows in groups.items()
    }


def evaluate(args):
    args.asset_root = args.asset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.worker_method:
        return run_one(args.worker_method, args)
    methods = expand_methods(args.methods)
    if args.model_path and EVOROUTER not in methods:
        raise ValueError("--model-path applies to EvoRouter")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        print(f"\nRunning {method}", flush=True)
        command = [
            sys.executable,
            "-m",
            "evorouter",
            "evaluate",
            "--worker-method",
            method,
            "--asset-root",
            str(args.asset_root),
            "--output-dir",
            str(args.output_dir),
            "--suite",
            args.suite,
            "--concurrency",
            str(args.concurrency),
            "--batch-size",
            str(args.batch_size),
            "--limit",
            str(args.limit),
            "--per-domain",
            str(args.per_domain),
            "--seed",
            str(args.seed),
        ]
        for key in ("device", "model_path"):
            if getattr(args, key):
                command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
        if args.datasets:
            command.extend(["--datasets", *args.datasets])
        for suite in (["main", "ood"] if args.suite == "all" else [args.suite]):
            subprocess.run([*command, "--worker-suite", suite], check=True)
    summarize(args.output_dir)


def run_one(method, args):
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    os.environ["EVOROUTER_ASSET_ROOT"] = str(args.asset_root)
    os.environ["PAPER_MODEL_POOL"] = ",".join(MODELS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for target, names in (
        ("URL", ("OPENAI_BASE_URL", "BASE_URL", "URL")),
        ("KEY", ("OPENAI_API_KEY", "API_KEY", "KEY")),
    ):
        value = next((os.environ[name] for name in names if os.environ.get(name)), None)
        if value:
            os.environ[target] = value
    os.environ.update(
        OPENAI_TIMEOUT="180",
        OPENAI_MAX_RETRIES="0",
        LLM_RETRY_ATTEMPTS="3",
        LLM_PROVIDER_CONCURRENCY_PER_MODEL="0",
    )
    groups = load_selected(args)
    items = [row for rows in groups.values() for row in rows]
    if not items:
        raise ValueError("No items match the selected datasets")
    output = args.output_dir / method
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "method": method,
        "asset_root": str(args.asset_root),
        "output_dir": str(output),
        "suite": args.suite,
        "datasets": args.datasets,
        "limit": args.limit,
        "per_domain": args.per_domain,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
        "timeout": 180,
        "max_tokens": 1024,
        "code_max_tokens": 4096,
        "expected": len(items),
    }
    if method == EVOROUTER:
        config["model_path"] = str(
            (args.model_path or args.asset_root / "weights" / "EvoRouter")
            .expanduser()
            .resolve()
        )
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Settings changed for {output}; choose another --output-dir")
    write_json(config_path, config)
    result_path = output / "results.jsonl"
    rows = read_jsonl(result_path) if result_path.exists() else []
    done = {row["id"] for row in rows}
    lookup = {row["id"]: row for row in items}
    with result_path.open("a", buffering=1, encoding="utf-8") as stream:
        for suite, records in groups.items():
            if args.worker_suite and suite != args.worker_suite:
                continue
            pending = [row for row in records if row["id"] not in done]
            if not pending:
                continue
            if method == EVOROUTER:
                if suite == "main":
                    from .coordinator import run_method
                else:
                    from .ood import run_method
                iterator = run_method(method, pending, config)
            else:
                from .baselines import run_method

                iterator = run_method(
                    method,
                    (
                        [baseline_item(row) for row in pending]
                        if suite == "main"
                        else pending
                    ),
                    config,
                )
            for row in iterator:
                item = lookup[row["id"]]
                row.setdefault("method", method)
                row.setdefault("suite", suite)
                row.setdefault("dataset", dataset_name(item))
                row.setdefault("task_type", item.get("task_type", "unknown"))
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                done.add(row["id"])
                summarize_method(output, rows, len(items))
                print(f"{method}: {len(rows)}/{len(items)}", flush=True)
    summarize_method(output, rows, len(items))
