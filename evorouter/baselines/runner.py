"""Load frozen baseline artifacts and execute their evaluation paths."""

from __future__ import annotations

import abc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re
import random
import sys
import threading
import time
from types import SimpleNamespace

from ._impl import core, scoring
from ._impl.paths import OUTPUT, UPSTREAM, WEIGHTS
from ._impl.prepare import read_jsonl

SHORT_ANSWER = (
    "Answer directly and concisely; no explanation. Respond as briefly as possible; give only the answer without reasoning. "
    + core.FINAL
)


def parallel(function, values, concurrency):
    def execute(value):
        candidate = value
        if isinstance(candidate, tuple):
            candidate = candidate[0]
        if isinstance(candidate, dict):
            candidate = candidate.get(
                "item", candidate.get("state", {}).get("item", candidate)
            )
        cap = (
            4096
            if isinstance(candidate, dict) and candidate.get("task_type") == "code"
            else None
        )
        token = core.WORKER_TOKEN_LIMIT.set(cap)
        try:
            return function(value)
        finally:
            core.WORKER_TOKEN_LIMIT.reset(token)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        yield from executor.map(execute, values)


def call_row(item, method, function, raw=False, **metadata):
    calls = []
    token = core.CALLS.set(calls)
    prediction, error = "", ""
    started = time.monotonic()
    try:
        prediction = function()
        if item["dataset"] in {"bbh_logical_deduction", "livecodebench"}:
            from evorouter.ood import score as ood_score

            result = ood_score(prediction, item, raw=raw)
        elif raw and item.get("budgetbench_task"):
            from MAR.Benchmarks.budgetbench import grade_budgetbench_prediction

            grade = grade_budgetbench_prediction(
                item["budgetbench_task"], prediction, item["budgetbench_root"]
            )
            result = dict(
                correct=bool(grade["success"]),
                score_value=float(grade["success"]),
                parsed_prediction=prediction,
                metric_name="native_budgetbench_success",
                evaluation=grade,
            )
        else:
            result = scoring.score(prediction, item)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:250]}"
        result = scoring.failure_score(item)
    finally:
        core.CALLS.reset(token)
    return dict(
        id=item["id"],
        dataset=item["dataset"],
        method=method,
        prediction=prediction,
        **result,
        error=error,
        calls=calls,
        elapsed_sec=time.monotonic() - started,
        **metadata,
    )


def routed_rows(method, decisions, api, config):
    def execute(decision):
        item, model = decision["item"], decision["model"]
        cap = (
            config.get("code_max_tokens", 4096)
            if item.get("task_type") == "code"
            else config.get("max_tokens", 1024)
        )
        return call_row(
            item,
            method,
            lambda: api.solve(item, model, SHORT_ANSWER, max_tokens=cap),
            model=model,
            router_call={
                k: v for k, v in decision.items() if k not in {"item", "model"}
            },
        )

    yield from parallel(execute, decisions, config.get("concurrency", 12))


def graph_decisions(items):
    import torch
    from sentence_transformers import SentenceTransformer
    from ._impl import graphrouter as module
    from ._impl.upstream import current_pool

    folder = WEIGHTS / "graphrouter"
    train = read_jsonl(folder / "calibration/train.jsonl")
    calibration = train + read_jsonl(folder / "calibration/validation.jsonl")
    outcomes = {
        model: {
            row["id"]: row
            for row in read_jsonl(folder / "calibration" / f"outcomes__{model}.jsonl")
        }
        for model in core.MODELS
    }
    encoder = SentenceTransformer(str(folder / "encoder"), device="cpu")
    q, t = module.embeddings(calibration, encoder)
    profiles = {p["Name"]: p for p in current_pool()}
    llm = torch.tensor(
        encoder.encode(
            [m + ": " + profiles[m]["Description"] for m in core.MODELS],
            show_progress_bar=False,
        )
    )
    effects = torch.tensor(
        [
            [float(outcomes[m][item["id"]]["correct"]) for m in core.MODELS]
            for item in calibration
        ]
    )
    costs = torch.tensor(
        [
            [
                float(outcomes[m][item["id"]].get("worker_flops_units", 0))
                for m in core.MODELS
            ]
            for item in calibration
        ]
    )
    costs = costs / costs.max().clamp(min=1)
    weights = torch.stack(
        [effects.flatten(), costs.flatten(), effects.flatten()], dim=-1
    )
    model = module.original_model()(384, 384, 8, 3)
    model.load_state_dict(
        torch.load(folder / "best.pt", map_location="cpu", weights_only=True)["model"]
    )
    model.eval()
    groups = defaultdict(list)
    for item in items:
        groups[item["dataset"]].append(item)
    n, m = len(calibration), len(core.MODELS)
    for group in groups.values():
        testq, testt = module.embeddings(group, encoder)
        total = n + len(group)
        edges = torch.stack(
            [
                torch.arange(total).repeat_interleave(m),
                torch.arange(m).repeat(total) + total,
            ]
        )
        visible = torch.arange(total * m) < len(train) * m
        target = torch.arange(total * m) >= n * m
        with torch.no_grad():
            probabilities = model(
                torch.cat([t, testt]),
                torch.cat([q, testq]),
                llm,
                edges,
                edge_mask=target,
                edge_can_see=visible,
                edge_weight=torch.cat([weights, torch.zeros(len(group) * m, 3)]),
            ).reshape(len(group), m)
        for item, index in zip(group, probabilities.argmax(1).tolist()):
            yield {"item": item, "model": core.MODELS[index]}


def agentrouter_rows(items, api):
    import torch
    import spacy
    from ._impl import agentrouter_official as module

    sys.path[:0] = [str(module.UP), str(module.UP / "scripts")]
    modules = module.modules()
    graph_agents, answer_agents = module.candidate_catalog(modules)
    modules.agents._throttle_until_available = lambda *args: None
    nlp = module.LockedNLP(spacy.load(WEIGHTS / "agentrouter/spacy_model"))
    bridge = module.ClientBridge(api, threading.BoundedSemaphore(32), core.MODELS[0])
    args = SimpleNamespace(sample_concurrency=8, request_concurrency=32)
    records = module.prepare(
        items, "test", OUTPUT, args, modules, nlp, graph_agents, answer_agents, bridge
    )
    del nlp
    encoder = modules.Embedder(
        str(module.BERT), pooling="mean", max_length=256, batch_size=64, device="cuda"
    )
    graphs = module.embed(records, "test", OUTPUT, modules, encoder)
    checkpoint = torch.load(
        WEIGHTS / "agentrouter/best.pt", map_location="cpu", weights_only=False
    )
    metadata = checkpoint["metadata"]
    for graph in graphs:
        for edge in metadata[1]:
            if edge not in graph.edge_types:
                graph[edge].edge_index = torch.empty((2, 0), dtype=torch.long)
    model = module.make_router(modules, answer_agents, metadata)
    model.model.load_state_dict(checkpoint["model"])
    for row, record in zip(
        module.infer(model, records, graphs, items, modules), records
    ):
        row["calls"] = record["calls"]
        yield row


def run(method, items, config):
    api = core.API(config.get("timeout", 180), 2, config.get("max_tokens", 1024))
    concurrency = config.get("concurrency", 12)
    if method.startswith("vanilla__") or method in {"vanilla", "io"}:
        from ._impl.vanilla_io_fulltest import messages

        model = (
            method.split("__", 1)[1] if "__" in method else config.get("model", "glm-5")
        )

        def execute(item):
            cap = (
                config.get("code_max_tokens", 4096)
                if item.get("task_type") == "code"
                else config.get("max_tokens", 1024)
            )
            return call_row(
                item,
                method,
                lambda: api.call(model, messages(item, model), max_tokens=cap),
                raw=True,
                model=model,
            )

        yield from parallel(execute, items, concurrency)
        return
    if method in core.FIXED:
        yield from parallel(
            lambda item: call_row(
                item,
                method,
                lambda: core.fixed(method, item, api, config.get("model", "glm-5")),
            ),
            items,
            concurrency,
        )
        return
    if method == "rcr_router":
        from ._impl.rcr_router import evaluate

        yield from parallel(
            lambda item: evaluate(item, api, rounds=1), items, concurrency
        )
        return
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(20260916)
    if method in {"daao", "masrouter"} and all(
        item["dataset"] not in {"bbh_logical_deduction", "livecodebench"}
        for item in items
    ):
        random.seed(20260916)
    if method == "routellm":
        import numpy as np
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from ._impl.upstream import load_definitions

        definitions = load_definitions(
            UPSTREAM / "RouteLLM/routellm/routers/routers.py",
            ["Router", "no_parallel", "BERTRouter"],
            dict(
                abc=abc,
                torch=torch,
                np=np,
                AutoModelForSequenceClassification=AutoModelForSequenceClassification,
                AutoTokenizer=AutoTokenizer,
            ),
        )
        router = definitions["BERTRouter"](str(WEIGHTS / "routellm/model"))
        router.model.eval()

        def decisions():
            for item in items:
                probability = float(
                    router.calculate_strong_win_rate(core.task_text(item))
                )
                yield dict(
                    item=item,
                    model="glm-5" if probability >= 0.5 else "qwen3.5-35b-a3b",
                    route_probability=probability,
                )

        yield from routed_rows(method, list(decisions()), api, config)
    elif method == "graphrouter":
        yield from routed_rows(method, list(graph_decisions(items)), api, config)
    elif method == "routerdc":
        from ._impl import routerdc as module

        router = module.Router()
        yield from routed_rows(method, router.route(items), api, config)
    elif method == "router_r1":
        from ._impl import router_r1 as module
        from ._impl.upstream import load_definitions

        backend = module.HFBackend()
        loop, pool = module.original_loop(), module.prompt_pool()
        parser = load_definitions(
            module.UP / "infer_vllm.py", ["get_query"], {"re": re}
        )["get_query"]
        yield from parallel(
            lambda item: module.evaluate_item(
                item, backend, api, loop, pool, parser, item["dataset"]
            ),
            items,
            min(concurrency, 4),
        )
    elif method == "xrouter":
        from ._impl import xrouter as module

        backend = module.BatchBackend()
        tools, system = module.tools(), module.official_system_prompt()
        yield from parallel(
            lambda item: module.evaluate(item, backend, api, tools, system),
            items,
            min(concurrency, 4),
        )
    elif method == "daao":
        torch.set_num_threads(2)
        from ._impl import daao as module

        controller, workflow = module.setup(api)
        checkpoint = torch.load(
            WEIGHTS / "daao/latest.pt", map_location="cpu", weights_only=False
        )
        controller.load_state_dict(checkpoint["model"])
        main_suite = all(
            item["dataset"] not in {"bbh_logical_deduction", "livecodebench"}
            for item in items
        )
        if main_suite and "rng" in checkpoint:
            torch.set_rng_state(checkpoint["rng"])
        controller.eval()

        def execute(item):
            with torch.no_grad():
                return module.evaluate_item(item, workflow)

        if main_suite:
            for item in sorted(items, key=lambda value: value["dataset"]):
                yield execute(item)
        else:
            yield from parallel(execute, items, min(concurrency, 16))
    elif method == "masrouter":
        torch.set_num_threads(2)
        import MAR
        import MAR.LLM
        import MAR.MasRouter
        import MAR.Agent

        for package, subdir in [
            (MAR, ""),
            (MAR.LLM, "LLM"),
            (MAR.MasRouter, "MasRouter"),
            (MAR.Agent, "Agent"),
        ]:
            package.__path__.append(str(UPSTREAM / "MasRouter/MAR" / subdir))
        if all(
            item["dataset"] not in {"bbh_logical_deduction", "livecodebench"}
            for item in items
        ):
            from ._impl import masrouter as module

            parts = module.setup(api)
            router = parts[0]
            checkpoint = torch.load(
                WEIGHTS / "masrouter/latest.pt", map_location="cpu", weights_only=False
            )
            router.load_state_dict(checkpoint["model"])
            if "rng" in checkpoint:
                torch.set_rng_state(checkpoint["rng"])
            router.eval()
            with torch.no_grad():
                for item in sorted(items, key=lambda value: value["dataset"]):
                    yield module.evaluate_item(item, *parts)
        else:
            from ._impl import masrouter_cost_v2 as module

            parts = module.setup(api)
            router = parts[0]
            router.load_state_dict(
                torch.load(
                    WEIGHTS / "masrouter/latest.pt",
                    map_location="cpu",
                    weights_only=False,
                )["model"]
            )
            router.eval()
            jobs = []
            for item in items:
                with torch.no_grad(), torch.random.fork_rng(devices=[]):
                    torch.manual_seed(
                        int(hashlib.sha256(item["id"].encode()).hexdigest()[:8], 16)
                    )
                    graphs, _ = module.plan([item], *parts[:-1])
                jobs.append((item, graphs[0]))
            yield from parallel(
                lambda job: module.execute_job(job, parts[-1], 0, 1), jobs, concurrency
            )
    elif method == "agentrouter":
        yield from agentrouter_rows(items, api)
    else:
        raise ValueError("Unknown baseline: " + method)
