from __future__ import annotations
from .paths import ASSETS, WEIGHTS
import ast, asyncio, base64, concurrent.futures, contextvars, hashlib, json, re, sys, threading, types
from pathlib import Path
from types import SimpleNamespace
import torch
from .core import CALLS, MODELS, VISION
from .prepare import write_json
from .scoring import score
from .upstream import current_pool
from .masrouter_cost_v2 import flops

BASE = ASSETS
UP = BASE / "upstream/AgentRouter"
OUT = WEIGHTS / "agentrouter"
BERT = WEIGHTS / "agentrouter/encoder"
CTX = contextvars.ContextVar("official_agent_call")
RESPONSE_FORMAT = contextvars.ContextVar("official_agent_response_format", default=None)
SEED = 20260929
ROLES = ["raw", "cot", "sc", "mad", "react_reflect", "summary"]
sys.path[:0] = [str(UP), str(UP / "scripts")]


def modules():
    from collections import Counter, defaultdict
    from typing import Any, Optional

    if "graph_build" not in sys.modules:
        package = types.ModuleType("graph_build")
        package.__path__ = [str(UP / "graph_build")]
        sys.modules["graph_build"] = package
    import graph_build.agents as agents
    import graph_build.builders as builders
    import graph_build.pyg_utils as pyg
    from graph_build.embedders import HFBertEmbedder
    from agentrouter_train import (
        RouterGNN,
        OpenAILLM,
        f1_score,
        canon_for_match,
        truncate,
        subgraph_blocks,
    )

    path = UP / "scripts/AgentTrainer.py"
    tree = ast.parse(path.read_text())
    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    )
    cls.name = "AnswerAggregator"
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_aggregate_answers", "_prepare_agent_job"}
    ]
    namespace = dict(
        defaultdict=defaultdict,
        Counter=Counter,
        canon_for_match=canon_for_match,
        Dict=dict,
        Any=Any,
        Optional=Optional,
        truncate=truncate,
        subgraph_blocks=subgraph_blocks,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(
        agents=agents,
        builders=builders,
        pyg=pyg,
        Embedder=HFBertEmbedder,
        RouterGNN=RouterGNN,
        OpenAILLM=OpenAILLM,
        f1_score=f1_score,
        AnswerAggregator=namespace["AnswerAggregator"],
    )


def candidate_catalog(mod):
    profiles = {p["Name"]: p for p in current_pool()}
    backbones = [
        dict(
            bb_id="BACKBONE::" + m,
            provider="openai",
            model=m,
            ctx=32768,
            desc=profiles[m]["Description"],
        )
        for m in MODELS
    ]
    graph_agents = mod.agents.expand_agents(backbones)
    released = json.loads((UP / "prompts/roles.json").read_text())
    templates = {}
    for role in ROLES:
        entries = [r for r in released if r["id"].endswith("::" + role)]
        if len({r["prompt"] for r in entries}) != 1:
            raise ValueError(
                "Backbone-specific strategy prompt requires explicit handling"
            )
        templates[role] = entries[0]
    answer_agents = []
    for agent in graph_agents:
        original = templates[agent["strategy"]]
        answer_agents.append(
            dict(
                original,
                id=agent["id"],
                backbone_id=agent["backbone_id"],
                model=agent["model"],
            )
        )
    assert len(graph_agents) == len(answer_agents) == 30
    return (graph_agents, answer_agents)


class ProviderExhausted(BaseException):
    """Do not let upstream JSON fallback reissue exhausted transport retries."""


class ClientBridge:

    def __init__(self, api, semaphore, graph_model):
        self.api = api
        self.semaphore = semaphore
        self.graph_model = graph_model
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        sdk_create = api.client.chat.completions.create

        def create_with_format(**kwargs):
            if RESPONSE_FORMAT.get() is not None:
                kwargs["response_format"] = RESPONSE_FORMAT.get()
            return sdk_create(**kwargs)

        api.client.chat.completions.create = create_with_format

    def create(self, model, messages, **kwargs):
        context = CTX.get()
        item = context["item"]
        stage = context["stage"]
        model = self.graph_model if stage == "graph" else model
        messages = json.loads(json.dumps(messages))
        if stage == "answer" and item.get("images"):
            if model in VISION:
                content = [{"type": "text", "text": messages[-1]["content"]}]
                for im in item["images"]:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(
                                    Path(im["path"]).read_bytes()
                                ).decode()
                            },
                        }
                    )
                messages[-1]["content"] = content
            else:
                messages[-1][
                    "content"
                ] += "\n[This text-only model cannot inspect attached images.]"
        local = []
        token = CALLS.set(local)
        fmt = RESPONSE_FORMAT.set(kwargs.get("response_format"))
        try:
            with self.semaphore:
                try:
                    text = self.api.call(
                        model,
                        messages,
                        max_tokens=4096 if item.get("task_type") == "code" else 1024,
                    )
                except RuntimeError as exc:
                    if (
                        local
                        and local[-1].get("status_code") == 400
                        and kwargs.get("response_format")
                    ):
                        raise
                    raise ProviderExhausted(str(exc)) from None
        finally:
            CALLS.reset(token)
            RESPONSE_FORMAT.reset(fmt)
            for call in local:
                call.update(
                    stage=stage,
                    agent_id=context["agent_id"],
                    response_format=kwargs.get("response_format"),
                )
            context["calls"].extend(local)
        last = local[-1]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                    finish_reason=last.get("finish_reason"),
                )
            ],
            usage=last.get("usage"),
        )


class LockedNLP:

    def __init__(self, nlp):
        self.nlp = nlp
        self.lock = threading.Lock()

    def __call__(self, text):
        with self.lock:
            return self.nlp(text)


def safe_id(item):
    return hashlib.sha256(item["id"].encode()).hexdigest()[:24]


def build_one(item, split, out, mod, nlp, graph_agents, answer_agents, bridge):
    folder = out / "cache" / split
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (safe_id(item) + ".json")
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["id"] != item["id"]:
            raise RuntimeError("Cache ID mismatch")
        return cached
    calls = []
    errors = []

    class Connector(mod.agents.AgentLLMConnector):

        def __init__(self):
            self.use_llm = True
            self.model = bridge.graph_model
            self.temperature = 1.0
            self.max_tokens = 1024
            self._client = bridge

        def choose(self, agent, *a, **kw):
            token = CTX.set(
                dict(item=item, stage="graph", agent_id=agent["id"], calls=calls)
            )
            try:
                return super().choose(agent, *a, **kw)
            except ProviderExhausted as exc:
                errors.append(dict(stage="graph", agent_id=agent["id"], error=str(exc)))
                return []
            finally:
                CTX.reset(token)

    question = item["question"]
    if item.get("choices"):
        question += "\nChoices:\n" + "\n".join(
            (f"{k}. {v}" for k, v in item["choices"].items())
        )
        question += "\nFor this multiple-choice task, put only the chosen option label in the JSON answer field."
    context = "\n\n".join(item.get("passages") or []) or question
    graph = mod.builders.build_graph_qa(
        dict(id=item["id"], question=question, context=context),
        nlp,
        None,
        graph_agents,
        Connector(),
        use_llm=True,
    )
    if graph is None:
        raise RuntimeError("Official graph builder returned no graph")
    graph["answer"] = None
    llm = mod.OpenAILLM(
        model=MODELS[0],
        temperature=1,
        max_tokens=1024,
        log_dir=str(out / "upstream_llm_logs"),
        max_workers=1,
    )
    llm._client = bridge
    llm.temperature = 1.0
    llm.set_agents({a["id"]: a for a in answer_agents})
    llm._log = lambda payload: None

    async def ask_in_context(model, system, user, max_answer_words=20):
        return llm.ask_json(model, system, user, max_answer_words=max_answer_words)

    llm.ask_json_async = ask_in_context
    helper = object.__new__(mod.AnswerAggregator)
    helper.agents_override = {a["id"]: a for a in answer_agents}

    def answer(agent):
        token = CTX.set(
            dict(item=item, stage="answer", agent_id=agent["id"], calls=calls)
        )
        try:
            job = helper._prepare_agent_job(graph, agent["id"], question)
            return (agent["id"], asyncio.run(llm.answer_async(**job)))
        except ProviderExhausted as exc:
            errors.append(dict(stage="answer", agent_id=agent["id"], error=str(exc)))
            return (agent["id"], "")
        finally:
            CTX.reset(token)

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        answers = dict(pool.map(answer, answer_agents))
    llm._executor.shutdown()
    result = dict(
        id=item["id"],
        dataset=item["dataset"],
        graph=graph,
        answers=answers,
        calls=calls,
        errors=errors,
        cache_file=str(path),
        worker_flops=flops(calls),
        graph_worker_flops=flops([c for c in calls if c["stage"] == "graph"]),
        answer_worker_flops=flops([c for c in calls if c["stage"] == "answer"]),
    )
    write_json(path, result)
    return result


def prepare(items, split, out, args, mod, nlp, graph_agents, answer_agents, bridge):
    records = []
    with concurrent.futures.ThreadPoolExecutor(args.sample_concurrency) as pool:
        jobs = [
            pool.submit(
                build_one, i, split, out, mod, nlp, graph_agents, answer_agents, bridge
            )
            for i in items
        ]
        for future in concurrent.futures.as_completed(jobs):
            record = future.result()
            records.append(record)
            write_json(
                out / "STATUS.json",
                dict(
                    state="preparing_" + split,
                    completed=len(records),
                    expected=len(items),
                    candidates=30,
                    request_concurrency=args.request_concurrency,
                ),
            )
            if len(records) % 10 == 0 or len(records) == len(items):
                print("prepared", split, len(records), len(items), flush=True)
    mapping = {r["id"]: r for r in records}
    return [mapping[i["id"]] for i in items]


def embed(records, split, out, mod, embedder):
    target = out / (split + ".pt")
    if target.exists():
        return torch.load(target, map_location="cpu", weights_only=False)
    graphs = []
    for index, record in enumerate(records):
        graph = record["graph"]
        emb = mod.pyg.embed_nodes(embedder, graph)
        text = [graph["question"]] + [
            a["model"] + " " + a.get("backbone_desc", "") + " " + a["prompt"]
            for a in graph["agents"]
        ]
        vecs = embedder.encode(text)
        emb["QUESTION::q"] = torch.tensor(vecs[0])
        for a, v in zip(graph["agents"], vecs[1:]):
            emb[a["id"]] = torch.tensor(v)
        data = mod.pyg.to_hetero(graph, emb)
        data.answer_text = ""
        data.sample_id = record["id"]
        if data["entity"].answer_mask.any():
            raise RuntimeError("Gold-derived entity mask must stay empty")
        graphs.append(data)
        if (index + 1) % 100 == 0:
            print("embedded", split, index + 1, flush=True)
    torch.save(graphs, target)
    return graphs


def make_router(mod, answer_agents, metadata):
    router = object.__new__(mod.AnswerAggregator)
    router.device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = sorted(
        {
            re.sub("[\\s\\-]+", "_", agent["backbone_id"].split("::", 1)[1].lower())
            for agent in answer_agents
        }
    )
    router.model = mod.RouterGNN(
        in_dim=768,
        hid_dim=256,
        num_layers=2,
        metadata=metadata,
        dropout=0.2,
        num_backbones=5,
        num_roles=6,
        bb_vocab=vocab,
        role_vocab=sorted(ROLES),
        use_instance_norm=True,
    ).to(router.device)
    return router


def infer(router, records, graphs, items, mod, topk=0):
    router.model.eval()
    rows = []
    for record, data, item in zip(records, graphs, items):
        data.answer_text = ""
        with torch.no_grad():
            prob = (
                torch.softmax(router.model(data.to(router.device))["logits"], dim=-1)
                .cpu()
                .tolist()
            )
        names = data.agent_names
        order = sorted(range(len(names)), key=lambda i: (-prob[i], i))
        selected = order[: topk or len(names)]
        prediction = router._aggregate_answers(
            {names[i]: record["answers"][names[i]] for i in selected},
            {names[i]: prob[i] for i in selected},
        )
        grade = score(prediction, item)
        rows.append(
            dict(
                id=item["id"],
                dataset=item["dataset"],
                method="agentrouter_official_current_pool",
                prediction=prediction,
                **grade,
                error="",
                component_errors=record["errors"],
                worker_flops=record["worker_flops"],
                graph_worker_flops=record["graph_worker_flops"],
                answer_worker_flops=record["answer_worker_flops"],
                selected_agents=[names[i] for i in selected],
                router_weights=dict(zip(names, prob)),
                paper_f1=mod.f1_score(prediction, str(item["answers"][0])),
                cache_file=record["cache_file"],
            )
        )
    return rows
