from __future__ import annotations
from .paths import WEIGHTS
import ast, asyncio, concurrent.futures, contextvars, threading, json, runpy, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import gammaln
from .prepare import ROOT
from .core import MODELS, CALLS, task_text, FINAL
from .scoring import score, failure_score
from .upstream import load_definitions

UP = ROOT / "upstream/DAAO"
CURRENT = contextvars.ContextVar("daao_item")
API_INSTANCE = [None]
LEARNING = contextvars.ContextVar("daao_learning", default=None)
TEMPLATE = UP / "daao/ext/maas/scripts/optimized/GSM8K/train/template"


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            str(WEIGHTS / "graphrouter/encoder"), device="cpu"
        )
        self.cache = {}
        self.lock = threading.Lock()

    def forward(self, s):
        key = tuple(s) if isinstance(s, list) else s
        with self.lock:
            if key not in self.cache:
                self.cache[key] = torch.tensor(
                    self.model.encode(s, show_progress_bar=False)
                ).float()
            return self.cache[key].clone()


def setup(api):
    API_INSTANCE[0] = api
    enc = Encoder()
    utils = load_definitions(
        UP / "daao/ext/maas/models/utils.py", ["sample_operators"], {"torch": torch}
    )
    source = UP / "daao/ext/maas/models/controller.py"
    tree = ast.parse(source.read_text())
    names = [
        x.name
        for x in tree.body
        if isinstance(x, (ast.FunctionDef, ast.ClassDef))
        and x.name != "difficulty_guided_vae_loss"
    ]
    ns = load_definitions(
        source,
        names,
        dict(
            torch=torch,
            nn=nn,
            F=F,
            np=np,
            gammaln=gammaln,
            List=List,
            Optional=object,
            sentence_encoder=enc,
            sample_operators=utils["sample_operators"],
            print=lambda *a, **k: None,
        ),
    )
    controller = ns["MultiLayerController"](device="cpu")
    schemas = runpy.run_path(str(TEMPLATE / "operator_an.py"))
    prompts = runpy.run_path(str(TEMPLATE / "op_prompt.py"))
    prompts["GENERATE_COT_PROMPT"] = (
        prompts["GENERATE_COT_PROMPT"]
        .replace("{", "{{")
        .replace("}", "}}")
        .replace("{{input}}", "{input}")
        .replace("{{instruction}}", "{instruction}")
    )
    from tenacity import retry, stop_after_attempt, wait_fixed
    import concurrent.futures, traceback
    from loguru import logger

    logger.remove()
    opns = dict(schemas, **{k: v for k, v in prompts.items() if not k.startswith("__")})
    opns.update(
        LLM=object,
        List=List,
        asyncio=asyncio,
        concurrent=concurrent,
        sys=sys,
        traceback=traceback,
        logger=logger,
        retry=retry,
        stop_after_attempt=stop_after_attempt,
        wait_fixed=wait_fixed,
    )
    opnames = [
        "Operator",
        "Generate",
        "GenerateCoT",
        "MultiGenerateCoT",
        "ScEnsemble",
        "Programmer",
        "SelfRefine",
    ]
    opns = load_definitions(TEMPLATE / "operator.py", opnames, opns)
    from .daao_native_fill import build_fill

    opns["Operator"]._fill_node = build_fill(api, CURRENT)

    async def exec_code(self, code, timeout=100):
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("code_sandbox.py"))],
            input=json.dumps({"code": code}),
            text=True,
            capture_output=True,
            timeout=8,
            env={"PATH": "/usr/bin:/bin"},
        )
        try:
            r = json.loads(result.stdout.strip().splitlines()[-1])
            return (r["status"], r["output"])
        except Exception:
            return ("Error", "Arithmetic sandbox could not execute the code")

    opns["Programmer"].exec_code = exec_code
    opns["Programmer"].__call__ = opns["Programmer"].__call__.__wrapped__
    mapping = {name: opns[name] for name in opnames if name != "Operator"}
    order = [
        "Generate",
        "GenerateCoT",
        "MultiGenerateCoT",
        "ScEnsemble",
        "Programmer",
        "SelfRefine",
    ]
    mapping = {n: mapping[n] for n in order}

    class Models:

        @staticmethod
        def default():
            return Models()

        def get_available_llms(self):
            return MODELS

        def get(self, name):
            return SimpleNamespace(model=name, api_type="openai")

    def create(config):
        return SimpleNamespace(
            model=config.model, cost_manager=SimpleNamespace(total_cost=0.0)
        )

    custom = SimpleNamespace(
        MATH_SOLVE_PROMPT="Solve the supplied task accurately and explain your reasoning. "
        + FINAL,
        REFINE_ANSWER_PROMPT="Use the program output and reasoning to provide a correct answer. "
        + FINAL,
    )
    workflow_path = UP / "daao/ext/maas/scripts/optimized/GSM8K/train/graph.py"
    tree = ast.parse(workflow_path.read_text())
    node = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Workflow")
    )

    class GradientFix(ast.NodeTransformer):

        def visit_Call(self, n):
            n = self.generic_visit(n)
            if (
                isinstance(n.func, ast.Attribute)
                and n.func.attr == "item"
                and isinstance(n.func.value, ast.Subscript)
                and isinstance(n.func.value.value, ast.Name)
                and (n.func.value.value.id == "log_probs_layers")
            ):
                return n.func.value
            return n

    node = GradientFix().visit(node)
    ast.fix_missing_locations(node)
    wn = dict(
        torch=torch,
        prompt_custom=custom,
        operator=SimpleNamespace(**mapping),
        operator_mapping=mapping,
        operator_names=order,
        create_llm_instance=create,
        CostManager=lambda: SimpleNamespace(total_cost=0),
        TokenCostManager=lambda: SimpleNamespace(total_cost=0),
        logger=logger,
        ModelsConfig=Models,
        LLMType=SimpleNamespace(OPENAI="openai", AZURE="azure"),
        print=lambda *a, **k: None,
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(workflow_path), "exec"),
        wn,
    )
    ops = enc(
        [
            prompts.get(
                {
                    "Generate": "GENERATE_COT_PROMPT",
                    "GenerateCoT": "GENERATE_COT_PROMPT",
                    "MultiGenerateCoT": "GENERATE_COT_PROMPT",
                    "ScEnsemble": "SC_ENSEMBLE_PROMPT",
                    "Programmer": "PYTHON_CODE_VERIFIER_PROMPT",
                    "SelfRefine": "SELFREFINE_PROMPT",
                }[n],
                n,
            )
            for n in order
        ]
    )
    from .upstream import current_pool

    profiles = {p["Name"]: p for p in current_pool()}
    llms = enc([m + ": " + profiles[m]["Description"] for m in MODELS])
    wf = object.__new__(wn["Workflow"])
    wf.controller = controller
    wf.models_config = Models()
    wf.operator_embeddings = ops
    wf.llm_embeddings = llms
    wf.selection_operator_names = order
    wf.llm_names = MODELS
    wf.llm = create(Models().get(MODELS[0]))
    wf.programmer = mapping["Programmer"](wf.llm)
    wf.sc_ensemble = mapping["ScEnsemble"](wf.llm)
    wf.custom = mapping["Generate"](wf.llm)
    return (controller, wf)


def evaluate_item(item, workflow):
    CURRENT.set(item)
    calls = []
    token = CALLS.set(calls)
    started = time.monotonic()
    prediction, error = ("", "")
    try:
        prediction = str(asyncio.run(workflow(task_text(item)))[0])
        result = score(prediction, item)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        result = failure_score(item)
    finally:
        CALLS.reset(token)
    return dict(
        id=item["id"],
        dataset=item["dataset"],
        method="daao",
        prediction=prediction,
        **result,
        error=error,
        calls=calls,
        elapsed_sec=time.monotonic() - started,
    )
