from __future__ import annotations
from .paths import WEIGHTS, OUTPUT
import ast, base64, contextvars, threading, json, os, sys, time
from pathlib import Path
import torch
from .prepare import ROOT, write_json
from .core import CALLS, VISION, FINAL, task_text
from .scoring import score, failure_score

UP = ROOT / "upstream/MasRouter"


def setup(api):
    sys.path.insert(0, str(UP))
    os.chdir(UP)
    from loguru import logger

    logger.remove()
    from sentence_transformers import SentenceTransformer
    import MAR.LLM.llm_embedding as embedding

    class LocalEncoder(torch.nn.Module):

        def __init__(self, device=None):
            super().__init__()
            self.model = SentenceTransformer(
                str(WEIGHTS / "graphrouter/encoder"), device="cpu"
            )
            self.device = "cpu"
            self.cache = {}
            self.lock = threading.Lock()

        def forward(self, sentences):
            key = tuple(sentences) if isinstance(sentences, list) else sentences
            with self.lock:
                if key not in self.cache:
                    self.cache[key] = torch.tensor(
                        self.model.encode(sentences, show_progress_bar=False)
                    ).float()
                return self.cache[key].clone()

    embedding.SentenceEncoder = LocalEncoder
    from MAR.LLM.llm_registry import LLMRegistry

    class Transport:

        def __init__(self, name):
            self.model_name = name

        def gen(self, msgs, *args, **kwargs):
            msgs = json.loads(json.dumps(msgs))
            item = CURRENT.get()
            if item["images"]:
                if self.model_name in VISION:
                    last = msgs[-1]
                    content = [{"type": "text", "text": last["content"]}]
                    for im in item["images"]:
                        content += [
                            {"type": "text", "text": f"Image {im['index']}:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(
                                        Path(im["path"]).read_bytes()
                                    ).decode()
                                },
                            },
                        ]
                    last["content"] = content
                else:
                    msgs[-1][
                        "content"
                    ] += "\nImages are unavailable to this text-only model."
            return api.call(self.model_name, msgs)

    LLMRegistry.get = classmethod(lambda cls, name=None: Transport(name))
    from collections import Counter
    from typing import Union, Literal, Optional, List
    import random
    import MAR.Utils.utils as utility
    from .upstream import load_definitions

    functions = load_definitions(
        UP / "MAR/Utils/utils.py",
        ["find_mode", "get_kwargs"],
        dict(
            Counter=Counter,
            Union=Union,
            Literal=Literal,
            Optional=Optional,
            List=List,
            random=random,
        ),
    )
    utility.find_mode = functions["find_mode"]
    utility.get_kwargs = functions["get_kwargs"]
    utility.plot_embedding_heatmap = utility.plot_row_similarity = (
        lambda *args, **kwargs: None
    )
    import types

    post_name = "MAR.Prompts.post_process"
    if post_name not in sys.modules:
        post_path = UP / "MAR/Prompts/post_process.py"
        post_tree = ast.parse(post_path.read_text())
        post_tree.body = [
            node
            for node in post_tree.body
            if not (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("langchain_community")
            )
        ]
        post_module = types.ModuleType(post_name)
        post_module.__file__ = str(post_path)
        exec(compile(post_tree, str(post_path), "exec"), post_module.__dict__)
        sys.modules[post_name] = post_module

    from MAR.Graph.graph import Graph
    import MAR.Agent.agent

    tree = ast.parse((UP / "MAR/Graph/graph.py").read_text())
    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Graph")
    )
    run_node = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    )
    for node in ast.walk(run_node):
        if isinstance(node, ast.ExceptHandler) and node.name == "e":
            node.body[:0] = ast.parse(
                "if isinstance(e, RuntimeError) and str(e).startswith('API failed'):\n raise"
            ).body
    ast.fix_missing_locations(run_node)
    ns = dict(Graph.run.__globals__)
    exec(
        compile(
            ast.Module(body=[run_node], type_ignores=[]),
            str(UP / "MAR/Graph/graph.py"),
            "exec",
        ),
        ns,
    )
    original_graph_run = ns["run"]

    def guarded_graph_run(graph, *args, **kwargs):
        try:
            return original_graph_run(graph, *args, **kwargs)
        except RuntimeError as exc:
            if not str(exc).startswith("API failed"):
                raise
            FAILED.get()["error"] = str(exc)
            return ([""], torch.tensor(0.0))

    Graph.run = guarded_graph_run
    import importlib

    post = importlib.import_module("MAR.Prompts.post_process")

    def supplied_wiki(raw, output):
        return (
            output + "\nProvided evidence:\n" + "\n\n".join(CURRENT.get()["passages"])
        )

    post.wiki = supplied_wiki

    def arithmetic_python(raw, output):
        import re, subprocess

        blocks = re.findall("```(?:python)?\\s*(.*?)```", output, re.S)
        if not blocks:
            return output
        proc = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("code_sandbox.py"))],
            input=json.dumps({"code": blocks[-1]}),
            text=True,
            capture_output=True,
            timeout=8,
            env={"PATH": "/usr/bin:/bin"},
        )
        return output + "\nArithmetic sandbox: " + proc.stdout[-2000:]

    post.python_execute = arithmetic_python
    from MAR.MasRouter.mas_router import MasRouter
    from MAR.Prompts.tasks_profile import tasks_profile

    profile_tree = ast.parse((UP / "MAR/Agent/reasoning_profile.py").read_text())
    reasoning_profile = ast.literal_eval(
        next(
            node.value
            for node in profile_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "reasoning_profile"
                for target in node.targets
            )
        )
    )
    router = MasRouter(max_agent=6, device="cpu").to("cpu")
    from .upstream import current_pool

    llms = current_pool()
    final = OUTPUT / "masrouter_final.json"
    write_json(
        final,
        {
            "system": "You are the top decision-maker. Analyze the provided solutions, find errors, and give the final answer.",
            "user": FINAL,
        },
    )
    return (router, tasks_profile, llms, reasoning_profile, str(final))


CURRENT = contextvars.ContextVar("masrouter_item")
FAILED = contextvars.ContextVar("masrouter_api_failure")


def evaluate_item(item, router, tasks, llms, collaborations, prompt):
    CURRENT.set(item)
    FAILED.set({"error": ""})
    calls = []
    token = CALLS.set(calls)
    started = time.monotonic()
    prediction, error = ("", "")
    category = 0 if item["dataset"] == "gsm8k" else 1
    try:
        outputs = router.forward(
            [task_text(item) + "\n" + FINAL],
            tasks,
            llms,
            collaborations,
            [category],
            prompt_file=prompt,
        )
        prediction = str(outputs[0][0])
        error = FAILED.get()["error"]
        result = failure_score(item) if error else score(prediction, item)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        result = failure_score(item)
    finally:
        CALLS.reset(token)
    return dict(
        id=item["id"],
        dataset=item["dataset"],
        method="masrouter",
        prediction=prediction,
        **result,
        error=error,
        calls=calls,
        elapsed_sec=time.monotonic() - started,
    )
