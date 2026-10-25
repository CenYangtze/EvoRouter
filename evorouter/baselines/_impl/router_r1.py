from __future__ import annotations
from .paths import WEIGHTS
import ast, runpy, threading, time
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .core import CALLS, MODELS, task_text
from .scoring import score, failure_score, VERSION
from .prepare import ROOT

MODEL = WEIGHTS / "router_r1/model"
UP = ROOT / "upstream/Router-R1"
_service_tree = ast.parse((UP / "router_r1/llm_agent/route_service.py").read_text())
AGENT_PROMPT = ast.literal_eval(
    next(
        (
            n.value
            for n in _service_tree.body
            if isinstance(n, ast.Assign)
            and any(
                (isinstance(t, ast.Name) and t.id == "AGENT_PROMPT" for t in n.targets)
            )
        )
    )
)


class HFBackend:
    """Batching compatibility surface for the unchanged original inference loop."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, local_files_only=True
            )
            .to("cuda")
            .eval()
        )
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.cv = threading.Condition()
        self.pending = []
        threading.Thread(target=self._worker, daemon=True).start()

    def generate(self, prompt, sampling_params):
        job = dict(prompt=prompt, params=sampling_params, event=threading.Event())
        with self.cv:
            self.pending.append(job)
            self.cv.notify()
        job["event"].wait()
        if "error" in job:
            raise RuntimeError(job["error"])
        return [SimpleNamespace(outputs=[SimpleNamespace(text=job["text"])])]

    def _worker(self):
        while True:
            with self.cv:
                while not self.pending:
                    self.cv.wait()
                self.cv.wait(timeout=0.05)
                batch = self.pending[:4]
                del self.pending[:4]
            try:
                params = batch[0]["params"]
                inputs = self.tokenizer(
                    [j["prompt"] for j in batch], padding=True, return_tensors="pt"
                ).to("cuda")
                if inputs.input_ids.shape[1] + params.max_tokens > 32768:
                    raise ValueError(
                        "Router-R1 context would exceed original 32768 window"
                    )
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=params.max_tokens,
                        do_sample=False,
                        repetition_penalty=1.0,
                        stop_strings=params.stop,
                        tokenizer=self.tokenizer,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                for job, tokens in zip(batch, output):
                    text = self.tokenizer.decode(
                        tokens[inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    ends = [text.index(stop) for stop in params.stop if stop in text]
                    job["text"] = text[: min(ends)] if ends else text
            except Exception as exc:
                for job in batch:
                    job["error"] = f"{type(exc).__name__}: {str(exc)[:250]}"
            finally:
                for job in batch:
                    job["event"].set()


def original_loop():
    tree = ast.parse((UP / "infer_vllm.py").read_text())
    main = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
        )
    )
    start = next(
        (
            i
            for i, n in enumerate(main.body)
            if isinstance(n, ast.Assign)
            and any((isinstance(t, ast.Name) and t.id == "question" for t in n.targets))
        )
    )
    return compile(
        ast.Module(body=main.body[start:], type_ignores=[]),
        str(UP / "infer_vllm.py"),
        "exec",
    )


def prompt_pool():
    original = runpy.run_path(str(UP / "data_process/prompt_pool.py"))[
        "PROMPT_TEMPLATE_QWEN"
    ]
    from .upstream import current_pool

    MODEL_POOL = current_pool()
    start = original.index("#### The Descriptions of Each LLM")
    end = original.index("If you find that no further external knowledge", start)
    original = (
        original[:start]
        + "#### The Descriptions of Each LLM\n\n"
        + "\n\n".join((m["Name"] + ": " + m["Description"] for m in MODEL_POOL))
        + "\n\n"
        + original[end:]
    )
    old = "Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct, Mistral-7B-Instruct, Mixtral-8x22B-Instruct, Gemma-2-27B-Instruct"
    original = original.replace(old, ", ".join(MODELS))
    return SimpleNamespace(PROMPT_TEMPLATE_QWEN=original)


def evaluate_item(item, backend, api, loop, pool, parser, dataset):
    start = time.monotonic()
    calls = []
    tok = CALLS.set(calls)
    error = ""
    trace = []
    ns = {}

    def route(query, api_base, api_key):
        model, sep, subquery = query.partition(":")
        model = model.strip().casefold()
        if not sep or model not in MODELS:
            trace.append("Routing observation: LLM Name Error")
            return "LLM Name Error"
        try:
            worker_item = dict(item, passages=[], images=[])
            return api.solve(
                worker_item,
                model,
                AGENT_PROMPT.format_map({"query": subquery}),
                system="You are a helpful assistant.",
            )
        except RuntimeError as exc:
            if not str(exc).startswith("API failed"):
                raise
            trace.append("Routing observation: API Request Error")
            return "API Request Error"

    try:
        ns = dict(
            args=SimpleNamespace(
                question=task_text(item), model_path=str(MODEL), api_base="", api_key=""
            ),
            torch=torch,
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *a, **k: backend.tokenizer
            ),
            LLM=lambda *a, **k: backend,
            SamplingParams=lambda **kw: SimpleNamespace(**dict(kw, temperature=0.0)),
            prompt_pool=pool,
            get_query=parser,
            route=route,
            print=lambda *x, **k: trace.append(" ".join(map(str, x))),
        )
        exec(loop, ns)
        pred = ns["all_output"]
        s = score(pred, item)
        if not ns.get("STOP", False):
            error = "No final answer within official 5-iteration limit"
            s = failure_score(item)
    except Exception as e:
        pred = ns.get("all_output", "")
        error = f"{type(e).__name__}: {str(e)[:300]}"
        s = failure_score(item)
    finally:
        CALLS.reset(tok)
    r = dict(
        scorer_version=VERSION,
        id=item["id"],
        dataset=dataset,
        method="router_r1",
        model=str(MODEL),
        prediction=pred,
        **s,
        error=error,
        calls=calls,
        trace=trace,
        elapsed_sec=time.monotonic() - start,
    )
    return r
