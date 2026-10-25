from __future__ import annotations
from .paths import WEIGHTS, UPSTREAM
import ast
import json
import re
import threading
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .core import CALLS, COEFFICIENTS, FINAL, MODELS, task_text
from .prepare import ROOT
from .scoring import VERSION, failure_score, score
from .upstream import current_pool

MODEL = WEIGHTS / "xrouter/model"
UPSTREAM = ROOT / "upstream/xRouter"
REVISION = "f4b1ff26e05d0ef0da55337e3f017d26e17f18e3"


def official_system_prompt():
    tree = ast.parse(
        (UPSTREAM / "data_preprocess/router_data_preprocess.py").read_text()
    )
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            (
                isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT"
                for target in node.targets
            )
        ):
            return eval(
                compile(
                    ast.Expression(node.value),
                    str(UPSTREAM / "data_preprocess/router_data_preprocess.py"),
                    "eval",
                ),
                {"__builtins__": {}},
            )
    raise ValueError("Official xRouter SYSTEM_PROMPT not found")


def tool_name(model):
    return "call_" + re.sub("[^a-zA-Z0-9_]", "_", model)


def tools():
    profiles = {entry["Name"]: entry for entry in current_pool()}
    output = []
    for model in MODELS:
        profile = profiles[model]
        output.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name(model),
                    "description": f"Call {model}. {profile['Description']} Relative FLOPs coefficient: {COEFFICIENTS[model]}. Hidden thinking is disabled.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "optimized_system_prompt": {
                                "type": "string",
                                "description": "Optimized system prompt for this model and task",
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature; runtime uses the shared fixed Worker temperature",
                            },
                        },
                        "required": ["optimized_system_prompt"],
                    },
                },
            }
        )
    output.append(
        {
            "type": "function",
            "function": {
                "name": "select_response",
                "description": "Select one previously executed model response as the final answer.",
                "parameters": {
                    "type": "object",
                    "properties": {"model_call_name": {"type": "string"}},
                    "required": ["model_call_name"],
                },
            },
        }
    )
    return output


class BatchBackend:

    def __init__(self, batch_size=4):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            local_files_only=True,
            device_map="auto",
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        self.batch_size = batch_size
        self.cv = threading.Condition()
        self.pending = []
        threading.Thread(target=self._worker, daemon=True).start()

    def generate(self, messages, available_tools, max_tokens=1024):
        job = {
            "messages": messages,
            "tools": available_tools,
            "max_tokens": max_tokens,
            "event": threading.Event(),
        }
        with self.cv:
            self.pending.append(job)
            self.cv.notify()
        job["event"].wait()
        if "error" in job:
            raise RuntimeError(job["error"])
        return (job["text"], job["usage"])

    def _worker(self):
        while True:
            with self.cv:
                while not self.pending:
                    self.cv.wait()
                self.cv.wait(timeout=0.05)
                batch = self.pending[: self.batch_size]
                del self.pending[: self.batch_size]
            try:
                prompts = [
                    self.tokenizer.apply_chat_template(
                        job["messages"],
                        tools=job["tools"],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for job in batch
                ]
                encoded = self.tokenizer(prompts, padding=True, return_tensors="pt").to(
                    self.model.device
                )
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **encoded,
                        max_new_tokens=max((job["max_tokens"] for job in batch)),
                        do_sample=False,
                        eos_token_id=[
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                        ],
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                prompt_width = encoded.input_ids.shape[1]
                for job, tokens, prompt in zip(batch, outputs, prompts):
                    generated = tokens[prompt_width : prompt_width + job["max_tokens"]]
                    job["text"] = self.tokenizer.decode(
                        generated, skip_special_tokens=True
                    )
                    job["usage"] = {
                        "prompt_tokens": len(self.tokenizer.encode(prompt)),
                        "completion_tokens": len(generated),
                        "total_tokens": len(self.tokenizer.encode(prompt))
                        + len(generated),
                    }
            except Exception as exc:
                for job in batch:
                    job["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            finally:
                for job in batch:
                    job["event"].set()


def parse_tool_calls(text):
    calls = []
    for body in re.findall("<tool_call>\\s*(.*?)\\s*</tool_call>", text, re.S):
        try:
            value = json.loads(body)
            if isinstance(value, dict) and value.get("name"):
                calls.append(value)
        except json.JSONDecodeError:
            continue
    return calls


def _normalized_call(call):
    """Normalize common checkpoint emissions without using task labels or answers."""
    if not isinstance(call, dict) or not call.get("name"):
        return None
    value = dict(call)
    arguments = value.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {key: item for key, item in value.items() if key != "name"}
    name = str(value["name"])
    aliases = {"call_qwen3_5_397b_a3b": "call_qwen3_5_397b_a17b"}
    return {"name": aliases.get(name, name), "arguments": arguments}


def first_structured_action(text):
    """Return only the first valid structured action from a router turn.

    The official serving stack exposes parsed message.tool_calls. Raw
    Transformers decoding can continue into a hallucinated conversation and
    emit later pseudo tool responses; executing every XML block is therefore
    not equivalent to the upstream runtime.
    """
    for call in parse_tool_calls(text):
        normalized = _normalized_call(call)
        if normalized:
            return normalized
    return None


def direct_router_content(text):
    """Remove router-private reasoning while preserving the candidate answer."""
    value = str(text).strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[1].strip()
    value = re.sub("^<think>.*?</think>\\s*", "", value, flags=re.S).strip()
    if "<tool_call>" in value:
        prefix = value.split("<tool_call>", 1)[0].strip()
        if prefix:
            value = prefix
    return value


def resolve_selection(wanted, history):
    names = [entry["function_name"] for entry in history]
    wanted = str(wanted or "").strip()
    candidates = [wanted]
    if wanted and (not wanted.startswith("call_")):
        candidates.extend(["call_" + wanted, tool_name(wanted)])
    for candidate in candidates:
        if candidate in names:
            return next(
                (
                    entry["response"]
                    for entry in reversed(history)
                    if entry["function_name"] == candidate
                )
            )
    if (
        wanted.lower()
        in {"self", "call_self", "answer", "answer_directly", "none", "no"}
        and len(history) == 1
    ):
        return history[-1]["response"]
    return None


def evaluate(item, backend, api, available_tools, system):
    started = time.monotonic()
    worker_calls = []
    token = CALLS.set(worker_calls)
    router_calls = []
    history = []
    last_response = ""
    error = ""
    conductor_question = task_text(item)
    if item.get("images"):
        conductor_question += (
            "\n[Images are omitted because the xRouter Conductor is text-only.]"
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "USER QUERY:\n" + conductor_question},
    ]
    function_to_model = {tool_name(model): model for model in MODELS}
    try:
        prediction = ""
        for turn in range(3):
            text, usage = backend.generate(messages, available_tools, 1024)
            action = first_structured_action(text)
            router_calls.append(
                {
                    "turn": turn + 1,
                    "response": text,
                    "parsed_action": action,
                    "usage": usage,
                    "max_tokens": 1024,
                    "temperature": 0.0,
                }
            )
            direct = direct_router_content(text)
            if not action or (
                action["name"] == "select_response" and (not history) and direct
            ):
                prediction = direct or text
                break
            call = action
            assistant_calls = [
                {
                    "id": f"call_{turn}_0",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call.get("arguments", {}),
                    },
                }
            ]
            messages.append(
                {"role": "assistant", "content": text, "tool_calls": assistant_calls}
            )
            name = call["name"]
            arguments = call.get("arguments") or {}
            call_id = f"call_{turn}_0"
            selected = None
            if name == "select_response":
                wanted = arguments.get("model_call_name")
                selected = resolve_selection(wanted, history)
                result = selected or f"Error: {wanted} was not previously called."
            elif name in function_to_model:
                model = function_to_model[name]
                worker_item = dict(item, passages=[], images=[])
                instruction = (
                    str(arguments.get("optimized_system_prompt") or "Solve accurately.")
                    + "\n"
                    + FINAL
                )
                result = api.solve(worker_item, model, instruction)
                last_response = result
                history.append(
                    {"function_name": name, "model": model, "response": result}
                )
            else:
                result = f"Error: unavailable tool {name}."
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": result}
            )
            if selected is not None:
                prediction = selected
            if prediction:
                break
        else:
            prediction = last_response
        if not prediction:
            prediction = last_response or ""
        result = score(prediction, item)
    except Exception as exc:
        prediction = last_response
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        result = failure_score(item)
    finally:
        CALLS.reset(token)
    return {
        "scorer_version": VERSION,
        "id": item["id"],
        "dataset": item["dataset"],
        "method": "xrouter",
        "model": str(MODEL),
        "prediction": prediction,
        **result,
        "error": error,
        "calls": worker_calls,
        "router_calls": router_calls,
        "route_history": history,
        "elapsed_sec": time.monotonic() - started,
    }
