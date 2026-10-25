from __future__ import annotations
import base64, contextvars, json, os, re, string, time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from openai import OpenAI
from dotenv import dotenv_values

ALL_MODELS = [
    "qwen3.5-35b-a3b",
    "qwen3.5-122b-a10b",
    "deepseek-v4-flash",
    "qwen3.5-397b-a17b",
    "glm-5",
]
ALL_COEFFICIENTS = dict(zip(ALL_MODELS, [3, 10, 13, 17, 40]))
_requested_models = (
    os.getenv("PAPER_MODEL_POOL", "").split(",")
    if os.getenv("PAPER_MODEL_POOL")
    else ALL_MODELS
)
MODELS = [model for model in ALL_MODELS if model in _requested_models]
COEFFICIENTS = {model: ALL_COEFFICIENTS[model] for model in MODELS}
VISION = set((m for m in MODELS if m.startswith("qwen")))
CALLS = contextvars.ContextVar("worker_calls", default=None)
WORKER_TOKEN_LIMIT = contextvars.ContextVar("worker_token_limit", default=None)
FINAL = "End your response with <answer>your final answer</answer>. For multiple choice, put only the choice label inside the tags."
SYSTEM = "You are a helpful assistant. Solve the provided task accurately. " + FINAL


def normalize_qa(s):
    s = str(s).lower()
    s = "".join((x for x in s if x not in string.punctuation))
    s = re.sub("\\b(a|an|the)\\b", " ", s)
    return " ".join(s.split())


def extract(text):
    text = re.sub("<think>.*?</think>", "", str(text), flags=re.S).strip()
    tags = re.findall("<answer>(.*?)</answer>", text, re.S)
    if tags:
        return tags[-1].strip()
    boxes = re.findall("\\\\boxed\\{([^{}]*)\\}", text)
    if boxes:
        return boxes[-1].strip()
    match = re.findall("(?:final answer|answer)\\s*(?:is|:)\\s*([^\\n]+)", text, re.I)
    if match:
        return match[-1].strip().strip("*. ")
    return text.strip()


def numeric(s):
    s = str(s).replace(",", "").replace("$", "").strip()
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def prediction_key(text, item):
    pred = extract(text)
    if item["kind"] == "mc":
        labels = list(item["choices"])
        clean = pred.strip().strip("().* ")
        if clean in labels:
            return clean
        match = re.fullmatch("([A-Z0-9])(?:[.) :\\-].*)?", clean, re.S)
        if match and match[1] in labels:
            return match[1]
        matches = [
            k
            for k, v in item["choices"].items()
            if normalize_qa(pred) == normalize_qa(v)
        ]
        return matches[0] if len(matches) == 1 else ""
    if item["kind"] == "numeric":
        if numeric(pred) is not None:
            return str(numeric(pred).normalize())
        if "<answer>" not in text:
            nums = re.findall("[-+]?\\d[\\d,]*(?:\\.\\d+)?", pred)
            if nums:
                return str(numeric(nums[-1]).normalize())
        return ""
    return normalize_qa(pred)


def task_text(item):
    text = item["question"]
    if item["choices"]:
        text += "\nChoices:\n" + "\n".join(
            (f"{k}. {v}" for k, v in item["choices"].items())
        )
    if item["passages"]:
        text += "\n\nProvided passages:\n" + "\n\n".join(
            (f"[{i + 1}] {p}" for i, p in enumerate(item["passages"]))
        )
    return text


def messages(item, model, instruction, context="", system=SYSTEM):
    text = task_text(item) + "\n\n" + instruction
    if context:
        text += "\n\nPrevious agent responses (assess critically):\n" + context
    images = item.get("images") or []
    if images and model in VISION:
        content = [{"type": "text", "text": text}]
        for img in images:
            content.extend(
                [
                    {"type": "text", "text": f"Image {img['index']}:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(Path(img["path"]).read_bytes()).decode()
                        },
                    },
                ]
            )
    else:
        content = text + (
            "\n[This model has no vision capability. Attached images are unavailable; answer using the text only.]"
            if images
            else ""
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def extra_body(model):
    if model.startswith("qwen"):
        return {"enable_thinking": False}
    if model in ["glm-5", "deepseek-v4-flash"]:
        return {"thinking": {"type": "disabled"}}
    if model == "step-3.5-flash":
        return {"reasoning_effort": "low"}
    return {}


def sanitized_messages(msgs):
    result = json.loads(json.dumps(msgs))
    for m in result:
        if isinstance(m.get("content"), list):
            for c in m["content"]:
                if c.get("type") == "image_url" and c["image_url"]["url"].startswith(
                    "data:"
                ):
                    import hashlib

                    url = c["image_url"]["url"]
                    c["image_url"]["url"] = (
                        "sha256:" + hashlib.sha256(url.encode()).hexdigest()
                    )
    return result


class API:

    def __init__(
        self,
        timeout=300,
        retries=2,
        max_tokens=32768,
        temperature=1.0,
        purpose="worker",
    ):
        if purpose not in {"worker", "judge"}:
            raise ValueError("Unknown API purpose")
        expected = 1.0 if purpose == "worker" else 0.0
        if temperature != expected:
            raise ValueError(f"Evaluation requires {purpose} temperature={expected:g}")
        import httpx

        env = {
            **dotenv_values(Path(__file__).resolve().parents[3] / ".env"),
            **os.environ,
        }
        base = env.get("OPENAI_BASE_URL") or env.get("BASE_URL") or env.get("URL")
        key = env.get("OPENAI_API_KEY") or env.get("API_KEY") or env.get("KEY")
        if not base or not key:
            raise RuntimeError(
                "Missing endpoint or credential in OPENAI_BASE_URL / OPENAI_API_KEY"
            )
        self.client = OpenAI(
            base_url=base,
            api_key=key,
            max_retries=0,
            timeout=timeout,
            http_client=httpx.Client(trust_env=False, timeout=timeout),
        )
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.purpose = purpose

    def call(self, model, msgs, *, max_tokens=None, temperature=None):
        expected = 1.0 if self.purpose == "worker" else 0.0
        if (
            temperature is not None
            and float(temperature) != expected
            or self.temperature != expected
        ):
            raise ValueError(
                f"Evaluation requires {self.purpose} temperature={expected:g}"
            )
        model = model.removesuffix("-nothinking")
        if model not in MODELS:
            raise ValueError("Model outside frozen pool: " + model)
        for attempt in range(self.retries + 1):
            start = time.monotonic()
            record = dict(
                purpose=self.purpose,
                model=model,
                attempt=attempt + 1,
                messages=sanitized_messages(msgs),
                max_tokens=WORKER_TOKEN_LIMIT.get() or max_tokens or self.max_tokens,
                temperature=expected,
                timeout=self.timeout,
                extra_body=extra_body(model),
            )
            try:
                kwargs = dict(
                    model=model,
                    messages=msgs,
                    max_tokens=record["max_tokens"],
                    temperature=record["temperature"],
                )
                if record["extra_body"]:
                    kwargs["extra_body"] = record["extra_body"]
                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                text = choice.message.content
                usage = response.usage.model_dump() if response.usage else {}
                record.update(
                    response=text,
                    reasoning_content=getattr(
                        choice.message, "reasoning_content", None
                    ),
                    finish_reason=choice.finish_reason,
                    usage=usage,
                )
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(
                        "Empty final response (reasoning-only output is not a final answer)"
                    )
                return text
            except Exception as e:
                record["error"] = type(e).__name__
                record["status_code"] = getattr(e, "status_code", None)
                if attempt == self.retries:
                    raise RuntimeError(
                        f"API failed after {attempt + 1} attempts: {type(e).__name__}"
                    ) from None
                time.sleep(min(2**attempt, 8))
            finally:
                record["elapsed_sec"] = time.monotonic() - start
                if CALLS.get() is not None:
                    CALLS.get().append(record)

    def solve(
        self, item, model, instruction, context="", system=SYSTEM, max_tokens=None
    ):
        return self.call(
            model,
            messages(item, model, instruction, context, system),
            max_tokens=max_tokens,
        )


ROLES = [
    "Solve independently with careful reasoning.",
    "Independently verify facts, calculations, and assumptions.",
    "Explore an alternative solution and check edge cases.",
]
FIXED = ["cot", "reflection", "chain", "debate", "fullconnected", "tree", "sc"]


def fixed(method, item, api, model="glm-5"):

    def call(prompt, context=""):
        return api.solve(item, model, prompt, context)

    def joined(xs):
        return "\n\n".join((f"Agent {i + 1}:\n{x}" for i, x in enumerate(xs)))

    if method == "vanilla":
        return call("Answer directly and concisely; no explanation. " + FINAL)
    if method == "cot":
        return call("Think through the problem step by step, then answer. " + FINAL)
    if method == "reflection":
        draft = call("Work out a solution step by step. " + FINAL)
        critique = call(
            "Critique your earlier solution. Identify concrete mistakes or verify its reasoning.",
            draft,
        )
        return call(
            "Revise your solution using the critique and give the final answer. "
            + FINAL,
            joined([draft, critique]),
        )
    if method == "chain":
        a = call("Analyze the problem and derive the necessary intermediate results.")
        b = call("Continue the previous analysis, verify it, and solve the problem.", a)
        return call(
            "Check the proposed solution and produce the final answer. " + FINAL,
            joined([a, b]),
        )
    if method in ["debate", "fullconnected"]:
        state = [call(role + " " + FINAL) for role in ROLES]
        new = []
        for i, role in enumerate(ROLES):
            if method == "debate":
                context = joined([state[i], state[(i + 1) % 3]])
            else:
                context = joined(state)
            new.append(
                call(
                    role
                    + " Review the previous round, resolve disagreements, and update your answer. "
                    + FINAL,
                    context,
                )
            )
        return call(
            "Act as the final adjudicator. Weigh the evidence in all final-round solutions and answer. "
            + FINAL,
            joined(new),
        )
    if method == "tree":
        root = call(
            "Outline two distinct promising solution approaches; do not choose a final answer yet."
        )
        leaves = [
            call(
                "Develop approach "
                + str(i + 1)
                + " from the outline into a complete checked solution. "
                + FINAL,
                root,
            )
            for i in range(2)
        ]
        return call(
            "Compare the two branches and provide the best-supported final answer. "
            + FINAL,
            joined(leaves),
        )
    if method == "sc":
        from .scoring import prediction_key

        answers = [
            call("Solve independently, thinking step by step. " + FINAL)
            for _ in range(5)
        ]
        keys = [prediction_key(x, item) for x in answers]
        counts = Counter((k for k in keys if k))
        if not counts:
            return answers[0]
        best = max(counts, key=lambda k: (counts[k], -keys.index(k)))
        return answers[keys.index(best)]
    raise ValueError(method)
