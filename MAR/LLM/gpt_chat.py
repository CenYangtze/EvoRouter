import json
from contextlib import nullcontext, contextmanager
from contextvars import ContextVar
import threading
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import OpenAI, AsyncOpenAI

from MAR.LLM.price import cost_count
from MAR.LLM.llm import LLM
from MAR.LLM.llm_registry import LLMRegistry

load_dotenv()
MINE_BASE_URL = os.getenv("BASE_URL")
MINE_API_KEYS = os.getenv("API_KEY")

_PROVIDER_SEMAPHORES = {}
_PROVIDER_SEMAPHORE_LOCK = threading.Lock()
_REQUEST_TIMEOUT = ContextVar("llm_request_timeout", default=None)
_OMIT_OUTPUT_TOKEN_LIMIT = ContextVar("llm_omit_output_token_limit", default=False)


@contextmanager
def omit_output_token_limit():
    """Omit the API output limit for this call only, not other Worker requests."""
    token = _OMIT_OUTPUT_TOKEN_LIMIT.set(True)
    try:
        yield
    finally:
        _OMIT_OUTPUT_TOKEN_LIMIT.reset(token)


@contextmanager
def request_timeout(seconds):
    """Override timeout for this call/thread without changing other Worker calls."""
    value = float(seconds)
    if value <= 0:
        raise ValueError("Request timeout must be positive")
    token = _REQUEST_TIMEOUT.set(value)
    try:
        yield
    finally:
        _REQUEST_TIMEOUT.reset(token)


def _provider_model_slot(model_name: str):
    """Optional per-process cap; torchrun multiplies it by local rank count."""

    limit = max(0, int(os.getenv("LLM_PROVIDER_CONCURRENCY_PER_MODEL", "0")))
    if limit <= 0:
        return nullcontext()
    key = _openai_api_model(model_name).lower()
    with _PROVIDER_SEMAPHORE_LOCK:
        semaphore = _PROVIDER_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _PROVIDER_SEMAPHORES[key] = semaphore
    return semaphore


def _openai_timeout():
    if _REQUEST_TIMEOUT.get() is not None:
        return _REQUEST_TIMEOUT.get()
    timeout = os.getenv("OPENAI_TIMEOUT") or os.getenv("ROUTER_TIMEOUT")
    return float(timeout) if timeout else 120.0


def _openai_max_retries():
    return max(0, int(os.getenv("OPENAI_MAX_RETRIES", "0")))


def _retry_attempts():
    return stop_after_attempt(max(1, int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))))


def _retry_wait():
    return wait_random_exponential(max=float(os.getenv("LLM_RETRY_MAX_WAIT", "100")))


def _message_content_to_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
        extra = message
    else:
        content = getattr(message, "content", None)
        extra = getattr(message, "model_extra", None) or {}
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(
                    str(
                        item.get("text")
                        or item.get("content")
                        or item.get("value")
                        or ""
                    )
                )
            else:
                parts.append(
                    str(
                        getattr(item, "text", None)
                        or getattr(item, "content", None)
                        or item
                    )
                )
        text = "\n".join(part for part in parts if part)
        if text.strip():
            return text
    for attr in ("reasoning_content", "reasoning", "refusal"):
        value = extra.get(attr) if isinstance(extra, dict) else None
        if value is None and not isinstance(message, dict):
            value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    tool_calls = extra.get("tool_calls") if isinstance(extra, dict) else None
    if tool_calls is None and not isinstance(message, dict):
        tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        try:
            return json.dumps(tool_calls, ensure_ascii=False, default=str)
        except TypeError:
            return str(tool_calls)
    return "" if content is None else str(content)


def _require_response_text(response: str) -> str:
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError("LLM provider returned an empty text response")
    return response


def _messages_to_text(messages: Union[List[Dict], str]) -> str:
    if isinstance(messages, str):
        return messages
    parts = []
    for item in messages:
        content = item.get("content", "") if isinstance(item, dict) else item
        if content is None:
            continue
        if isinstance(content, str):
            parts.append(content)
        else:
            try:
                parts.append(json.dumps(content, ensure_ascii=False))
            except TypeError:
                parts.append(str(content))
    return "".join(parts)


def _usage_token_counts(chat_completion: Any):
    usage = getattr(chat_completion, "usage", None)
    if usage is None and isinstance(chat_completion, dict):
        usage = chat_completion.get("usage")
    if usage is None:
        return None, None

    def value(*names):
        for name in names:
            token_count = (
                usage.get(name)
                if isinstance(usage, dict)
                else getattr(usage, name, None)
            )
            if token_count is not None:
                return int(token_count)
        return None

    return value("prompt_tokens", "input_tokens"), value(
        "completion_tokens", "output_tokens"
    )


def _openai_extra_body(model_name: str) -> Dict[str, Any]:
    model_key = (model_name or "").lower()
    if model_key == "glm-5":
        return {"thinking": {"type": "disabled"}}
    if model_key == "claude-sonnet-5":
        return {"thinking": {"type": "disabled"}}
    if model_key == "deepseek-v4-pro-0813":
        return {"thinking": {"type": "disabled"}}
    if model_key.endswith("-nothinking") and "deepseek" in model_key:
        return {"thinking": {"type": "disabled"}}
    if model_key.endswith("-thinking") and "deepseek" in model_key:
        return {"thinking": {"type": "enabled"}}
    # The bare Worker name is also used by intermediate agents in auto mode.
    # Preserve explicit -thinking aliases (notably the Insight extractor) above.
    if model_key == "deepseek-v4-flash":
        return {"thinking": {"type": "disabled"}}
    if "qwen" in model_key:
        return {"enable_thinking": False}
    if model_key == "step-3.5-flash":
        # The provider ignores thinking-disable switches for this model.
        # Request low effort explicitly; this is not a separate token budget
        # and does not guarantee that the provider honors the effort level.
        return {"reasoning_effort": "low"}
    return {}


def _openai_api_model(model_name: str) -> str:
    model_key = (model_name or "").lower()
    if "deepseek" in model_key:
        for suffix in ("-nothinking", "-thinking"):
            if model_key.endswith(suffix):
                return model_name[: -len(suffix)]
    return model_name


@LLMRegistry.register("ALLChat")
class ALLChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @retry(wait=_retry_wait(), stop=_retry_attempts())
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        request_kwargs = {
            "messages": messages,
            "model": _openai_api_model(self.model_name),
            "temperature": temperature,
        }
        if not _OMIT_OUTPUT_TOKEN_LIMIT.get():
            request_kwargs["max_tokens"] = max_tokens
        extra_body = _openai_extra_body(self.model_name)
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        with OpenAI(
            base_url=os.environ.get("URL"),
            api_key=os.environ.get("KEY"),
            timeout=_openai_timeout(),
            max_retries=_openai_max_retries(),
        ) as client:
            with _provider_model_slot(self.model_name):
                chat_completion = client.chat.completions.create(**request_kwargs)
        response = _require_response_text(
            _message_content_to_text(chat_completion.choices[0].message)
        )
        prompt = _messages_to_text(messages)
        prompt_tokens, completion_tokens = _usage_token_counts(chat_completion)
        cost_count(
            prompt,
            response,
            self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return response

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        request_kwargs = {
            "messages": messages,
            "model": _openai_api_model(self.model_name),
            "temperature": temperature,
        }
        if not _OMIT_OUTPUT_TOKEN_LIMIT.get():
            request_kwargs["max_tokens"] = max_tokens
        extra_body = _openai_extra_body(self.model_name)
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        async with AsyncOpenAI(
            base_url=os.environ.get("URL"),
            api_key=os.environ.get("KEY"),
            timeout=_openai_timeout(),
            max_retries=_openai_max_retries(),
        ) as client:
            chat_completion = await client.chat.completions.create(**request_kwargs)
        response = _require_response_text(
            _message_content_to_text(chat_completion.choices[0].message)
        )
        prompt = _messages_to_text(messages)
        prompt_tokens, completion_tokens = _usage_token_counts(chat_completion)
        cost_count(
            prompt,
            response,
            self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return response
