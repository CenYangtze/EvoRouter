import gc
import os
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from MAR.LLM.llm import LLM
from MAR.LLM.llm_registry import LLMRegistry


def _resolve_model_path(model_name: str) -> str:
    model_name = (
        model_name[len("local:") :] if model_name.startswith("local:") else model_name
    )
    return os.path.abspath(os.path.expanduser(model_name))


def _dtype_from_env() -> torch.dtype:
    dtype = os.getenv("LOCAL_HF_DTYPE", "bfloat16").lower()
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    return torch.bfloat16


def _messages_to_prompt(tokenizer: Any, messages: Union[List[Dict], str]) -> str:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


@LLMRegistry.register("LocalHFChat")
class LocalHFChat(LLM):
    _cache: Dict[Tuple[str, str, torch.dtype], Tuple[Any, Any]] = {}
    _cache_lock = threading.Lock()
    _generation_lock = threading.Lock()

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = _resolve_model_path(model_name)

    @classmethod
    def _release_cached_models_locked(cls) -> None:
        """Fully release the resident model before another one is loaded."""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        while cls._cache:
            _, (tokenizer, model) = cls._cache.popitem()
            del model
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def clear_cache(cls) -> None:
        # Generation is serialized, so acquiring this lock also guarantees that
        # no active call still owns references to the cached model.
        with cls._generation_lock:
            with cls._cache_lock:
                cls._release_cached_models_locked()

    def _load(self) -> Tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = os.getenv("LOCAL_HF_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = _dtype_from_env() if device.startswith("cuda") else torch.float32
        cache_key = (self.model_path, device, dtype)
        if cache_key not in self._cache:
            with self._cache_lock:
                if cache_key not in self._cache:
                    # Only one local Worker backbone may remain resident.  This
                    # release must finish before from_pretrained starts allocating
                    # the replacement model, otherwise mixed-backbone rollouts can
                    # transiently require memory for both models and OOM.
                    self._release_cached_models_locked()
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    model.to(device)
                    model.eval()
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    self._cache[cache_key] = (tokenizer, model)
        return self._cache[cache_key]

    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        if max_tokens is None:
            max_tokens = min(self.DEFAULT_MAX_TOKENS, 2048)
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        with self._generation_lock:
            tokenizer, model = self._load()
            prompt = _messages_to_prompt(tokenizer, messages)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            do_sample = temperature is not None and temperature > 1e-6
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = max(float(temperature), 1e-6)
                generation_kwargs["top_p"] = float(os.getenv("LOCAL_HF_TOP_P", "0.9"))
            else:
                generation_config = deepcopy(model.generation_config)
                generation_config.do_sample = False
                generation_config.temperature = None
                generation_config.top_p = None
                generation_config.top_k = None
                generation_kwargs["generation_config"] = generation_config
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        return self.gen(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            num_comps=num_comps,
        )
