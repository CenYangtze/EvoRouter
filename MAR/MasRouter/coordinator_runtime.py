"""Local Coordinator inference, dynamic batching, and execution traces."""

import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from MAR.MasRouter.llm_coordinator_router import LLMCoordinatorRouter


@dataclass
class CoordinatorStep:
    request_type: str
    messages: List[Dict[str, str]]
    prompt_text: str
    response_text: str
    max_tokens: int
    temperature: float
    accepted: bool = True
    rejection_reason: str = ""


def resolve_local_model_path(model_path: str) -> str:
    return os.path.abspath(os.path.expanduser(model_path))


class CoordinatorPolicy:

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = resolve_local_model_path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self._dtype(dtype)
        self._generation_lock = threading.Lock()
        self._judging_batch_condition = threading.Condition()
        self._judging_batch_queue: List[Dict[str, Any]] = []
        self._judging_batch_thread: Optional[threading.Thread] = None
        self._judging_batch_shutdown = False
        self._judging_batch_max_size = 1
        self._judging_batch_wait_seconds = 0.0
        self._judging_batch_requests = 0
        self._judging_generation_batches = 0
        self._judging_batch_size_counts: Dict[int, int] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype if self.device.startswith("cuda") else torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.use_cache = False
        self.supports_thinking_switch = (
            str(getattr(self.model.config, "model_type", "")).strip().lower() == "qwen3"
            or "qwen3" in str(self.model_path).lower()
        )
        self.model.to(self.device)

        self.model.eval()

    def configure_judging_batching(
        self,
        max_batch_size: int = 1,
        wait_ms: float = 50.0,
    ) -> None:
        """Batch concurrent Judging requests without concurrent model access.

        Rollout threads enqueue final-stage requests and wait on individual
        events.  One daemon owns local-model generation, combines compatible
        requests into a GPU batch, and returns each response to its caller.
        This preserves the model/tokenizer lock while avoiding serial forward
        passes through the shared Coordinator.
        """

        max_batch_size = int(max_batch_size)
        wait_ms = float(wait_ms)
        if max_batch_size < 1:
            raise ValueError("Judging batch size must be at least 1")
        if wait_ms < 0:
            raise ValueError("Judging batch wait must be non-negative")
        with self._judging_batch_condition:
            if self._judging_batch_queue:
                raise RuntimeError(
                    "Cannot reconfigure Judging batching with pending requests"
                )
            self._judging_batch_max_size = max_batch_size
            self._judging_batch_wait_seconds = wait_ms / 1000.0
            self._judging_batch_shutdown = False
            if max_batch_size > 1 and (
                self._judging_batch_thread is None
                or not self._judging_batch_thread.is_alive()
            ):
                self._judging_batch_thread = threading.Thread(
                    target=self._judging_batch_worker,
                    name="coordinator-judging-batcher",
                    daemon=True,
                )
                self._judging_batch_thread.start()

    @staticmethod
    def _judging_request_key(request: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            int(request["max_new_tokens"]),
            float(request["temperature"]),
            float(request["top_p"]),
            bool(request["enable_thinking"]),
        )

    def _judging_batch_worker(self) -> None:
        while True:
            with self._judging_batch_condition:
                while (
                    not self._judging_batch_queue and not self._judging_batch_shutdown
                ):
                    self._judging_batch_condition.wait()
                if self._judging_batch_shutdown and not self._judging_batch_queue:
                    return

                first_key = self._judging_request_key(self._judging_batch_queue[0])
                deadline = time.monotonic() + self._judging_batch_wait_seconds
                while True:
                    compatible_count = sum(
                        self._judging_request_key(request) == first_key
                        for request in self._judging_batch_queue
                    )
                    if compatible_count >= self._judging_batch_max_size:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._judging_batch_condition.wait(timeout=remaining)

                batch = []
                remaining_queue = []
                for request in self._judging_batch_queue:
                    if (
                        len(batch) < self._judging_batch_max_size
                        and self._judging_request_key(request) == first_key
                    ):
                        batch.append(request)
                    else:
                        remaining_queue.append(request)
                self._judging_batch_queue = remaining_queue

            try:
                with self._generation_lock:
                    prompt_texts, responses = self.generate_batch(
                        messages_batch=[request["messages"] for request in batch],
                        max_new_tokens=first_key[0],
                        temperature=first_key[1],
                        top_p=first_key[2],
                        enable_thinking=first_key[3],
                    )
                if len(prompt_texts) != len(batch) or len(responses) != len(batch):
                    raise RuntimeError(
                        "Batched Coordinator Judging returned "
                        f"{len(prompt_texts)} prompts/{len(responses)} responses "
                        f"for {len(batch)} requests"
                    )
                for request, prompt_text, response in zip(
                    batch, prompt_texts, responses
                ):
                    request["result"] = (prompt_text, response)
            except BaseException as exc:
                for request in batch:
                    request["error"] = exc
            finally:
                with self._judging_batch_condition:
                    batch_size = len(batch)
                    self._judging_batch_requests += batch_size
                    self._judging_generation_batches += 1
                    self._judging_batch_size_counts[batch_size] = (
                        self._judging_batch_size_counts.get(batch_size, 0) + 1
                    )
                for request in batch:
                    request["event"].set()

    def generate_concurrent(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        enable_thinking: bool = False,
    ) -> Tuple[str, str]:
        """Generate one response while dynamically batching concurrent callers."""

        if self._judging_batch_max_size <= 1:
            with self._generation_lock:
                return self.generate(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                )

        request = {
            "messages": deepcopy(messages),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "enable_thinking": bool(enable_thinking),
            "event": threading.Event(),
            "result": None,
            "error": None,
        }
        with self._judging_batch_condition:
            if self._judging_batch_shutdown:
                raise RuntimeError("Coordinator Judging batcher has been shut down")
            self._judging_batch_queue.append(request)
            self._judging_batch_condition.notify_all()
        request["event"].wait()
        if request["error"] is not None:
            raise request["error"]
        if request["result"] is None:
            raise RuntimeError("Coordinator Judging batcher returned no result")
        return request["result"]

    def generate_judging(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        enable_thinking: bool = False,
    ) -> Tuple[str, str]:
        """Backward-compatible name for dynamically batched generation."""

        return self.generate_concurrent(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )

    def judging_batching_stats(self) -> Dict[str, Any]:
        with self._judging_batch_condition:
            requests = int(self._judging_batch_requests)
            batches = int(self._judging_generation_batches)
            return {
                "enabled": bool(self._judging_batch_max_size > 1),
                "configured_max_batch_size": int(self._judging_batch_max_size),
                "wait_ms": float(self._judging_batch_wait_seconds * 1000.0),
                "requests": requests,
                "generation_batches": batches,
                "mean_batch_size": requests / batches if batches else 0.0,
                "batch_size_counts": dict(self._judging_batch_size_counts),
            }

    def shutdown_judging_batching(self) -> None:
        with self._judging_batch_condition:
            self._judging_batch_shutdown = True
            self._judging_batch_condition.notify_all()
            thread = self._judging_batch_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    @staticmethod
    def _dtype(dtype: str) -> torch.dtype:
        dtype = dtype.lower()
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        return torch.bfloat16

    def format_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        enable_thinking: bool = False,
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.supports_thinking_switch:
                kwargs["enable_thinking"] = bool(enable_thinking)
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    @torch.no_grad()
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        enable_thinking: bool = False,
    ) -> Tuple[str, str]:
        self.model.eval()
        prompt_text = self.format_messages(messages, enable_thinking=enable_thinking)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 1e-6
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(float(temperature), 1e-6)
            generation_kwargs["top_p"] = top_p
        else:
            generation_config = deepcopy(self.model.generation_config)
            generation_config.do_sample = False
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
            generation_kwargs["generation_config"] = generation_config
        output_ids = self.model.generate(
            **inputs,
            **generation_kwargs,
        )
        response_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        response_text = self.tokenizer.decode(
            response_ids, skip_special_tokens=True
        ).strip()
        self.model.eval()
        return prompt_text, response_text

    @torch.no_grad()
    def generate_many(
        self,
        messages: List[Dict[str, str]],
        count: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        enable_thinking: bool = False,
    ) -> Tuple[str, List[str]]:
        count = max(1, int(count))
        if count == 1:
            prompt_text, response = self.generate(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )
            return prompt_text, [response]
        if temperature <= 1e-6:
            raise ValueError(
                "generate_many requires a positive temperature when count > 1"
            )

        self.model.eval()
        prompt_text = self.format_messages(messages, enable_thinking=enable_thinking)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(float(temperature), 1e-6),
            top_p=top_p,
            num_return_sequences=count,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        responses = [
            self.tokenizer.decode(
                sequence[prompt_length:], skip_special_tokens=True
            ).strip()
            for sequence in output_ids
        ]
        self.model.eval()
        return prompt_text, responses

    @torch.no_grad()
    def generate_batch(
        self,
        messages_batch: Sequence[List[Dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        enable_thinking: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Generate one response for each distinct Coordinator prompt in a GPU batch."""

        messages_batch = list(messages_batch)
        if not messages_batch:
            return [], []
        if len(messages_batch) == 1:
            prompt_text, response = self.generate(
                messages=messages_batch[0],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )
            return [prompt_text], [response]

        self.model.eval()
        prompt_texts = [
            self.format_messages(messages, enable_thinking=enable_thinking)
            for messages in messages_batch
        ]
        previous_padding_side = self.tokenizer.padding_side
        try:
            # Left padding places generation immediately after each prompt.
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            do_sample = temperature > 1e-6
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = max(float(temperature), 1e-6)
                generation_kwargs["top_p"] = top_p
            else:
                generation_config = deepcopy(self.model.generation_config)
                generation_config.do_sample = False
                generation_config.temperature = None
                generation_config.top_p = None
                generation_config.top_k = None
                generation_kwargs["generation_config"] = generation_config
            output_ids = self.model.generate(**inputs, **generation_kwargs)
            prompt_length = inputs["input_ids"].shape[-1]
            responses = [
                self.tokenizer.decode(
                    sequence[prompt_length:],
                    skip_special_tokens=True,
                ).strip()
                for sequence in output_ids
            ]
        finally:
            self.tokenizer.padding_side = previous_padding_side
            self.model.eval()
        return prompt_texts, responses


class CoordinatorRouter(LLMCoordinatorRouter):
    def __init__(
        self, coordinator_policy: CoordinatorPolicy, *args: Any, **kwargs: Any
    ):
        kwargs.setdefault("coordinator_model", coordinator_policy.model_path)
        super().__init__(*args, **kwargs)
        self.coordinator_policy = coordinator_policy
        self.coordinator_steps: List[CoordinatorStep] = []
        self._queued_coordinator_outputs: Dict[
            str,
            List[Tuple[str, str, Optional[float]]],
        ] = {}

    def queue_coordinator_output(
        self,
        request_type: str,
        prompt_text: str,
        response_text: str,
        temperature: Optional[float] = None,
    ) -> None:
        self._queued_coordinator_outputs.setdefault(request_type, []).append(
            (prompt_text, response_text, temperature)
        )

    def run(self, query: str, prompt_file: str = "") -> Dict[str, Any]:
        self.coordinator_steps = []
        return super().run(query=query, prompt_file=prompt_file)

    def _call_coordinator(
        self, messages: List[Dict[str, str]], request_type: str
    ) -> str:
        max_tokens = self._coordinator_max_tokens_for_request(request_type)
        queued = self._queued_coordinator_outputs.get(request_type) or []
        if queued:
            prompt_text, response, queued_temperature = queued.pop(0)
            if not prompt_text:
                prompt_text = self.coordinator_policy.format_messages(messages)
            sampling_temperature = (
                self.coordinator_temperature
                if queued_temperature is None
                else float(queued_temperature)
            )
        else:
            sampling_temperature = self.coordinator_temperature
            use_dynamic_batching = request_type == "final" or bool(
                getattr(
                    self.coordinator_policy,
                    "batch_all_coordinator_requests",
                    False,
                )
            )
            if use_dynamic_batching and hasattr(
                self.coordinator_policy,
                "generate_concurrent",
            ):
                prompt_text, response = self.coordinator_policy.generate_concurrent(
                    messages=messages,
                    max_new_tokens=max_tokens,
                    temperature=sampling_temperature,
                )
            else:
                with self.coordinator_policy._generation_lock:
                    prompt_text, response = self.coordinator_policy.generate(
                        messages=messages,
                        max_new_tokens=max_tokens,
                        temperature=sampling_temperature,
                    )
        self.coordinator_steps.append(
            CoordinatorStep(
                request_type=request_type,
                messages=deepcopy(messages),
                prompt_text=prompt_text,
                response_text=response,
                max_tokens=max_tokens,
                temperature=sampling_temperature,
            )
        )
        self._record_event(
            "llm_call",
            {
                "request_type": request_type,
                "model": self.coordinator_model,
                "temperature": sampling_temperature,
                "max_tokens": max_tokens,
                "messages": deepcopy(messages),
                "response": response,
                "trace_context": {
                    "coordinator_stage": request_type,
                    "local_coordinator": True,
                },
            },
        )
        return response

    def _mark_last_coordinator_attempt_rejected(
        self,
        request_type: str,
        reason: str,
    ) -> None:
        for step in reversed(self.coordinator_steps):
            if step.request_type == request_type and step.accepted:
                step.accepted = False
                step.rejection_reason = str(reason)
                return


def create_policy(model_path, *, device=None, dtype="bfloat16"):
    """Load the downloaded EvoRouter model and its tokenizer."""
    return CoordinatorPolicy(
        model_path=str(model_path),
        device=device,
        dtype=dtype,
    )
