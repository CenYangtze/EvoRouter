import base64
import json
import math
import mimetypes
import os
import re
import runpy
import time
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from dotenv import load_dotenv
from loguru import logger

from MAR.LLM import LLMRegistry
from MAR.Skills import select_skills_for_task
from MAR.Utils.globals import CompletionTokens, Cost, PromptTokens

COORDINATOR_MODEL = "assets/weights/EvoRouter"
DEFAULT_COORDINATOR_PLAN_MAX_TOKENS = 3072
DEFAULT_COORDINATOR_FINAL_MAX_TOKENS = 1024

MODEL_POOL = [
    {
        "Name": "glm-5",
        "Description": (
            "A MoE model for complex software engineering and agentic reasoning, combining large-scale "
            "pretraining, asynchronous reinforcement learning, and sparse attention for "
            "efficient long-context processing. This model does not support multimodal input."
        ),
        "CostCoefficient": 40.0,
        "CostAccounting": "flops",
        "SupportsVision": False,
    },
    {
        "Name": "qwen3.5-397b-a17b",
        "Description": (
            "Qwen’s flagship native multimodal MoE model, combining hybrid linear/full attention with "
            "strong reasoning, coding, agentic, multilingual, and vision capabilities."
        ),
        "CostCoefficient": 17.0,
        "CostAccounting": "flops",
        "SupportsVision": True,
    },
    {
        "Name": "deepseek-v4-flash",
        "Description": (
            "A compact MoE model delivering near-Pro reasoning and strong agent performance at lower "
            "latency and cost, with 1M context and sparse-attention optimizations for efficient long-context inference. "
            "This model does not support multimodal input."
        ),
        "CostCoefficient": 13.0,
        "CostAccounting": "flops",
        "SupportsVision": False,
    },
    {
        "Name": "step-3.5-flash",
        "Description": (
            "An efficiency-focused sparse-MoE model for fast reasoning, coding, and agents, using "
            "multi-token prediction and hybrid attention to achieve high throughput with only 11B parameters "
            "activated per token. This model does not support multimodal input."
        ),
        "CostCoefficient": 11.0,
        "CostAccounting": "flops",
        "SupportsVision": False,
    },
    {
        "Name": "qwen3.5-122b-a10b",
        "Description": (
            "A mid-sized native multimodal MoE model balancing capability and deployment cost, with "
            "hybrid attention, strong reasoning and agent abilities, long-context support, and broad multilingual "
            "coverage."
        ),
        "CostCoefficient": 10.0,
        "CostAccounting": "flops",
        "SupportsVision": True,
    },
    {
        "Name": "qwen3.5-35b-a3b",
        "Description": (
            "A lightweight native multimodal MoE model designed for efficient deployment, retaining "
            "Qwen3.5’s reasoning, coding, agentic, multilingual, and vision capabilities while activating just 3B "
            "parameters."
        ),
        "CostCoefficient": 3.0,
        "CostAccounting": "flops",
        "SupportsVision": True,
    },
]

# BudgetBench v3 has one Worker pool and one accounting contract: FLOPs.
# Keep the helper as a small compatibility surface for evaluation scripts, but
# reject the retired token profile instead of silently changing accounting.
FLOPS_MODEL_POOL = MODEL_POOL
MODEL_POOL_PROFILES = {"flops": MODEL_POOL}


def model_pool_for_profile(profile: str = "flops") -> List[Dict[str, Any]]:
    normalized = str(profile).strip().lower()
    if normalized != "flops":
        raise ValueError("Only the 'flops' Worker model-pool profile is supported")
    return deepcopy(MODEL_POOL)


MAS_STATE_READY = "Ready"
MAS_STATE_RUNNING = "Running"
MAS_STATE_JUDGING = "Judging"
MAS_STATE_FINISHED = "Finished"

FIXED_WORKER_TEMPERATURE = 1.0
MAX_MAS_ROUNDS = 4
MAX_VISIBLE_SKILLS = 7
INFORMATION_SEARCH_SKILL = "InformationSearch"
VISUAL_EVIDENCE_SKILL = "VisualEvidence"
VISION_UNAVAILABLE_TEXT = "你不具有多模态分析能力，所以不可见当前任务图片。"

COORDINATOR_SYSTEM_PROMPT_TEMPLATE = """You are a multi-agent router. Based on the incoming task, you need to design an appropriate multi-agent system that will execute and answer according to your plan.

Your plan should simultaneously balance task difficulty and budget, designing an efficient multi-agent architecture while ensuring accuracy.

After receiving the task requirements, you must first organize your reasoning inside the <analysis> and </analysis> tags. The reasoning must thoroughly consider the task type and difficulty to assign appropriate parameters to the multi-agent system.

Once reasoning is complete, you need to output the multi-agent design plan in JSON format within the <plan> and </plan> tags. This includes two global parameters, collaboration mode and rounds, as well as the ID, base model, skill, and prompt for each agent. The standard format is as follows:
{"collaboration":"value","rounds":1,"agents":[{"id":"value","model":"value","skill":"value","prompt":"value"},{"id":"value","model":"value","skill":"value","prompt":"value"},…]}

Every Worker automatically receives the original question and answer choices. Passages and images remain available only through their authorized Skills.

Inside JSON strings, escape every literal backslash as a double backslash, including mathematical notation; escape embedded quotation marks and newlines as valid JSON. Follow the advertised agent-count and rounds requirements for the chosen collaboration.

After the multi-agent system completes its execution, you will receive the running results. You must first organize your reasoning inside the <analysis> and </analysis> tags, and finally provide the direct response within the <answer> and </answer> tags. This step is primarily to standardize the output format according to the prompt's instructions; you do not need to perform additional reasoning or modify the answer—simply check the answer for completeness.

Available Models:
{{models}}
Available Skills (One per agent):
{{skills}}
Available Collaboration Modes (and parameter requirements):
{{collaborations}}"""

WORKER_SYSTEM_PROMPT = """You are a helpful assistant.
You are participating in a multi-agent reasoning process where a conductor organizes a multi-agent architecture to solve a problem, and you will act as one of the agents involved in solving it.

Your task is to:

+ Complete the requirements of the instruction by combining responses received from other agents (if any) and the skill provided by the conductor, offering a concise explanation; or

+ Provide helpful and relevant context, background knowledge, or insights related to the instruction, even if you cannot fully answer it.

Important Constraints:

+ Keep your response clear, concise, and informative (preferably under 1024 tokens). Your response will help guide other agents in their subsequent steps.

+ Stay strictly on-topic. Do not include irrelevant or generic content."""

STATIC_DESCRIPTION_CATEGORIES = (
    "collaborations",
    "parameters",
    "models",
    "skills",
)
STATIC_PARAMETER_DESCRIPTIONS = {
    "collaboration": "Use one available collaboration name.",
    "rounds": "Loop count. It is ignored for IO, counts iterations for CoT/Chain/Reflection, and counts synchronous rounds for FullConnected/Debate.",
    "agents": "Use one or more distinct worker roles.",
    "model": "Use one available model name per worker.",
    "skill": "Use one available Skill name per worker.",
    "temperature": "Fixed at 1.0 for every Worker call.",
    "prompt": "Give each worker a concrete subtask.",
}


IO_WORKER_INSTRUCTION = "Collaboration Mode: IO. Directly provide the precise result according to the instruction, along with a brief explanation."
CHAIN_FIRST_WORKER_INSTRUCTION = (
    "Collaboration Mode: Chain, first Agent, iteration {round_index} of {total_rounds}. Develop and refine your assigned "
    "result across your own iterations. On the final iteration, provide a clear result the next Agent can use."
)
CHAIN_MIDDLE_WORKER_INSTRUCTION = (
    "Collaboration Mode: Chain, intermediate Agent, iteration {round_index} of {total_rounds}. Use the previous Agent's "
    "final result as upstream context throughout your iterations, refine your own prior response when it is appended, "
    "and on the final iteration provide a clear result for the downstream Agent."
)
CHAIN_FINAL_WORKER_INSTRUCTION = (
    "Collaboration Mode: Chain, final Agent, iteration {round_index} of {total_rounds}. Use the previous Agent's final "
    "result as upstream context throughout your iterations, refine your own prior response when it is appended, and on "
    "the final iteration provide the final result of the chain call."
)
COT_WORKER_INSTRUCTION = (
    "Collaboration Mode: CoT, iteration {round_index} of {total_rounds}. Pay attention to the iteration progress, "
    "provide the answer in the final round of reasoning, and reason carefully according to the instruction in the "
    "previous rounds."
)
FULL_CONNECTED_INITIAL_INSTRUCTION = (
    "Collaboration Mode: FullConnected, round 1 of {total_rounds}. Independently solve the instruction assigned to "
    "you, and produce a clear draft for all peers to review in subsequent rounds."
)
FULL_CONNECTED_REVISION_INSTRUCTION = (
    "Collaboration Mode: FullConnected, round {round_index} of {total_rounds}. Integrate previous-round responses from "
    "yourself and every peer, resolve conflicts, and return a revised result that reflects the full group evidence."
)
DEBATE_INITIAL_INSTRUCTION = (
    "Collaboration Mode: Debate, round 1 of {total_rounds}. Independently solve the instruction assigned to you, and "
    "produce a clear draft for all peers to review in subsequent rounds."
)
DEBATE_REBUTTAL_INSTRUCTION = (
    "Collaboration Mode: Debate, round {round_index} of {total_rounds}. Directly test the other debaters' previous-round "
    "claims, rebut material errors, acknowledge stronger evidence, and update your position."
)
DEBATE_FINAL_INSTRUCTION = (
    "Collaboration Mode: Debate, final round {round_index} of {total_rounds}. Address the strongest remaining "
    "counterarguments and return your final result."
)
REFLECTION_INITIAL_INSTRUCTION = (
    "Collaboration Mode: Reflection, iteration 1 of {total_rounds}. Produce a careful initial answer and make the "
    "reasoning and possible uncertainty clear enough for your subsequent self-reflection."
)
REFLECTION_REVISION_INSTRUCTION = (
    "Collaboration Mode: Reflection, iteration {round_index} of {total_rounds}. Evaluate your previous response for "
    "concrete errors or omissions and decide whether it needs modification. If it is already sound, retain it; "
    "otherwise return a corrected, complete answer."
)

FULL_CONNECTED_SELF_CONTEXT = "Your response from the previous round:\n{response}"
FULL_CONNECTED_PEER_CONTEXT = (
    "Other agents' responses from the previous round:\n{responses}"
)
DEBATE_SELF_CONTEXT = "Your position from the previous round:\n{response}"
DEBATE_PEER_CONTEXT = "Other debaters' positions from the previous round:\n{responses}"
REFLECTION_SELF_CONTEXT = (
    "Your response from the previous Reflection round:\n{response}"
)


def _names(items: List[Dict[str, Any]]) -> List[str]:
    return [item["Name"] for item in items]


def _load_reasoning_profile() -> List[Dict[str, Any]]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    profile_path = os.path.join(root, "MAR", "Agent", "reasoning_profile.py")
    return runpy.run_path(profile_path)["reasoning_profile"]


class LLMCoordinatorRouter(torch.nn.Module):
    """Execute the Coordinator plan and collect Worker responses."""

    def __init__(
        self,
        coordinator_model: str = COORDINATOR_MODEL,
        model_pool: Optional[List[Dict[str, Any]]] = None,
        skill_pool: Optional[List[Dict[str, Any]]] = None,
        collaboration_pool: Optional[List[Dict[str, Any]]] = None,
        max_agent: int = 0,
        min_agent: int = 0,
        mas_update_freq: int = 1,
        max_rounds: int = 0,
        coordinator_temperature: float = 0.0,
        agent_temperature: float = FIXED_WORKER_TEMPERATURE,
        terminal_temperature: Optional[float] = FIXED_WORKER_TEMPERATURE,
        coordinator_max_tokens: Optional[int] = None,
        agent_max_tokens: int = 4096,
        terminal_max_tokens: Optional[int] = None,
        capture_events: bool = False,
        max_mas_revisions: int = 0,
        workflow_protocol: bool = False,
        few_shot_workflows: bool = True,
        update_during_workflow: bool = False,
        review_after_workflow: bool = False,
        gemini_terminal_mode: str = "nothinking",
        gemini_intermediate_mode: str = "auto",
        deepseek_terminal_mode: str = "nothinking",
        deepseek_intermediate_mode: str = "nothinking",
        state_format_retries: int = 2,
        state_think_turn_limit: int = 4,
        static_definition_updates: bool = False,
        static_update_retries: int = 2,
        coordinator_plan_max_tokens: Optional[int] = None,
        coordinator_final_max_tokens: Optional[int] = None,
    ):
        super().__init__()
        load_dotenv()
        self.coordinator_model = coordinator_model
        self.model_pool = model_pool or deepcopy(MODEL_POOL)
        self._auto_select_skills = skill_pool is None
        self.skill_pool = list(skill_pool or select_skills_for_task())
        if not self.skill_pool or len(self.skill_pool) > MAX_VISIBLE_SKILLS:
            raise ValueError(
                "Coordinator Skill pool must contain between one and seven task-relevant Skills"
            )
        self.collaboration_pool = collaboration_pool or deepcopy(
            _load_reasoning_profile()
        )
        self._validate_catalog_descriptions(self.model_pool, "Model")
        self._validate_catalog_descriptions(self.skill_pool, "Skill")
        self.max_agent = None
        self.min_agent = None
        # Each evaluation performs planning, Worker execution, and final synthesis.
        self.mas_update_freq = 0
        self.mas_updates_enabled = False
        self.max_rounds = 1
        self.coordinator_temperature = coordinator_temperature
        self.agent_temperature = FIXED_WORKER_TEMPERATURE
        self.terminal_temperature = FIXED_WORKER_TEMPERATURE
        self.coordinator_plan_max_tokens = self._resolve_coordinator_token_limit(
            stage_value=coordinator_plan_max_tokens,
            shared_value=coordinator_max_tokens,
            default=DEFAULT_COORDINATOR_PLAN_MAX_TOKENS,
            name="coordinator_plan_max_tokens",
        )
        self.coordinator_final_max_tokens = self._resolve_coordinator_token_limit(
            stage_value=coordinator_final_max_tokens,
            shared_value=coordinator_max_tokens,
            default=DEFAULT_COORDINATOR_FINAL_MAX_TOKENS,
            name="coordinator_final_max_tokens",
        )
        # Backward-compatible alias for callers that still inspect the former
        # shared limit. Planning is the conservative interpretation.
        self.coordinator_max_tokens = self.coordinator_plan_max_tokens
        self.agent_max_tokens = agent_max_tokens
        self.terminal_max_tokens = (
            int(terminal_max_tokens)
            if terminal_max_tokens is not None and int(terminal_max_tokens) > 0
            else int(agent_max_tokens)
        )
        self.capture_events = bool(capture_events)
        self.episode_assets: List[Dict[str, Any]] = []
        self.episode_asset_root: Optional[Path] = None
        self.max_mas_revisions = 0
        self.workflow_protocol = False
        self.few_shot_workflows = bool(few_shot_workflows)
        self.update_during_workflow = False
        self.review_after_workflow = False
        if gemini_terminal_mode not in {"auto", "thinking", "nothinking"}:
            raise ValueError(
                "gemini_terminal_mode must be auto, thinking, or nothinking"
            )
        if gemini_intermediate_mode not in {"auto", "thinking", "nothinking"}:
            raise ValueError(
                "gemini_intermediate_mode must be auto, thinking, or nothinking"
            )
        if deepseek_terminal_mode not in {"auto", "thinking", "nothinking"}:
            raise ValueError(
                "deepseek_terminal_mode must be auto, thinking, or nothinking"
            )
        if deepseek_intermediate_mode not in {"auto", "thinking", "nothinking"}:
            raise ValueError(
                "deepseek_intermediate_mode must be auto, thinking, or nothinking"
            )
        self.gemini_terminal_mode = gemini_terminal_mode
        self.gemini_intermediate_mode = gemini_intermediate_mode
        self.deepseek_terminal_mode = deepseek_terminal_mode
        self.deepseek_intermediate_mode = deepseek_intermediate_mode
        self.state_format_retries = 0
        # Kept as a serialized/CLI compatibility field. MAS states now require
        # an actionable terminal tag in every accepted Coordinator response.
        self.state_think_turn_limit = max(0, int(state_think_turn_limit))
        # Post-task static-definition maintenance is intentionally suspended.
        # Keep the constructor argument for compatibility with older commands.
        self.static_definition_updates = False
        self.static_update_retries = max(0, int(static_update_retries))

        self.model_names = _names(self.model_pool)
        self.skill_names = _names(self.skill_pool)
        self.collaboration_names = _names(self.collaboration_pool)
        self.static_descriptions = self._preset_static_descriptions()
        self.static_definition_version = 1
        self.trace: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        self.current_state: Optional[str] = None
        self.state_history: List[Dict[str, Any]] = []

    def set_model_pool(self, model_pool: Sequence[Dict[str, Any]]) -> None:
        """Swap the visible worker order between episodes without rebuilding the policy."""
        self._validate_catalog_descriptions(model_pool, "Model")
        current_descriptions = self.static_descriptions.get("models", {})
        self.model_pool = deepcopy(list(model_pool))
        self.model_names = _names(self.model_pool)
        self.static_descriptions["models"] = {
            item["Name"]: current_descriptions.get(
                item["Name"],
                str(item.get("Description", "")),
            )
            for item in self.model_pool
        }

    def set_skill_pool(self, skill_pool: Sequence[Dict[str, Any]]) -> None:
        """Expose only the task-relevant Skills for the current episode."""

        selected = deepcopy(list(skill_pool))
        if not selected or len(selected) > MAX_VISIBLE_SKILLS:
            raise ValueError(
                "Each task must expose between one and seven relevant Skills"
            )
        self._validate_catalog_descriptions(selected, "Skill")
        self.skill_pool = selected
        self.skill_names = _names(self.skill_pool)
        if hasattr(self, "static_descriptions"):
            self.static_descriptions["skills"] = {
                item["Name"]: str(item.get("Description", ""))
                for item in self.skill_pool
            }

    def set_episode_assets(
        self,
        assets: Sequence[Dict[str, Any]] | None,
        asset_root: str | Path | None = None,
    ) -> None:
        """Set Worker-only evidence; Coordinator prompt builders never read it."""

        self.episode_assets = deepcopy(list(assets or []))
        self.episode_asset_root = (
            Path(asset_root).expanduser().resolve() if asset_root is not None else None
        )

    def validate_and_normalize_external_plan(
        self,
        raw_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply the deployment validator without contaminating episode quality."""

        previous_quality = getattr(self, "_quality", None)
        self._quality = self._new_quality_state()
        try:
            raw_plan = self._repair_plan_execution_shape(raw_plan)
            self._validate_raw_plan(raw_plan)
            if self._has_quality_failure():
                issues = "; ".join(
                    str(item) for item in self._quality.get("issues", [])
                )
                raise ValueError(issues or "Invalid external MAS plan")
            normalized = self._normalize_plan(
                deepcopy(raw_plan),
                fallback=self._fallback_plan(),
            )
            if self._has_quality_failure():
                issues = "; ".join(
                    str(item) for item in self._quality.get("issues", [])
                )
                raise ValueError(issues or "External MAS plan normalization failed")
            return normalized
        finally:
            if previous_quality is None:
                del self._quality
            else:
                self._quality = previous_quality

    @staticmethod
    def _validate_catalog_descriptions(
        pool: Sequence[Dict[str, Any]],
        kind: str,
    ) -> None:
        max_length = 256 if kind == "Model" else 100
        for item in pool:
            description = str(item.get("Description", "")).strip()
            if not description:
                raise ValueError(
                    f"{kind} {item.get('Name', '<unknown>')} needs a description"
                )
            if len(description) > max_length:
                raise ValueError(
                    f"{kind} {item.get('Name', '<unknown>')} description exceeds {max_length} characters"
                )

    def _select_default_skills_for_query(self, query: str) -> None:
        if not self._auto_select_skills:
            return
        text = str(query or "")
        lower = text.lower()
        math_like = bool(
            re.search(
                r"\b(?:calculate|compute|equation|probability|geometry|algebra|integer|number)\b|[=+*/]",
                lower,
            )
        )
        choice_like = "choices:" in lower or bool(re.search(r"\n\s*[A-D][.)]\s", text))
        tags = []
        output_format = ""
        if choice_like:
            tags.append("multiple_choice_reasoning")
        if math_like:
            tags.append("mathematics")
            output_format = "number"
        selected = select_skills_for_task(
            task_type="reasoning" if (math_like or choice_like) else "qa",
            capability_tags=tags,
            output_format=output_format,
            instruction=text,
        )
        self.set_skill_pool(selected)

    def run(self, query: str, prompt_file: str = "") -> Dict[str, Any]:
        self._select_default_skills_for_query(query)
        self._events = []
        self.current_state = None
        self.state_history = []
        self._record_event(
            "run_start",
            {
                "query": query,
                "coordinator_model": self.coordinator_model,
                "coordinator_plan_max_tokens": self.coordinator_plan_max_tokens,
                "coordinator_final_max_tokens": self.coordinator_final_max_tokens,
                "model_pool": self.model_pool,
                "skill_pool": self.skill_pool,
                "collaboration_pool": self.collaboration_pool,
                "max_agent": self.max_agent,
                "min_agent": self.min_agent,
                "max_rounds": self.max_rounds,
                "mas_update_freq": self.mas_update_freq,
                "mas_updates_enabled": self.mas_updates_enabled,
                "max_mas_revisions": self.max_mas_revisions,
                "workflow_protocol": self.workflow_protocol,
                "update_during_workflow": self.update_during_workflow,
                "review_after_workflow": self.review_after_workflow,
                "state_format_retries": self.state_format_retries,
                "state_think_turn_limit": self.state_think_turn_limit,
                "static_definition_updates": self.static_definition_updates,
                "static_definition_scope": "router_instance",
                "static_definition_persistent": False,
                "static_update_retries": self.static_update_retries,
                "static_definition_version": self.static_definition_version,
                "static_descriptions": deepcopy(self.static_descriptions),
            },
        )
        self._quality = self._new_quality_state()
        if self.workflow_protocol:
            return self._run_workflow_protocol(query=query, prompt_file=prompt_file)
        return self._run_mas_state_machine(query=query, prompt_file=prompt_file)

    @staticmethod
    def _new_quality_state() -> Dict[str, Any]:
        return {
            "format_errors": 0,
            "json_parse_failures": 0,
            "non_strict_json": 0,
            "invalid_plan": 0,
            "invalid_update": 0,
            "problem_nodes": 0,
            "compiled_repairs": 0,
            "repairs": [],
            "issues": [],
            "state_retries": 0,
            "state_think_turns": 0,
        }

    def _run_workflow_protocol(self, query: str, prompt_file: str) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        plan = self._create_workflow_plan(query)
        self._record_event("initial_plan", {"plan": deepcopy(plan)})

        if self._has_quality_failure():
            return self._finish_run(query, "", plan, history)

        revision_count = 0
        while True:
            self._execute_mas_plan(
                query=query,
                plan=plan,
                history=history,
                workflow_index=revision_count,
            )
            if self._has_quality_failure():
                return self._finish_run(query, "", plan, history)

            if plan.get("protocol") == "workflow_v2" and not self.review_after_workflow:
                final_answer = self._terminal_workflow_answer(query, plan, history)
                return self._finish_run(query, final_answer, plan, history)

            final_answer, revision_plan = self._finalize_or_revise(
                query=query,
                plan=plan,
                history=history,
                prompt_file=prompt_file,
            )
            if final_answer is not None:
                return self._finish_run(query, final_answer, plan, history)

            if revision_plan is None:
                self._quality_issue(
                    "format_errors",
                    "final response contained neither answer nor correction MAS",
                )
                return self._finish_run(query, "", plan, history)

            if (
                self.max_mas_revisions is not None
                and revision_count >= self.max_mas_revisions
            ):
                self._quality_issue(
                    "invalid_plan",
                    f"Coordinator requested more than {self.max_mas_revisions} MAS revision(s)",
                )
                return self._finish_run(query, "", plan, history)

            revision_count += 1
            plan = revision_plan
            self._record_event(
                "revision_plan",
                {
                    "revision_index": revision_count,
                    "plan": deepcopy(plan),
                },
            )

    def _run_mas_state_machine(self, query: str, prompt_file: str) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        self._transition_state(None, MAS_STATE_READY, trigger="run_start")
        plan = self._create_initial_plan(query)
        self._record_event("initial_plan", {"plan": deepcopy(plan)})
        if self._has_quality_failure():
            return self._finish_run(query, "", plan, history)

        self._transition_state(MAS_STATE_READY, MAS_STATE_RUNNING, trigger="plan")
        executed, plan_exhausted = self._execute_mas_state_chunk(
            query=query,
            plan=plan,
            history=history,
            cycle_index=0,
        )
        self._record_event(
            "running_chunk_complete",
            {
                "cycle_index": 0,
                "worker_calls": executed,
                "plan_exhausted": plan_exhausted,
                "plan": deepcopy(plan),
            },
        )
        if self._has_quality_failure():
            return self._finish_run(query, "", plan, history)

        self._transition_state(
            MAS_STATE_RUNNING,
            MAS_STATE_JUDGING,
            trigger="mas_complete",
        )
        final_answer, revision_plan = self._finalize_or_revise(
            query=query,
            plan=plan,
            history=history,
            prompt_file=prompt_file,
        )
        if revision_plan is not None:
            self._quality_issue(
                "invalid_plan",
                "The two-stage protocol does not permit answer-stage replanning",
            )
        if self._has_quality_failure() or final_answer is None:
            return self._finish_run(query, "", plan, history)
        self._transition_state(MAS_STATE_JUDGING, MAS_STATE_FINISHED, trigger="answer")
        return self._finish_run(query, final_answer, plan, history)

    def _transition_state(
        self,
        previous: Optional[str],
        current: str,
        trigger: str,
    ) -> None:
        if self.current_state is not None and previous != self.current_state:
            raise RuntimeError(
                f"Invalid Coordinator state transition source: expected {self.current_state}, got {previous}"
            )
        transition = {
            "from": previous,
            "to": current,
            "trigger": trigger,
            "index": len(self.state_history),
        }
        self.current_state = current
        self.state_history.append(transition)
        self._record_event("state_transition", deepcopy(transition))

    def _execute_mas_state_chunk(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
        cycle_index: int,
    ) -> Tuple[int, bool]:
        """Run one MAS chunk before an update activation or final Judging."""
        previous_round_outputs: Dict[str, str] = {}
        node_run_count = 0
        round_idx = 0
        total_rounds = self._round_count(plan)
        active_agents = [
            agent for agent in plan.get("agents", []) if not agent.get("disabled")
        ]
        planned_calls = total_rounds * len(active_agents)
        chunk_limit = (
            self.mas_update_freq if self.mas_updates_enabled else planned_calls
        )
        chunk_label: Any = (
            chunk_limit if self.mas_updates_enabled else "all (updates disabled)"
        )
        logger.info(
            f"LLMCoordinatorRouter state=Running collaboration={plan['collaboration']} "
            f"rounds={total_rounds} chunk_limit={chunk_label}"
        )
        logger.info(
            f"LLMCoordinatorRouter agents={[agent['id'] for agent in plan['agents']]}"
        )

        if plan["collaboration"] == "Chain":
            node_run_count = self._execute_chain_plan(
                query=query,
                plan=plan,
                history=history,
                workflow_index=cycle_index,
                call_limit=chunk_limit,
                prior_cycle_context=self._prior_mas_cycle_context(history, cycle_index),
            )
            return node_run_count, node_run_count >= planned_calls

        while round_idx < total_rounds and node_run_count < chunk_limit:
            current_round_outputs: Dict[str, str] = {}
            for agent_index, agent in enumerate(plan["agents"]):
                if agent.get("disabled"):
                    continue
                context = self._agent_context(
                    plan=plan,
                    agent_index=agent_index,
                    current_round_outputs=current_round_outputs,
                    previous_round_outputs=previous_round_outputs,
                    history=history,
                    round_idx=round_idx,
                )
                prior_cycle_context = self._prior_mas_cycle_context(
                    history, cycle_index
                )
                if prior_cycle_context:
                    context = "\n\n".join(
                        part for part in (prior_cycle_context, context) if part
                    )
                output = self._run_agent(
                    query,
                    plan,
                    agent,
                    context,
                    round_idx,
                    is_terminal=self._is_terminal_agent(plan, agent),
                )
                current_round_outputs[agent["id"]] = output
                history.append(
                    {
                        "round": round_idx + 1,
                        "workflow_index": cycle_index,
                        "agent_id": agent["id"],
                        "agent_name": agent.get("name", agent["id"]),
                        "model": agent["model"],
                        "mode": agent.get("mode", "auto"),
                        "skill": agent["skill"],
                        "prompt": agent.get("prompt", ""),
                        "depends_on": deepcopy(agent.get("depends_on", [])),
                        "output": output,
                    }
                )
                node_run_count += 1
                if node_run_count >= chunk_limit:
                    break
            previous_round_outputs = current_round_outputs
            round_idx += 1

        return node_run_count, node_run_count >= planned_calls

    def _execute_chain_plan(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_index: int,
        call_limit: Optional[int] = None,
        prior_cycle_context: str = "",
    ) -> int:
        """Run every Chain Agent for all its iterations before transferring its final result."""
        total_rounds = self._round_count(plan)
        active_agents = [
            agent for agent in plan.get("agents", []) if not agent.get("disabled")
        ]
        maximum_calls = (
            call_limit if call_limit is not None else total_rounds * len(active_agents)
        )
        node_run_count = 0
        upstream_context = ""

        for agent in active_agents:
            own_previous_response = ""
            for round_idx in range(total_rounds):
                context = "\n\n".join(
                    part for part in (prior_cycle_context, upstream_context) if part
                )
                output = self._run_agent(
                    query,
                    plan,
                    agent,
                    context,
                    round_idx,
                    is_terminal=self._is_terminal_agent(plan, agent),
                    previous_response=own_previous_response,
                )
                history.append(
                    {
                        "round": round_idx + 1,
                        "workflow_index": workflow_index,
                        "agent_id": agent["id"],
                        "agent_name": agent.get("name", agent["id"]),
                        "model": agent["model"],
                        "mode": agent.get("mode", "auto"),
                        "skill": agent["skill"],
                        "prompt": agent.get("prompt", ""),
                        "depends_on": deepcopy(agent.get("depends_on", [])),
                        "output": output,
                    }
                )
                own_previous_response = output
                node_run_count += 1
                if node_run_count >= maximum_calls:
                    return node_run_count
            upstream_context = f"- [#1] {agent['id']}: {own_previous_response}"

        return node_run_count

    def _prior_mas_cycle_context(
        self,
        history: List[Dict[str, Any]],
        cycle_index: int,
    ) -> str:
        if cycle_index <= 0:
            return ""
        prior_items = [
            item for item in history if int(item.get("workflow_index", 0)) < cycle_index
        ][-8:]
        if not prior_items:
            return ""
        evidence = [
            {
                "cycle": int(item.get("workflow_index", 0)),
                "agent_id": item.get("agent_id"),
                "prompt": item.get("prompt", ""),
                "output": str(item.get("output", ""))[:2000],
            }
            for item in prior_items
        ]
        return (
            "Retained worker evidence from earlier MAS cycles. Treat it as fallible and improve on it:\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2)
        )

    def _execute_mas_plan(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_index: int = 0,
    ) -> None:
        if plan.get("protocol") == "workflow_v2":
            self._execute_workflow_plan(
                query=query,
                plan=plan,
                history=history,
                workflow_index=workflow_index,
            )
            return

        if plan.get("collaboration") == "Chain":
            logger.info(
                f"LLMCoordinatorRouter collaboration=Chain rounds={self._round_count(plan)}"
            )
            logger.info(
                f"LLMCoordinatorRouter agents={[agent['id'] for agent in plan['agents']]}"
            )
            self._execute_chain_plan(
                query=query,
                plan=plan,
                history=history,
                workflow_index=workflow_index,
            )
            return

        previous_round_outputs: Dict[str, str] = {}
        node_run_count = 0

        logger.info(
            f"LLMCoordinatorRouter collaboration={plan['collaboration']} rounds={self._round_count(plan)}"
        )
        logger.info(
            f"LLMCoordinatorRouter agents={[agent['id'] for agent in plan['agents']]}"
        )

        round_idx = 0
        while round_idx < self._round_count(plan):
            current_round_outputs: Dict[str, str] = {}
            idx = 0
            while idx < len(plan["agents"]):
                agent = plan["agents"][idx]
                if agent.get("disabled"):
                    idx += 1
                    continue

                context = self._agent_context(
                    plan=plan,
                    agent_index=idx,
                    current_round_outputs=current_round_outputs,
                    previous_round_outputs=previous_round_outputs,
                    history=history,
                    round_idx=round_idx,
                )
                output = self._run_agent(
                    query,
                    plan,
                    agent,
                    context,
                    round_idx,
                    is_terminal=self._is_terminal_agent(plan, agent),
                )
                current_round_outputs[agent["id"]] = output
                history.append(
                    {
                        "round": round_idx + 1,
                        "workflow_index": workflow_index,
                        "agent_id": agent["id"],
                        "agent_name": agent.get("name", agent["id"]),
                        "model": agent["model"],
                        "mode": agent.get("mode", "auto"),
                        "skill": agent["skill"],
                        "prompt": agent.get("prompt", ""),
                        "depends_on": deepcopy(agent.get("depends_on", [])),
                        "output": output,
                    }
                )
                node_run_count += 1

                if (
                    self.mas_updates_enabled
                    and node_run_count % self.mas_update_freq == 0
                ):
                    update = self._evaluate_update(query, plan, history)
                    plan = self._apply_update(plan, update)
                    self._record_event(
                        "plan_update",
                        {
                            "after_node_run_count": node_run_count,
                            "update": deepcopy(update),
                            "plan_after_update": deepcopy(plan),
                        },
                    )
                    if self._has_quality_failure():
                        return

                idx += 1

            previous_round_outputs = current_round_outputs
            round_idx += 1

    def _execute_workflow_plan(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_index: int,
    ) -> None:
        """Execute an explicit dependency DAG in topological list order."""
        node_run_count = 0
        idx = 0
        logger.info(
            "LLMCoordinatorRouter workflow_v2 steps="
            f"{[agent['id'] for agent in plan.get('agents', [])]}"
        )
        while idx < len(plan.get("agents", [])):
            agent = plan["agents"][idx]
            if agent.get("disabled"):
                idx += 1
                continue

            context = self._workflow_agent_context(
                agent=agent,
                history=history,
                workflow_index=workflow_index,
            )
            output = self._run_agent(
                query=query,
                plan=plan,
                agent=agent,
                context=context,
                round_idx=0,
                is_terminal=self._is_terminal_agent(plan, agent),
            )
            history.append(
                {
                    "round": 1,
                    "workflow_index": workflow_index,
                    "agent_id": agent["id"],
                    "agent_name": agent.get("name", agent["id"]),
                    "model_id": agent.get("model_id"),
                    "model": agent["model"],
                    "mode": agent.get("mode", "auto"),
                    "skill": agent["skill"],
                    "prompt": agent.get("prompt", ""),
                    "depends_on": deepcopy(agent.get("depends_on", [])),
                    "output": output,
                }
            )
            node_run_count += 1

            has_remaining = any(
                not candidate.get("disabled")
                for candidate in plan.get("agents", [])[idx + 1 :]
            )
            if (
                self.update_during_workflow
                and self.mas_updates_enabled
                and has_remaining
                and node_run_count % self.mas_update_freq == 0
            ):
                update = self._evaluate_update(query, plan, history)
                plan = self._apply_update(plan, update)
                self._record_event(
                    "plan_update",
                    {
                        "workflow_index": workflow_index,
                        "after_node_run_count": node_run_count,
                        "update": deepcopy(update),
                        "plan_after_update": deepcopy(plan),
                    },
                )
                if self._has_quality_failure():
                    return
            idx += 1

    def _workflow_agent_context(
        self,
        agent: Dict[str, Any],
        history: List[Dict[str, Any]],
        workflow_index: int,
    ) -> str:
        dependencies = list(agent.get("depends_on") or [])
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in history:
            if item.get("workflow_index") != workflow_index:
                continue
            agent_id = str(item.get("agent_id") or "")
            if agent_id:
                by_id[agent_id] = item

        parts = []
        if workflow_index > 0:
            previous_items = [
                item
                for item in history
                if int(item.get("workflow_index", 0)) < workflow_index
            ]
            if previous_items:
                previous = previous_items[-1]
                parts.append(
                    "Previous workflow terminal response under review:\n"
                    f"{previous.get('output', '')}"
                )
        for dependency in dependencies:
            item = by_id.get(str(dependency))
            if not item:
                continue
            parts.append(
                f"Prior step {item['agent_id']} subtask:\n{item.get('prompt', '')}\n\n"
                f"Prior step {item['agent_id']} result:\n{self._extract_worker_result(item.get('output', ''))}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _is_terminal_agent(plan: Dict[str, Any], agent: Dict[str, Any]) -> bool:
        active = [
            candidate
            for candidate in plan.get("agents", [])
            if not candidate.get("disabled")
        ]
        return bool(active and active[-1].get("id") == agent.get("id"))

    def _finish_run(
        self,
        query: str,
        final_answer: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        answer_committed = bool(final_answer) and (
            self.current_state == MAS_STATE_FINISHED or self.workflow_protocol
        )
        if answer_committed:
            self._record_event(
                "answer_committed",
                {
                    "final_answer": final_answer,
                    "state": self.current_state,
                    "immutable": True,
                },
            )
        static_update = self._static_update_skipped(
            "feature_suspended" if answer_committed else "answer_not_committed"
        )
        result = {
            "query": query,
            "final_answer": final_answer,
            "final_plan": plan,
            "history": history,
            "state": self.current_state,
            "state_history": deepcopy(self.state_history),
            "quality": deepcopy(getattr(self, "_quality", {})),
            "static_definition_update": static_update,
            "static_definition_version": self.static_definition_version,
            "static_descriptions": deepcopy(self.static_descriptions),
        }
        self._record_event(
            "run_finish",
            {
                "final_answer": final_answer,
                "final_plan": deepcopy(plan),
            },
        )
        return result

    @staticmethod
    def _empty_static_description_changes() -> Dict[str, Dict[str, str]]:
        return {category: {} for category in STATIC_DESCRIPTION_CATEGORIES}

    def _preset_static_descriptions(self) -> Dict[str, Dict[str, str]]:
        return {
            "collaborations": {
                item["Name"]: str(item.get("Description", ""))
                for item in self.collaboration_pool
            },
            "parameters": deepcopy(STATIC_PARAMETER_DESCRIPTIONS),
            "models": {
                item["Name"]: str(item.get("Description", ""))
                for item in self.model_pool
            },
            "skills": {
                item["Name"]: str(item.get("Description", ""))
                for item in self.skill_pool
            },
        }

    def _static_description_names(self) -> Dict[str, set]:
        return {
            "collaborations": set(self.collaboration_names),
            "parameters": set(STATIC_PARAMETER_DESCRIPTIONS),
            "models": set(self.model_names),
            "skills": set(self.skill_names),
        }

    @staticmethod
    def _usage_snapshot() -> Dict[str, float]:
        return {
            "cost": float(Cost.instance().value),
            "prompt_tokens": float(PromptTokens.instance().value),
            "completion_tokens": float(CompletionTokens.instance().value),
        }

    @staticmethod
    def _restore_usage(snapshot: Dict[str, float]) -> None:
        Cost.instance().value = snapshot["cost"]
        PromptTokens.instance().value = snapshot["prompt_tokens"]
        CompletionTokens.instance().value = snapshot["completion_tokens"]

    def _static_update_skipped(self, reason: str) -> Dict[str, Any]:
        return {
            "status": "skipped",
            "action": "keep",
            "reason": reason,
            "attempts": 0,
            "changes": self._empty_static_description_changes(),
            "version_before": self.static_definition_version,
            "version_after": self.static_definition_version,
            "scope": "router_instance",
            "persistent": False,
            "usage": {"cost": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0},
            "errors": [],
        }

    def _run_static_definition_update(
        self,
        query: str,
        final_answer: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        version_before = self.static_definition_version
        usage_before = self._usage_snapshot()
        result = self._static_update_skipped("not_run")
        result.update(
            {
                "status": "failed",
                "action": "keep",
                "reason": "Static definition maintenance did not complete.",
                "version_before": version_before,
                "errors": [],
            }
        )
        conversation = self._static_definition_update_messages(
            query=query,
            final_answer=final_answer,
            plan=plan,
            history=history,
        )
        self._record_event(
            "static_definition_update_start",
            {
                "answer_already_committed": True,
                "version": version_before,
                "current_descriptions": deepcopy(self.static_descriptions),
            },
        )

        try:
            for attempt in range(1, self.static_update_retries + 2):
                result["attempts"] = attempt
                try:
                    response = self._call_coordinator(
                        conversation,
                        request_type="static_update",
                    )
                except Exception as exc:
                    error = f"static_update API call failed: {exc}"
                    result["errors"].append(error)
                    self._record_event(
                        "static_definition_update_rejected",
                        {"attempt": attempt, "reason": error, "response": ""},
                    )
                    continue

                payload, error = self._parse_static_definition_update(response)
                if error is not None:
                    result["errors"].append(error)
                    self._mark_last_coordinator_attempt_rejected("static_update", error)
                    self._record_event(
                        "static_definition_update_rejected",
                        {"attempt": attempt, "reason": error, "response": response},
                    )
                    if attempt <= self.static_update_retries:
                        conversation.extend(
                            [
                                {"role": "assistant", "content": response},
                                {
                                    "role": "user",
                                    "content": (
                                        "STATIC DEFINITION UPDATE REJECTED\n"
                                        f"Reason: {error}\n"
                                        "The task answer is already committed. Regenerate only one valid "
                                        "<static_update> strict JSON response using the maintenance schema."
                                    ),
                                },
                            ]
                        )
                    continue

                action = payload["action"]
                changes = self._empty_static_description_changes()
                if action == "update":
                    changes = self._apply_static_description_update(
                        payload["descriptions"]
                    )
                result.update(
                    {
                        "status": "updated" if any(changes.values()) else "kept",
                        "action": action,
                        "reason": payload["reason"],
                        "changes": changes,
                        "version_after": self.static_definition_version,
                    }
                )
                break
        except Exception as exc:
            result["errors"].append(
                f"static definition update could not be applied: {exc}"
            )
        finally:
            usage_after = self._usage_snapshot()
            result["usage"] = {
                key: max(0.0, usage_after[key] - usage_before[key])
                for key in usage_before
            }
            self._restore_usage(usage_before)

        result["version_after"] = self.static_definition_version
        self._record_event(
            "static_definition_update_finish",
            deepcopy(result),
        )
        return result

    def _static_definition_update_messages(
        self,
        query: str,
        final_answer: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        context = {
            "completed_task": self._coordinator_task_view(query),
            "committed_answer": final_answer,
            "final_mas": plan,
            "worker_outputs": self._compact_history_items(history),
            "current_description_version": self.static_definition_version,
        }
        allowed_names = {
            category: sorted(names)
            for category, names in self._static_description_names().items()
        }
        empty_descriptions = self._empty_static_description_changes()
        keep_example = {
            "action": "keep",
            "reason": "This trajectory does not provide general evidence that the current usage guidance should change.",
            "descriptions": empty_descriptions,
        }
        update_example = {
            "action": "update",
            "reason": "The trajectory provides a reusable rule for choosing deterministic worker behavior.",
            "descriptions": {
                "collaborations": {},
                "parameters": {
                    "temperature": "Prefer lower values when a worker must verify a precise, already-grounded result; use higher values only when genuinely diverse candidates are useful."
                },
                "models": {},
                "skills": {},
            },
        }
        dynamic = (
            "STATIC_DEFINITION_MAINTENANCE_REQUEST\n"
            "[DYNAMIC DEFINITION]\n"
            "PHASE: Post-task static definition maintenance.\n"
            "The final answer below has already been committed and is immutable. Do not answer, revise, judge, or "
            "resubmit the task. Decide only whether this completed trajectory provides reusable evidence for improving "
            "future routing descriptions about WHEN TO USE a collaboration mode, MAS parameter, LLM Backbone, or Skill.\n\n"
            "Only description guidance is mutable. Never change names, pool membership, schemas, value domains, model "
            "capability or price facts, Skill instructions, the completed MAS, or execution behavior. Generalize across "
            "tasks: do not mention this task's entities, answer, dataset, or other example-specific details. When evidence "
            "is weak, choose keep. Any accepted description is available only to later tasks handled by this Router "
            "instance and is discarded when the current program run ends. The system STATIC DEFINITION above contains "
            "the complete current description working copy, initialized from the configured presets at version 1.\n\n"
            f"ALLOWED DESCRIPTION NAMES:\n{json.dumps(allowed_names, ensure_ascii=False, indent=2)}\n\n"
            "RESPONSE FORMAT: output exactly one <static_update> tag and nothing outside it. Its payload must be one strict "
            "JSON object with exactly action, reason, and descriptions. action is keep or update. descriptions must contain "
            "exactly collaborations, parameters, models, and skills, each mapped to an object. For keep, all four objects "
            "must be empty. For update, include at least one allowed exact name mapped directly to its complete replacement "
            "when-to-use description; omit unchanged names.\n\n"
            "2-SHOT EXAMPLES:\n"
            "Example 1 situation: no broadly reusable routing lesson was established.\n"
            f"Example 1 response:\n<static_update>{json.dumps(keep_example, ensure_ascii=False, separators=(',', ':'))}</static_update>\n"
            "Example 2 situation: a reusable parameter-selection lesson was established.\n"
            f"Example 2 response:\n<static_update>{json.dumps(update_example, ensure_ascii=False, separators=(',', ':'))}</static_update>\n\n"
            "COMPLETED TRAJECTORY:\n"
            f"{json.dumps(context, ensure_ascii=False, indent=2)}\n"
            "[/DYNAMIC DEFINITION]"
        )
        return self._mas_state_messages(dynamic)

    def _parse_static_definition_update(
        self,
        response: Any,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(response, str):
            return None, "static_update response is not a string"
        match = re.fullmatch(
            r"\s*<static_update>\s*(.*?)\s*</static_update>\s*",
            response,
            flags=re.DOTALL,
        )
        if match is None:
            return (
                None,
                "response must contain exactly one <static_update> tag and no outside text",
            )
        try:
            raw = json.loads(match.group(1).strip())
        except Exception as exc:
            return None, f"<static_update> payload is not strict JSON: {exc}"
        if not isinstance(raw, dict):
            return None, "<static_update> JSON must be an object"
        expected_keys = {"action", "reason", "descriptions"}
        if set(raw) != expected_keys:
            return None, f"<static_update> must contain exactly {sorted(expected_keys)}"
        action = raw.get("action")
        if action not in {"keep", "update"}:
            return None, "static_update action must be keep or update"
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return None, "static_update reason must be a non-empty string"
        descriptions = raw.get("descriptions")
        if not isinstance(descriptions, dict) or set(descriptions) != set(
            STATIC_DESCRIPTION_CATEGORIES
        ):
            return (
                None,
                f"descriptions must contain exactly {list(STATIC_DESCRIPTION_CATEGORIES)}",
            )

        allowed_names = self._static_description_names()
        normalized = self._empty_static_description_changes()
        for category in STATIC_DESCRIPTION_CATEGORIES:
            values = descriptions.get(category)
            if not isinstance(values, dict):
                return None, f"descriptions.{category} must be an object"
            for name, description in values.items():
                if name not in allowed_names[category]:
                    return None, f"unknown {category} description name: {name}"
                if not isinstance(description, str) or not description.strip():
                    return (
                        None,
                        f"description for {category}.{name} must be a non-empty string",
                    )
                description = description.strip()
                if len(description) > 4000:
                    return (
                        None,
                        f"description for {category}.{name} exceeds 4000 characters",
                    )
                normalized[category][name] = description

        has_changes = any(
            normalized[category] for category in STATIC_DESCRIPTION_CATEGORIES
        )
        if action == "keep" and has_changes:
            return None, "keep action requires all description objects to be empty"
        if action == "update" and not has_changes:
            return None, "update action requires at least one description"
        return {
            "action": action,
            "reason": reason.strip(),
            "descriptions": normalized,
        }, None

    def _apply_static_description_update(
        self,
        descriptions: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, Any]]:
        candidate = deepcopy(self.static_descriptions)
        changes: Dict[str, Dict[str, Any]] = {
            category: {} for category in STATIC_DESCRIPTION_CATEGORIES
        }
        for category in STATIC_DESCRIPTION_CATEGORIES:
            for name, description in descriptions.get(category, {}).items():
                previous = candidate[category].get(name, "")
                if previous == description:
                    continue
                candidate[category][name] = description
                changes[category][name] = {
                    "previous": previous,
                    "current": description,
                }
        if not any(changes.values()):
            return changes

        self.static_descriptions = candidate
        self.static_definition_version += 1
        return changes

    def _has_quality_failure(self) -> bool:
        quality = getattr(self, "_quality", {}) or {}
        failure_keys = (
            "format_errors",
            "json_parse_failures",
            "non_strict_json",
            "invalid_plan",
            "invalid_update",
            "problem_nodes",
        )
        return any(int(quality.get(key, 0) or 0) > 0 for key in failure_keys)

    def _create_initial_plan(self, query: str) -> Dict[str, Any]:
        if self.workflow_protocol:
            return self._create_workflow_plan(query)

        fallback = self._fallback_plan()
        tag, value = self._run_state_until_terminal(
            state=MAS_STATE_READY,
            messages=self.mas_plan_messages(query),
            request_type="plan",
            terminal_tags=("plan",),
            plan=None,
        )
        if tag != "plan" or not isinstance(value, dict):
            return fallback
        return value

    def plan_messages(self, query: str) -> List[Dict[str, str]]:
        """Build the planning prompt for the Router's selected protocol."""
        if self.workflow_protocol:
            return self.workflow_plan_messages(query)
        return self.mas_plan_messages(query)

    def mas_plan_messages(self, query: str) -> List[Dict[str, str]]:
        """Build the Ready-state prompt used by both inference and batched RL."""
        return self._mas_state_messages(self._ready_dynamic_definition(query))

    def _mas_state_messages(self, dynamic_definition: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self._mas_static_definition()},
            {"role": "user", "content": dynamic_definition},
        ]

    def _mas_static_definition(self) -> str:
        return (
            COORDINATOR_SYSTEM_PROMPT_TEMPLATE.replace(
                "{{models}}", self._format_pool(self.model_pool, category="models")
            )
            .replace(
                "{{skills}}", self._format_pool(self.skill_pool, category="skills")
            )
            .replace(
                "{{collaborations}}",
                self._format_pool(self.collaboration_pool, category="collaborations"),
            )
        )

    @staticmethod
    def _ordered_mas_design_instruction() -> str:
        return (
            "ORDERED MAS DESIGN IN <analysis> (whenever this response will emit <plan>):\n"
            "1. Collaboration and agents: first choose the collaboration mode, agent count, and distinct worker roles. "
            "Obey the collaboration agent-count rules and every explicit user-required agent count.\n"
            "2. Skills: next assign one available Skill to each agent according to that role.\n"
            "3. Models: then assign one available model to each agent.\n"
            # Boldly use a large-scale, high-capacity model for "
            # "agents that need broad world knowledge, complex reasoning, code writing, or high-reliability synthesis or "
            # "verification; do not under-allocate model capacity for difficult work.
            "4. Parameters: last set rounds and concrete per-agent prompts.\n"
            "Apply any [HINDSIGHT_SKILL] as the primary guide throughout these four decisions, then make the <plan> JSON "
            "exactly match the ordered design.\n"
        )

    def _ready_dynamic_definition(self, query: str) -> str:
        return (
            f"{self._coordinator_task_view(query)}\n\n"
            "Please carefully analyze the task step-by-step. Output your response strictly in the following format:\n"
            "You must close </analysis> before opening <plan>. Never place <plan> inside <analysis>.\n"
            "<analysis>[Provide your internal reasoning, problem breakdown, and strategic planning here.]</analysis>\n"
            "<plan>[Provide the final execution plan here.]</plan>"
        )

    def _ready_state_example_mas(self) -> Dict[str, Any]:
        capable_model = self.model_names[
            self._preferred_model_id("qwen3.5-27b", default=0)
        ]
        flagship_model = self.model_names[
            self._preferred_model_id(
                "qwen3.5-397b-a17b", default=len(self.model_names) - 1
            )
        ]
        return {
            "collaboration": (
                "Chain"
                if "Chain" in self.collaboration_names
                else self.collaboration_names[0]
            ),
            "rounds": 1,
            "agents": [
                {
                    "id": "agent_1",
                    "model": capable_model,
                    "skill": self._preferred_skill("ProblemDecomposition"),
                    "prompt": (
                        "Map the exact scientific-contribution -> physicist -> birth city -> country relation chain, "
                        "identify ambiguity risks, and return a concise resolution checklist."
                    ),
                },
                {
                    "id": "agent_2",
                    "model": flagship_model,
                    "skill": self._preferred_skill("DomainKnowledgeQA"),
                    "prompt": (
                        "Follow the planned relation chain, resolve each entity using reliable world knowledge, and return "
                        "the supported city and country with the intermediate identity."
                    ),
                },
                {
                    "id": "agent_3",
                    "model": flagship_model,
                    "skill": self._preferred_skill("AnswerSynthesis"),
                    "prompt": (
                        "Independently verify the physicist, birthplace, and country links, correct any upstream error, "
                        "and return only the requested city, country answer."
                    ),
                },
            ],
        }

    def _state_example_mas(self) -> Dict[str, Any]:
        first_model = self.model_names[0]
        second_model = self.model_names[1] if len(self.model_names) > 1 else first_model
        return {
            "collaboration": (
                "Chain"
                if "Chain" in self.collaboration_names
                else self.collaboration_names[0]
            ),
            "rounds": 1,
            "agents": [
                {
                    "id": "agent_1",
                    "model": first_model,
                    "skill": self._preferred_skill("EvidenceGrounding"),
                    "prompt": "Resolve the key evidence and return a focused intermediate result.",
                },
                {
                    "id": "agent_2",
                    "model": second_model,
                    "skill": self._preferred_skill("ConsistencyVerification"),
                    "prompt": "Verify the prior reasoning and produce the requested final response.",
                },
            ],
        }

    def _running_state_messages(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        status = self._status_xml(
            self._status_payload(
                query=query,
                plan=plan,
                history=history,
                recent_only=False,
            )
        )
        dynamic = (
            "ROUTER_STATE_REQUEST\n"
            "[DYNAMIC DEFINITION]\n"
            "CURRENT STATE: Running\n"
            "The execution environment has reached the configured worker-call activation point. Inspect the current "
            "MAS and all worker outputs below.\n\n"
            f"{status}\n\n"
            "ALLOWED OUTPUT TAGS IN Running: <analysis> and <update>.\n"
            "- <analysis>brief analysis of the observed execution</analysis>: reason about the update inside this response.\n"
            "- <update>strict Update JSON</update>: apply one update, then transition Running -> Judging. Use no_change "
            "when no structural or assignment change is justified.\n"
            "REQUIRED RESPONSE SHAPE: <analysis>...</analysis><update>...</update>. Do not stop after analysis; finish the Running decision in this same "
            "response. Whenever a valid update is present, it is applied and the state transitions to Judging. Do not answer the task in Running.\n\n"
            "2-SHOT EXAMPLES:\n"
            "The examples demonstrate state control only; make the update from the actual status above.\n"
            "Example 1 situation: a worker result is incomplete and its assignment needs a targeted clarification.\n"
            "Example 1 response:\n"
            "<analysis>The first worker needs a more explicit evidence-checking assignment.</analysis>\n"
            '<update>{"type":"update_prompt","target_agent_id":"agent_1","prompt":"Recheck the disputed evidence and report only supported conclusions.","reason":"The observed result lacks evidence verification."}</update>\n'
            "Example 2 situation: the current MAS already reflects the useful roles and no concrete change is needed.\n"
            "Example 2 response:\n"
            "<analysis>The observed outputs do not justify changing the current MAS.</analysis>\n"
            '<update>{"type":"no_change","reason":"The observed outputs do not justify changing the current MAS."}</update>\n'
            "[/DYNAMIC DEFINITION]"
        )
        return self._mas_state_messages(dynamic)

    def _judging_state_messages(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
        prompt_hint: str = "",
    ) -> List[Dict[str, str]]:
        dynamic = (
            f"{self._coordinator_task_view(query)}\n\n"
            "Here is your multi-agent system plan:\n"
            f"{json.dumps(self._public_plan(plan), ensure_ascii=False, separators=(',', ':'))}\n\n"
            "Execution output from the multi-agent system: "
            f"{self._execution_output(plan, history)}\n\n"
            "Based on the output provided by the MAS, determine the final answer.\n"
            "Output your response strictly in the following format:\n"
            "<analysis>[Perform simple analysis and synthesis, and extract the answer]</analysis>\n"
            "<answer>[Output final answer clearly and concisely here]</answer>"
        )
        return self._mas_state_messages(dynamic)

    @staticmethod
    def _public_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "collaboration": str(plan.get("collaboration", "")),
            "rounds": int(plan.get("rounds", 1) or 1),
            "agents": [
                {
                    "id": str(agent.get("id", "")),
                    "model": str(agent.get("model", "")),
                    "skill": str(agent.get("skill", "")),
                    "prompt": str(agent.get("prompt", "")),
                }
                for agent in plan.get("agents", [])
                if isinstance(agent, dict)
            ],
        }

    @staticmethod
    def _execution_output(
        plan: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
    ) -> str:
        items = [item for item in history if isinstance(item, dict)]
        if not items:
            return ""
        latest_workflow_index = max(
            int(item.get("workflow_index", 0)) for item in items
        )
        items = [
            item
            for item in items
            if int(item.get("workflow_index", 0)) == latest_workflow_index
        ]
        if not items:
            return ""

        collaboration = str(plan.get("collaboration") or "")
        if collaboration == "IO":
            return str(items[-1].get("output", ""))
        if collaboration == "CoT":
            # CoT Judging sees the chronological call results concatenated
            # directly, without artificial round/agent tags.
            return "\n\n".join(str(item.get("output", "")) for item in items)
        if collaboration == "Chain":
            return str(items[-1].get("output", ""))
        if collaboration == "Reflection":
            return str(items[-1].get("output", ""))
        if collaboration in {"FullConnected", "Debate"}:
            final_round = max(int(item.get("round", 1)) for item in items)
            final_items = [
                item for item in items if int(item.get("round", 1)) == final_round
            ]
            return "\n".join(
                f"- {item.get('agent_id', '')}: {item.get('output', '')}"
                for item in final_items
            )
        return str(items[-1].get("output", ""))

    def _run_state_until_terminal(
        self,
        state: str,
        messages: List[Dict[str, str]],
        request_type: str,
        terminal_tags: Sequence[str],
        plan: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Any]:
        conversation = deepcopy(messages)
        rejected_attempts = 0
        terminal_tag_set = set(terminal_tags)

        while True:
            response = self._call_coordinator(conversation, request_type=request_type)
            tag, payload, error_key, error = self._parse_state_response(
                response=response,
                state=state,
                allowed_tags=terminal_tag_set,
            )
            value: Any = None
            if error is None and tag in terminal_tag_set:
                value, error_key, error = self._decode_state_terminal(
                    state=state,
                    tag=tag,
                    payload=payload,
                    plan=plan,
                )

            if error is not None:
                rejected_attempts += 1
                self._quality["state_retries"] = (
                    int(self._quality.get("state_retries", 0)) + 1
                )
                self._mark_last_coordinator_attempt_rejected(request_type, error)
                self._record_event(
                    "state_output_rejected",
                    {
                        "state": state,
                        "request_type": request_type,
                        "attempt": rejected_attempts,
                        "reason": error,
                        "response": response,
                    },
                )
                if rejected_attempts > self.state_format_retries:
                    self._quality_issue(
                        error_key or "format_errors",
                        f"{state} state exhausted format retries: {error}",
                    )
                    return None, None
                conversation.extend(
                    [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": self._state_rejection_message(
                                state=state,
                                reason=error,
                                terminal_tags=terminal_tags,
                            ),
                        },
                    ]
                )
                continue

            self._record_event(
                "state_output",
                {
                    "state": state,
                    "request_type": request_type,
                    "tag": tag,
                    "response": response,
                },
            )
            return tag, value

    def _parse_state_response(
        self,
        response: Any,
        state: str,
        allowed_tags: Sequence[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        if not isinstance(response, str):
            return None, None, "format_errors", f"{state}: response is not a string"

        if state in {MAS_STATE_READY, MAS_STATE_RUNNING, MAS_STATE_JUDGING}:
            expected = {
                MAS_STATE_READY: "plan",
                MAS_STATE_RUNNING: "update",
                MAS_STATE_JUDGING: "answer",
            }[state]
            if expected not in {str(tag).lower() for tag in allowed_tags}:
                return (
                    None,
                    None,
                    "format_errors",
                    f"{state}: expected <{expected}> is not allowed",
                )
            pattern = re.compile(
                rf"\s*<analysis>\s*(?P<analysis>.+?)\s*</analysis>\s*"
                rf"<{expected}>\s*(?P<payload>.+?)\s*</{expected}>\s*",
                flags=re.IGNORECASE | re.DOTALL,
            )
            match = pattern.fullmatch(response)
            if match is None:
                return (
                    None,
                    None,
                    "format_errors",
                    f"{state}: response must be exactly <analysis>...</analysis><{expected}>...</{expected}>",
                )
            return expected, match.group("payload").strip(), None, None

        allowed = {str(tag).lower() for tag in allowed_tags}
        opening_pattern = re.compile(
            r"<\s*(analysis|plan|update|answer)\s*>",
            flags=re.IGNORECASE,
        )
        openings = [
            match
            for match in opening_pattern.finditer(response)
            if not self._position_inside_double_quoted_text(response, match.start())
        ]
        if not openings:
            payload = response.strip()
            fenced = re.fullmatch(
                r"\s*```(?:text|markdown|md)?\s*(.*?)\s*```\s*",
                payload,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if fenced is not None:
                payload = fenced.group(1).strip()
            if payload and "analysis" in allowed:
                self._compiler_repair(
                    f"{state}: treated nonempty untagged state output as analysis-only"
                )
                return "analysis", payload, None, None
            return (
                None,
                None,
                "format_errors",
                f"{state}: response contains no actionable state tag",
            )

        blocks: List[Dict[str, Any]] = []
        generic_closing_pattern = re.compile(
            r"<\s*/\s*(?:analysis|plan|update|answer)\s*>",
            flags=re.IGNORECASE,
        )
        for index, opening in enumerate(openings):
            tag = opening.group(1).lower()
            matching_close = re.search(
                rf"<\s*/\s*{re.escape(tag)}\s*>",
                response[opening.end() :],
                flags=re.IGNORECASE,
            )
            if matching_close is not None:
                payload_end = opening.end() + matching_close.start()
                block_end = opening.end() + matching_close.end()
                closed = True
            else:
                next_opening = (
                    openings[index + 1].start()
                    if index + 1 < len(openings)
                    else len(response)
                )
                generic_close = generic_closing_pattern.search(
                    response, opening.end(), next_opening
                )
                payload_end = (
                    generic_close.start() if generic_close is not None else next_opening
                )
                block_end = (
                    generic_close.end() if generic_close is not None else payload_end
                )
                closed = False
            payload = response[opening.end() : payload_end].strip()
            if response.lstrip().startswith("```") and not closed:
                payload = re.sub(r"\s*```\s*$", "", payload).strip()
            blocks.append(
                {
                    "tag": tag,
                    "payload": payload,
                    "start": opening.start(),
                    "end": block_end,
                    "closed": closed,
                }
            )

        terminal_blocks = [
            block
            for block in blocks
            if block["tag"] in allowed and block["tag"] != "analysis"
        ]
        analysis_blocks = [
            block
            for block in blocks
            if block["tag"] == "analysis" and "analysis" in allowed
        ]
        candidates = terminal_blocks or analysis_blocks
        if not candidates:
            detected = [block["tag"] for block in blocks]
            return (
                detected[-1],
                blocks[-1]["payload"],
                "format_errors",
                f"{state}: detected tags {detected} are not currently allowed; allowed tags are {sorted(allowed)}",
            )

        nonempty_candidates = [block for block in candidates if block["payload"]]
        selected = (nonempty_candidates or candidates)[-1]
        tag = str(selected["tag"])
        payload = str(selected["payload"]).strip()
        if not payload:
            return tag, payload, "format_errors", f"{state}: <{tag}> payload is empty"

        if len(blocks) > 1:
            self._compiler_repair(
                f"{state}: selected the last actionable <{tag}> from a multi-tag response"
            )
        if not selected["closed"]:
            self._compiler_repair(
                f"{state}: accepted <{tag}> with a missing or mismatched closing tag"
            )
        if "```" in response:
            self._compiler_repair(
                f"{state}: ignored Markdown fences around state output"
            )
        return tag, payload, None, None

    @staticmethod
    def _position_inside_double_quoted_text(text: str, position: int) -> bool:
        in_string = False
        escaped = False
        for char in text[:position]:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
        return in_string

    def _decode_state_terminal(
        self,
        state: str,
        tag: Optional[str],
        payload: Optional[str],
        plan: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Optional[str], Optional[str]]:
        if tag == "answer":
            return str(payload or "").strip(), None, None
        if tag == "plan" and state == MAS_STATE_READY:
            try:
                raw = self.parse_plan_json(str(payload or ""))
            except ValueError as exc:
                return (
                    None,
                    "json_parse_failures",
                    f"{state}: strict plan JSON parse failed: {exc}",
                )
            if not isinstance(raw, dict):
                return (
                    None,
                    "json_parse_failures",
                    f"{state}: plan payload must be one JSON object",
                )
            json_error = None
        else:
            raw, json_error = self._decode_repairable_state_json(
                payload=str(payload or ""),
                state=state,
                tag=str(tag or ""),
            )
        if json_error is not None:
            return None, "json_parse_failures", json_error

        raw = self._normalize_state_resource_names(raw, tag=str(tag or ""))
        if tag == "plan":
            raw = self._repair_plan_execution_shape(raw)

        quality_snapshot = deepcopy(self._quality)
        if tag == "plan":
            self._validate_raw_plan(raw)
        elif tag == "update":
            self._validate_raw_update(raw)
            self._validate_update_against_plan(raw, plan or {})
        else:
            return None, "format_errors", f"{state}: unsupported terminal tag <{tag}>"

        failure_keys = (
            "format_errors",
            "json_parse_failures",
            "non_strict_json",
            "invalid_plan",
            "invalid_update",
            "problem_nodes",
        )
        increased = [
            key
            for key in failure_keys
            if int(self._quality.get(key, 0)) > int(quality_snapshot.get(key, 0))
        ]
        if increased:
            new_issues = self._quality.get("issues", [])[
                len(quality_snapshot.get("issues", [])) :
            ]
            self._quality = quality_snapshot
            error_key = increased[-1]
            reason = (
                "; ".join(str(issue) for issue in new_issues)
                or f"invalid <{tag}> payload"
            )
            return None, error_key, f"{state}: {reason}"

        if tag == "plan":
            return self._normalize_plan(raw, fallback=self._fallback_plan()), None, None
        return self._normalize_update(raw), None, None

    def _decode_repairable_state_json(
        self,
        payload: str,
        state: str,
        tag: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        original = str(payload or "").strip()
        cleaned = original
        fence_match = re.fullmatch(
            r"\s*```(?:json|javascript|js)?\s*(.*?)\s*```\s*",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        fence_removed = fence_match is not None
        if fence_match is not None:
            cleaned = fence_match.group(1).strip()
        elif re.match(
            r"^\s*```(?:json|javascript|js)?\s*", cleaned, flags=re.IGNORECASE
        ):
            cleaned = re.sub(
                r"^\s*```(?:json|javascript|js)?\s*",
                "",
                cleaned,
                count=1,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
            fence_removed = True

        candidates: List[Tuple[str, bool]] = [(cleaned, False)]
        balanced = self._first_json_object(cleaned)
        if (
            balanced
            and balanced.strip() != cleaned.strip()
            and not cleaned.lstrip().startswith("[")
        ):
            candidates.append((balanced, True))

        last_error: Optional[Exception] = None
        saw_non_object = False
        for candidate, extracted in candidates:
            repaired = self._repair_invalid_json_escapes(candidate)
            variants = [(repaired, repaired != candidate), (candidate, False)]
            seen_variants = set()
            for candidate_text, escaped in variants:
                if candidate_text in seen_variants:
                    continue
                seen_variants.add(candidate_text)
                try:
                    parsed = json.loads(candidate_text)
                except Exception as exc:
                    last_error = exc
                    continue
                if not isinstance(parsed, dict):
                    saw_non_object = True
                    continue
                repairs = []
                if fence_removed:
                    repairs.append("removed a Markdown JSON fence")
                if extracted:
                    repairs.append("extracted the first balanced JSON object")
                if escaped:
                    repairs.append("escaped LaTeX-style or invalid JSON backslashes")
                if repairs:
                    self._compiler_repair(f"{state}: <{tag}> " + "; ".join(repairs))
                return parsed, None

        if saw_non_object:
            return None, f"{state}: <{tag}> JSON must be an object"
        detail = str(last_error) if last_error is not None else "no JSON object found"
        return None, f"{state}: <{tag}> is not strict or repairable JSON: {detail}"

    @staticmethod
    def _canonical_selector_name(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    @staticmethod
    def _selector_tokens(value: Any) -> set:
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
        return set(re.findall(r"[a-z]+\d*|\d+[a-z]+|\d+", expanded.lower()))

    def _unique_pool_name(
        self,
        value: Any,
        names: Sequence[str],
        aliases: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        if value in names:
            return value

        canonical = self._canonical_selector_name(value)
        canonical_names = {name: self._canonical_selector_name(name) for name in names}
        exact = [
            name
            for name, normalized in canonical_names.items()
            if normalized == canonical
        ]
        if len(exact) == 1:
            return exact[0]

        alias_target = (aliases or {}).get(canonical)
        if alias_target in names:
            return alias_target

        if len(canonical) >= 3:
            contained = [
                name
                for name, normalized in canonical_names.items()
                if canonical in normalized
                or (len(normalized) >= 3 and normalized in canonical)
            ]
            if len(contained) == 1:
                return contained[0]

        ignored_tokens = {"agent", "llm", "model", "single", "the", "use", "using"}
        token_matches = []
        name_tokens = {name: self._selector_tokens(name) for name in names}
        for token in self._selector_tokens(value) - ignored_tokens:
            if len(token) < 2:
                continue
            matches = {
                name
                for name, tokens in name_tokens.items()
                if token in tokens
                or (
                    len(token) >= 3
                    and any(
                        token in choice_token or choice_token in token
                        for choice_token in tokens
                    )
                )
            }
            if matches:
                token_matches.append(matches)
        if token_matches:
            intersection = set.intersection(*token_matches)
            if len(intersection) == 1:
                return next(iter(intersection))
        return None

    def _normalize_named_field(
        self,
        container: Dict[str, Any],
        key: str,
        names: Sequence[str],
        label: str,
        aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        original = container.get(key)
        matched = self._unique_pool_name(original, names, aliases=aliases)
        if matched is None or original == matched:
            return
        container[key] = matched
        self._compiler_repair(f"normalized {label} from {original!r} to {matched!r}")

    def _normalize_state_resource_names(
        self,
        raw: Optional[Dict[str, Any]],
        tag: str,
    ) -> Dict[str, Any]:
        value = deepcopy(raw) if isinstance(raw, dict) else {}
        collaboration_aliases = {
            "single": "IO",
            "singleagent": "IO",
            "singleagentio": "IO",
            "direct": "IO",
            "chainofthought": "CoT",
            "singleagentcot": "CoT",
            "fullyconnected": "FullConnected",
            "fullconnection": "FullConnected",
        }
        if tag == "plan":
            self._normalize_named_field(
                value,
                "collaboration",
                self.collaboration_names,
                "collaboration",
                aliases=collaboration_aliases,
            )
            agents = value.get("agents")
            if isinstance(agents, list):
                for index, agent in enumerate(agents, start=1):
                    if not isinstance(agent, dict):
                        continue
                    self._normalize_named_field(
                        agent,
                        "model",
                        self.model_names,
                        f"agent_{index} model",
                    )
                    self._normalize_named_field(
                        agent,
                        "skill",
                        self.skill_names,
                        f"agent_{index} skill",
                    )
            return value

        if tag != "update":
            return value

        allowed_updates = (
            "no_change",
            "update_model",
            "update_temperature",
            "update_skill",
            "update_prompt",
            "set_num_rounds",
            "remove_node",
            "add_node",
        )
        action_value = value.get("type", value.get("action"))
        normalized_action = self._unique_pool_name(action_value, allowed_updates)
        if normalized_action is not None:
            if action_value != normalized_action:
                self._compiler_repair(
                    f"normalized update type from {action_value!r} to {normalized_action!r}"
                )
            value["type"] = normalized_action
        action = value.get("type", value.get("action"))
        if action == "update_model":
            self._normalize_named_field(
                value, "model", self.model_names, "update model"
            )
        elif action == "update_skill":
            self._normalize_named_field(
                value, "skill", self.skill_names, "update skill"
            )
        elif action == "add_node" and isinstance(value.get("new_agent"), dict):
            self._normalize_named_field(
                value["new_agent"],
                "model",
                self.model_names,
                "new agent model",
            )
            self._normalize_named_field(
                value["new_agent"],
                "skill",
                self.skill_names,
                "new agent skill",
            )
        return value

    def _validate_update_against_plan(
        self,
        update: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> None:
        agents = [agent for agent in plan.get("agents", []) if isinstance(agent, dict)]
        agent_ids = {str(agent.get("id")) for agent in agents}
        action = update.get("type", update.get("action"))
        if action in {
            "update_model",
            "update_temperature",
            "update_skill",
            "update_prompt",
            "remove_node",
        }:
            target_id = str(update.get("target_agent_id") or "")
            if target_id not in agent_ids:
                self._quality_issue(
                    "invalid_update", f"update target not found: {target_id}"
                )
            if action == "remove_node" and len(agents) <= 1:
                self._quality_issue(
                    "invalid_update", "remove_node would leave no active agents"
                )
        if action == "add_node":
            new_agent = update.get("new_agent") or {}
            new_id = (
                str(new_agent.get("id") or "") if isinstance(new_agent, dict) else ""
            )
            if new_id in agent_ids:
                self._quality_issue(
                    "problem_nodes", f"add_node duplicate agent id: {new_id}"
                )

    @staticmethod
    def _state_rejection_message(
        state: str,
        reason: str,
        terminal_tags: Sequence[str],
    ) -> str:
        return (
            "[STATE OUTPUT REJECTED]\n"
            f"CURRENT STATE REMAINS: {state}\n"
            f"REJECTION REASON: {reason}\n"
            f"Regenerate the complete state action in one response. Emit <analysis>...</analysis> followed by one of "
            f"{list(terminal_tags)}, and follow the terminal schema. Do not stop after <analysis>.\n"
            "[/STATE OUTPUT REJECTED]"
        )

    def _create_workflow_plan(self, query: str) -> Dict[str, Any]:
        fallback = self._fallback_plan(protocol="workflow_v2")
        messages = self.workflow_plan_messages(query)
        response = self._call_coordinator(messages, request_type="plan")
        raw_plan = self._extract_tagged_json(
            response,
            tag="workflow",
            fallback=fallback,
            request_type="plan",
        )
        self._validate_raw_plan(raw_plan)
        return self._normalize_plan(raw_plan, fallback=fallback)

    def workflow_plan_messages(self, query: str) -> List[Dict[str, str]]:
        """Build the Coordinator planning messages."""
        examples = self._workflow_few_shot_examples() if self.few_shot_workflows else ""
        prompt = {
            "role": "user",
            "content": (
                "ROUTER_WORKFLOW_REQUEST\n"
                "Your role is to obtain an answer to the user question by iteratively querying powerful language models "
                "with different capabilities. Do not answer the user question yourself. You are given the question and a "
                "list of numbered workers with metadata. Output an ordered workflow whose workers produce the answer.\n\n"
                "Each workflow step has four operational elements: a worker model, its reasoning mode, a focused natural-language "
                "subtask, and an access list represented by depends_on that determines which earlier step responses the worker sees. A "
                "subtask may ask a worker to solve the question from scratch, refine previous work, verify a proposal, or do "
                "any other task that helps a later worker answer correctly. The first selected worker receives the original "
                "question and its subtask. Later workers also receive exactly the earlier messages declared in depends_on. "
                "The response of the final worker is returned to the user. Its subtask must require a final response that "
                "follows the original formatting instructions inside <answer>...</answer>.\n\n"
                "Choose the workers, number of steps, subtasks, and communication topology from the user question and the "
                "available worker metadata. A worker attempting independently should have an empty depends_on list. A worker "
                "that uses earlier work must declare those earlier step ids. Use one step or multiple steps according to what "
                "will improve the final answer; do not create redundant subtasks. Set mode to thinking only when the extra "
                "reasoning is likely to improve correctness enough to justify its cost; otherwise use nothinking or auto. "
                "Only workers whose metadata supports both modes act on this field; other workers treat it as auto.\n\n"
                "Strict output protocol:\n"
                "1. Output one or more <analysis>...</analysis> blocks.\n"
                "2. Then output exactly one <workflow>...</workflow> block.\n"
                "3. The workflow block contains one strict JSON object, without markdown or extra text.\n"
                "4. Every dependency must name an earlier step from the same workflow.\n"
                "5. If a prompt refers to #N or step_N, its depends_on must explicitly include step_N. "
                "Never mention an earlier result without declaring the edge that makes it visible.\n\n"
                "JSON schema:\n"
                "{\n"
                '  "steps": [\n'
                '    {"id":"step_1","model_id":0,"mode":"auto|thinking|nothinking",'
                '"depends_on":[],"prompt":"one concrete subtask"}\n'
                "  ]\n"
                "}\n\n"
                "Available workers (select by integer model_id):\n"
                f"{self._format_indexed_model_pool()}\n\n"
                f"{examples}"
                f"USER TASK:\n{self._coordinator_task_view(query)}"
            ),
        }
        return [prompt]

    def _evaluate_update(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if plan.get("protocol") == "workflow_v2":
            return self._evaluate_workflow_update(query, plan, history)

        fallback = {
            "type": "no_change",
            "action": "no_change",
            "reason": "Keep current MAS.",
        }
        tag, value = self._run_state_until_terminal(
            state=MAS_STATE_RUNNING,
            messages=self._running_state_messages(query, plan, history),
            request_type="update",
            terminal_tags=("update",),
            plan=plan,
        )
        if tag != "update" or not isinstance(value, dict):
            return fallback
        return value

    def _evaluate_workflow_update(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        status_payload = self._status_payload(
            query=query,
            plan=plan,
            history=history,
            recent_only=True,
        )
        status = self._status_xml(status_payload)
        completed_step_ids = [
            str(value) for value in status_payload.get("completed_step_ids", [])
        ]
        remaining_step_ids = [
            str(value) for value in status_payload.get("remaining_step_ids", [])
        ]
        existing_step_ids = {
            str(agent.get("id"))
            for agent in plan.get("agents", [])
            if isinstance(agent, dict) and agent.get("id")
        }
        new_step_index = 1
        while f"step_{new_step_index}" in existing_step_ids:
            new_step_index += 1
        new_step_id = f"step_{new_step_index}"
        target_step_id = remaining_step_ids[0] if remaining_step_ids else ""
        correction_dependencies = []
        for step_id in (
            completed_step_ids[-1] if completed_step_ids else "",
            target_step_id,
        ):
            if step_id and step_id not in correction_dependencies:
                correction_dependencies.append(step_id)
        compared_steps = (
            " and ".join(correction_dependencies) or "the available worker outputs"
        )
        fallback = {"type": "no_change", "reason": "Keep the remaining workflow."}
        correction_example = {
            "type": "add_node",
            "new_step": {
                "id": new_step_id,
                "model_id": self._preferred_model_id("qwen3.5-397b-a17b", 0),
                "skill": self._preferred_skill("AnswerSynthesis"),
                "depends_on": correction_dependencies,
                "prompt": (
                    f"Solve the original task independently, then re-evaluate the evidence and compare {compared_steps}; "
                    "return only <answer>the corrected short final answer</answer>."
                ),
            },
            "reason": "The faulty step is already completed, so append a correction terminal instead of mutating history.",
        }
        schema_examples = ['{"type":"no_change","reason":"..."}']
        if target_step_id:
            schema_examples.extend(
                [
                    json.dumps(
                        {
                            "type": "update_model",
                            "target_agent_id": target_step_id,
                            "model_id": self._preferred_model_id(
                                "qwen3.5-397b-a17b", 0
                            ),
                            "reason": "...",
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {
                            "type": "update_skill",
                            "target_agent_id": target_step_id,
                            "skill": self._preferred_skill("EvidenceGrounding"),
                            "reason": "...",
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {
                            "type": "update_prompt",
                            "target_agent_id": target_step_id,
                            "prompt": "new focused subtask",
                            "reason": "...",
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {
                            "type": "remove_node",
                            "target_agent_id": target_step_id,
                            "reason": "...",
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ]
            )
        schema_examples.append(
            json.dumps(correction_example, ensure_ascii=False, separators=(",", ":"))
        )
        one_shot = (
            "UPDATE RESPONSE EXAMPLES (use only ids that are legal in the current status):\n"
            f"Example A status: completed_step_ids={completed_step_ids}, remaining_step_ids={remaining_step_ids}, "
            "and no concrete failure was observed.\n"
            "<analysis>The completed step produced a usable result and the remaining DAG already covers verification.</analysis>\n"
            '<update>{"type":"no_change","reason":"No concrete execution failure requires changing the remaining steps."}</update>\n'
            f"Example B status: completed_step_ids={completed_step_ids}, remaining_step_ids={remaining_step_ids}, "
            "and a completed output is demonstrably wrong.\n"
            "<analysis>The faulty completed step is immutable. Let the remaining verifier finish, then append a correction terminal "
            "that can see both results.</analysis>\n"
            f"<update>{json.dumps(correction_example, ensure_ascii=False, separators=(',', ':'))}</update>\n\n"
            if self.few_shot_workflows
            else ""
        )
        prompt = {
            "role": "user",
            "content": (
                "ROUTER_WORKFLOW_UPDATE_REQUEST\n"
                "You are the Coordinator LLM observing a workflow during execution. The state is inside "
                "<status>...</status>. Decide whether one targeted change to the not-yet-run steps is useful. "
                "Do not change the workflow merely because an intermediate answer is uncertain; make a change only "
                "when the observed output reveals a concrete missing role, failed subtask, or unsuitable worker.\n\n"
                f"{status}\n\n"
                "Strict output: one or more <analysis>...</analysis> blocks followed by exactly one "
                "<update>strict JSON object</update> block.\n"
                "For any targeted change, target_agent_id must be one of status.remaining_step_ids; completed steps "
                "cannot be changed retroactively.\n"
                f"For this request the legal target ids are {remaining_step_ids}; the next unused add_node id is {new_step_id}.\n"
                "If a completed step is demonstrably faulty, do not target it. Use add_node to append a new correction "
                "terminal whose dependencies include the relevant completed result and a remaining verifier/synthesis step; "
                "its prompt must return <answer>...</answer>.\n"
                "Allowed schemas instantiated with legal ids for this request:\n"
                f"{chr(10).join(schema_examples)}\n\n"
                "Available workers:\n"
                f"{self._format_indexed_model_pool()}\n\n"
                "Available skills:\n"
                f"{self._format_skill_catalog()}\n\n"
                f"{one_shot}"
            ),
        }
        response = self._call_coordinator([prompt], request_type="update")
        update = self._extract_tagged_json(
            response,
            tag="update",
            fallback=fallback,
            request_type="update",
        )
        current_workflow_index = max(
            (
                int(item.get("workflow_index", 0))
                for item in history
                if isinstance(item, dict)
            ),
            default=0,
        )
        completed_ids = {
            str(item.get("agent_id"))
            for item in history
            if isinstance(item, dict)
            and int(item.get("workflow_index", 0)) == current_workflow_index
            and item.get("agent_id")
        }
        self._validate_raw_workflow_update(update, plan, completed_ids=completed_ids)
        return self._normalize_workflow_update(update, plan)

    def _apply_update(
        self, plan: Dict[str, Any], update: Dict[str, Any]
    ) -> Dict[str, Any]:
        if plan.get("protocol") == "workflow_v2":
            return self._apply_workflow_update(plan, update)

        action = update.get("action", "no_change")
        plan.setdefault("updates", []).append(update)
        if action == "no_change":
            return plan

        agents = plan["agents"]
        target_id = update.get("target_agent_id")
        target = next((agent for agent in agents if agent["id"] == target_id), None)

        if action == "update_model" and target is not None:
            target["model"] = self._valid_model(update.get("model"), target["model"])
        elif action == "update_temperature" and target is not None:
            target["temperature"] = self._temperature(
                update.get("temperature", target["temperature"])
            )
        elif action == "update_skill" and target is not None:
            target["skill"] = self._valid_skill(update.get("skill"), target["skill"])
        elif action == "update_prompt" and target is not None:
            prompt = str(update.get("prompt") or "")
            target["prompt"] = prompt
            target.pop("instruction", None)
        elif action == "set_num_rounds":
            try:
                plan["num_rounds"] = max(
                    1, int(update.get("num_rounds", plan.get("num_rounds", 1)))
                )
            except Exception:
                self._quality_issue(
                    "invalid_update",
                    f"invalid set_num_rounds value: {update.get('num_rounds')}",
                )
        elif action == "remove_node" and target is not None and len(agents) > 1:
            plan["agents"] = [agent for agent in agents if agent["id"] != target_id]
        elif action == "add_node":
            new_agent = self._normalize_agent(
                update.get("new_agent", {}), len(agents) + 1
            )
            existing_ids = {agent["id"] for agent in agents}
            if new_agent["id"] in existing_ids:
                self._quality_issue(
                    "problem_nodes", f"add_node duplicate agent id: {new_agent['id']}"
                )
                new_agent["id"] = self._next_agent_id(agents)
            agents.append(new_agent)
        elif (
            action
            in {
                "update_model",
                "update_temperature",
                "update_skill",
                "update_prompt",
                "remove_node",
            }
            and target is None
        ):
            self._quality_issue(
                "invalid_update", f"update target not found: {target_id}"
            )
        elif action == "remove_node" and target is not None and len(agents) <= 1:
            self._quality_issue(
                "invalid_update", "remove_node would leave no active agents"
            )

        logger.info(f"LLMCoordinatorRouter update={update}")
        return plan

    def _apply_workflow_update(
        self,
        plan: Dict[str, Any],
        update: Dict[str, Any],
    ) -> Dict[str, Any]:
        action = update.get("action", update.get("type", "no_change"))
        plan.setdefault("updates", []).append(deepcopy(update))
        if action == "no_change":
            return plan

        agents = plan.get("agents", [])
        target_id = str(update.get("target_agent_id") or "")
        target = next((agent for agent in agents if agent.get("id") == target_id), None)
        if action == "update_model" and target is not None:
            model_id = self._valid_model_id(update.get("model_id"))
            target["model_id"] = model_id
            target["model"] = self.model_names[model_id]
        elif action == "update_skill" and target is not None:
            target["skill"] = self._valid_skill(update.get("skill"), target["skill"])
        elif action == "update_prompt" and target is not None:
            target["prompt"] = str(update.get("prompt") or "")
            target.pop("instruction", None)
        elif action == "remove_node" and target is not None:
            target["disabled"] = True
        elif action == "add_node":
            known_ids = [agent.get("id") for agent in agents]
            new_agent = self._normalize_workflow_agent(
                update.get("new_step", {}),
                len(agents) + 1,
                known_ids,
            )
            agents.append(new_agent)
        elif action != "no_change":
            self._quality_issue(
                "invalid_update", f"workflow update target not found: {target_id}"
            )

        plan["collaboration"] = self._infer_workflow_collaboration(
            [agent for agent in agents if not agent.get("disabled")]
        )
        logger.info(f"LLMCoordinatorRouter workflow update={update}")
        return plan

    def _run_agent(
        self,
        query: str,
        plan: Dict[str, Any],
        agent: Dict[str, Any],
        context: str,
        round_idx: int,
        is_terminal: bool = False,
        previous_response: str = "",
    ) -> str:
        skill = self._skill_profile(agent["skill"])
        model_profile = self._model_profile(str(agent["model"]))
        supports_vision = bool(model_profile.get("SupportsVision"))
        node_prompt = str(agent.get("prompt") or "")
        collaboration = str(plan.get("collaboration") or "")
        iterative_previous_response = str(previous_response or "")
        if collaboration == "CoT" and round_idx > 0:
            iterative_previous_response = context
        visible_context = "" if collaboration == "CoT" else context
        protocol_instruction = self._collaboration_worker_instruction(
            plan=plan,
            agent=agent,
            round_idx=round_idx,
        )
        context_heading = (
            "Your prior Reflection context:"
            if collaboration == "Reflection"
            else "Information provided by other agents:"
        )
        user_prompt = (
            f"{protocol_instruction}\n\n"
            f"Original question and answer choices:\n{self._original_question_view(query)}\n\n"
            f"{context_heading}\n\n"
            f"{visible_context}\n\n"
            f"{self._worker_skill_section(skill, supports_vision=supports_vision, original_question=query)}"
            f"Here is the instruction you need to assist with: {node_prompt}"
        )
        if iterative_previous_response:
            # The Agent's previous response must be the literal final content
            # in its next User Prompt: no header, XML tag, or other marker.
            user_prompt = f"{user_prompt}\n\n{iterative_previous_response}"
        api_model = self._agent_api_model(agent, is_terminal=is_terminal)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": WORKER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._worker_user_content(
                    user_prompt,
                    skill,
                    supports_vision=supports_vision,
                ),
            },
        ]
        return self._call_llm(
            model=api_model,
            messages=messages,
            temperature=FIXED_WORKER_TEMPERATURE,
            max_tokens=self._worker_max_tokens(api_model, is_terminal=is_terminal),
            request_type="agent",
            trace_context={
                "round": round_idx + 1,
                "agent_id": agent["id"],
                "agent_name": agent.get("name", agent["id"]),
                "skill": agent["skill"],
                "collaboration": plan["collaboration"],
                "configured_model": agent["model"],
                "cost_coefficient": float(model_profile.get("CostCoefficient", 1.0)),
                "cost_accounting": str(model_profile.get("CostAccounting", "flops")),
                "api_model": api_model,
            },
        )

    def _collaboration_worker_instruction(
        self,
        plan: Dict[str, Any],
        agent: Dict[str, Any],
        round_idx: int,
    ) -> str:
        collaboration = str(plan.get("collaboration") or "")
        total_rounds = self._round_count(plan)
        round_index = round_idx + 1
        values = {
            "round_index": round_index,
            "total_rounds": total_rounds,
        }
        if collaboration == "IO":
            return IO_WORKER_INSTRUCTION
        if collaboration == "CoT":
            return COT_WORKER_INSTRUCTION.format(**values)
        if collaboration == "Chain":
            active_agent_ids = [
                str(candidate.get("id"))
                for candidate in plan.get("agents", [])
                if not candidate.get("disabled")
            ]
            agent_id = str(agent.get("id"))
            if active_agent_ids and agent_id == active_agent_ids[0]:
                return CHAIN_FIRST_WORKER_INSTRUCTION.format(**values)
            if active_agent_ids and agent_id == active_agent_ids[-1]:
                return CHAIN_FINAL_WORKER_INSTRUCTION.format(**values)
            return CHAIN_MIDDLE_WORKER_INSTRUCTION.format(**values)
        if collaboration == "FullConnected":
            template = (
                FULL_CONNECTED_INITIAL_INSTRUCTION
                if round_idx == 0
                else FULL_CONNECTED_REVISION_INSTRUCTION
            )
            return template.format(**values)
        if collaboration == "Debate":
            if round_idx == 0:
                template = DEBATE_INITIAL_INSTRUCTION
            elif round_index == total_rounds:
                template = DEBATE_FINAL_INSTRUCTION
            else:
                template = DEBATE_REBUTTAL_INSTRUCTION
            return template.format(**values)
        if collaboration == "Reflection":
            template = (
                REFLECTION_INITIAL_INSTRUCTION
                if round_idx == 0
                else REFLECTION_REVISION_INSTRUCTION
            )
            return template.format(**values)
        return ""

    def _worker_skill_text(
        self,
        skill: Dict[str, Any],
        *,
        supports_vision: bool = True,
        original_question: str = "",
    ) -> str:
        text = f"{skill['Name']}: {skill['Instruction']}"
        if skill["Name"] == INFORMATION_SEARCH_SKILL and self.episode_assets:
            passages = self._format_episode_assets(images=False)
            if passages:
                text += "\n\nPassages:\n" + passages
        if skill["Name"] == VISUAL_EVIDENCE_SKILL and self.episode_assets:
            if not supports_vision:
                text += "\n\n" + VISION_UNAVAILABLE_TEXT
            else:
                images = self._format_episode_assets(images=True)
                if images:
                    text += "\n\nImages attached to this Worker:\n" + images
        return text

    def _worker_skill_section(
        self,
        skill: Dict[str, Any],
        *,
        supports_vision: bool = True,
        original_question: str = "",
    ) -> str:
        """Render the optional Worker Skill block in the MAS user prompt."""

        return (
            "Here is the skill provided by the conductor: "
            f"{self._worker_skill_text(skill, supports_vision=supports_vision, original_question=original_question)}\n"
        )

    @staticmethod
    def _asset_is_image(asset: Any) -> bool:
        if not isinstance(asset, dict):
            return False
        kind = str(asset.get("kind") or "").casefold()
        media_type = str(asset.get("media_type") or "").casefold()
        path = str(asset.get("path") or "").casefold()
        return (
            kind == "image"
            or media_type.startswith("image/")
            or Path(path).suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        )

    def _format_episode_assets(self, *, images: bool) -> str:
        rendered = []
        for index, asset in enumerate(self.episode_assets):
            if self._asset_is_image(asset) != images:
                continue
            if isinstance(asset, dict):
                default_id = f"image-{index}" if images else f"passage-{index}"
                asset_id = str(asset.get("asset_id") or default_id)
                if images:
                    metadata = asset.get("metadata") or {}
                    detail = str(
                        metadata.get("description")
                        or asset.get("path")
                        or "visual asset"
                    )
                    rendered.append(f"[{asset_id}] {detail}")
                    continue
                content = asset.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
            else:
                asset_id = f"passage-{index}"
                content = str(asset)
            if str(content).strip():
                rendered.append(f"[{asset_id}]\n{str(content).strip()}")
        return "\n\n".join(rendered)

    def _worker_user_content(
        self,
        user_prompt: str,
        skill: Dict[str, Any],
        *,
        supports_vision: bool = True,
    ) -> str | List[Dict[str, Any]]:
        if skill["Name"] != VISUAL_EVIDENCE_SKILL:
            return user_prompt
        if not supports_vision:
            return user_prompt
        image_blocks = self._visual_asset_blocks()
        if not image_blocks:
            return user_prompt
        return [{"type": "text", "text": user_prompt}, *image_blocks]

    def _visual_asset_blocks(self) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for asset in self.episode_assets:
            if not self._asset_is_image(asset):
                continue
            image_url = self._image_asset_data_url(asset)
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        return blocks

    def _image_asset_data_url(self, asset: Dict[str, Any]) -> str:
        content = asset.get("content")
        if isinstance(content, str) and content.startswith("data:image/"):
            return content
        path_value = str(asset.get("path") or "").strip()
        if not path_value:
            raise ValueError(
                f"Image asset {asset.get('asset_id', '<unknown>')} needs a path or data URL"
            )
        image_path = Path(path_value).expanduser()
        if not image_path.is_absolute() and self.episode_asset_root is not None:
            image_path = self.episode_asset_root / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Worker image asset does not exist: {image_path}")
        media_type = str(asset.get("media_type") or "").strip()
        if not media_type.startswith("image/"):
            media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def _run_workflow_agent(
        self,
        query: str,
        plan: Dict[str, Any],
        agent: Dict[str, Any],
        context: str,
        round_idx: int,
        is_terminal: bool,
    ) -> str:
        skill = self._skill_profile(agent["skill"])
        model_profile = self._model_profile(str(agent["model"]))
        supports_vision = bool(model_profile.get("SupportsVision"))
        node_prompt = str(agent.get("prompt") or "")
        if is_terminal:
            system_prompt = (
                "You are the final worker in a model-orchestration workflow. Execute the assigned subtask using the "
                "original user task and the authorized earlier responses. Treat earlier responses as fallible and do "
                "any checking needed for correctness. Return exactly one <answer>...</answer> block whose contents obey "
                "the original user's output contract, with no text outside that block."
            )
        else:
            system_prompt = (
                "You are a worker in a model-orchestration workflow. Execute the assigned subtask using the original "
                "user task and only the authorized earlier responses supplied below. Treat earlier responses as fallible. "
                "Return complete reasoning, evidence, calculations, code, or other artifacts that will help downstream "
                "workers; no special wrapper is required."
            )
        user_prompt = (
            f"Original user task:\n{self._worker_task_view(query)}\n\n"
            f"Authorized earlier responses:\n{context or 'None.'}\n\n"
            f"{self._worker_skill_section(skill, supports_vision=supports_vision, original_question=query)}"
            f"Assigned subtask:\n{node_prompt}"
        )
        api_model = self._agent_api_model(agent, is_terminal=is_terminal)
        return self._call_llm(
            model=api_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": self._worker_user_content(
                        user_prompt,
                        skill,
                        supports_vision=supports_vision,
                    ),
                },
            ],
            temperature=FIXED_WORKER_TEMPERATURE,
            max_tokens=self._worker_max_tokens(api_model, is_terminal=is_terminal),
            request_type="agent",
            trace_context={
                "round": round_idx + 1,
                "agent_id": agent["id"],
                "agent_name": agent.get("name", agent["id"]),
                "skill": agent["skill"],
                "collaboration": plan["collaboration"],
                "configured_model": agent["model"],
                "cost_coefficient": float(model_profile.get("CostCoefficient", 1.0)),
                "cost_accounting": str(model_profile.get("CostAccounting", "flops")),
                "reasoning_mode": agent.get("mode", "auto"),
                "api_model": api_model,
            },
        )

    def _worker_max_tokens(self, model: str, *, is_terminal: bool) -> int:
        # Step's unavoidable reasoning shares the generation allowance with
        # its answer. Apply this exception only to Worker calls, in all modes.
        if str(model).lower() == "step-3.5-flash":
            return 4096
        return self.terminal_max_tokens if is_terminal else self.agent_max_tokens

    def _agent_api_model(self, agent: Dict[str, Any], is_terminal: bool) -> str:
        model = str(agent.get("model") or "")
        requested_mode = str(agent.get("mode") or "auto").lower()
        if requested_mode not in {"auto", "thinking", "nothinking"}:
            requested_mode = "auto"
        if model == "gemini-2.5-flash":
            mode = requested_mode
            if mode == "auto":
                mode = (
                    self.gemini_terminal_mode
                    if is_terminal
                    else self.gemini_intermediate_mode
                )
            if mode != "auto":
                return f"{model}-{mode}"
        if model == "deepseek-v4-flash":
            mode = requested_mode
            if mode == "auto":
                mode = (
                    self.deepseek_terminal_mode
                    if is_terminal
                    else self.deepseek_intermediate_mode
                )
            if mode != "auto":
                return f"{model}-{mode}"
        return model

    def _finalize_or_revise(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
        prompt_file: str = "",
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        # Code is whitespace-sensitive. Select the terminal Worker by the public
        # plan order, never by hidden-test success, and preserve its output.
        if self._is_code_task(query) or getattr(
            self, "_code_verbatim_submission", False
        ):
            return self._terminal_workflow_answer(query, plan, history), None
        if plan.get("protocol") == "workflow_v2":
            return self._review_workflow_result(query, plan, history)

        prompt_hint = ""
        if prompt_file and os.path.exists(prompt_file):
            try:
                prompt_hint = json.dumps(
                    json.load(open(prompt_file, "r", encoding="utf-8")),
                    ensure_ascii=False,
                )
            except Exception:
                prompt_hint = ""

        tag, value = self._run_state_until_terminal(
            state=MAS_STATE_JUDGING,
            messages=self._judging_state_messages(
                query=query,
                plan=plan,
                history=history,
                prompt_hint=prompt_hint,
            ),
            request_type="final",
            terminal_tags=("answer",),
            plan=plan,
        )
        if tag == "answer" and isinstance(value, str):
            return value, None
        return None, None

    def _review_workflow_result(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        active_ids = [
            agent["id"] for agent in plan.get("agents", []) if not agent.get("disabled")
        ]
        fallback_step_id = active_ids[-1] if active_ids else ""
        fallback = {
            "action": "accept",
            "answer_step_id": fallback_step_id,
            "reason": "Use the terminal worker response.",
        }
        one_shot = (
            "ONE-SHOT RESPONSE EXAMPLE (use the actual current step id shown here; copy no answer content):\n"
            "<analysis>The terminal response is supported by its dependencies and already follows the answer contract.</analysis>\n"
            f'<decision>{{"action":"accept","answer_step_id":"{fallback_step_id}",'
            '"reason":"The terminal worker produced the supported final answer."}</decision>\n\n'
            if self.few_shot_workflows
            else ""
        )
        status = self._status_xml(
            self._status_payload(
                query=query,
                plan=plan,
                history=history,
                recent_only=False,
            )
        )
        prompt = {
            "role": "user",
            "content": (
                "ROUTER_WORKFLOW_REVIEW_REQUEST\n"
                "You are the Coordinator LLM reviewing a completed worker workflow. The state is inside "
                "<status>...</status>. Do not solve or rewrite the answer yourself. Decide whether a worker response "
                "can be returned, or whether a targeted correction workflow is needed.\n\n"
                f"{status}\n\n"
                "Strict output protocol:\n"
                "1. Output one or more <analysis>...</analysis> blocks.\n"
                "2. Then output exactly one <decision>...</decision> block containing one strict JSON object.\n"
                "3. To accept, select the step whose response should be returned verbatim after extracting its "
                "<answer> block.\n"
                "4. To revise, provide only new correction steps. They may independently retry disputed hops and "
                "must end with a worker instructed to return <answer>short final answer</answer>.\n\n"
                "Accepted schemas:\n"
                '{"action":"accept","answer_step_id":"step_3","reason":"..."}\n'
                '{"action":"revise","steps":[{"id":"step_1","model_id":0,'
                '"skill":"allowed skill","depends_on":[],"prompt":"..."}],"reason":"..."}\n\n'
                "Available workers:\n"
                f"{self._format_indexed_model_pool()}\n\n"
                "Available skills:\n"
                f"{self._format_skill_catalog()}\n\n"
                f"{one_shot}"
            ),
        }
        response = self._call_coordinator([prompt], request_type="final")
        decision = self._extract_tagged_json(
            response,
            tag="decision",
            fallback=fallback,
            request_type="final",
        )
        action = decision.get("action") if isinstance(decision, dict) else None
        if action == "accept":
            step_id = str(decision.get("answer_step_id") or "")
            if step_id not in active_ids:
                self._quality_issue(
                    "invalid_plan", f"final decision selected unknown step: {step_id}"
                )
                return "", None
            selected = next(
                (
                    item
                    for item in reversed(history)
                    if str(item.get("agent_id")) == step_id
                ),
                None,
            )
            if selected is None:
                self._quality_issue(
                    "invalid_plan",
                    f"final decision selected unexecuted step: {step_id}",
                )
                return "", None
            answer = self._extract_worker_answer(
                selected.get("output", ""),
                code_task=self._is_code_task(query),
            )
            if not answer:
                self._quality_issue(
                    "format_errors",
                    f"worker {step_id} did not provide an extractable answer",
                )
                return "", None
            return answer, None

        if action == "revise":
            raw_plan = {
                "steps": decision.get("steps"),
            }
            self._validate_raw_plan(raw_plan)
            return None, self._normalize_plan(
                raw_plan,
                fallback=self._fallback_plan(protocol="workflow_v2"),
            )

        self._quality_issue("invalid_plan", f"invalid final decision action: {action}")
        return "", None

    def _terminal_workflow_answer(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> str:
        active_ids = [
            agent.get("id")
            for agent in plan.get("agents", [])
            if not agent.get("disabled")
        ]
        terminal_id = active_ids[-1] if active_ids else ""
        selected = next(
            (
                item
                for item in reversed(history)
                if str(item.get("agent_id")) == str(terminal_id)
            ),
            None,
        )
        answer = (
            self._extract_worker_answer(
                selected.get("output", ""),
                code_task=self._is_code_task(query)
                or getattr(self, "_code_verbatim_submission", False),
            )
            if selected
            else ""
        )
        if not answer:
            self._quality_issue(
                "format_errors",
                f"terminal worker {terminal_id or '<missing>'} did not provide an extractable answer",
            )
        return answer

    @staticmethod
    def _extract_worker_answer(text: Any, code_task: bool = False) -> str:
        text = text if isinstance(text, str) else str(text or "")
        matches = re.findall(
            r"<answer>\s*(.*?)\s*</answer\s*>?",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if matches:
            candidate = matches[-1].strip()
            candidate = re.sub(
                r"</?answer\s*>?", "", candidate, flags=re.IGNORECASE
            ).strip()
            return candidate
        opened = re.findall(
            r"<answer>\s*(.*)\Z" if code_task else r"<answer>\s*([^\n<]+)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if opened:
            return opened[-1].strip()
        if code_task:
            return text.strip()
        labelled = re.findall(
            r"(?:final\s+answer|answer)\s*[:：]\s*(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if labelled:
            return labelled[-1].strip().strip('"').strip("'")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            candidate = re.sub(r"^[-*]\s*", "", lines[-1]).strip()
            if 0 < len(candidate.split()) <= 20:
                return candidate.strip('"').strip("'")
        return text.strip()

    @staticmethod
    def _extract_worker_result(text: Any) -> str:
        text = text if isinstance(text, str) else str(text or "")
        matches = re.findall(
            r"<result>\s*(.*?)\s*</result>", text, flags=re.DOTALL | re.IGNORECASE
        )
        if matches:
            return matches[-1].strip()
        return text.strip()

    def _agent_context(
        self,
        plan: Dict[str, Any],
        agent_index: int,
        current_round_outputs: Dict[str, str],
        previous_round_outputs: Dict[str, str],
        history: List[Dict[str, Any]],
        round_idx: int = 0,
    ) -> str:
        collaboration = plan["collaboration"]
        agents = plan["agents"]
        visible: List[Tuple[str, str]] = []

        if collaboration == "CoT" and round_idx > 0:
            return str(previous_round_outputs.get(str(agents[agent_index]["id"]), ""))
        if collaboration == "Chain":
            if agent_index > 0:
                prev_agent = agents[agent_index - 1]["id"]
                if prev_agent in current_round_outputs:
                    visible.append((prev_agent, str(current_round_outputs[prev_agent])))
        elif collaboration == "Reflection" and round_idx > 0:
            agent_id = str(agents[agent_index]["id"])
            own_response = str(previous_round_outputs.get(agent_id, ""))
            return (
                REFLECTION_SELF_CONTEXT.format(response=own_response)
                if own_response
                else ""
            )
        elif collaboration in {"FullConnected", "Debate"} and round_idx > 0:
            agent_id = str(agents[agent_index]["id"])
            own_response = str(previous_round_outputs.get(agent_id, ""))
            peer_responses = [
                (
                    str(candidate["id"]),
                    str(previous_round_outputs.get(str(candidate["id"]), "")),
                )
                for candidate in agents
                if not candidate.get("disabled")
                and str(candidate.get("id")) != agent_id
            ]
            peer_text = "\n".join(
                f"- {peer_id}: {response}"
                for peer_id, response in peer_responses
                if response
            )
            if collaboration == "FullConnected":
                templates = FULL_CONNECTED_SELF_CONTEXT, FULL_CONNECTED_PEER_CONTEXT
            else:
                templates = DEBATE_SELF_CONTEXT, DEBATE_PEER_CONTEXT
            parts = []
            if own_response:
                parts.append(templates[0].format(response=own_response))
            if peer_text:
                parts.append(templates[1].format(responses=peer_text))
            return "\n\n".join(parts)

        return "\n".join(
            f"- [#{index}] {agent_id}: {output}"
            for index, (agent_id, output) in enumerate(visible, start=1)
        )

    def _round_count(self, plan: Dict[str, Any]) -> int:
        collaboration = str(plan.get("collaboration") or "")
        if collaboration == "IO":
            return 1
        rounds = self._parse_rounds_value(plan.get("rounds")) or 1
        minimum = 2 if collaboration in {"FullConnected", "Debate", "Reflection"} else 1
        return min(MAX_MAS_ROUNDS, max(minimum, rounds))

    def _normalize_plan(
        self, raw_plan: Dict[str, Any], fallback: Dict[str, Any]
    ) -> Dict[str, Any]:
        source = raw_plan if isinstance(raw_plan, dict) else fallback
        collaboration = self._valid_collaboration(source.get("collaboration"))
        raw_agents = source.get("agents")
        if not isinstance(raw_agents, list) or not raw_agents:
            raw_agents = fallback["agents"]

        agents = [
            self._normalize_agent(raw_agent, i)
            for i, raw_agent in enumerate(raw_agents, start=1)
        ]
        if collaboration in {"IO", "CoT", "Reflection"}:
            agents = agents[:1]
        return {
            "collaboration": collaboration,
            "rounds": self._normalized_rounds(collaboration, source.get("rounds", 1)),
            "agents": agents or deepcopy(fallback["agents"]),
        }

    @classmethod
    def _normalized_rounds(cls, collaboration: str, value: Any) -> int:
        if collaboration == "IO":
            return 1
        rounds = cls._parse_rounds_value(value) or 1
        minimum = 2 if collaboration in {"FullConnected", "Debate", "Reflection"} else 1
        return min(MAX_MAS_ROUNDS, max(minimum, rounds))

    @staticmethod
    def _parse_rounds_value(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 1 else None
        if isinstance(value, float):
            return int(value) if value.is_integer() and value >= 1 else None
        if isinstance(value, str) and re.fullmatch(r"[1-9]\d*", value.strip()):
            return int(value.strip())
        return None

    def _normalize_workflow_plan(
        self,
        raw_plan: Dict[str, Any],
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_plan = raw_plan if isinstance(raw_plan, dict) else {}
        raw_steps = raw_plan.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raw_steps = raw_plan.get("agents")
        if not isinstance(raw_steps, list) or not raw_steps:
            raw_steps = fallback.get("agents", [])

        agents: List[Dict[str, Any]] = []
        known_ids: List[str] = []
        for index, raw_agent in enumerate(raw_steps, start=1):
            agent = self._normalize_workflow_agent(raw_agent, index, known_ids)
            agents.append(agent)
            known_ids.append(agent["id"])

        if not agents:
            agents = deepcopy(fallback.get("agents") or [self._default_agent(1)])
        return {
            "protocol": "workflow_v2",
            "collaboration": self._infer_workflow_collaboration(agents),
            "num_rounds": 1,
            "agents": agents,
            "updates": [],
        }

    def _normalize_workflow_agent(
        self,
        raw_agent: Dict[str, Any],
        index: int,
        known_ids: Sequence[str],
    ) -> Dict[str, Any]:
        raw_agent = raw_agent if isinstance(raw_agent, dict) else {}
        agent_id = str(raw_agent.get("id") or f"step_{index}")
        agent_id = re.sub(r"[^A-Za-z0-9_-]", "_", agent_id)[:40] or f"step_{index}"
        model_id = self._valid_model_id(
            raw_agent.get("model_id", raw_agent.get("model"))
        )
        node_prompt = str(
            raw_agent.get(
                "prompt",
                raw_agent.get("subtask", raw_agent.get("instruction", "")),
            )
            or ""
        )
        raw_dependencies = raw_agent.get("depends_on", raw_agent.get("access_list", []))
        if raw_dependencies == "all":
            dependencies = list(known_ids)
        elif isinstance(raw_dependencies, list):
            dependencies = [
                str(item) for item in raw_dependencies if str(item) in known_ids
            ]
        else:
            dependencies = []
        skill_default = (
            "GeneralReasoning"
            if "GeneralReasoning" in self.skill_names
            else self.skill_names[0]
        )
        reasoning_mode = str(
            raw_agent.get("mode", raw_agent.get("reasoning_mode", "auto")) or "auto"
        ).lower()
        if reasoning_mode not in {"auto", "thinking", "nothinking"}:
            reasoning_mode = "auto"
        return {
            "id": agent_id,
            "name": str(raw_agent.get("name") or agent_id),
            "model_id": model_id,
            "model": self.model_names[model_id],
            "skill": self._valid_skill(raw_agent.get("skill"), skill_default),
            "mode": reasoning_mode,
            "temperature": self._temperature(
                raw_agent.get("temperature", self.agent_temperature)
            ),
            "depends_on": dependencies,
            "prompt": node_prompt,
        }

    @staticmethod
    def _infer_workflow_collaboration(agents: Sequence[Dict[str, Any]]) -> str:
        if len(agents) <= 1:
            return "IO"
        ids = [agent.get("id") for agent in agents]
        is_chain = all(
            list(agent.get("depends_on") or []) == [ids[index - 1]]
            for index, agent in enumerate(agents[1:], start=1)
        )
        if is_chain:
            return "Chain"
        final_dependencies = set(agents[-1].get("depends_on") or [])
        if len(final_dependencies) >= 2:
            return "FullConnected"
        if any(len(agent.get("depends_on") or []) > 1 for agent in agents):
            return "FullConnected"
        return "Chain"

    def _normalize_agent(self, raw_agent: Dict[str, Any], index: int) -> Dict[str, Any]:
        raw_agent = raw_agent if isinstance(raw_agent, dict) else {}
        agent_id = str(raw_agent.get("id") or f"agent_{index}")
        agent_id = re.sub(r"[^A-Za-z0-9_-]", "_", agent_id)[:40] or f"agent_{index}"
        node_prompt = str(
            raw_agent.get("prompt", raw_agent.get("instruction", "")) or ""
        )
        return {
            "id": agent_id,
            "model": self._valid_model(raw_agent.get("model"), self.model_names[0]),
            "skill": self._valid_skill(raw_agent.get("skill"), self.skill_names[0]),
            "prompt": node_prompt,
        }

    def _normalize_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(update, dict):
            return {
                "type": "no_change",
                "action": "no_change",
                "reason": "Invalid update object.",
            }
        action = update.get("type", update.get("action", "no_change"))
        allowed = {
            "no_change",
            "update_model",
            "update_temperature",
            "update_skill",
            "update_prompt",
            "set_num_rounds",
            "remove_node",
            "add_node",
        }
        if action not in allowed:
            action = "no_change"
        update["type"] = action
        update["action"] = action
        if action == "update_model":
            update["model"] = self._valid_model(
                update.get("model"), self.model_names[0]
            )
        elif action == "update_skill":
            update["skill"] = self._valid_skill(
                update.get("skill"), self.skill_names[0]
            )
        elif action == "update_temperature":
            update["temperature"] = self._temperature(
                update.get("temperature", self.agent_temperature)
            )
        elif action == "update_prompt":
            update["prompt"] = str(update.get("prompt") or "")
        elif action == "set_num_rounds":
            try:
                update["num_rounds"] = max(1, int(update.get("num_rounds", 1)))
            except Exception:
                update["num_rounds"] = 1
        elif action == "add_node":
            update["new_agent"] = self._normalize_agent(update.get("new_agent", {}), 1)
        return update

    def _normalize_workflow_update(
        self,
        update: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        update = deepcopy(update) if isinstance(update, dict) else {}
        action = update.get("type", "no_change")
        allowed = {
            "no_change",
            "update_model",
            "update_skill",
            "update_prompt",
            "remove_node",
            "add_node",
        }
        if action not in allowed:
            action = "no_change"
        update["type"] = action
        update["action"] = action
        if action == "update_model":
            update["model_id"] = self._valid_model_id(update.get("model_id"))
        elif action == "update_skill":
            update["skill"] = self._valid_skill(
                update.get("skill"),
                self._preferred_skill("GeneralReasoning"),
            )
        elif action == "update_prompt":
            update["prompt"] = str(update.get("prompt") or "")
        elif action == "add_node":
            update["new_step"] = deepcopy(update.get("new_step") or {})
        return update

    def _fallback_plan(self, protocol: str = "legacy") -> Dict[str, Any]:
        if protocol == "workflow_v2":
            return self._fallback_workflow_plan()
        agent_count = 2
        return {
            "collaboration": (
                "Chain"
                if "Chain" in self.collaboration_names
                else self.collaboration_names[0]
            ),
            "rounds": 1,
            "agents": [
                self._default_agent(index) for index in range(1, agent_count + 1)
            ],
        }

    def _fallback_workflow_plan(self) -> Dict[str, Any]:
        reasoner_id = self._preferred_model_id("gemini-2.5-flash", default=0)
        independent_id = self._preferred_model_id(
            "qwen3.5-397b-a17b", default=reasoner_id
        )
        planner_id = self._preferred_model_id("qwen3.5-27b", default=reasoner_id)
        steps = [
            {
                "id": "step_1",
                "model_id": planner_id,
                "skill": self._preferred_skill("ProblemDecomposition"),
                "depends_on": [],
                "prompt": "Decompose the question into an ordered chain of factual subquestions and end with <result>the chain</result>.",
            },
            {
                "id": "step_2",
                "model_id": independent_id,
                "skill": self._preferred_skill("DomainKnowledgeQA"),
                "depends_on": [],
                "prompt": "Solve independently, track every intermediate entity, and end with <result>the proposed answer</result>.",
            },
            {
                "id": "step_3",
                "model_id": reasoner_id,
                "skill": self._preferred_skill("AnswerSynthesis"),
                "depends_on": ["step_1", "step_2"],
                "prompt": "Reconcile the prior work, correct entity-chain mistakes, and return only <answer>short final answer</answer>.",
            },
        ]
        normalized = self._normalize_workflow_plan(
            {"steps": steps},
            {"agents": []},
        )
        return normalized

    def _default_agent(self, index: int) -> Dict[str, Any]:
        model = self.model_names[(index - 1) % len(self.model_names)]
        skill = self.skill_names[(index - 1) % len(self.skill_names)]
        return {
            "id": f"agent_{index}",
            "model": model,
            "skill": skill,
            "prompt": "Complete the assigned task using the selected Skill and available Agent information.",
        }

    def _call_coordinator(
        self, messages: List[Dict[str, str]], request_type: str
    ) -> str:
        max_tokens = self._coordinator_max_tokens_for_request(request_type)
        return self._call_llm(
            model=self.coordinator_model,
            messages=messages,
            temperature=self.coordinator_temperature,
            max_tokens=max_tokens,
            request_type=request_type,
            trace_context={"coordinator_stage": request_type},
        )

    @staticmethod
    def _resolve_coordinator_token_limit(
        stage_value: Optional[int],
        shared_value: Optional[int],
        default: int,
        name: str,
    ) -> int:
        value = stage_value if stage_value is not None else shared_value
        resolved = int(default if value is None else value)
        if resolved <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return resolved

    def _coordinator_max_tokens_for_request(self, request_type: str) -> int:
        if str(request_type).lower() == "final":
            return self.coordinator_final_max_tokens
        return self.coordinator_plan_max_tokens

    def _mark_last_coordinator_attempt_rejected(
        self,
        request_type: str,
        reason: str,
    ) -> None:
        """Allow a policy wrapper to mark an unsuccessful generation attempt."""
        return None

    def _call_llm(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        request_type: str,
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        started = time.monotonic()
        usage_before = self._usage_snapshot()
        try:
            llm = LLMRegistry.get(model)
            response = llm.gen(messages, max_tokens=max_tokens, temperature=temperature)
            response = response if isinstance(response, str) else str(response)
        except Exception as exc:
            message = self._format_llm_call_error(model, request_type, exc)
            self._record_event(
                "llm_error",
                {
                    "request_type": request_type,
                    "model": model,
                    "error": message,
                    "raw_error": type(exc).__name__,
                    "trace_context": trace_context or {},
                },
            )
            raise RuntimeError(message) from exc
        usage_after = self._usage_snapshot()
        usage = {
            "input_tokens": max(
                0,
                int(
                    round(usage_after["prompt_tokens"] - usage_before["prompt_tokens"])
                ),
            ),
            "reasoning_tokens": 0,
            "output_tokens": max(
                0,
                int(
                    round(
                        usage_after["completion_tokens"]
                        - usage_before["completion_tokens"]
                    )
                ),
            ),
            "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
        }
        self._record_event(
            "llm_call",
            {
                "request_type": request_type,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": self._trace_safe_messages(messages),
                "response": response,
                "usage": usage,
                "trace_context": trace_context or {},
            },
        )
        return response

    @staticmethod
    def _trace_safe_messages(
        messages: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep image metadata in traces without duplicating base64 payloads."""

        safe_messages = deepcopy(list(messages))
        for message in safe_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image_url":
                    continue
                image_url = block.get("image_url")
                if not isinstance(image_url, dict):
                    continue
                url = str(image_url.get("url") or "")
                if url.startswith("data:image/"):
                    media_type = url[5:].split(";", 1)[0]
                    image_url["url"] = f"data:{media_type};base64,<omitted>"
        return safe_messages

    def _format_llm_call_error(
        self, model: str, request_type: str, exc: Exception
    ) -> str:
        error = exc
        last_attempt = getattr(exc, "last_attempt", None)
        if last_attempt is not None:
            try:
                inner_exc = last_attempt.exception()
            except Exception:
                inner_exc = None
            if inner_exc is not None:
                error = inner_exc
        message = type(error).__name__
        status = getattr(error, "status_code", None)
        if status is not None:
            message += f" (HTTP {status})"
        return (
            f"LLM call failed during '{request_type}' with model '{model}'. "
            f"Provider/API error: {message}"
        )

    def _status_payload(
        self,
        query: str,
        plan: Dict[str, Any],
        history: List[Dict[str, Any]],
        recent_only: bool = False,
        prompt_hint: str = "",
    ) -> Dict[str, Any]:
        history_items = self._compact_history_items(
            history[-4:] if recent_only else history
        )
        payload = {
            "task": self._coordinator_task_view(query),
            "current_mas": plan,
            "node_outputs": history_items,
        }
        if plan.get("protocol") == "workflow_v2":
            current_workflow_index = max(
                (
                    int(item.get("workflow_index", 0))
                    for item in history
                    if isinstance(item, dict)
                ),
                default=0,
            )
            completed_ids = {
                str(item.get("agent_id"))
                for item in history
                if isinstance(item, dict)
                and int(item.get("workflow_index", 0)) == current_workflow_index
                and item.get("agent_id")
            }
            active_ids = [
                str(agent.get("id"))
                for agent in plan.get("agents", [])
                if isinstance(agent, dict)
                and not agent.get("disabled")
                and agent.get("id")
            ]
            payload["completed_step_ids"] = [
                step_id for step_id in active_ids if step_id in completed_ids
            ]
            payload["remaining_step_ids"] = [
                step_id for step_id in active_ids if step_id not in completed_ids
            ]
        if recent_only:
            payload["status_scope"] = "recent_node_outputs"
        else:
            payload["status_scope"] = "all_node_outputs"
        if prompt_hint:
            payload["optional_final_answer_prompt"] = prompt_hint
        return payload

    @staticmethod
    def _coordinator_task_view(query: str) -> str:
        """Keep long evidence in worker prompts while routing from the question itself."""
        return re.sub(
            r"\n\s*<evidence>.*?</evidence>\s*$",
            "",
            str(query or ""),
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    @staticmethod
    def _is_code_task(query: str) -> bool:
        text = str(query or "")
        return bool(
            re.search(
                r"(?:Task domain:\s*(?:code_generation|livecodebench|coding)\b|"
                r"Output contract:\s*complete_executable_python3_program\b)",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _original_question_view(query: str) -> str:
        """Extract the original question from the evaluation query."""
        text = LLMCoordinatorRouter._coordinator_task_view(query)
        text = re.sub(
            r"(?:^|\n)[ \t]*Task (?:running budget|Budget):[ \t]*(?:High|Medium|Low)\.[^\n]*(?:\n|$)",
            "\n",
            text,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\AQuestion:[ \t]*", "", text.strip(), count=1)

    @staticmethod
    def _worker_task_view(query: str) -> str:
        """Remove routing-policy prose while retaining the raw question and all evidence."""
        text = str(query or "").strip()
        contract_lines = []
        for label in ("Task domain", "Output contract"):
            match = re.search(
                rf"(?:^|\n){re.escape(label)}:\s*([^\n]+)",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                contract_lines.append(f"{label}: {match.group(1).strip()}")
        question_match = re.search(
            r"(?:^|\n)Question:\s*(.*?)(?=\n\s*<evidence>|$)",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not question_match:
            return text
        question = question_match.group(1).strip()
        prefix = "\n".join(contract_lines)
        if prefix:
            prefix += "\n\n"
        evidence_match = re.search(
            r"<evidence>\s*(.*?)\s*</evidence>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not evidence_match:
            return f"{prefix}Question: {question}"
        return f"{prefix}Question: {question}\n\nEvidence passages:\n{evidence_match.group(1).strip()}"

    def _status_xml(self, payload: Dict[str, Any]) -> str:
        return (
            "<status>\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n</status>"
        )

    def _extract_tag(
        self,
        text: str,
        tag: str,
        request_type: str,
        required: bool = True,
    ) -> Optional[str]:
        if not isinstance(text, str):
            if required:
                self._quality_issue(
                    "format_errors", f"{request_type}: response is not a string"
                )
            return None
        pattern = re.compile(
            rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.DOTALL
        )
        matches = pattern.findall(text)
        if not matches:
            if required:
                self._quality_issue(
                    "format_errors", f"{request_type}: missing <{tag}> block"
                )
            return None
        if len(matches) > 1:
            self._quality_issue(
                "format_errors", f"{request_type}: multiple <{tag}> blocks"
            )
        return matches[0]

    def _ensure_analysis_before_terminal(
        self, text: str, tags: List[str], request_type: str
    ) -> None:
        if not isinstance(text, str):
            self._quality_issue(
                "format_errors", f"{request_type}: response is not a string"
            )
            return
        analysis_matches = list(
            re.finditer(r"<analysis>\s*(.*?)\s*</analysis>", text, re.DOTALL)
        )
        if not analysis_matches:
            self._quality_issue(
                "format_errors", f"{request_type}: missing <analysis> block"
            )
            return
        terminal_positions = []
        for tag in tags:
            match = re.search(rf"<{re.escape(tag)}>", text)
            if match:
                terminal_positions.append((match.start(), tag))
        if not terminal_positions:
            self._quality_issue(
                "format_errors",
                f"{request_type}: missing terminal tag from {tags}",
            )
            return
        first_terminal_start, first_tag = min(terminal_positions)
        if analysis_matches[-1].end() > first_terminal_start:
            self._quality_issue(
                "format_errors",
                f"{request_type}: <analysis> must appear before <{first_tag}>",
            )

    def _extract_tagged_json(
        self,
        text: str,
        tag: str,
        fallback: Dict[str, Any],
        request_type: str,
    ) -> Dict[str, Any]:
        self._ensure_analysis_before_terminal(
            text, tags=[tag], request_type=request_type
        )
        tagged_text = self._extract_tag(
            text, tag, request_type=request_type, required=True
        )
        if tagged_text is None:
            return deepcopy(fallback)
        return self._parse_json_text(
            tagged_text, fallback=fallback, request_type=request_type
        )

    def _parse_json_text(
        self, text: str, fallback: Dict[str, Any], request_type: str
    ) -> Dict[str, Any]:
        if not isinstance(text, str):
            self._quality_issue(
                "json_parse_failures", f"{request_type}: JSON payload is not a string"
            )
            return deepcopy(fallback)
        strict_text = text.strip()
        repaired_text = self._repair_invalid_json_escapes(strict_text)
        if repaired_text != strict_text:
            try:
                parsed = json.loads(repaired_text)
                if isinstance(parsed, dict):
                    self._compiler_repair(
                        f"{request_type}: escaped LaTeX-style backslashes inside string values"
                    )
                    return parsed
            except Exception:
                pass
        try:
            parsed = json.loads(strict_text)
            if isinstance(parsed, dict):
                return parsed
            self._quality_issue(
                "json_parse_failures", f"{request_type}: JSON payload is not an object"
            )
            return deepcopy(fallback)
        except Exception:
            recovered = self._first_json_object(text)
            if recovered:
                for candidate, repaired in (
                    (recovered, False),
                    (self._repair_invalid_json_escapes(recovered), True),
                ):
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            self._quality_issue(
                                "non_strict_json",
                                f"{request_type}: recovered JSON from non-strict tagged payload",
                            )
                            if repaired and candidate != recovered:
                                self._compiler_repair(
                                    f"{request_type}: escaped non-JSON backslashes in recovered JSON"
                                )
                            return parsed
                    except Exception:
                        pass
            self._quality_issue(
                "json_parse_failures", f"{request_type}: failed to parse tagged JSON"
            )
            logger.warning(
                f"Failed to parse coordinator tagged JSON, using fallback. Raw response: {str(text)[:500]}"
            )
            return deepcopy(fallback)

    def _extract_json(
        self, text: str, fallback: Dict[str, Any], request_type: str = "json"
    ) -> Dict[str, Any]:
        if not isinstance(text, str):
            self._quality_issue(
                "json_parse_failures", f"{request_type}: response is not a string"
            )
            return deepcopy(fallback)
        strict_text = text.strip()
        candidates = [(text, True)]
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            candidates.insert(0, (fenced.group(1), False))
        balanced = self._first_json_object(text)
        if balanced and balanced.strip() != strict_text:
            candidates.insert(0, (balanced, False))
        for candidate, is_strict in candidates:
            repaired_candidate = self._repair_invalid_json_escapes(candidate)
            payloads = [(candidate, False)]
            if repaired_candidate != candidate:
                payloads.insert(0, (repaired_candidate, True))
            for payload, repaired in payloads:
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        if not is_strict:
                            self._quality_issue(
                                "non_strict_json",
                                f"{request_type}: recovered JSON from non-strict response",
                            )
                        if repaired and payload != candidate:
                            self._compiler_repair(
                                f"{request_type}: escaped non-JSON backslashes inside string values"
                            )
                        return parsed
                except Exception:
                    continue
        self._quality_issue(
            "json_parse_failures", f"{request_type}: failed to parse JSON"
        )
        logger.warning(
            f"Failed to parse coordinator JSON, using fallback. Raw response: {text[:500]}"
        )
        return deepcopy(fallback)

    def _quality_issue(self, key: str, message: str) -> None:
        quality = getattr(self, "_quality", None)
        if quality is None:
            return
        quality[key] = int(quality.get(key, 0)) + 1
        quality.setdefault("issues", []).append(message)

    def _compiler_repair(self, message: str) -> None:
        quality = getattr(self, "_quality", None)
        if quality is None:
            return
        quality["compiled_repairs"] = int(quality.get("compiled_repairs", 0)) + 1
        quality.setdefault("repairs", []).append(message)

    @staticmethod
    def _repair_invalid_json_escapes(text: str) -> str:
        """Repair string escapes before JSON decoding can silently corrupt TeX."""
        ambiguous_tex = set(
            (
                "frac fbox framebox footnotesize forall flat begin beta bar boxed binom "
                "big Big bigg Bigg bigl bigr biggl biggr boldsymbol bf bmod backslash "
                "nabla nu neq ne not neg ni nmid notin newline newcommand nrightarrow "
                "rho right rangle rbrace rceil rfloor root rvert rm renewcommand "
                "rightarrow rightharpoonup rightleftharpoons text textrm textbf textit "
                "texttt textcolor textnormal textsf theta times tan tanh tau to tilde "
                "tfrac tbinom top triangle triangleright tiny tt"
            ).split()
        )
        text = str(text)
        out, inside, i = [], False, 0
        while i < len(text):
            char = text[i]
            if char == '"':
                inside = not inside
                out.append(char)
                i += 1
            elif inside and char == "\\" and i + 1 < len(text):
                nxt = text[i + 1]
                command = re.match(r"[A-Za-z]+", text[i + 1 :])
                tex = command is not None and command.group(0) in ambiguous_tex
                valid = nxt in '"\\/bfnrt' or (
                    nxt == "u" and re.match(r"[0-9a-fA-F]{4}", text[i + 2 :])
                )
                if valid and not tex:
                    out.extend((char, nxt))
                    i += 2
                else:
                    out.append("\\\\")
                    i += 1
            elif inside and ord(char) < 32:
                out.append(json.dumps(char)[1:-1])
                i += 1
            else:
                out.append(char)
                i += 1
        return "".join(out)

    def parse_plan_json(self, payload: str) -> Dict[str, Any]:
        """Parse one complete plan; repair string escapes, never invent JSON fields."""

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        def reject_constant(value):
            raise ValueError(f"non-standard JSON constant {value}")

        repaired = self._repair_invalid_json_escapes(payload)
        raw = json.loads(
            repaired, object_pairs_hook=unique_object, parse_constant=reject_constant
        )
        if not isinstance(raw, dict):
            raise ValueError("plan must be one JSON object")
        if repaired != payload:
            self._compiler_repair(
                "Escaped TeX or invalid JSON string controls/backslashes"
            )
        return raw

    def _repair_plan_execution_shape(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Keep every Agent, correcting only collaboration/count/round incompatibility."""
        plan = deepcopy(plan)
        if not isinstance(plan, dict):
            return plan
        agents, collaboration = plan.get("agents"), plan.get("collaboration")
        if (
            not isinstance(agents, list)
            or not agents
            or not isinstance(collaboration, str)
        ):
            return plan
        count = len(agents)
        mapping = (
            {"IO": "Chain", "CoT": "Chain", "Reflection": "FullConnected"}
            if count > 1
            else {"Chain": "CoT", "FullConnected": "Reflection", "Debate": "Reflection"}
        )
        replacement = mapping.get(collaboration)
        if replacement is not None:
            plan["collaboration"] = replacement
            self._compiler_repair(
                f"Mapped {collaboration} to {replacement} for {count} existing Agents; all Agents retained"
            )
            collaboration = replacement
        value = plan.get("rounds")
        integer = None
        if not isinstance(value, bool) and isinstance(value, (int, float, str)):
            try:
                numeric = float(value)
                if math.isfinite(numeric) and numeric.is_integer():
                    integer = int(numeric)
            except (ValueError, TypeError, OverflowError):
                pass
        if integer is not None and collaboration in self.collaboration_names:
            minimum = (
                2 if collaboration in {"FullConnected", "Debate", "Reflection"} else 1
            )
            normalized = (
                1
                if collaboration == "IO"
                else min(MAX_MAS_ROUNDS, max(minimum, integer))
            )
            if normalized != value:
                self._compiler_repair(
                    f"Normalized {collaboration} rounds={value!r} to {normalized}"
                )
            plan["rounds"] = normalized
        return plan

    def _validate_raw_plan(self, plan: Dict[str, Any]) -> None:
        if not isinstance(plan, dict):
            self._quality_issue("invalid_plan", "plan is not a JSON object")
            return
        required_plan_fields = {"collaboration", "rounds", "agents"}
        missing_plan_fields = sorted(required_plan_fields - set(plan))
        extra_plan_fields = sorted(set(plan) - required_plan_fields)
        if missing_plan_fields:
            self._quality_issue(
                "invalid_plan", f"plan missing fields: {missing_plan_fields}"
            )
        if extra_plan_fields:
            self._quality_issue(
                "invalid_plan",
                f"plan contains unsupported fields: {extra_plan_fields}",
            )
        collaboration = plan.get("collaboration")
        if collaboration not in self.collaboration_names:
            self._quality_issue(
                "invalid_plan", f"invalid collaboration: {collaboration}"
            )

        if "rounds" in plan and collaboration != "IO":
            rounds = self._parse_rounds_value(plan.get("rounds"))
            if rounds is not None:
                if (
                    collaboration in {"FullConnected", "Debate", "Reflection"}
                    and rounds < 2
                ):
                    self._quality_issue(
                        "invalid_plan",
                        f"{collaboration} requires rounds >= 2",
                    )
            else:
                self._quality_issue(
                    "invalid_plan",
                    f"{collaboration} rounds must be a positive integer",
                )

        agents = plan.get("agents")
        if not isinstance(agents, list) or not agents:
            self._quality_issue("invalid_plan", "plan agents must be a non-empty list")
            return
        if collaboration in {"IO", "CoT", "Reflection"} and len(agents) != 1:
            self._quality_issue(
                "invalid_plan", f"{collaboration} requires exactly one agent"
            )
        if collaboration in {"Chain", "FullConnected", "Debate"} and len(agents) < 2:
            self._quality_issue(
                "invalid_plan", f"{collaboration} requires at least two agents"
            )
        seen_ids = set()
        required = {"id", "model", "skill", "prompt"}
        for index, agent in enumerate(agents, start=1):
            if not isinstance(agent, dict):
                self._quality_issue("problem_nodes", f"agent_{index} is not an object")
                continue
            missing = sorted(key for key in required if key not in agent)
            extra = sorted(key for key in agent if key not in required)
            if missing:
                self._quality_issue(
                    "problem_nodes",
                    f"{agent.get('id', f'agent_{index}')} missing fields: {missing}",
                )
            if extra:
                self._quality_issue(
                    "problem_nodes",
                    f"{agent.get('id', f'agent_{index}')} contains unsupported fields: {extra}",
                )
            agent_id = str(agent.get("id", "")).strip()
            if not agent_id:
                self._quality_issue("problem_nodes", f"agent_{index} has an empty id")
            if agent_id in seen_ids:
                self._quality_issue("problem_nodes", f"duplicate agent id: {agent_id}")
            seen_ids.add(agent_id)
            if agent.get("model") not in self.model_names:
                self._quality_issue(
                    "problem_nodes",
                    f"{agent_id or index} invalid model: {agent.get('model')}",
                )
            if agent.get("skill") not in self.skill_names:
                self._quality_issue(
                    "problem_nodes",
                    f"{agent_id or index} invalid skill: {agent.get('skill')}",
                )
            if (
                not isinstance(agent.get("prompt"), str)
                or not agent.get("prompt", "").strip()
            ):
                self._quality_issue(
                    "problem_nodes", f"{agent_id or index} has an empty prompt"
                )

    def _validate_raw_workflow(self, plan: Dict[str, Any]) -> None:
        steps = plan.get("steps")
        if steps is None and plan.get("protocol") == "workflow_v2":
            steps = plan.get("agents")
        if not isinstance(steps, list) or not steps:
            self._quality_issue(
                "invalid_plan", "workflow steps must be a non-empty list"
            )
            return

        seen_ids = set()
        required = {"model_id", "prompt"}
        skill_default = (
            "GeneralReasoning"
            if "GeneralReasoning" in self.skill_names
            else self.skill_names[0]
        )
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                self._quality_issue("problem_nodes", f"step_{index} is not an object")
                continue
            step_id = str(step.get("id") or "")
            missing = sorted(key for key in required if key not in step)
            if missing:
                self._quality_issue(
                    "problem_nodes",
                    f"{step_id or f'step_{index}'} missing fields: {missing}",
                )
            if not step_id:
                step_id = f"step_{index}"
                step["id"] = step_id
                self._compiler_repair(f"assigned missing workflow id: {step_id}")
            if step_id in seen_ids:
                self._quality_issue(
                    "problem_nodes", f"duplicate workflow step id: {step_id}"
                )

            model_id = step.get("model_id")
            try:
                parsed_model_id = int(model_id)
                if isinstance(model_id, bool) or not 0 <= parsed_model_id < len(
                    self.model_names
                ):
                    raise ValueError
            except Exception:
                self._quality_issue(
                    "problem_nodes", f"{step_id or index} invalid model_id: {model_id}"
                )

            if not step.get("skill"):
                step["skill"] = skill_default
            elif step.get("skill") not in self.skill_names:
                old_skill = step.get("skill")
                step["skill"] = skill_default
                self._compiler_repair(
                    f"normalized {step_id} skill from {old_skill!r} to {skill_default}"
                )
            mode = str(step.get("mode", step.get("reasoning_mode", "auto"))).lower()
            if mode not in {"auto", "thinking", "nothinking"}:
                self._compiler_repair(
                    f"normalized {step_id} reasoning mode from {mode!r} to 'auto'"
                )
                mode = "auto"
            step["mode"] = mode
            if (
                not isinstance(step.get("prompt"), str)
                or not step.get("prompt", "").strip()
            ):
                self._quality_issue(
                    "problem_nodes", f"{step_id or index} has an empty prompt"
                )

            dependencies = step.get("depends_on")
            if dependencies is None:
                dependencies = []
                step["depends_on"] = dependencies
                self._compiler_repair(f"assigned missing depends_on for {step_id}")
            elif (
                isinstance(dependencies, str) and dependencies.strip().lower() == "all"
            ):
                dependencies = list(seen_ids)
                step["depends_on"] = dependencies
                self._compiler_repair(f"expanded depends_on='all' for {step_id}")
            if not isinstance(dependencies, list):
                self._quality_issue(
                    "problem_nodes",
                    f"{step_id or index} depends_on must be a list",
                )
            else:
                dependency_ids = {str(dependency) for dependency in dependencies}
                for dependency in dependencies:
                    dependency_id = str(dependency)
                    if dependency_id not in seen_ids:
                        self._quality_issue(
                            "problem_nodes",
                            f"{step_id or index} dependency is not an earlier step: {dependency_id}",
                        )
                prompt = str(step.get("prompt") or "")
                referenced_ids = {
                    f"step_{int(reference)}"
                    for reference in re.findall(r"(?<!\w)#(\d+)\b", prompt)
                }
                referenced_ids.update(re.findall(r"\bstep_\d+\b", prompt))
                for referenced_id in sorted(referenced_ids - dependency_ids):
                    if referenced_id in seen_ids:
                        dependencies.append(referenced_id)
                        dependency_ids.add(referenced_id)
                        self._compiler_repair(
                            f"added omitted dependency {referenced_id} to {step_id}"
                        )
                    else:
                        self._quality_issue(
                            "problem_nodes",
                            f"{step_id or index} references unavailable step: {referenced_id}",
                        )
            seen_ids.add(step_id)

    def _validate_raw_update(self, update: Dict[str, Any]) -> None:
        if not isinstance(update, dict):
            self._quality_issue("invalid_update", "update is not a JSON object")
            return
        if "type" not in update:
            self._quality_issue("invalid_update", "update missing required type field")
        action = update.get("type", update.get("action"))
        allowed = {
            "no_change",
            "update_model",
            "update_temperature",
            "update_skill",
            "update_prompt",
            "set_num_rounds",
            "remove_node",
            "add_node",
        }
        if action not in allowed:
            self._quality_issue("invalid_update", f"invalid update type: {action}")
            return
        if action in {
            "update_model",
            "update_temperature",
            "update_skill",
            "update_prompt",
            "remove_node",
        } and not update.get("target_agent_id"):
            self._quality_issue("invalid_update", f"{action} missing target_agent_id")
        if action == "update_model" and update.get("model") not in self.model_names:
            self._quality_issue(
                "invalid_update", f"invalid update model: {update.get('model')}"
            )
        if action == "update_skill" and update.get("skill") not in self.skill_names:
            self._quality_issue(
                "invalid_update", f"invalid update skill: {update.get('skill')}"
            )
        if action == "update_prompt" and not isinstance(update.get("prompt"), str):
            self._quality_issue(
                "invalid_update", f"invalid update prompt: {update.get('prompt')}"
            )
        if action == "set_num_rounds":
            try:
                rounds = int(update.get("num_rounds"))
                if rounds < 1:
                    self._quality_issue(
                        "invalid_update", f"set_num_rounds out of range: {rounds}"
                    )
            except Exception:
                self._quality_issue(
                    "invalid_update",
                    f"invalid set_num_rounds value: {update.get('num_rounds')}",
                )
        if action == "update_temperature":
            try:
                temperature = float(update.get("temperature"))
                if temperature < 0 or temperature > 2:
                    self._quality_issue(
                        "invalid_update",
                        f"update temperature out of range: {temperature}",
                    )
            except Exception:
                self._quality_issue(
                    "invalid_update",
                    f"invalid update temperature: {update.get('temperature')}",
                )
        if action == "add_node":
            new_agent = update.get("new_agent")
            if not isinstance(new_agent, dict):
                self._quality_issue(
                    "invalid_update", "add_node new_agent is not an object"
                )
                return
            required = {"id", "name", "model", "skill", "temperature", "prompt"}
            missing = sorted(key for key in required if key not in new_agent)
            if missing:
                self._quality_issue(
                    "problem_nodes", f"add_node new_agent missing fields: {missing}"
                )
            if new_agent.get("model") not in self.model_names:
                self._quality_issue(
                    "problem_nodes", f"add_node invalid model: {new_agent.get('model')}"
                )
            if new_agent.get("skill") not in self.skill_names:
                self._quality_issue(
                    "problem_nodes", f"add_node invalid skill: {new_agent.get('skill')}"
                )
            try:
                temperature = float(new_agent.get("temperature"))
                if temperature < 0 or temperature > 2:
                    self._quality_issue(
                        "problem_nodes",
                        f"add_node temperature out of range: {temperature}",
                    )
            except Exception:
                self._quality_issue(
                    "problem_nodes",
                    f"add_node invalid temperature: {new_agent.get('temperature')}",
                )

    def _validate_raw_workflow_update(
        self,
        update: Dict[str, Any],
        plan: Dict[str, Any],
        completed_ids: Optional[Sequence[str]] = None,
    ) -> None:
        if not isinstance(update, dict):
            self._quality_issue("invalid_update", "workflow update is not an object")
            return
        action = update.get("type")
        allowed = {
            "no_change",
            "update_model",
            "update_skill",
            "update_prompt",
            "remove_node",
            "add_node",
        }
        if action not in allowed:
            self._quality_issue(
                "invalid_update", f"invalid workflow update type: {action}"
            )
            return

        agents = plan.get("agents", [])
        agent_ids = [str(agent.get("id")) for agent in agents]
        completed_id_set = {str(step_id) for step_id in (completed_ids or [])}
        if action in {"update_model", "update_skill", "update_prompt", "remove_node"}:
            target_id = str(update.get("target_agent_id") or "")
            if target_id not in agent_ids:
                self._quality_issue(
                    "invalid_update", f"workflow update target not found: {target_id}"
                )
            elif target_id in completed_id_set:
                self._quality_issue(
                    "invalid_update",
                    f"workflow update target already completed: {target_id}",
                )
        if action == "update_model":
            try:
                model_id = int(update.get("model_id"))
                if isinstance(update.get("model_id"), bool) or not 0 <= model_id < len(
                    self.model_names
                ):
                    raise ValueError
            except Exception:
                self._quality_issue(
                    "invalid_update",
                    f"invalid workflow model_id: {update.get('model_id')}",
                )
        elif action == "update_skill" and update.get("skill") not in self.skill_names:
            self._quality_issue(
                "invalid_update", f"invalid workflow skill: {update.get('skill')}"
            )
        elif action == "update_prompt":
            if (
                not isinstance(update.get("prompt"), str)
                or not update.get("prompt", "").strip()
            ):
                self._quality_issue("invalid_update", "workflow update_prompt is empty")
            target_id = str(update.get("target_agent_id") or "")
            target = next(
                (agent for agent in agents if str(agent.get("id")) == target_id),
                None,
            )
            if target is not None:
                dependency_ids = {str(value) for value in target.get("depends_on", [])}
                prompt = str(update.get("prompt") or "")
                referenced_ids = {
                    f"step_{int(reference)}"
                    for reference in re.findall(r"(?<!\w)#(\d+)\b", prompt)
                }
                referenced_ids.update(re.findall(r"\bstep_\d+\b", prompt))
                for referenced_id in sorted(referenced_ids - dependency_ids):
                    self._quality_issue(
                        "invalid_update",
                        f"workflow update_prompt references {referenced_id} but target dependencies omit it",
                    )
        elif action == "add_node":
            step = update.get("new_step")
            if not isinstance(step, dict):
                self._quality_issue(
                    "invalid_update", "add_node new_step is not an object"
                )
                return
            required = {"id", "model_id", "skill", "depends_on", "prompt"}
            missing = sorted(key for key in required if key not in step)
            if missing:
                self._quality_issue(
                    "problem_nodes", f"add_node new_step missing fields: {missing}"
                )
            step_id = str(step.get("id") or "")
            if not step_id or step_id in agent_ids:
                self._quality_issue(
                    "problem_nodes", f"add_node invalid or duplicate id: {step_id}"
                )
            try:
                model_id = int(step.get("model_id"))
                if isinstance(step.get("model_id"), bool) or not 0 <= model_id < len(
                    self.model_names
                ):
                    raise ValueError
            except Exception:
                self._quality_issue(
                    "problem_nodes",
                    f"add_node invalid model_id: {step.get('model_id')}",
                )
            if step.get("skill") not in self.skill_names:
                self._quality_issue(
                    "problem_nodes", f"add_node invalid skill: {step.get('skill')}"
                )
            dependencies = step.get("depends_on")
            if not isinstance(dependencies, list):
                self._quality_issue(
                    "problem_nodes", "add_node depends_on must be a list"
                )
            else:
                dependency_ids = {str(dependency) for dependency in dependencies}
                for dependency in dependencies:
                    if str(dependency) not in agent_ids:
                        self._quality_issue(
                            "problem_nodes",
                            f"add_node dependency not found: {dependency}",
                        )
                prompt = str(step.get("prompt") or "")
                referenced_ids = {
                    f"step_{int(reference)}"
                    for reference in re.findall(r"(?<!\w)#(\d+)\b", prompt)
                }
                referenced_ids.update(re.findall(r"\bstep_\d+\b", prompt))
                for referenced_id in sorted(referenced_ids - dependency_ids):
                    self._quality_issue(
                        "problem_nodes",
                        f"add_node prompt references {referenced_id} but depends_on omits it",
                    )
            if (
                not isinstance(step.get("prompt"), str)
                or not step.get("prompt", "").strip()
            ):
                self._quality_issue("problem_nodes", "add_node prompt is empty")

    def _first_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                char = text[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            start = text.find("{", start + 1)
        return None

    def _valid_collaboration(self, name: Any) -> str:
        if isinstance(name, str) and name in self.collaboration_names:
            return name
        return (
            "Chain"
            if "Chain" in self.collaboration_names
            else self.collaboration_names[0]
        )

    def _valid_model(self, name: Any, default: str) -> str:
        if isinstance(name, str) and name in self.model_names:
            return name
        return default if default in self.model_names else self.model_names[0]

    def _valid_model_id(self, value: Any) -> int:
        if isinstance(value, str) and value in self.model_names:
            return self.model_names.index(value)
        try:
            model_id = int(value)
        except Exception:
            model_id = 0
        if 0 <= model_id < len(self.model_names):
            return model_id
        return 0

    def _preferred_model_id(self, name: str, default: int = 0) -> int:
        if name in self.model_names:
            return self.model_names.index(name)
        if not self.model_names:
            return 0
        return max(0, min(int(default), len(self.model_names) - 1))

    def _valid_skill(self, name: Any, default: str) -> str:
        if isinstance(name, str) and name in self.skill_names:
            return name
        return default if default in self.skill_names else self.skill_names[0]

    def _preferred_skill(self, name: str) -> str:
        if name in self.skill_names:
            return name
        return (
            "GeneralReasoning"
            if "GeneralReasoning" in self.skill_names
            else self.skill_names[0]
        )

    def _temperature(self, value: Any) -> float:
        del value
        return FIXED_WORKER_TEMPERATURE

    def _skill_profile(self, name: str) -> Dict[str, Any]:
        for skill in self.skill_pool:
            if skill["Name"] == name:
                return skill
        return self.skill_pool[0]

    def _model_profile(self, name: str) -> Dict[str, Any]:
        for model in self.model_pool:
            if model["Name"] == name:
                return model
        return self.model_pool[0]

    def _next_agent_id(self, agents: List[Dict[str, Any]]) -> str:
        existing = {agent["id"] for agent in agents}
        i = len(agents) + 1
        while f"agent_{i}" in existing:
            i += 1
        return f"agent_{i}"

    def _format_pool(
        self,
        pool: List[Dict[str, Any]],
        category: str = "",
    ) -> str:
        lines = []
        for item in pool:
            if category == "models":
                coefficient = float(item.get("CostCoefficient", 1.0))
                coefficient_text = f"{coefficient:g}"
                lines.append(
                    f"- {item['Name']}, {coefficient_text}: "
                    f"{str(item.get('Description', '')).strip()}"
                )
                continue
            if category in {"collaborations", "skills"}:
                lines.append(
                    f"- {item['Name']}: {str(item.get('Description', '')).strip()}"
                )
                continue
            description = self.static_descriptions.get(category, {}).get(
                item["Name"],
                str(item.get("Description", "")),
            )
            extra = item.get("Instruction") or item.get("Cost") or ""
            lines.append(f"- {item['Name']}: {description} {extra}".strip())
        return "\n".join(lines)

    def _format_parameter_definitions(self) -> str:
        lines = []
        current = self.static_descriptions.get("parameters", {})
        for name, preset in STATIC_PARAMETER_DESCRIPTIONS.items():
            description = current.get(name, preset)
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    def _format_indexed_model_pool(self) -> str:
        lines = []
        for model_id, item in enumerate(self.model_pool):
            lines.append(f"- model_id {model_id}: {item['Name']}")
        return "\n".join(lines)

    def _format_skill_catalog(self) -> str:
        return "\n".join(f"- {item['Name']}" for item in self.skill_pool)

    def _workflow_few_shot_examples(self) -> str:
        if not self.model_names:
            return ""
        model_a = 0
        model_b = 1 if len(self.model_names) > 1 else model_a
        model_c = 2 if len(self.model_names) > 2 else model_a
        model_d = 3 if len(self.model_names) > 3 else model_a
        countdown_question_1 = (
            "Using the numbers [76, 58, 53], create an equation that equals 81. Use each number once "
            "with basic arithmetic and return only the equation inside <answer> tags."
        )
        countdown_example_1 = {
            "steps": [
                {
                    "id": "step_1",
                    "model_id": model_a,
                    "skill": self._preferred_skill("MathReasoning"),
                    "depends_on": [],
                    "prompt": "Attempt the original equation puzzle from scratch and end with <result>a valid candidate equation</result>.",
                },
                {
                    "id": "step_2",
                    "model_id": model_c,
                    "skill": self._preferred_skill("ConsistencyVerification"),
                    "depends_on": ["step_1"],
                    "prompt": "Check #1 against every number-use and arithmetic constraint, and end with <result>a corrected equation if needed</result>.",
                },
                {
                    "id": "step_3",
                    "model_id": model_a,
                    "skill": self._preferred_skill("AnswerSynthesis"),
                    "depends_on": ["step_1", "step_2"],
                    "prompt": "Use #1 and #2 to return the most accurate response in the original format: <answer>equation only</answer>.",
                },
            ],
        }
        medical_question = (
            "Does brain-derived neurotrophic factor enhance contraction of intestinal muscle strips induced by "
            "SP and CGRP in mice? Answer choices: A. Yes B. No"
        )
        medical_example = {
            "steps": [
                {
                    "id": "step_1",
                    "model_id": model_b,
                    "skill": self._preferred_skill("DomainKnowledgeQA"),
                    "depends_on": [],
                    "prompt": "Answer the original biomedical question independently with A or B; end with <result>the candidate letter</result>.",
                },
                {
                    "id": "step_2",
                    "model_id": model_a,
                    "skill": self._preferred_skill("DomainKnowledgeQA"),
                    "depends_on": [],
                    "prompt": "Independently answer the original biomedical question with A or B; end with <result>the candidate letter</result>.",
                },
                {
                    "id": "step_3",
                    "model_id": model_c,
                    "skill": self._preferred_skill("ConsistencyVerification"),
                    "depends_on": ["step_1", "step_2"],
                    "prompt": "Check #1 and #2 and return the supported answer according to the original format inside <answer>...</answer>.",
                },
            ],
        }
        limit_question = (
            "Evaluate lim[t->0] (1/ln(1+t) + 1/ln(1-t)). Return the final answer inside <answer> tags "
            "using LaTeX notation."
        )
        limit_example = {
            "steps": [
                {
                    "id": "step_1",
                    "model_id": model_d,
                    "skill": self._preferred_skill("MathReasoning"),
                    "depends_on": [],
                    "prompt": "Solve the original limit independently, for example with a series expansion; end with <result>the derivation and candidate</result>.",
                },
                {
                    "id": "step_2",
                    "model_id": model_c,
                    "skill": self._preferred_skill("MathReasoning"),
                    "depends_on": [],
                    "prompt": "Independently solve the original limit with a sound alternative derivation; end with <result>the derivation and candidate</result>.",
                },
                {
                    "id": "step_3",
                    "model_id": model_b,
                    "skill": self._preferred_skill("ConsistencyVerification"),
                    "depends_on": ["step_1", "step_2"],
                    "prompt": "Verify #1 and #2, using another method if useful, and end with <result>the checked result</result>.",
                },
                {
                    "id": "step_4",
                    "model_id": model_a,
                    "skill": self._preferred_skill("AnswerSynthesis"),
                    "depends_on": ["step_1", "step_2", "step_3"],
                    "prompt": "Reconcile #1, #2, and #3 and return the correct final response according to the original formatting instructions inside <answer>...</answer>.",
                },
            ],
        }
        countdown_question_2 = (
            "Using the numbers [19, 43, 34], create an equation that equals 10. Use each number once with basic "
            "arithmetic and return the equation inside <answer> tags."
        )
        countdown_example_2 = {
            "steps": [
                {
                    "id": "step_1",
                    "model_id": model_d,
                    "skill": self._preferred_skill("MathReasoning"),
                    "depends_on": [],
                    "prompt": "Attempt the original equation puzzle and end with <result>a candidate equation</result>.",
                },
                {
                    "id": "step_2",
                    "model_id": model_a,
                    "skill": self._preferred_skill("ConsistencyVerification"),
                    "depends_on": ["step_1"],
                    "prompt": "Verify #1 against all constraints and end with <result>a corrected equation if necessary</result>.",
                },
                {
                    "id": "step_3",
                    "model_id": model_b,
                    "skill": self._preferred_skill("AnswerSynthesis"),
                    "depends_on": ["step_1", "step_2"],
                    "prompt": "Use #1 and #2 to return the equation according to the original instructions inside <answer>...</answer>.",
                },
            ],
        }
        for example in (
            countdown_example_1,
            medical_example,
            limit_example,
            countdown_example_2,
        ):
            for step in example["steps"]:
                step.pop("skill", None)
                step["mode"] = "auto"
        for step in limit_example["steps"][:3]:
            step["mode"] = "thinking"
        limit_example["steps"][-1]["mode"] = "nothinking"
        return (
            "WORKFLOW EXAMPLES (OOD examples adapted from the reference Conductor implementation; learn the output "
            "format and coordination space, not a dataset-to-workflow rule). Model ids are deliberately rotated across "
            "roles and are not a permanent role-to-model mapping. Choose from the current worker metadata for the user task:\n"
            f"Example 1 question: {countdown_question_1}\n"
            "Example 1 response:\n<analysis>Use an initial attempt, constraint check, and final refinement.</analysis>\n"
            f"<workflow>{json.dumps(countdown_example_1, ensure_ascii=False)}</workflow>\n"
            f"Example 2 question: {medical_question}\n"
            "Example 2 response:\n<analysis>Use independent recall attempts and a final check.</analysis>\n"
            f"<workflow>{json.dumps(medical_example, ensure_ascii=False)}</workflow>\n"
            f"Example 3 question: {limit_question}\n"
            "Example 3 response:\n<analysis>Use independent derivations, verification, and synthesis.</analysis>\n"
            f"<workflow>{json.dumps(limit_example, ensure_ascii=False)}</workflow>\n"
            f"Example 4 question: {countdown_question_2}\n"
            "Example 4 response:\n<analysis>Use an attempt, verification, and final formatting pass.</analysis>\n"
            f"<workflow>{json.dumps(countdown_example_2, ensure_ascii=False)}</workflow>\n\n"
        )

    def _compact_history_items(
        self, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        compact = []
        for item in history:
            compact.append(
                {
                    "round": item.get("round"),
                    "agent_id": item.get("agent_id"),
                    "agent_name": item.get("agent_name"),
                    "model": item.get("model"),
                    "mode": item.get("mode", "auto"),
                    "skill": item.get("skill"),
                    "output": str(item.get("output", ""))[:2000],
                }
            )
        return compact

    def _compact_history(self, history: List[Dict[str, Any]]) -> str:
        return json.dumps(
            self._compact_history_items(history), ensure_ascii=False, indent=2
        )

    def _record_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.capture_events:
            return
        self._events.append(
            {
                "index": len(self._events) + 1,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "type": event_type,
                "payload": payload,
            }
        )
