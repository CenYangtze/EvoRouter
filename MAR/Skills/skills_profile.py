from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

skills_profile = [
    {
        "Name": "GeneralReasoning",
        "Description": "Interprets instructions and solves bounded tasks with a concise, logically sound chain.",
        "Instruction": (
            "Frame the user's core objective in one sentence. Identify known facts, working assumptions, and constraints. "
            "Solve using a minimal, logically sound reasoning chain. If ambiguity exists, explicitly state your working assumption. "
            "Strictly follow requested formatting guidelines, avoid speculative details, and pass unresolved uncertainties "
            "to downstream specialized agents when needed."
        ),
    },
    {
        "Name": "InstructionCompliance",
        "Description": "Tracks every explicit constraint and produces an answer that satisfies each one exactly.",
        "Instruction": (
            "Extract a checklist of every explicit requirement, including content, scope, count, tone, length, and output format. "
            "Resolve the task while preserving all constraints, then audit the candidate response against the checklist. Correct any "
            "omission or conflict before returning the requested result, without adding unrequested framing or filler."
        ),
    },
    {
        "Name": "ConversationContinuity",
        "Description": "Uses dialogue history to resolve references and answer the latest turn consistently.",
        "Instruction": (
            "Identify the latest user request and the earlier turns it depends on. Carry forward established facts, definitions, style, "
            "and prior outputs; resolve pronouns and revision requests precisely; and answer only the newest request. Do not repeat the "
            "whole conversation or contradict commitments established in prior turns."
        ),
    },
    {
        "Name": "CreativeComposition",
        "Description": "Produces original, coherent writing with the requested genre, voice, structure, and detail.",
        "Instruction": (
            "Translate the requested genre, audience, tone, viewpoint, length, and structural constraints into a concise writing plan. "
            "Create specific and internally consistent content with concrete details rather than generic filler. Preserve continuity, "
            "avoid imitation of living authors, and revise the draft until every requested stylistic and factual constraint is met."
        ),
    },
    {
        "Name": "ProblemDecomposition",
        "Description": "Splits multi-step tasks into ordered subgoals, dependencies, and precise handoffs.",
        "Instruction": (
            "Produce an actionable execution blueprint rather than solving the task directly. Restate the core goal, "
            "split the work into ordered or parallel subtasks, highlight potential risks or missing context, and explicitly "
            "define the evidence, constraints, and intermediate artifacts that downstream agents must produce."
        ),
    },
    {
        "Name": "FactChecking",
        "Description": "Checks factual claims, entities, dates, and relations against explicit evidence.",
        "Instruction": (
            "Decompose candidate statements into checkable atomic claims. For each claim, classify it as supported, "
            "contradicted, or uncertain, linking it directly to exact source snippets or verified facts. Separate proven evidence "
            "from unsupported inference, resolve conflicting statements by evaluating source credibility, and report "
            "the precise correction needed for absolute factual accuracy."
        ),
    },
    {
        "Name": "StructuredExtraction",
        "Description": "Extracts exact entities, relations, quantities, and attributes from supplied text.",
        "Instruction": (
            "Strictly conform to the target schema or output structure. Normalize names, dates, and units into standard formats. "
            "If a field is missing or uncertain, output null or an explicit uncertainty note rather than guessing. Deduplicate entries, "
            "preserve exact values where precision matters, and ensure the final output is valid and machine-readable."
        ),
    },
    {
        "Name": "DomainKnowledgeQA",
        "Description": "Answers precise factual and conceptual questions using relevant domain knowledge.",
        "Instruction": (
            "Identify the relevant domain and required level of precision. Retrieve canonical concepts, frameworks, and facts. "
            "Distinguish established facts from debated interpretations. Explicitly mention domain boundary conditions "
            "(e.g., jurisdiction, time period, version), and keep the response tightly focused on the query without conversational filler."
        ),
    },
    {
        "Name": "MathReasoning",
        "Description": "Solves arithmetic, algebra, geometry, and probability with checked derivations.",
        "Instruction": (
            "Define variables, state governing equations or logical constraints, and derive the solution step-by-step. "
            "Keep all intermediate values and arithmetic visible. Track units, signs, domains, and rounding conventions carefully. "
            "Perform a sanity/boundary check against the original problem statement to eliminate calculation errors."
        ),
    },
    {
        "Name": "MathematicalModeling",
        "Description": "Converts verbal problems into variables, equations, constraints, and solvable models.",
        "Instruction": (
            "Explicitly state all simplifying assumptions before formulating the model. Map real-world quantities to mathematical "
            "variables, define constraints and objective functions, choose an appropriate model class, and explain how variations "
            "in key parameters affect the outcome. Clearly distinguish exact mathematical conclusions from qualitative approximations."
        ),
    },
    {
        "Name": "AlgorithmDesign",
        "Description": "Designs correct algorithms, invariants, data structures, and complexity bounds.",
        "Instruction": (
            "Clarify inputs, outputs, and performance constraints. Describe the core algorithmic strategy, justify correctness "
            "using invariants or logical arguments, and analyze both time and space complexity. Explicitly call out boundary conditions, "
            "corner cases, and trade-offs compared to simpler baseline approaches."
        ),
    },
    {
        "Name": "CodeSynthesis",
        "Description": "Writes complete executable code that follows the required interface and constraints.",
        "Instruction": (
            "Adhere strictly to specified function signatures, interfaces, and language conventions. Implement the simplest correct "
            "solution, handling edge cases, boundary conditions, and invalid inputs gracefully. Mentally trace provided examples against "
            "the code. Return only the requested executable code or clean code blocks without unnecessary commentary."
        ),
    },
    {
        "Name": "CodeDebugging",
        "Description": "Finds root causes of code failures and produces minimal verified fixes.",
        "Instruction": (
            "Follow an OBSERVE -> HYPOTHESIZE -> TEST -> FIX workflow. Identify the exact root cause of the failure, explain why "
            "the bug occurred under specific conditions, and provide the minimal correct code fix. Include a verification step or test case "
            "proving the bug is resolved without introducing regressions."
        ),
    },
    {
        "Name": "CodeReview",
        "Description": "Audits code for correctness, security, performance, and concrete edge cases.",
        "Instruction": (
            "Summarize the intended software behavior first. Prioritize findings by severity: security flaws and correctness bugs first, "
            "followed by performance bottlenecks and maintainability concerns. Provide concrete code counterexamples, edge-case triggers, "
            "and actionable refactoring recommendations for each issue."
        ),
    },
    {
        "Name": "ScientificAnalysis",
        "Description": "Applies scientific laws and mechanisms to physical, chemical, and biological questions.",
        "Instruction": (
            "Identify fundamental governing laws, variables, and system boundary conditions. Reason logically from mechanisms to "
            "consequences, distinguish correlation from causality, and quantify measurement uncertainties or assumptions. "
            "Avoid overstating conclusions beyond what empirical evidence or domain theory strictly supports."
        ),
    },
    {
        "Name": "CriticalReflection",
        "Description": "Finds logical gaps, contradictions, calculation errors, and unmet constraints.",
        "Instruction": (
            "Systematically audit candidate answers against the original user prompt and constraints. Search for logical gaps, "
            "numerical inconsistencies, unstated assumptions, or formatting errors. Provide a prioritized list of material flaws "
            "and concrete fixes, or issue a clean verification signal if no issues exist."
        ),
    },
    {
        "Name": "AnswerSynthesis",
        "Description": "Reconciles Worker results into one supported, concise, format-compliant answer.",
        "Instruction": (
            "Synthesize all verified worker contributions into a cohesive response. Reconcile minor conflicts by prioritizing "
            "higher-confidence evidence, strip out redundant internal communication artifacts, and enforce the final output style "
            "and format strictly as requested by the user."
        ),
    },
    {
        "Name": "InformationSearch",
        "Description": "Finds answer-relevant facts and multi-hop links in the supplied passages.",
        "Instruction": (
            "Read the supplied passages for the assigned question. Locate exact entities, dates, quantities, and relations; "
            "follow necessary links across passages; distinguish explicit evidence from inference; and return only the "
            "answer-relevant evidence and conclusion. Treat the supplied passages as authoritative when they determine the answer."
        ),
    },
    {
        "Name": "VisualEvidence",
        "Description": "Inspects supplied images and extracts visual evidence required for the assigned question.",
        "Instruction": (
            "Inspect every image attached to this Worker. Read charts, diagrams, tables, labels, spatial relations, "
            "and other visual details needed by the assigned question. Distinguish directly observed evidence from "
            "inference, cross-check quantities and option labels, and return the answer-relevant visual evidence and conclusion."
        ),
    },
    {
        "Name": "EntityResolution",
        "Description": "Resolves aliases and identity links so every entity in a reasoning chain is exact.",
        "Instruction": (
            "List each entity required by the assigned relation chain, resolve aliases or ambiguous references, and verify that "
            "attributes belong to the same entity. Preserve exact names, dates, locations, and relation direction. Explicitly flag "
            "any unresolved identity instead of silently substituting a similar entity."
        ),
    },
    {
        "Name": "MultiHopReasoning",
        "Description": "Connects several evidence-backed relations without skipping or reversing a hop.",
        "Instruction": (
            "Express the required chain as ordered hops. Resolve each hop from available evidence, carry only verified entities "
            "into the next hop, and check that relation direction and intermediate identities remain consistent. Return the compact "
            "chain and its supported endpoint."
        ),
    },
    {
        "Name": "LogicalDeduction",
        "Description": "Derives conclusions from explicit rules, constraints, cases, and necessary implications.",
        "Instruction": (
            "Translate the assigned problem into explicit propositions or constraints. Derive only valid implications, test boundary "
            "and counterexample cases, eliminate contradictions, and verify that the conclusion is necessary rather than merely plausible."
        ),
    },
    {
        "Name": "ChoiceEvaluation",
        "Description": "Tests every answer option against the question and returns the uniquely supported choice.",
        "Instruction": (
            "Determine exactly what the question asks, evaluate every option under the same criteria, and record the decisive support "
            "or contradiction for each. Check negations and exception wording carefully, then return the uniquely supported option label."
        ),
    },
]


_SKILL_BY_NAME = {item["Name"]: item for item in skills_profile}


def _skills(names: Iterable[str]) -> list[dict[str, Any]]:
    selected = [
        deepcopy(_SKILL_BY_NAME[name]) for name in names if name in _SKILL_BY_NAME
    ]
    if len(selected) > 7:
        raise ValueError("A task may expose at most seven Skills to the Coordinator")
    return selected


def select_skills_for_task(
    task: Mapping[str, Any] | None = None,
    *,
    task_type: str = "",
    capability_tags: Iterable[str] = (),
    output_format: str = "",
    instruction: str = "",
) -> list[dict[str, Any]]:
    """Return a compact, task-relevant Coordinator Skill catalog.

    BudgetBench metadata is used only by the trusted runner to choose the
    visible catalog. It is never serialized into the Coordinator question.
    """

    task = task or {}
    family = str(task.get("family") or task_type or "").strip().lower()
    tags = {
        str(value).strip().lower()
        for value in (task.get("capability_tags") or capability_tags or ())
        if str(value).strip()
    }
    contract = task.get("output_contract") if isinstance(task, Mapping) else None
    fmt = str((contract or {}).get("format") or output_format or "").strip().lower()
    text = str(task.get("instruction") or instruction or "").lower()
    assets = task.get("assets") if isinstance(task, Mapping) else None
    has_images = any(
        isinstance(asset, Mapping)
        and (
            str(asset.get("kind") or "").casefold() == "image"
            or str(asset.get("media_type") or "").casefold().startswith("image/")
        )
        for asset in (assets or ())
    )

    if (
        family == "multimodal"
        or has_images
        or "visual_question_answering" in tags
        or "image_understanding" in tags
    ):
        return _skills(
            (
                "VisualEvidence",
                "ChoiceEvaluation",
                "StructuredExtraction",
                "MathReasoning",
                "ScientificAnalysis",
                "AnswerSynthesis",
            )
        )

    if family in {"code", "programming"} or "code_generation" in tags:
        return _skills(
            (
                "AlgorithmDesign",
                "CodeSynthesis",
                "CodeDebugging",
                "CodeReview",
                "ProblemDecomposition",
                "AnswerSynthesis",
            )
        )

    if (
        family in {"instruction_following", "conversation"}
        or "instruction_following" in tags
    ):
        return _skills(
            (
                "InstructionCompliance",
                "ConversationContinuity",
                "CreativeComposition",
                "GeneralReasoning",
                "DomainKnowledgeQA",
                "AnswerSynthesis",
            )
        )

    if family == "qa" or "question_answering" in tags:
        if "evidence_retrieval" in tags or bool(task.get("assets")):
            return _skills(
                (
                    "InformationSearch",
                    "EntityResolution",
                    "MultiHopReasoning",
                    "StructuredExtraction",
                    "FactChecking",
                    "AnswerSynthesis",
                )
            )
        return _skills(
            (
                "DomainKnowledgeQA",
                "EntityResolution",
                "FactChecking",
                "MultiHopReasoning",
                "GeneralReasoning",
                "AnswerSynthesis",
            )
        )

    is_choice = "multiple_choice_reasoning" in tags or "choices:" in text
    is_math = (
        "mathematics" in tags
        or fmt == "number"
        or any(
            tag in tags
            for tag in (
                "arithmetic",
                "algebra",
                "prealgebra",
                "intermediate_algebra",
                "geometry",
                "number_theory",
                "precalculus",
                "counting_&_probability",
            )
        )
    )
    if is_choice:
        return _skills(
            (
                "ChoiceEvaluation",
                "LogicalDeduction",
                (
                    "ScientificAnalysis"
                    if "science_reasoning" in tags
                    else "DomainKnowledgeQA"
                ),
                "FactChecking",
                "ProblemDecomposition",
                "AnswerSynthesis",
            )
        )
    if is_math:
        return _skills(
            (
                "MathReasoning",
                "MathematicalModeling",
                "LogicalDeduction",
                "ProblemDecomposition",
                "ScientificAnalysis" if "science_reasoning" in tags else "FactChecking",
                "AnswerSynthesis",
            )
        )
    return _skills(
        (
            "LogicalDeduction",
            "ProblemDecomposition",
            "DomainKnowledgeQA",
            "FactChecking",
            "ScientificAnalysis",
            "AnswerSynthesis",
        )
    )
