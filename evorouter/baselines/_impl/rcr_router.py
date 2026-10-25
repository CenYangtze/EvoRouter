from __future__ import annotations
import math
import re
from dataclasses import dataclass
from .core import CALLS, FINAL
from .scoring import VERSION, failure_score, score

PAPER = "arXiv:2508.04903v3"
SOURCE = "https://arxiv.org/abs/2508.04903"
MODEL = "glm-5"
ROUNDS = 3
BUDGET = 2048
ROLE_KEYWORDS = {
    "planner": {
        "question",
        "goal",
        "plan",
        "decompose",
        "constraint",
        "relation",
        "hop",
        "option",
    },
    "searcher": {
        "passage",
        "evidence",
        "fact",
        "entity",
        "date",
        "number",
        "calculate",
        "verify",
    },
    "recommender": {
        "answer",
        "candidate",
        "evidence",
        "result",
        "conclusion",
        "choice",
    },
}
STAGE_TYPES = {
    "planner": {"query", "feedback", "candidate"},
    "searcher": {"context", "plan", "evidence"},
    "recommender": {"context", "plan", "evidence", "candidate"},
}


@dataclass
class MemoryItem:
    text: str
    role: str
    kind: str
    timestamp: int


def token_count(text):
    return max(1, math.ceil(len(str(text)) / 4))


def terms(text):
    return set(re.findall("[a-z0-9]+", str(text).casefold()))


def route_memory(memory, role, stage, query, now, budget=BUDGET):
    query_terms = terms(query)
    keywords = ROLE_KEYWORDS[role]
    scored = []
    for index, item in enumerate(memory):
        item_terms = terms(item.text)
        role_score = float(bool(item_terms & keywords) or item.role == role)
        stage_score = float(item.kind in STAGE_TYPES[stage])
        recency = math.exp(-0.5 * max(0, now - item.timestamp))
        semantic = len(item_terms & query_terms) / max(
            1, math.sqrt(len(item_terms) * len(query_terms))
        )
        scored.append((role_score + stage_score + recency + semantic, -index, item))
    selected = []
    used = 0
    for value, _, item in sorted(
        scored, reverse=True, key=lambda row: (row[0], row[1])
    ):
        cost = token_count(item.text)
        if used + cost <= budget:
            selected.append((value, item))
            used += cost
        elif not selected:
            chars = max(32, (budget - used) * 4)
            selected.append(
                (
                    value,
                    MemoryItem(item.text[:chars], item.role, item.kind, item.timestamp),
                )
            )
            used = budget
        if used >= budget:
            break
    return (selected, used)


def memory_text(selected):
    return "\n\n".join(
        (
            f"[{item.kind}; source={item.role}; importance={value:.3f}]\n{item.text}"
            for value, item in selected
        )
    )


def initial_memory(item):
    choices = "\n".join(
        (f"{key}. {value}" for key, value in (item.get("choices") or {}).items())
    )
    memory = [
        MemoryItem(
            item["question"] + ("\nChoices:\n" + choices if choices else ""),
            "user",
            "query",
            0,
        )
    ]
    memory.extend(
        (
            MemoryItem(passage, "source", "context", 0)
            for passage in item.get("passages") or []
        )
    )
    if item.get("images"):
        memory.append(
            MemoryItem(
                "Attached images are unavailable to the text-only GLM-5 agents.",
                "source",
                "context",
                0,
            )
        )
    return memory


def evaluate(item, api, rounds=ROUNDS):
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    calls = []
    token = CALLS.set(calls)
    error = ""
    prediction = ""
    routes = []
    memory = initial_memory(item)
    worker_item = dict(item, passages=[], images=[])
    try:
        for round_index in range(1, rounds + 1):
            selected, used = route_memory(
                memory, "planner", "planner", item["question"], round_index, BUDGET
            )
            context = memory_text(selected)
            routes.append(
                {
                    "round": round_index,
                    "role": "planner",
                    "memory_tokens": used,
                    "items": len(selected),
                }
            )
            plan = api.solve(
                worker_item,
                MODEL,
                f"You are the Planner in a role-aware multi-agent QA system. Round {round_index}: decompose the task into concise subgoals and identify what evidence or computation is needed. Do not give a final answer.",
                context=context,
                system="You are a planning agent. Return a concise actionable plan.",
            )
            memory.append(MemoryItem(plan, "planner", "plan", round_index))
            selected, used = route_memory(
                memory,
                "searcher",
                "searcher",
                item["question"] + " " + plan,
                round_index,
                BUDGET,
            )
            context = memory_text(selected)
            routes.append(
                {
                    "round": round_index,
                    "role": "searcher",
                    "memory_tokens": used,
                    "items": len(selected),
                }
            )
            evidence = api.solve(
                worker_item,
                MODEL,
                f"You are the Searcher in a role-aware multi-agent QA system. Round {round_index}: follow the plan, inspect only the routed memory, extract decisive evidence, calculations, and a candidate answer. Be concise.",
                context=context,
                system="You are an evidence extraction and verification agent.",
            )
            memory.append(MemoryItem(evidence, "searcher", "evidence", round_index))
            selected, used = route_memory(
                memory,
                "recommender",
                "recommender",
                item["question"] + " " + evidence,
                round_index,
                BUDGET,
            )
            context = memory_text(selected)
            routes.append(
                {
                    "round": round_index,
                    "role": "recommender",
                    "memory_tokens": used,
                    "items": len(selected),
                }
            )
            prediction = api.solve(
                worker_item,
                MODEL,
                f"You are the Recommender and final decision agent. Round {round_index}: synthesize the routed evidence, resolve conflicts, and answer the original task. Answer directly and concisely; no explanation. {FINAL}",
                context=context,
            )
            memory.append(
                MemoryItem(prediction, "recommender", "candidate", round_index)
            )
        result = score(prediction, item)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        result = failure_score(item)
    finally:
        CALLS.reset(token)
    return {
        "scorer_version": VERSION,
        "id": item["id"],
        "dataset": item["dataset"],
        "method": "rcr_router",
        "model": MODEL,
        "prediction": prediction,
        **result,
        "error": error,
        "calls": calls,
        "memory_routes": routes,
        "final_memory_items": len(memory),
    }
