import json
import re
from collections import Counter
from copy import deepcopy


INITIAL_INSIGHTS = [
    {"id": "I1", "text": "Different task bottlenecks call for different dimensions of expansion."},
    {"id": "I2", "text": "Every expansion dimension has a knee point; blindly enlarging the configuration often produces disappointing results."},
    {"id": "I3", "text": (
        "The best downstream Worker configuration depends on the quality of upstream organization. "
        "Accurate subtask decomposition, context routing, and skill assignment can reduce downstream "
        "computation. Poor organization can cause repeated propagation and rapidly amplify errors, "
        "including with large models and tightly coupled multi-agent collaboration."
    )},
]

SYSTEM_PROMPT = """You maintain general, practical routing Insights for future multi-agent Coordinators using completed routing trajectories.

You receive previous Insights and recent Queries with MAS configurations, Worker results, correctness outcomes, and FLOPs costs. Revise wording, add conditions and practical details, or introduce Insights while preserving these directions:
I1. Choose expansion dimensions from observable task bottlenecks.
I2. Identify the knee point of each expansion dimension and avoid blindly enlarging configurations.
I3. Account for how upstream task decomposition, context routing, and skill assignment affect downstream computation and error propagation.

Every Query is retained. Configurations and results may be sampled, compressed, and deduplicated by configuration/outcome. original_candidate_count, outcome_counts, and collaboration_counts describe the original candidates. Displayed representative counts differ from actual frequencies; omitted text may still exist in the trajectory.

Requirements:
- Keep each Insight within 100 words. Merge similar content while preserving I1, I2, and I3.
- Use task features observable at deployment and give conditional, actionable routing advice.
- Generalize across correctness, computation cost, and error propagation.
- Describe conditions for increasing or reducing model capability, Agent count, rounds, Skills, and collaboration complexity.
- Exclude task answers, task-specific proper names, sample IDs, dataset statistics, accuracy percentages, Skill usage rates, and model rankings.
- Ground rules in repeatable task characteristics and compare different configurations on the same question.
- Distinguish configuration format failures from incorrect answers produced by valid configurations. Attribute format failures to configuration generation; exclude fallback configurations from configuration-performance evidence.
- Retain observable applicability conditions, knee points, or evidence requirements for IO, CoT, Chain, FullConnected, Debate, and Reflection, including infrequent modes.
- Discuss conditional choices across model capability levels and multimodal support. Model choices must depend on observable requirements.
- Write Insights and change_summary in English.

Preserve IDs I1, I2, and I3. Number additions from I4 onward and keep 3 to 8 Insights. Output exactly one JSON object with keys insights and change_summary. Each insights entry has exactly id and text. Return the JSON object directly.
{"insights":[{"id":"I1","text":"..."},{"id":"I2","text":"..."},{"id":"I3","text":"..."}],"change_summary":"..."}"""


def compact_text(value, limit):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    if limit <= 24:
        return text[:max(0, limit - 1)] + "…"
    half = max(1, (limit - 24) // 2)
    return text[:half] + "\n...[COMPACTED]...\n" + text[-half:]


def compact_evidence(evidence, max_chars):
    text_fields = {"query", "prompt", "worker_results", "final_result", "plan_rejection_reason"}

    def trim(value, limit, field=None):
        if isinstance(value, str):
            return compact_text(value, limit) if field in text_fields else value
        if isinstance(value, list):
            return [trim(item, limit, field) for item in value]
        if isinstance(value, dict):
            return {key: trim(item, limit, key) for key, item in value.items()}
        return value

    def identity(row):
        plan = row.get("configuration") or {}
        agents = [(agent.get("model"), agent.get("skill"), agent.get("prompt")) for agent in plan.get("agents", [])]
        return json.dumps([plan.get("collaboration"), plan.get("rounds"), agents, row["correct"], row["format_valid"]], sort_keys=True, ensure_ascii=False)

    def features(row):
        plan = row.get("configuration") or {}
        return {("outcome", row["correct"]), ("mode", plan.get("collaboration")), *(("model", agent.get("model")) for agent in plan.get("agents", []))}

    groups = []
    for query in evidence:
        distinct = {}
        for row in query["rollouts"]:
            key = identity(row)
            if key not in distinct:
                distinct[key] = dict(deepcopy(row), similar_configuration_count=0)
            distinct[key]["similar_configuration_count"] += 1
        ordered, seen = [], set()
        remaining = list(distinct.values())
        while remaining:
            best = max(remaining, key=lambda row: len(features(row) - seen))
            ordered.append(best)
            seen.update(features(best))
            remaining.remove(best)
        groups.append((query, ordered))
    for cap, chars in [(6, 600), (4, 450), (3, 350), (2, 300), (1, 240), (1, 120), (1, 80), (1, 48), (1, 32), (1, 16)]:
        result = []
        for query, ordered in groups:
            originals = query["rollouts"]
            row = dict(
                query=compact_text(query["query"], chars * 2),
                rollouts=trim(ordered[:cap], chars),
                original_candidate_count=len(originals),
                distinct_configuration_outcomes=len(ordered),
                outcome_counts=dict(Counter("correct" if item["correct"] else "incorrect" for item in originals)),
                collaboration_counts=dict(Counter(str((item.get("configuration") or {}).get("collaboration", "unavailable")) for item in originals)),
            )
            result.append(row)
        if len(json.dumps(result, ensure_ascii=False)) <= max_chars:
            return result
    raise ValueError("Insight evidence exceeds the configured input limit")


def parse_insights(response):
    source = re.sub(r"<think>.*?</think>", "", str(response), flags=re.IGNORECASE | re.DOTALL)
    decoder = json.JSONDecoder()
    payload = None
    for match in re.finditer(r"\{", source):
        try:
            candidate, _ = decoder.raw_decode(source[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "insights" in candidate:
            payload = candidate
            break
    if payload is None or set(payload) != {"insights", "change_summary"}:
        raise ValueError("Insight output requires exactly insights and change_summary")
    rows = payload["insights"]
    if not isinstance(rows, list) or not 3 <= len(rows) <= 8:
        raise ValueError("Expected 3 to 8 Insights")
    insights, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "text"}:
            raise ValueError("Insight entries require exactly id and text")
        identity = str(row["id"]).strip()
        text = " ".join(str(row["text"]).split())
        if not re.fullmatch(r"I[1-9]\d*", identity) or identity in seen:
            raise ValueError("Invalid or duplicate Insight ID")
        if not text or len(text.split()) > 100:
            raise ValueError("Each Insight must contain 1 to 100 words")
        if re.search(r"\d+(?:\.\d+)?\s*%", text):
            raise ValueError("Insight percentages are excluded")
        insights.append(dict(id=identity, text=text))
        seen.add(identity)
    if not {"I1", "I2", "I3"}.issubset(seen):
        raise ValueError("Updated Insights must preserve I1, I2, and I3")
    summary = " ".join(str(payload["change_summary"]).split())
    if not summary:
        raise ValueError("Insight change_summary must be nonempty")
    return dict(insights=sorted(insights, key=lambda row: int(row["id"][1:])), change_summary=summary)


class InsightManager:
    def __init__(self, model="deepseek-v4-pro-thinking", refresh_queries=80, max_input_chars=96000, timeout=1800, retries=5, temperature=0.2):
        if refresh_queries <= 0 or max_input_chars <= 0 or timeout <= 0 or retries < 0:
            raise ValueError("Invalid Insight maintenance settings")
        self.model = str(model)
        self.refresh_queries = int(refresh_queries)
        self.max_input_chars = int(max_input_chars)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.temperature = float(temperature)
        self.insights = deepcopy(INITIAL_INSIGHTS)
        self.version = 0
        self.completed_queries = 0
        self.pending = []

    def state_dict(self):
        return deepcopy(dict(
            insights=self.insights,
            version=self.version,
            completed_queries=self.completed_queries,
            pending=self.pending,
            configuration={key: getattr(self, key) for key in ("model", "refresh_queries", "max_input_chars", "timeout", "retries", "temperature")},
        ))

    def load_state_dict(self, state):
        if state.get("configuration") != self.state_dict()["configuration"]:
            raise ValueError("Insight maintenance settings differ from the checkpoint")
        parsed = parse_insights(json.dumps(dict(insights=state["insights"], change_summary="Restored training state")))
        self.insights = parsed["insights"]
        self.version = int(state["version"])
        self.completed_queries = int(state["completed_queries"])
        self.pending = deepcopy(state["pending"])

    def add_query(self, item, outcomes):
        task = item.get("budgetbench_task") or item.get("task") or {}
        query = item.get("query") or task.get("instruction") or item.get("instruction") or item.get("question") or ""
        rows = []
        for index, outcome in enumerate(outcomes):
            if outcome.get("provider_failure"):
                continue
            result = outcome.get("result") or {}
            history = list(result.get("history") or [])
            valid = not bool(outcome.get("invalid_structure"))
            rows.append(dict(
                rollout_index=index,
                format_valid=valid,
                correct=bool(outcome.get("answer_correct")),
                flops_cost=float(outcome.get("worker_flops", 0.0)),
                configuration=deepcopy(outcome.get("final_mas") or {}) if valid else None,
                evidence_status="executed_valid_configuration" if valid else "plan_generation_failure_excluded_from_configuration_statistics",
                worker_results=[compact_text(row.get("output", ""), 400) for row in history[-4:] if isinstance(row, dict)],
                final_result=compact_text(result.get("final_answer", ""), 500),
            ))
        self.pending.append(dict(query=str(query), rollouts=rows))
        self.completed_queries += 1

    def _request(self, messages):
        from MAR.LLM import LLMRegistry
        from MAR.LLM.gpt_chat import omit_output_token_limit, request_timeout

        with request_timeout(self.timeout), omit_output_token_limit():
            return LLMRegistry.get(self.model).gen(messages, max_tokens=None, temperature=self.temperature)

    def _refresh(self, evidence):
        errors, responses = [], []
        try:
            compact = compact_evidence(evidence, self.max_input_chars)
            base = [dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=(
                "Previous Insights:\n" + json.dumps(self.insights, ensure_ascii=False)
                + "\n\nRecent routing evidence:\n" + json.dumps(compact, ensure_ascii=False)
                + "\n\nUpdate the Insights with conditional resource adjustments and knee-point criteria. Preserve I1, I2, and I3. Return the required English JSON object."
            ))]
        except (TypeError, ValueError) as error:
            return dict(updated=False, attempts=0, errors=[str(error)], version=self.version)
        for attempt in range(self.retries + 1):
            messages = deepcopy(base)
            if responses:
                messages.extend([
                    dict(role="assistant", content=responses[-1]),
                    dict(role="user", content=f"Correct the previous response: {errors[-1]}. Return 3 to 8 English Insights, each within 100 words. Merge similar content and preserve I1, I2, I3. Exclude answers, sample IDs, percentages, and model rankings. Return exactly insights and change_summary in JSON."),
                ])
            try:
                response = str(self._request(messages))
                responses.append(response)
                parsed = parse_insights(response)
                self.insights = parsed["insights"]
                self.version += 1
                return dict(updated=True, attempts=attempt + 1, errors=errors, version=self.version, change_summary=parsed["change_summary"])
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        return dict(updated=False, attempts=self.retries + 1, errors=errors, version=self.version)

    def refresh_if_due(self):
        reports = []
        while len(self.pending) >= self.refresh_queries:
            evidence = self.pending[:self.refresh_queries]
            del self.pending[:self.refresh_queries]
            report = self._refresh(evidence)
            report["completed_queries"] = self.completed_queries
            reports.append(report)
        return reports
