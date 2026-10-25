import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import threading


DOMAINS = ("qa", "knowledge", "commonsense", "math", "multimodal")
MODES = ("IO", "CoT", "Chain", "FullConnected", "Debate", "Reflection")
READY_CONTRACT = (
    "Return exactly one nonempty <analysis>...</analysis> block followed by one <plan>...</plan> block, "
    "with nothing outside them. Keep analysis to a routing rationale of at most 120 words: identify task "
    "requirements, justify the collaboration by its exact name, and explain roles, models, skills and rounds. "
    "Do not solve the question, calculate or guess its answer, or put an answer in agent instructions. "
    "The plan must be strict JSON with exactly collaboration, rounds and agents. Each agent must have exactly "
    "id, model, skill and prompt. Use one listed skill per agent and task-specific instructions. Escape "
    "backslashes, quotation marks and newlines correctly within JSON strings. No task budget is imposed. "
    "Choose 1–3 agents and 1–4 rounds: IO uses one agent and one round; CoT uses one agent; Reflection uses "
    "one agent and at least two rounds; Chain uses two or three agents; FullConnected and Debate use two "
    "or three agents and at least two rounds. Every Worker automatically receives the question and supplied "
    "passages; image-capable Workers also receive supplied images. If VisualEvidence is available, assign "
    "it to at least one image-capable Worker. The Coordinator cannot see the image itself. "
)
JUDGING_CONTRACT = (
    "Return exactly one nonempty <analysis>...</analysis> block followed by one <answer>...</answer> block, "
    "with nothing outside them. Keep analysis under 60 words and use it to check completeness and extract "
    "the actual Workers' conclusion. Follow the question's answer contract inside <answer>, such as a "
    "single choice label or a numeric result. Do not solve the task afresh, introduce an unsupported "
    "answer, or replace a failed Worker result with your own solution. If the Workers did not establish "
    "an answer, state that explicitly."
)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def quota_slots(target):
    remainder = target - target // 2
    return {"autonomous": target // 2, **{
        mode: remainder // len(MODES) + (index < remainder % len(MODES))
        for index, mode in enumerate(MODES)
    }}


def canonical_target(router, response, stage, constraint):
    terminal = "plan" if stage == "Ready" else "answer"
    match = re.fullmatch(
        rf"\s*<analysis>\s*(.+?)\s*</analysis>\s*<{terminal}>\s*(.+?)\s*</{terminal}>\s*",
        str(response), flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Expected complete analysis and {terminal} blocks")
    analysis, payload = (value.strip() for value in match.groups())
    if re.search(r"</?(analysis|plan|answer|think|mas)\b", analysis, re.IGNORECASE):
        raise ValueError("Nested Coordinator tags in analysis")
    if stage == "Judging":
        if re.search(r"</?(analysis|plan|answer|think|mas)\b", payload, re.IGNORECASE):
            raise ValueError("Nested Coordinator tags in final answer")
        return f"<analysis>{analysis}</analysis><answer>{payload}</answer>"
    raw = router.parse_plan_json(payload)
    plan = router.validate_and_normalize_external_plan(raw)
    if raw.get("collaboration") != plan["collaboration"] or raw.get("rounds") != plan["rounds"]:
        raise ValueError("Collaboration, agent count and rounds must agree without normalization")
    if plan["collaboration"].casefold() not in analysis.casefold():
        raise ValueError("Analysis must name the selected collaboration")
    if not 1 <= len(plan["agents"]) <= 3 or not 1 <= plan["rounds"] <= 4:
        raise ValueError("Collection requires 1–3 agents and 1–4 rounds")
    if constraint and (
        plan["collaboration"] != constraint["collaboration"]
        or plan["agents"][0]["model"] != constraint["model"]
    ):
        raise ValueError("The requested configuration-coverage choice was not followed")
    vision_models = {model["Name"] for model in router.model_pool if model.get("SupportsVision")}
    if "VisualEvidence" in router.skill_names and not any(
        agent["skill"] == "VisualEvidence" and agent["model"] in vision_models for agent in plan["agents"]
    ):
        raise ValueError("Image evidence requires a VisualEvidence Worker with image input support")
    return f"<analysis>{analysis}</analysis><plan>{json.dumps(plan, ensure_ascii=False, separators=(',', ':'))}</plan>"


class APIPolicy:
    def __init__(self, model, constraint, format_retries, timeout):
        self.model_path = model
        self.constraint = constraint
        self.format_retries = format_retries
        self.timeout = timeout
        self._generation_lock = threading.Lock()
        self.batch_all_coordinator_requests = False
        self.router = None
        self.completed_stages = 0

    def format_messages(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def generate(self, messages, max_new_tokens, temperature, **kwargs):
        from MAR.LLM import LLMRegistry
        from MAR.LLM.gpt_chat import request_timeout

        stage = "Ready" if self.completed_stages == 0 else "Judging"
        generation = deepcopy(messages)
        contract = READY_CONTRACT if stage == "Ready" else JUDGING_CONTRACT
        if stage == "Ready":
            if self.constraint:
                contract += (
                    f"For this configuration-coverage sample, use collaboration {self.constraint['collaboration']} "
                    f"and assign {self.constraint['model']} to your first agent. Design task-appropriate roles "
                    "and skills within this configuration; choose remaining models and rounds yourself. Explain "
                    "why this concrete configuration addresses the task. Do not mention this sampling instruction "
                    "or the collection process in your response."
                )
            else:
                contract += "Choose the collaboration, models, roles and rounds yourself based on this task."
        generation[-1]["content"] += "\n\n" + contract
        response, error = "", ""
        for attempt in range(self.format_retries + 1):
            current = deepcopy(generation)
            if error:
                current.extend([
                    {"role": "assistant", "content": response},
                    {"role": "user", "content":
                     f"Your previous response was not usable: {error}\n"
                     "Return a corrected complete response following the same contract. Align analysis with the "
                     "actual plan. Do not add commentary outside the required tags."},
                ])
            with request_timeout(self.timeout):
                response = LLMRegistry.get(self.model_path).gen(
                    current, max_tokens=max_new_tokens, temperature=temperature,
                )
            try:
                target = canonical_target(self.router, response, stage, self.constraint)
            except ValueError as exc:
                error = str(exc)
                continue
            self.completed_stages += 1
            return self.format_messages(messages), target
        raise ValueError(f"Teacher format retries exhausted: {error}")


def build_sft_records(item, outcome, split, slot, teacher, episode_id):
    from .data import question_key

    records = []
    for ordinal, step in enumerate(outcome["steps"]):
        stage = "Ready" if step.request_type == "plan" else "Judging"
        identity = hashlib.sha256(f"{episode_id}\0coordinator\0{stage}\0{ordinal}".encode()).hexdigest()[:24]
        records.append({
            "schema_version": "evorouter.sft.v1", "id": identity,
            "episode_id": episode_id, "role": "coordinator", "stage": stage,
            "messages": deepcopy(step.messages), "response": step.response_text,
            "metadata": {
                "task_id": str(item.get("task_id") or item["budgetbench_task"]["task_id"]),
                "item_id": str(item["id"]), "task_type": item["task_type"],
                "split": split, "teacher_model": teacher, "slot": slot,
                "question_key": question_key(item),
            },
        })
    return records


def collect_task(item, split, slot, constraint, args):
    from .runtime import RolloutSettings, build_training_router, run_episode

    settings = RolloutSettings(
        temperature=0.6, plan_max_tokens=3072, final_max_tokens=1024,
        workers=1, generation_batch=1, format_retries=0, provider_retries=0,
    )
    attempts = []
    for attempt in range(args.attempts_per_task):
        policy = APIPolicy(args.teacher_model, constraint, args.format_retries, args.teacher_timeout)
        policy.router = build_training_router(policy, item, settings)
        try:
            outcome = run_episode(policy, item, settings)
            steps = outcome.get("steps") or []
            result = outcome.get("result") or {}
            plan = outcome.get("final_mas") or result.get("final_plan") or {}
            history = result.get("history") or []
            planned = {str(agent["id"]) for agent in plan.get("agents", [])}
            executed = {str(entry.get("agent_id")) for entry in history}
            reasons = []
            if outcome.get("invalid_structure"):
                reasons.append("invalid_structure")
            if outcome.get("provider_failure"):
                reasons.append("provider_failure")
            if not outcome.get("answer_correct") or not outcome.get("terminated"):
                reasons.append("incorrect_or_incomplete")
            if [step.request_type for step in steps] != ["plan", "final"] or not all(step.accepted for step in steps):
                reasons.append("not_two_accepted_stages")
            if not history or planned != executed:
                reasons.append("worker_execution_incomplete")
            attempts.append({"attempt": attempt, "accepted": not reasons, "reasons": reasons})
            if reasons:
                continue
            task_id = str(item.get("task_id") or item["budgetbench_task"]["task_id"])
            episode_id = f"sft::{split}::{task_id}::{attempt}"
            return {
                "episode_id": episode_id, "task_id": task_id, "split": split,
                "task_type": item["task_type"], "slot": slot, "final_plan": plan,
                "records": build_sft_records(item, outcome, split, slot, args.teacher_model, episode_id),
            }, attempts
        except Exception as exc:
            attempts.append({"attempt": attempt, "accepted": False, "error_type": type(exc).__name__})
    return None, attempts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect outcome-verified two-stage Coordinator SFT trajectories.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--teacher-model", default="deepseek-v4-pro-nothinking")
    parser.add_argument("--train-per-domain", type=int, default=400)
    parser.add_argument("--dev-per-domain", type=int, default=40)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--attempts-per-task", type=int, default=2)
    parser.add_argument("--format-retries", type=int, default=3)
    parser.add_argument("--teacher-timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.train_per_domain, args.workers, args.attempts_per_task) < 1:
        raise ValueError("Training quota, workers, and attempts per task must be positive")
    if args.dev_per_domain < 0 or args.format_retries < 0 or args.teacher_timeout <= 0:
        raise ValueError("Invalid validation quota, retries, or timeout")
    from .data import load_split, question_key, read_jsonl
    from .runtime import configure_environment
    from evorouter.coordinator import load_catalogs
    import fcntl

    configure_environment()
    if "deepseek" in args.teacher_model.lower() and not args.teacher_model.endswith(("-thinking", "-nothinking")):
        args.teacher_model += "-nothinking"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "collection.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pool = next(iter(load_catalogs().values()))["model_pool"]
    model_names = [model["Name"] for model in pool]
    vision_names = [model["Name"] for model in pool if model.get("SupportsVision")]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_pool=model_names, ready_contract=READY_CONTRACT, judging_contract=JUDGING_CONTRACT)
    config_path = output / "collection_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ("data_dir", "teacher_model", "train_per_domain", "dev_per_domain", "seed",
                    "model_pool", "ready_contract", "judging_contract"):
            if previous.get(key) != config[key]:
                raise ValueError(f"Collection resume configuration differs: {key}")
    else:
        write_json(config_path, config)
    accepted_path, completed_path = output / "accepted_trajectories.jsonl", output / "completed_tasks.jsonl"
    accepted = read_jsonl(accepted_path) if accepted_path.exists() else []
    completed = read_jsonl(completed_path) if completed_path.exists() else []
    seen_ids = {record["task_id"] for record in accepted + completed}
    counts = Counter((record["split"], record["task_type"], record["slot"]) for record in accepted)
    model_counts = Counter((record["task_type"], agent["model"])
                           for record in accepted for agent in record["final_plan"]["agents"])
    targets, buckets, all_ids, all_questions = {}, {}, set(), set()
    for split, target in (("dev", args.dev_per_domain), ("train", args.train_per_domain)):
        rows = load_split(args.data_dir, split)
        random.Random(args.seed + (split == "dev")).shuffle(rows)
        for domain in DOMAINS:
            buckets[(split, domain)] = []
            for mode, number in quota_slots(target).items():
                targets[(split, domain, mode)] = number
        for item in rows:
            task_id = str(item.get("task_id") or item["budgetbench_task"]["task_id"])
            question = question_key(item)
            if task_id in all_ids or question in all_questions:
                continue
            all_ids.add(task_id)
            all_questions.add(question)
            if item["task_type"] in DOMAINS and task_id not in seen_ids:
                buckets[(split, item["task_type"])].append(item)
    pending, pending_counts, pending_models = {}, Counter(), Counter()

    def export_records():
        for split in ("train", "dev"):
            path = output / f"{split}.sft.jsonl"
            temporary = path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for episode in accepted:
                    if episode["split"] == split:
                        for record in episode["records"]:
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary.replace(path)

    def persist(status):
        remaining = {"/".join(key): max(0, target - counts[key])
                     for key, target in targets.items() if target > counts[key]}
        write_json(output / "collection_summary.json", {
            "status": status, "accepted_trajectories": len(accepted),
            "target_trajectories": sum(targets.values()), "sft_records": 2 * len(accepted),
            "counts": {"/".join(key): value for key, value in counts.items()}, "remaining": remaining,
        })
        return remaining

    def next_job():
        available = [key for key, target in targets.items()
                     if counts[key] + pending_counts[key] < target and buckets[key[:2]]]
        if not available:
            return None
        key = min(available, key=lambda value: ((counts[value] + pending_counts[value]) / targets[value], value))
        item = buckets[key[:2]].pop()
        constraint = None
        if key[2] != "autonomous":
            candidates = list(vision_names if key[1] == "multimodal" and key[2] in ("IO", "CoT", "Reflection") else model_names)
            random.Random(f"{args.seed}:{item['id']}").shuffle(candidates)
            model = min(candidates, key=lambda name: model_counts[(key[1], name)] + pending_models[(key[1], name)])
            constraint = {"collaboration": key[2], "model": model}
        return key, item, constraint

    persist("running")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            while len(pending) < args.workers:
                job = next_job()
                if job is None:
                    break
                key, item, constraint = job
                future = executor.submit(collect_task, item, key[0], key[2], constraint, args)
                pending[future] = (key, item, constraint)
                pending_counts[key] += 1
                if constraint:
                    pending_models[(key[1], constraint["model"])] += 1
            if not pending:
                break
            finished, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                key, item, constraint = pending.pop(future)
                pending_counts[key] -= 1
                if constraint:
                    pending_models[(key[1], constraint["model"])] -= 1
                episode, attempts = future.result()
                if episode:
                    append_jsonl(accepted_path, episode)
                    accepted.append(episode)
                    counts[key] += 1
                    for agent in episode["final_plan"]["agents"]:
                        model_counts[(key[1], agent["model"])] += 1
                completed_record = {
                    "task_id": str(item.get("task_id") or item["budgetbench_task"]["task_id"]),
                    "attempts": attempts,
                }
                append_jsonl(completed_path, completed_record)
            persist("running")
            if finished:
                export_records()
                print(f"accepted={len(accepted)}/{sum(targets.values())} in_flight={len(pending)}", flush=True)
    export_records()
    complete = all(counts[key] >= target for key, target in targets.items())
    persist("complete" if complete else "incomplete_quotas")
    if not complete:
        raise SystemExit("Candidate questions exhausted before collection quotas were met")


if __name__ == "__main__":
    main()
