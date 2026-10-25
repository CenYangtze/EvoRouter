import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist

from .data import _sft_exclusions, load_rl_tasks, question_key, stratified_order
from .insights import InsightManager
from .objectives import compute_advantages, encode_response, teacher_messages
from .runtime import RolloutSettings, TrainingPolicy, configure_environment, run_episode


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default=str(ROOT / "configs/training/rl.json"))
    known, _ = preliminary.parse_known_args(argv)
    defaults = json.loads(Path(known.config).read_text())
    parser = argparse.ArgumentParser(description="Train EvoRouter")
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--teacher-device")
    parser.add_argument("--data-dir", default=str(Path(os.getenv("EVOROUTER_ASSET_ROOT", ROOT / "assets")) / "data/budgetbench"))
    parser.add_argument("--sft-source")
    parser.add_argument("--cost-budget-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deepspeed", default=str(ROOT / "configs/training/deepspeed.json"))
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    for key, value in defaults.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    if not args.sft_source:
        candidate = Path(args.model) / "sft_manifest.json"
        if not candidate.is_file():
            parser.error("Provide --sft-source with the SFT records or sft_manifest.json")
        args.sft_source = str(candidate)
    for key in ("max_steps", "questions_per_step", "rollouts_per_question", "micro_batch_size", "max_length", "opd_max_length", "logit_chunk_size", "rollout_workers", "generation_batch_size", "save_steps", "save_total_limit"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.rollouts_per_question < 2:
        parser.error("Group-relative training requires at least two rollouts per question")
    for key in ("kl_coefficient", "opd_coefficient", "analysis_weight", "cost_coefficient", "cost_cap", "weight_decay"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be nonnegative")
    if args.learning_rate <= 0 or args.temperature <= 0:
        parser.error("Learning rate and sampling temperature must be positive")
    return args


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def broadcast(value):
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def reduce_numbers(values, device, operation=dist.ReduceOp.SUM):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=operation)
    return tensor.cpu().tolist()


def format_messages(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def make_records(groups, tokenizer, teacher_tokenizer, insights, args):
    records = []
    skipped = 0
    for _, outcomes in groups:
        advantages = compute_advantages([row["reward"] for row in outcomes], std_floor=args.advantage_std_floor)
        for outcome, advantage in zip(outcomes, advantages):
            for step in outcome["steps"]:
                if step.request_type not in {"plan", "update", "final"}:
                    continue
                response = step.response_text
                encoded = encode_response(tokenizer, step.prompt_text, response, args.max_length)
                record = dict(rl=encoded, advantage=advantage, student_opd=None, teacher_opd=None)
                if args.opd_coefficient > 0 and not outcome.get("invalid_structure") and step.accepted and "<plan>" in response:
                    try:
                        student = encode_response(tokenizer, step.prompt_text, response, args.opd_max_length)
                        if not student["opd_valid"]:
                            raise ValueError("No valid OPD mask")
                        messages = teacher_messages(step.messages, insights, outcome["final_mas"], bool(outcome["answer_correct"]))
                        teacher = encode_response(teacher_tokenizer, format_messages(teacher_tokenizer, messages), response, args.opd_max_length, privileged=True)
                        student_active = [(label, kind) for label, kind in zip(student["labels"], student["kinds"]) if label != -100]
                        teacher_active = [(label, kind) for label, kind in zip(teacher["labels"], teacher["kinds"]) if label != -100]
                        if student_active != teacher_active:
                            raise RuntimeError("Teacher and Student response token alignment differs")
                        record.update(student_opd=student, teacher_opd=teacher)
                    except ValueError:
                        skipped += 1
                records.append(record)
    return records, skipped


def normalization(records, device):
    counts = [0, 0, 0, 0]
    for row in records:
        response_tokens = row["rl"]["counts"]["response_tokens"]
        counts[0] += response_tokens if row["advantage"] != 0 else 0
        counts[1] += response_tokens
        if row["student_opd"] is not None:
            counts[2] += row["student_opd"]["counts"]["json_value_tokens"]
            counts[3] += row["student_opd"]["counts"]["analysis_tokens"]
    values = reduce_numbers(counts, device)
    return dict(zip(("rl", "kl", "plan", "analysis"), values), world_size=dist.get_world_size())


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())


def scientific_config(args):
    keys = ("model", "teacher_model", "max_steps", "questions_per_step", "rollouts_per_question", "learning_rate", "weight_decay", "gradient_clipping", "kl_coefficient", "opd_coefficient", "analysis_weight", "cost_coefficient", "cost_cap", "cost_budget_file", "advantage_std_floor", "max_length", "opd_max_length", "temperature", "plan_max_tokens", "final_max_tokens", "insight_refresh_queries", "insight_model", "seed")
    values = {key: getattr(args, key) for key in keys}
    values["cost_budget_sha256"] = hashlib.sha256(Path(args.cost_budget_file).read_bytes()).hexdigest() if args.cost_budget_file else None
    return values


def save_checkpoint(engine, tokenizer, manager, args, step, position, fingerprint):
    destination = Path(args.output_dir)
    tag = f"step_{step:06d}"
    state = dict(step=step, position=position, data_fingerprint=fingerprint, configuration=scientific_config(args), world_size=dist.get_world_size(), rng=gather(rng_state()), insights=manager.state_dict())
    engine.save_checkpoint(str(destination / "checkpoints"), tag=tag, client_state=state)
    if dist.get_rank() == 0:
        export = destination / "checkpoints" / tag / "model"
        engine.module.student.save_pretrained(export, safe_serialization=True)
        tokenizer.save_pretrained(export)
        shutil.copy2(destination / "sft_manifest.json", export / "sft_manifest.json")
        (destination / "latest_model.txt").write_text(str(export.resolve()) + "\n")
        checkpoints = sorted((destination / "checkpoints").glob("step_*"))
        for old in checkpoints[:-args.save_total_limit]:
            if old.is_dir():
                shutil.rmtree(old)
    dist.barrier()


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args):
    configure_environment()
    items = load_rl_tasks(args.data_dir, args.sft_source)
    ordered = stratified_order(items, args.seed)
    fingerprint = hashlib.sha256(json.dumps([(str(item.get("task_id", item["id"])), question_key(item)) for item in ordered], ensure_ascii=False).encode()).hexdigest()
    if len(items) < args.questions_per_step * args.max_steps:
        raise ValueError("The training pool must contain enough distinct non-SFT questions for this run")
    if args.dry_run:
        print(json.dumps(dict(training_questions=len(items), sft_source=args.sft_source, configuration=scientific_config(args), data_fingerprint=fingerprint), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("RL training requires a CUDA GPU")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    import deepspeed
    from .backend import create_engine
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False, timeout=timedelta(hours=12))
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.questions_per_step % world:
        raise ValueError("questions_per_step must be divisible by the number of training processes")
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    if not args.resume and (output / "run_config.json").exists():
        raise ValueError("Use a new output directory or pass --resume for an existing run")
    engine, tokenizer, teacher_tokenizer = create_engine(args)
    manager = InsightManager(model=args.insight_model, refresh_queries=args.insight_refresh_queries)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    step, position = 0, 0
    if args.resume:
        checkpoint, state = engine.load_checkpoint(str(output / "checkpoints"))
        if not checkpoint or not state:
            raise ValueError("No complete DeepSpeed checkpoint is available")
        if state["data_fingerprint"] != fingerprint or state["configuration"] != scientific_config(args) or state["world_size"] != world:
            raise ValueError("Resume requires the same data, training configuration, and world size")
        step, position = state["step"], state["position"]
        manager.load_state_dict(state["insights"])
        restore_rng(state["rng"][rank])
        if rank == 0:
            metrics = output / "metrics.jsonl"
            if metrics.exists():
                rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
                metrics.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= step))
    elif rank == 0:
        (output / "run_config.json").write_text(json.dumps(dict(vars(args), training_questions=len(items), data_fingerprint=fingerprint, world_size=world), indent=2) + "\n")
    if rank == 0:
        task_ids, question_keys = _sft_exclusions(args.sft_source)
        (output / "sft_manifest.json").write_text(json.dumps(dict(task_ids=sorted(task_ids), question_keys=sorted(question_keys)), ensure_ascii=False, indent=2) + "\n")
    settings = RolloutSettings(temperature=args.temperature, plan_max_tokens=args.plan_max_tokens, final_max_tokens=args.final_max_tokens, workers=args.rollout_workers, generation_batch=args.generation_batch_size, cost_coefficient=args.cost_coefficient, cost_cap=args.cost_cap, cost_budget_file=args.cost_budget_file)
    policy = TrainingPolicy(engine.module.student, tokenizer, args.model, max_prompt_length=args.max_length)
    policy.configure_judging_batching(max_batch_size=args.generation_batch_size, wait_ms=100)
    per_rank = args.questions_per_step // world
    dummy_id = tokenizer.eos_token_id
    dummy = dict(rl=dict(input_ids=[dummy_id, dummy_id], labels=[-100, dummy_id], kinds=[0, 0], counts=dict(response_tokens=1)), advantage=0.0, student_opd=None, teacher_opd=None, dummy=True)
    try:
        with ThreadPoolExecutor(max_workers=args.rollout_workers) as executor:
            while step < args.max_steps:
                started = time.monotonic()
                selected = ordered[position:position + args.questions_per_step]
                position += args.questions_per_step
                local_items = selected[rank * per_rank:(rank + 1) * per_rank]
                engine.eval()
                submitted = [(item, [executor.submit(run_episode, policy, deepcopy(item), settings) for _ in range(args.rollouts_per_question)]) for item in local_items]
                groups, failure = [], None
                try:
                    for item, futures in submitted:
                        outcomes = [future.result() for future in futures]
                        if not any(row["provider_failure"] for row in outcomes):
                            groups.append((item, outcomes))
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                errors = [error for error in gather(failure) if error]
                if errors:
                    raise RuntimeError("Rollout collection failed: " + errors[0])
                policy.assert_idle()
                complete_groups = sum(len(value) for value in gather([item["id"] for item, _ in groups]))
                if complete_groups == 0:
                    raise RuntimeError("Every rollout group failed; resume after restoring the model API")
                rollout_seconds = time.monotonic() - started
                try:
                    records, skipped = make_records(groups, tokenizer, teacher_tokenizer, manager.insights, args)
                    failure = None
                except Exception as exc:
                    records, skipped = [], 0
                    failure = f"{type(exc).__name__}: {exc}"
                errors = [error for error in gather(failure) if error]
                if errors:
                    raise RuntimeError("Training record preparation failed: " + errors[0])
                denominators = normalization(records, engine.device)
                if denominators["kl"] == 0:
                    raise RuntimeError("No Coordinator response tokens were collected")
                microbatches = [records[index:index + args.micro_batch_size] for index in range(0, len(records), args.micro_batch_size)]
                count = int(reduce_numbers([len(microbatches)], engine.device, dist.ReduceOp.MAX)[0])
                components = {}
                engine.train()
                for index in range(count):
                    engine.set_gradient_accumulation_boundary(index == count - 1)
                    loss, metrics = engine(microbatches[index] if index < len(microbatches) else [dummy], denominators)
                    engine.backward(loss, scale_wrt_gas=False)
                    for key, value in metrics.items():
                        components[key] = components.get(key, 0.0) + float(value)
                learning_rate = engine.get_lr()[0]
                engine.step()
                step += 1
                evidence = [(item, [{key: value for key, value in outcome.items() if key != "steps"} for outcome in outcomes]) for item, outcomes in groups]
                all_evidence = gather(evidence) if args.opd_coefficient > 0 else []
                refresh_reports = []
                if rank == 0 and args.opd_coefficient > 0:
                    for rank_groups in all_evidence:
                        for item, outcomes in rank_groups:
                            manager.add_query(item, outcomes)
                    refresh_reports = manager.refresh_if_due()
                if args.opd_coefficient > 0:
                    manager.load_state_dict(broadcast(manager.state_dict() if rank == 0 else None))
                keys = sorted(components)
                totals = reduce_numbers([components[key] for key in keys], engine.device)
                episodes = [row for _, outcomes in groups for row in outcomes]
                aggregate = reduce_numbers([len(episodes), sum(row["answer_correct"] for row in episodes), sum(row["reward"] for row in episodes), skipped], engine.device)
                if rank == 0:
                    reduced = dict(zip(keys, totals))
                    row = dict(step=step, learning_rate=learning_rate, loss=reduced["total_loss"], **reduced, complete_queries=complete_groups, episodes=int(aggregate[0]), accuracy=aggregate[1] / max(1, aggregate[0]), reward=aggregate[2] / max(1, aggregate[0]), skipped_opd_examples=int(aggregate[3]), insight_version=manager.version, rollout_seconds=rollout_seconds, elapsed_seconds=time.monotonic() - started)
                    append_jsonl(output / "metrics.jsonl", row)
                    for report in refresh_reports:
                        append_jsonl(output / "insight_updates.jsonl", dict(step=step, **report))
                    (output / "insights.json").write_text(json.dumps(manager.insights, ensure_ascii=False, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                if step % args.save_steps == 0 or step == args.max_steps:
                    save_checkpoint(engine, tokenizer, manager, args, step, position, fingerprint)
    finally:
        policy.shutdown_judging_batching()
    if rank == 0:
        engine.module.student.save_pretrained(output / "final", safe_serialization=True)
        tokenizer.save_pretrained(output / "final")
        shutil.copy2(output / "sft_manifest.json", output / "final/sft_manifest.json")
    dist.barrier()


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
