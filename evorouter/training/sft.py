import argparse
import json
import os
from pathlib import Path
import shutil


def read_records(path):
    records = []
    identities = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            identity = str(record.get("id", ""))
            if not identity or identity in identities:
                raise ValueError(f"Missing or repeated SFT ID at line {line_number}")
            identities.add(identity)
            if record.get("role", "coordinator") != "coordinator":
                continue
            messages = record.get("messages")
            response = record.get("response")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"SFT record {identity} requires messages")
            if any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError(f"Invalid messages in SFT record {identity}")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"SFT record {identity} requires a response")
            stage = record.get("stage")
            terminal = "plan" if stage == "Ready" else "answer" if stage == "Judging" else None
            if terminal is None or f"<{terminal}>" not in response:
                raise ValueError(f"Invalid Coordinator stage in SFT record {identity}")
            records.append(record)
    if not records:
        raise ValueError(f"No Coordinator SFT records in {path}")
    return records


def record_identity(records):
    task_ids, questions = set(), set()
    for record in records:
        metadata = record.get("metadata") or {}
        task_id = metadata.get("task_id")
        if task_id is not None:
            task_ids.add(str(task_id))
        question = metadata.get("question_key") or metadata.get("question")
        if question:
            questions.add(" ".join(str(question).casefold().split()))
    return task_ids, questions


def tokenize_record(record, tokenizer, max_length):
    prompt = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(
        record["response"] + (tokenizer.eos_token or ""), add_special_tokens=False,
    )["input_ids"]
    if not prompt_ids or not response_ids or len(response_ids) >= max_length:
        raise ValueError(f"SFT response does not fit the context: {record['id']}")
    budget = max_length - len(response_ids)
    if len(prompt_ids) > budget:
        head = max(1, budget // 4)
        tail = budget - head
        prompt_ids = prompt_ids[:head] + (prompt_ids[-tail:] if tail else [])
    return {
        "input_ids": prompt_ids + response_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(response_ids)),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def collator_for(tokenizer):
    def collate(rows):
        import torch

        length = max(len(row["input_ids"]) for row in rows)
        result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in rows:
            padding = length - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
    return collate


def response_statistics(logits, labels):
    import torch
    import torch.nn.functional as functional

    if isinstance(logits, tuple):
        logits = logits[0]
    targets = labels[:, 1:].contiguous()
    losses = functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1), ignore_index=-100, reduction="none",
    ).reshape_as(targets)
    counts = targets.ne(-100).sum(dim=1)
    return torch.stack((losses.double().sum(dim=1), counts.double()), dim=1)


def response_metrics(prediction):
    values = prediction.predictions
    tokens = float(values[:, 1].sum())
    if tokens <= 0:
        raise ValueError("Validation contains no supervised response tokens")
    return {"response_nll": float(values[:, 0].sum()) / tokens, "response_tokens": int(tokens)}


def export_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    ignored = {"trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "scaler.pt"}
    for path in source.iterdir():
        if path.is_file() and path.name not in ignored and not path.name.startswith("rng_state"):
            if path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja", ".tiktoken"}:
                shutil.copy2(path, destination / path.name)
    if not any(destination.glob("*.safetensors")) and not (destination / "pytorch_model.bin").is_file():
        raise ValueError(f"The selected checkpoint has no full model weights: {source}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the Coordinator on verified two-stage trajectories.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--dev-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--deepspeed", type=Path)
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.epochs, args.batch_size, args.micro_batch_size, args.max_length) <= 0:
        raise ValueError("Epochs, batch sizes, and maximum length must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    divisor = args.micro_batch_size * world_size
    if args.batch_size % divisor:
        raise ValueError("Global batch size must be divisible by micro batch size times world size")
    train_records = read_records(args.train_file)
    dev_records = read_records(args.dev_file)
    train_ids, train_questions = record_identity(train_records)
    dev_ids, dev_questions = record_identity(dev_records)
    if train_ids & dev_ids or train_questions & dev_questions:
        raise ValueError("SFT train and validation tasks overlap")
    if any(not (record.get("metadata") or {}).get("task_id") for record in train_records):
        raise ValueError("Training SFT records require metadata.task_id for RL exclusions")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a padding or EOS token")
    tokenizer.padding_side = "right"
    train_rows = [tokenize_record(row, tokenizer, args.max_length) for row in train_records]
    dev_rows = [tokenize_record(row, tokenizer, args.max_length) for row in dev_records]
    arguments = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.batch_size // divisor,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio,
        weight_decay=0.0, max_grad_norm=1.0, optim="adamw_torch", bf16=args.dtype == "bfloat16",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        metric_for_best_model="response_nll", greater_is_better=False,
        load_best_model_at_end=False, logging_steps=10, report_to=[], seed=args.seed,
        data_seed=args.seed, remove_unused_columns=False, dataloader_num_workers=0,
        deepspeed=str(args.deepspeed) if args.deepspeed else None,
        ddp_find_unused_parameters=False, ddp_backend=args.distributed_backend, local_rank=args.local_rank,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
    )
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    class ResponseTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
            statistics = response_statistics(outputs.logits, inputs["labels"])
            loss = (statistics[:, 0] / statistics[:, 1].clamp_min(1)).mean().float()
            return (loss, outputs) if return_outputs else loss

    trainer = ResponseTrainer(
        model=model, args=arguments, train_dataset=train_rows, eval_dataset=dev_rows,
        processing_class=tokenizer, data_collator=collator_for(tokenizer),
        preprocess_logits_for_metrics=response_statistics, compute_metrics=response_metrics,
    )
    trainer.model_accepts_loss_kwargs = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task_ids": sorted(train_ids), "question_keys": sorted(train_questions),
        "train_records": len(train_records), "dev_records": len(dev_records),
        "epochs": args.epochs, "global_batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "scheduler": "cosine", "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length, "seed": args.seed, "response_only": True,
        "base_model": args.model,
    }
    manifest_path = args.output_dir / "sft_manifest.json"
    if trainer.is_world_process_zero():
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            for key in ("task_ids", "question_keys", "train_records", "dev_records", "base_model"):
                if previous.get(key) != manifest[key]:
                    raise ValueError(f"SFT output belongs to a different dataset or model: {key}")
        elif any(args.output_dir.glob("checkpoint-*")):
            raise ValueError("Existing SFT checkpoints require their original sft_manifest.json")
        if any(args.output_dir.glob("checkpoint-*")) and not args.resume_from_checkpoint:
            raise ValueError("Use --resume-from-checkpoint to continue an existing SFT run")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    trainer.accelerator.wait_for_everyone()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        selected = trainer.state.best_model_checkpoint
        if not selected:
            raise RuntimeError("No validation-selected SFT checkpoint was saved")
        export_checkpoint(selected, args.output_dir / "best")
        tokenizer.save_pretrained(args.output_dir / "best")
        manifest.update(best_response_nll=float(trainer.state.best_metric), optimizer_steps=trainer.state.global_step)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for destination in (args.output_dir / "best", args.output_dir / "final"):
            shutil.copy2(manifest_path, destination / "sft_manifest.json")
        print(json.dumps({"best_model": str(args.output_dir / "best"), "optimizer_steps": trainer.state.global_step}))
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
