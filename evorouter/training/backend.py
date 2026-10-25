import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .objectives import full_reverse_kl, sampled_reference_kl


def prepare_awq():
    import transformers.activations as activations

    if not hasattr(activations, "PytorchGELUTanh"):
        class PytorchGELUTanh(nn.GELU):
            def __init__(self):
                super().__init__(approximate="tanh")

        activations.PytorchGELUTanh = PytorchGELUTanh


def validate_tokenizers(student, teacher, student_config, teacher_config):
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("OPD requires identical Student and Teacher token-to-ID mappings")
    if student_config.vocab_size != teacher_config.vocab_size:
        raise ValueError("Student and Teacher output vocabulary sizes differ")
    if student.eos_token_id != teacher.eos_token_id:
        raise ValueError("Student and Teacher EOS IDs differ")
    if not student.is_fast or not teacher.is_fast:
        raise ValueError("OPD requires fast tokenizers")
    left = json.loads(student.backend_tokenizer.to_str())
    right = json.loads(teacher.backend_tokenizer.to_str())
    for key in ("model", "normalizer", "pre_tokenizer", "added_tokens"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Student and Teacher tokenizer {key} differs")


def response_hidden(model, encoded):
    device = next(model.parameters()).device
    inputs = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    labels = torch.tensor(encoded["labels"], dtype=torch.long, device=device)
    kinds = torch.tensor(encoded["kinds"], dtype=torch.int8, device=device)
    output = model.base_model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, return_dict=True)
    mask = labels[1:] != -100
    return output.last_hidden_state[0, :-1][mask], labels[1:][mask], kinds[1:][mask]


def selected_logp(head, hidden, targets):
    logits = head(hidden).float()
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected - torch.logsumexp(logits, dim=-1)


class RoutingObjective(nn.Module):
    def __init__(self, student, reference=None, teacher=None, kl_coefficient=0.001, opd_coefficient=0.03, analysis_weight=0.25, logit_chunk_size=128):
        super().__init__()
        self.student = student
        self.frozen = SimpleNamespace(reference=reference, teacher=teacher)
        self.kl_coefficient = float(kl_coefficient)
        self.opd_coefficient = float(opd_coefficient)
        self.analysis_weight = float(analysis_weight)
        self.logit_chunk_size = int(logit_chunk_size)
        if self.logit_chunk_size < 1:
            raise ValueError("logit_chunk_size must be positive")
        if min(self.kl_coefficient, self.opd_coefficient, self.analysis_weight) < 0:
            raise ValueError("Objective coefficients must be nonnegative")
        if self.kl_coefficient > 0 and reference is None:
            raise ValueError("KL requires a frozen SFT reference")
        if self.opd_coefficient > 0 and teacher is None:
            raise ValueError("OPD requires a frozen external Teacher")
        for model in (reference, teacher):
            if model is not None:
                model.requires_grad_(False)
                model.eval()

    def reference_logp(self, encoded):
        reference = self.frozen.reference
        with torch.no_grad():
            hidden, targets, _ = response_hidden(reference, encoded)
            head = reference.get_output_embeddings()
            values = [selected_logp(head, hidden[start:start + self.logit_chunk_size], targets[start:start + self.logit_chunk_size]) for start in range(0, len(targets), self.logit_chunk_size)]
            return torch.cat(values)

    def policy_losses(self, encoded, advantage, denominators):
        reference_logp = self.reference_logp(encoded) if self.kl_coefficient > 0 else None
        hidden, targets, kinds = response_hidden(self.student, encoded)
        head = self.student.get_output_embeddings()
        rl_denominator = max(float(denominators["rl"]), 1.0)
        kl_denominator = max(float(denominators["kl"]), 1.0)
        terms = []
        for start in range(0, len(targets), self.logit_chunk_size):
            stop = start + self.logit_chunk_size
            references = reference_logp[start:stop].to(hidden.device) if reference_logp is not None else hidden.new_zeros(len(targets[start:stop]))

            def chunk(values, token_ids, reference_scores):
                scores = selected_logp(head, values, token_ids)
                rl = -float(advantage) * scores.sum() / rl_denominator
                kl = sampled_reference_kl(scores, reference_scores).sum() / kl_denominator if self.kl_coefficient > 0 else scores.sum() * 0
                return torch.stack((rl, kl))

            terms.append(checkpoint(chunk, hidden[start:stop], targets[start:stop], references, use_reentrant=False))
        if not terms:
            raise ValueError("A Coordinator response must contain trainable tokens")
        return torch.stack(terms).sum(0), (hidden, targets, kinds)

    def distillation_losses(self, student_encoded, teacher_encoded, denominators, cached=None):
        teacher = self.frozen.teacher
        with torch.no_grad():
            teacher_hidden, teacher_targets, teacher_kinds = response_hidden(teacher, teacher_encoded)
            teacher_mask = teacher_kinds > 0
            teacher_hidden = teacher_hidden[teacher_mask]
            teacher_targets = teacher_targets[teacher_mask]
            teacher_kinds = teacher_kinds[teacher_mask]
        student_hidden, targets, kinds = cached if cached is not None else response_hidden(self.student, student_encoded)
        mask = kinds > 0
        student_hidden, targets, kinds = student_hidden[mask], targets[mask], kinds[mask]
        if not torch.equal(targets, teacher_targets.to(targets.device)) or not torch.equal(kinds, teacher_kinds.to(kinds.device)):
            raise ValueError("Student and Teacher OPD response positions do not align")
        if not torch.any(kinds == 1) or not torch.any(kinds == 2):
            raise ValueError("OPD requires analysis and plan-value tokens")
        head = self.student.get_output_embeddings()
        teacher_head = teacher.get_output_embeddings()
        plan_denominator = max(float(denominators["plan"]), 1.0)
        analysis_denominator = max(float(denominators["analysis"]), 1.0)
        terms = []

        def chunk(student_values, teacher_values, token_kinds):
            with torch.no_grad():
                teacher_logits = teacher_head(teacher_values).to(student_values.device)
            divergence = full_reverse_kl(head(student_values), teacher_logits)
            return torch.stack((divergence[token_kinds == 1].sum() / plan_denominator, divergence[token_kinds == 2].sum() / analysis_denominator))

        for start in range(0, len(kinds), self.logit_chunk_size):
            stop = start + self.logit_chunk_size
            terms.append(checkpoint(chunk, student_hidden[start:stop], teacher_hidden[start:stop], kinds[start:stop], use_reentrant=False))
        return torch.stack(terms).sum(0)

    def dummy_loss(self, record):
        device = next(self.student.parameters()).device
        token = int(record["rl"]["input_ids"][0])
        inputs = torch.tensor([[token]], device=device)
        hidden = self.student.base_model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, return_dict=True).last_hidden_state
        return (hidden.sum() + self.student.get_output_embeddings()(hidden).sum()) * 0.0

    def forward(self, records, denominators):
        if not records:
            raise ValueError("An optimizer microbatch requires a record or a dummy")
        total = None
        components = dict(rl_loss=0.0, kl_loss=0.0, kl_weighted_loss=0.0, opd_plan_loss=0.0, opd_analysis_loss=0.0, opd_loss=0.0, opd_weighted_loss=0.0, total_loss=0.0)
        for record in records:
            if record.get("dummy"):
                loss = self.dummy_loss(record)
            else:
                policy, cached = self.policy_losses(record["rl"], record["advantage"], denominators)
                rl, kl = policy.unbind()
                loss = rl + self.kl_coefficient * kl
                components["rl_loss"] += float(rl.detach())
                components["kl_loss"] += float(kl.detach())
                components["kl_weighted_loss"] += self.kl_coefficient * float(kl.detach())
                student_opd, teacher_opd = record.get("student_opd"), record.get("teacher_opd")
                if self.opd_coefficient > 0 and student_opd is not None and teacher_opd is not None:
                    reuse = cached if student_opd["input_ids"] == record["rl"]["input_ids"] else None
                    plan, analysis = self.distillation_losses(student_opd, teacher_opd, denominators, reuse).unbind()
                    opd = plan + self.analysis_weight * analysis
                    loss = loss + self.opd_coefficient * opd
                    components["opd_plan_loss"] += float(plan.detach())
                    components["opd_analysis_loss"] += float(analysis.detach())
                    components["opd_loss"] += float(opd.detach())
                    components["opd_weighted_loss"] += self.opd_coefficient * float(opd.detach())
                components["total_loss"] += float(loss.detach())
            total = loss if total is None else total + loss
        return total * int(denominators["world_size"]), components


def resolved_deepspeed_config(args, world_size):
    config = json.loads(Path(args.deepspeed).read_text())
    stage = int(config.get("zero_optimization", {}).get("stage", 0))
    if stage not in (0, 2):
        raise ValueError("Training supports ZeRO stage 2, or stage 0 for small debugging runs")
    config["bf16"] = dict(config.get("bf16", {}), enabled=args.dtype == "bfloat16")
    config["fp16"] = dict(config.get("fp16", {}), enabled=args.dtype == "float16")
    config["train_micro_batch_size_per_gpu"] = int(args.micro_batch_size)
    config["gradient_accumulation_steps"] = 1
    config["train_batch_size"] = int(args.micro_batch_size) * int(world_size)
    config["gradient_clipping"] = float(args.gradient_clipping)
    parameters = dict(lr=float(args.learning_rate), betas=[0.9, 0.999], eps=1e-8, weight_decay=float(args.weight_decay))
    if stage == 0:
        parameters["torch_adam"] = True
    config["optimizer"] = dict(type="AdamW", params=parameters)
    config.pop("scheduler", None)
    return config


def create_engine(args):
    import deepspeed
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Training requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    student_config = AutoConfig.from_pretrained(args.model)
    if getattr(student_config, "quantization_config", None):
        raise ValueError("Student initialization requires unquantized model weights")
    student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map={"": str(device)}, attn_implementation="sdpa")
    student.requires_grad_(True)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    teacher_device = torch.device(args.teacher_device) if getattr(args, "teacher_device", None) else device
    cuda_devices = sorted({device.index, teacher_device.index if teacher_device.index is not None else device.index}) if teacher_device.type == "cuda" else [device.index]
    reference, teacher, teacher_tokenizer = None, None, None
    with torch.random.fork_rng(devices=cuda_devices):
        if float(args.kl_coefficient) > 0:
            reference = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map={"": str(device)}, attn_implementation="sdpa")
            reference.config.use_cache = False
            reference.requires_grad_(False)
            reference.eval()
        if float(args.opd_coefficient) > 0:
            teacher_config = AutoConfig.from_pretrained(args.teacher_model)
            teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)
            validate_tokenizers(tokenizer, teacher_tokenizer, student_config, teacher_config)
            quantization = getattr(teacher_config, "quantization_config", None)
            teacher_dtype = torch.bfloat16
            if quantization:
                if quantization.get("quant_method") != "awq" or quantization.get("bits") != 4:
                    raise ValueError("Quantized Teachers require AWQ INT4 weights")
                prepare_awq()
                teacher_dtype = torch.float16
            teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=teacher_dtype, device_map={"": str(teacher_device)}, attn_implementation="sdpa")
            teacher.config.use_cache = False
            teacher.requires_grad_(False)
            teacher.eval()
    module = RoutingObjective(student, reference, teacher, args.kl_coefficient, args.opd_coefficient, args.analysis_weight, args.logit_chunk_size)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    config = resolved_deepspeed_config(args, world_size)

    def scheduler(optimizer):
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=int(args.max_steps))

    engine, _, _, _ = deepspeed.initialize(model=module, model_parameters=student.parameters(), config=config, lr_scheduler=scheduler, dist_init_required=False)
    return engine, tokenizer, teacher_tokenizer
