from __future__ import annotations
from .paths import WEIGHTS, UPSTREAM
import importlib.util
import sys
import time
import torch
from transformers import DebertaV2Model, DebertaV2Tokenizer
from .core import FINAL, MODELS, task_text

UPSTREAM = UPSTREAM / "RouterDC"
ENCODER = WEIGHTS / "routerdc/encoder"
TRAIN_DIR = WEIGHTS / "routerdc"
SEED = 20260920
WORKER_INSTRUCTION = (
    "Answer directly and concisely; no explanation. Respond as briefly as possible; give only the answer without reasoning. "
    + FINAL
)


def original_router_class():
    sys.path.insert(0, str(UPSTREAM))
    path = UPSTREAM / "train_router_mdeberta.py"
    spec = importlib.util.spec_from_file_location("paper_official_routerdc", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.RouterModule


def router_text(item):
    text = task_text(item)
    if item.get("images"):
        text += "\n[Attached images are unavailable to this text-only router.]"
    return text


def tokenize(tokenizer, items, max_length=512):
    return tokenizer(
        [router_text(item) for item in items],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )


class Router:

    def __init__(self, batch_size=32, max_length=512):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(
            ENCODER, local_files_only=True, truncation_side="left"
        )
        encoder = DebertaV2Model.from_pretrained(ENCODER, local_files_only=True)
        self.model = original_router_class()(
            encoder,
            hidden_state_dim=768,
            node_size=len(MODELS),
            similarity_function="cos",
        ).to("cuda")
        self.model.load_state_dict(
            torch.load(TRAIN_DIR / "best.pt", map_location="cuda", weights_only=True)
        )
        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length

    def route(self, items):
        encoded = tokenize(self.tokenizer, items, self.max_length)
        results = []
        began = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(items), self.batch_size):
                inputs = {
                    key: value[start : start + self.batch_size].to("cuda")
                    for key, value in encoded.items()
                }
                logits, _ = self.model(**inputs)
                for row in logits.float().cpu().tolist():
                    chosen = max(range(len(MODELS)), key=lambda index: row[index])
                    results.append(
                        {"model": MODELS[chosen], "scores": dict(zip(MODELS, row))}
                    )
        latency = (time.monotonic() - began) / max(len(items), 1)
        return [
            {"item": item, **decision, "router_latency_sec_share": latency}
            for item, decision in zip(items, results)
        ]
