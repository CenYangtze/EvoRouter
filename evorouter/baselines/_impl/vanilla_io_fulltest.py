from __future__ import annotations
from .paths import ASSETS, OUTPUT
import base64
import os
from pathlib import Path

FIVE_MODELS = (
    "qwen3.5-35b-a3b",
    "qwen3.5-122b-a10b",
    "deepseek-v4-flash",
    "qwen3.5-397b-a17b",
    "glm-5",
)
os.environ["PAPER_MODEL_POOL"] = ",".join(FIVE_MODELS)
from .core import VISION
from MAR.Benchmarks.budgetbench import grade_budgetbench_prediction

DATA_ROOT = ASSETS / "data/budgetbench"
OUTPUT_ROOT = OUTPUT
INSTRUCTION = "Give the answer directly."
TEXT_ONLY = "[This model has no vision capability. Attached images are unavailable; answer using the text only.]"
SIX = ("musique", "mmlu_pro", "math", "arc_challenge", "supergpqa", "mmmu")


def messages(item, model):
    text = item["budgetbench_task"]["instruction"].strip()
    if item.get("passages"):
        text += "\n\nProvided passages:\n" + "\n\n".join(
            (
                f"[{index}] {passage}"
                for index, passage in enumerate(item["passages"], 1)
            )
        )
    pictures = item.get("images") or []
    if pictures and model not in VISION:
        text += "\n\n" + "\n".join(
            (f"Image {picture['index']}: unavailable." for picture in pictures)
        )
        text += "\n" + TEXT_ONLY
    text += "\n\n" + INSTRUCTION
    content = text
    if pictures and model in VISION:
        content = [{"type": "text", "text": text}]
        for picture in pictures:
            encoded = base64.b64encode(Path(picture["path"]).read_bytes()).decode()
            content.extend(
                [
                    {"type": "text", "text": f"Image {picture['index']}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + encoded},
                    },
                ]
            )
    return [{"role": "user", "content": content}]


def grade_raw(item, prediction):
    return grade_budgetbench_prediction(
        item["budgetbench_task"], prediction, item["budgetbench_root"]
    )
