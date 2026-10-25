"""Read evaluation splits and resolve their local assets."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import re


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_budgetbench(data_dir):
    directory = Path(data_dir).expanduser().resolve()
    public = read_jsonl(directory / "test.inputs.jsonl")
    labels = {
        str(row["id"]): row for row in read_jsonl(directory / "test.labels.jsonl")
    }
    grader_root = (
        Path(__file__).resolve().parents[1] / "third_party" / "masrouter-budgetbench"
    )
    items = []
    for record in public:
        item = deepcopy(record)
        label = labels[str(item["id"])]
        for key in ("answer", "budgetbench_task", "budgetbench_condition"):
            item[key] = deepcopy(label[key])
        item["budgetbench_root"] = str(grader_root)
        item["budgetbench_asset_root"] = str(directory)
        for assets in (
            item.get("worker_assets", []),
            item["budgetbench_task"].get("assets", []),
        ):
            for asset in assets:
                if asset.get("path"):
                    relative = Path(asset["path"])
                    if relative.is_absolute():
                        parts = relative.parts
                        if "assets" not in parts:
                            raise ValueError(
                                f"Image path must be relative to the dataset directory: {relative}"
                            )
                        relative = Path(*parts[parts.index("assets") :])
                    asset["path"] = str(relative)
        items.append(item)
    return items


def baseline_item(raw):
    """Use the shared routing input format without changing the question."""
    task = raw["budgetbench_task"]
    text = task["instruction"]
    choices = {}
    if "\nChoices:\n" in text:
        question, option_text = text.split("\nChoices:\n", 1)
        option_text = option_text.rsplit("\nReturn only", 1)[0]
        markers = list(re.finditer(r"(?m)^([A-Z0-9])\. ", option_text))
        for index, marker in enumerate(markers):
            end = (
                markers[index + 1].start()
                if index + 1 < len(markers)
                else len(option_text)
            )
            choices[marker[1]] = option_text[marker.end() : end].strip()
    else:
        question = text.rsplit("\nReturn only", 1)[0]
    images, passages = [], []
    for asset in raw.get("worker_assets") or task.get("assets", []):
        if asset.get("kind") == "image":
            path = Path(asset["path"])
            if not path.is_absolute():
                path = Path(raw["budgetbench_asset_root"]) / path
            images.append({"index": len(images) + 1, "path": str(path)})
        elif asset.get("content"):
            passages.append(str(asset["content"]))
    return {
        "id": raw["id"],
        "dataset": task["provenance"]["source_name"],
        "question": question,
        "choices": choices,
        "answers": [str(raw["answer"])],
        "kind": "mc" if choices else "numeric" if raw["task_type"] == "math" else "qa",
        "passages": passages,
        "images": images,
        "task_type": raw["task_type"],
        "budgetbench_task": task,
        "budgetbench_root": raw["budgetbench_root"],
        "raw": raw,
    }


def select_items(items, datasets=None, limit=0, per_domain=0, seed=20260906):
    rows = list(items)
    if datasets:
        selected = set(datasets)
        rows = [row for row in rows if dataset_name(row) in selected]
    if per_domain:
        groups = {}
        for item in rows:
            groups.setdefault(item.get("task_type", "unknown"), []).append(item)
        rng = random.Random(seed)
        rows = []
        for name in sorted(groups):
            group = list(groups[name])
            if len(group) < per_domain:
                raise ValueError(
                    f"Domain {name} has {len(group)} items; requested {per_domain}"
                )
            rows.extend(rng.sample(group, per_domain))
        rng.shuffle(rows)
    return rows[:limit] if limit else rows


def dataset_name(item):
    return item.get("dataset") or item["budgetbench_task"]["provenance"]["source_name"]
