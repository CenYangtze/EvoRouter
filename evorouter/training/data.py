from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import random


def read_jsonl(path):
    rows = []
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{number}")
            rows.append(row)
    return rows


def _indexed(rows, path):
    result = {}
    for row in rows:
        identifier = str(row.get("id") or "")
        if not identifier or identifier in result:
            raise ValueError(f"Missing or duplicate item ID in {path}: {identifier!r}")
        result[identifier] = row
    return result


def _asset_path(value, directory):
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            path = path.relative_to(directory)
        except ValueError:
            if "assets" not in path.parts:
                raise ValueError(f"Dataset asset must be located inside {directory}")
            path = Path(*path.parts[path.parts.index("assets") :])
    resolved = (directory / path).resolve()
    if not resolved.is_relative_to(directory):
        raise ValueError(f"Dataset asset escapes the dataset directory: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset asset is missing: {resolved}")
    return str(path)


def load_split(data_dir, split):
    if split not in {"train", "dev", "validation", "test"}:
        raise ValueError(f"Unknown dataset split: {split}")
    directory = Path(data_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    inputs = directory / f"{split}.inputs.jsonl"
    labels = directory / f"{split}.labels.jsonl"
    combined = directory / f"{split}.jsonl"
    if inputs.is_file() or labels.is_file():
        public = _indexed(read_jsonl(inputs), inputs)
        private = _indexed(read_jsonl(labels), labels)
        if public.keys() != private.keys():
            raise ValueError(f"Input and label IDs differ for the {split} split")
        records = []
        for identifier, row in public.items():
            item = deepcopy(row)
            for key in ("answer", "budgetbench_task", "budgetbench_condition"):
                if key not in private[identifier]:
                    raise ValueError(f"Label {identifier} has no {key}")
                item[key] = deepcopy(private[identifier][key])
            if private[identifier].get("split", split) != split:
                raise ValueError(f"Label {identifier} belongs to a different split")
            records.append(item)
    else:
        records = list(_indexed(read_jsonl(combined), combined).values())
    grader_root = Path(__file__).resolve().parents[2] / "third_party" / "masrouter-budgetbench"
    items = []
    for record in records:
        item = deepcopy(record)
        if item.get("split", split) != split:
            raise ValueError(f"Item {item['id']} belongs to a different split")
        task = item.get("budgetbench_task")
        if not isinstance(task, dict) or not str(task.get("instruction") or "").strip():
            raise ValueError(f"Item {item['id']} has no task instruction")
        if task.get("split", split) != split:
            raise ValueError(f"Task {item['id']} belongs to a different split")
        condition = item.get("budgetbench_condition")
        if not isinstance(condition, dict) or "weighted_tokens" not in condition.get("budget", {}):
            raise ValueError(f"Item {item['id']} has no BudgetBench condition")
        item["split"] = split
        item["task_id"] = str(item.get("task_id") or task.get("task_id") or str(item["id"]).split("::")[0])
        item["task_type"] = str(item.get("task_type") or task.get("family") or "unknown")
        item["budgetbench_root"] = str(grader_root)
        item["budgetbench_asset_root"] = str(directory)
        for assets in (item.get("worker_assets", []), task.get("assets", [])):
            for asset in assets or []:
                if isinstance(asset, dict) and asset.get("path"):
                    asset["path"] = _asset_path(asset["path"], directory)
        items.append(item)
    return items


def question_key(item):
    metadata = item.get("metadata") or {}
    key = item.get("question_key") or metadata.get("question_key")
    if key:
        return " ".join(str(key).casefold().split())
    instruction = (item.get("budgetbench_task") or {}).get("instruction")
    if not instruction:
        instruction = item.get("question") or item.get("query") or ""
        if str(instruction).startswith("Question:"):
            instruction = str(instruction)[len("Question:") :]
    return " ".join(str(instruction).casefold().split())


def _sft_exclusions(source):
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        candidates = [path / "sft_manifest.json", path / "manifest.json"]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        task_ids = {
            str(row.get("task_id") or (row.get("metadata") or {}).get("task_id") or "")
            for row in rows
        }
        keys = {question_key(row) for row in rows}
        task_ids.discard("")
        keys.discard("")
    else:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("SFT manifest must be a JSON object")
        task_ids = {str(value) for value in manifest.get("task_ids", []) if str(value)}
        keys = {" ".join(str(value).casefold().split()) for value in manifest.get("question_keys", []) if str(value).strip()}
    if not task_ids and not keys:
        raise ValueError("SFT exclusions must include task IDs or question keys")
    return task_ids, keys


def load_rl_tasks(data_dir, sft_source):
    items = load_split(data_dir, "train")
    task_ids, keys = _sft_exclusions(sft_source)
    keys.update(question_key(item) for item in items if item["task_id"] in task_ids)
    seen = set()
    selected = []
    for item in items:
        key = question_key(item)
        if not key or item["task_id"] in task_ids or key in keys or key in seen:
            continue
        seen.add(key)
        selected.append(item)
    if not selected:
        raise ValueError("No distinct training questions remain after excluding SFT tasks")
    return selected


def stratified_order(items, seed=20260906):
    rows = list(items)
    random.Random(seed).shuffle(rows)
    buckets = {}
    for item in rows:
        value = item.get("task_type")
        if value is None or isinstance(value, str) and not value.strip():
            raise ValueError("Training items require a nonempty task_type for stratification")
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        buckets.setdefault(key, deque()).append(item)
    active = deque(buckets.values())
    ordered = []
    while active:
        bucket = active.popleft()
        ordered.append(bucket.popleft())
        if bucket:
            active.append(bucket)
    return ordered
