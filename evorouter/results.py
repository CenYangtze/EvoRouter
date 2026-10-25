"""Write aggregate evaluation results."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

from .data import read_jsonl


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def successful(row):
    return bool(row.get("task_success", row.get("correct", False))) and not row.get(
        "error"
    )


def worker_cost(row):
    for key in ("weighted_worker_tokens", "worker_flops", "worker_cost"):
        if key in row:
            return float(row[key])
    return 0.0


def aggregate(rows, expected=None):
    count = len(rows)
    return {
        "completed": count,
        "expected": expected if expected is not None else count,
        "correct": sum(successful(row) for row in rows),
        "accuracy": sum(successful(row) for row in rows) / count if count else None,
        "mean_worker_flops": (
            sum(worker_cost(row) for row in rows) / count if count else None
        ),
        "errors": sum(bool(row.get("error")) for row in rows),
    }


def summarize_method(directory, rows=None, expected=None):
    directory = Path(directory)
    rows = read_jsonl(directory / "results.jsonl") if rows is None else rows
    summary = aggregate(rows, expected)
    for field, section in (
        ("suite", "suites"),
        ("dataset", "datasets"),
        ("task_type", "domains"),
    ):
        groups = defaultdict(list)
        for row in rows:
            groups[row.get(field, row.get("family", "unknown"))].append(row)
        summary[section] = {
            name: aggregate(values) for name, values in sorted(groups.items())
        }
    write_json(directory / "summary.json", summary)
    return summary


def summarize(output_dir):
    root = Path(output_dir)
    summaries = {}
    records = []
    for source in sorted(root.glob("*/results.jsonl")):
        config_file = source.parent / "config.json"
        config = json.loads(config_file.read_text()) if config_file.exists() else {}
        summary = summarize_method(source.parent, expected=config.get("expected"))
        name = source.parent.name
        summaries[name] = summary
        records.append(
            {
                "method": name,
                "group": "all",
                **aggregate(read_jsonl(source), summary["expected"]),
            }
        )
        for section in ("suites", "domains", "datasets"):
            for group, value in summary[section].items():
                records.append({"method": name, "group": group, **value})
    if not summaries:
        raise FileNotFoundError(f"No evaluation results under {root}")
    write_json(root / "summary.json", summaries)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "method",
                "group",
                "completed",
                "expected",
                "correct",
                "accuracy",
                "mean_worker_flops",
                "errors",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Results: {root / 'summary.csv'}")
    return summaries
