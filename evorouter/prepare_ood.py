"""Convert official BBH and LiveCodeBench downloads into evaluation inputs."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil


def prepare_bbh(source, indices):
    examples = json.loads(Path(source).read_text(encoding="utf-8"))["examples"]
    rows = []
    contract = "Return only the correct option letter, without explanation."
    for index in indices:
        example = examples[index]
        choices = dict(re.findall(r"^\(([A-Z])\) (.+)$", example["input"], re.M))
        letter = example["target"].strip("() ")
        question = example["input"].split("Options:")[0].strip() + "\n\n" + contract
        instruction = (
            question
            + "\nChoices:\n"
            + "\n".join(f"{key}. {value}" for key, value in choices.items())
        )
        rows.append(
            dict(
                id=f"ood2:bbh_logical_deduction:{index}",
                dataset="bbh_logical_deduction",
                question=question,
                choices=choices,
                images=[],
                passages=[],
                answers=[choices[letter]],
                task_type="commonsense",
                kind="mc",
                answer_type="text",
                evaluator="ood_v1",
                answer_contract_version=2,
                budgetbench_task=dict(
                    family="commonsense",
                    instruction=instruction,
                    assets=[],
                    capability_tags=["logical_reasoning", "deductive_reasoning"],
                    output_contract=dict(format="text", description=contract),
                ),
            )
        )
    return rows


def prepare_code(source, question_ids):
    selected = set(question_ids)
    records = {}
    with Path(source).open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            example = json.loads(line)
            identifier = str(example["question_id"])
            if identifier in selected:
                records[identifier] = (offset, example)
    missing = selected - records.keys()
    if missing:
        raise ValueError(
            "LiveCodeBench source is missing question IDs: "
            + ", ".join(sorted(missing))
        )
    rows = []
    for identifier in question_ids:
        offset, example = records[identifier]
        starter = str(example.get("starter_code") or "").strip()
        metadata = json.loads(example.get("metadata") or "{}")
        function_task = bool(metadata.get("func_name"))
        contract = (
            "Return the complete Python solution implementing the supplied interface, without explanation."
            if function_task
            else "Return a complete executable Python3 program reading standard input and writing standard output, without explanation."
        )
        question = "Task domain: code_generation\n"
        if not function_task:
            question += "Output contract: complete_executable_python3_program\n"
        question += (
            str(example.get("question_title") or "")
            + "\n\n"
            + str(example["question_content"])
        )
        if starter:
            question += "\n\nStarter code:\n" + starter
        question += "\n\n" + contract
        rows.append(
            dict(
                id="ood2:livecodebench:" + identifier,
                dataset="livecodebench",
                question=question,
                choices={},
                images=[],
                passages=[],
                answers=["tests_pass"],
                task_type="code",
                kind="code",
                answer_type="code",
                evaluator="ood_v1",
                code_evaluator="livecodebench",
                answer_contract_version=4,
                grader=dict(
                    source_path="sources/livecodebench_test6.jsonl",
                    byte_offset=offset,
                    question_id=identifier,
                ),
                budgetbench_task=dict(
                    family="code",
                    instruction=question,
                    assets=[],
                    capability_tags=[
                        "programming",
                        "code_generation",
                        "algorithm_design",
                    ],
                    output_contract=dict(format="python", description=contract),
                ),
            )
        )
    return rows


def prepare(asset_root, bbh_source=None, livecodebench_source=None):
    specifications = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/ood_items.json").read_text()
    )
    output = Path(asset_root).expanduser().resolve() / "data/ood"
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    if bbh_source:
        groups["bbh_logical_deduction"] = prepare_bbh(
            Path(bbh_source).expanduser(),
            specifications["bbh_logical_deduction"]["indices"],
        )
    if livecodebench_source:
        source = Path(livecodebench_source).expanduser().resolve()
        groups["livecodebench"] = prepare_code(
            source, specifications["livecodebench"]["question_ids"]
        )
        destination = output / "sources/livecodebench_test6.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination.resolve():
            shutil.copyfile(source, destination)
    for name, rows in groups.items():
        destination = output / f"{name}.jsonl"
        with destination.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} tasks -> {destination}")
