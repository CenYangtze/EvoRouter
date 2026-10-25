from __future__ import annotations

import json
import math
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _normalise_text(value: Any, *, case_sensitive: bool = False) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if not case_sensitive:
        text = text.casefold()
    return text


def _normalise_qa_answer(value: Any) -> str:
    """Apply the standard open-domain QA exact-match normalization."""

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _decode_prediction(prediction: Any) -> Any:
    if not isinstance(prediction, str):
        return prediction
    stripped = prediction.strip()
    if not stripped:
        return ""
    if stripped[0] in "{\"" or stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return stripped


def _extract_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric answer")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping) and "answer" in value:
        return _extract_number(value["answer"])
    text = str(value).strip().replace(",", "")
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1].strip()
    latex_fraction = re.fullmatch(
        r"\s*(-?)\\(?:d?frac)\{([+-]?\d+)\}\{(\d+)\}\s*", text
    )
    if latex_fraction and int(latex_fraction.group(3)):
        sign = -1 if latex_fraction.group(1) else 1
        return sign * int(latex_fraction.group(2)) / int(latex_fraction.group(3))
    plain_fraction = re.fullmatch(r"\s*([+-]?\d+)\s*/\s*(\d+)\s*", text)
    if plain_fraction and int(plain_fraction.group(2)):
        return int(plain_fraction.group(1)) / int(plain_fraction.group(2))
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not matches:
        raise ValueError(f"no numeric value found in {value!r}")
    # Numeric task contracts ask for a final answer. Using the last numeric
    # literal supports a short derivation without mistaking an intermediate
    # quantity for the submitted answer.
    return float(matches[-1])


def _result(score: float, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    bounded = max(0.0, min(1.0, float(score)))
    return {
        "score": bounded,
        "success": math.isclose(bounded, 1.0, abs_tol=1e-12),
        "details": details or {},
    }


def grade_exact(reference: Any, prediction: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    decoded = _decode_prediction(prediction)
    if isinstance(decoded, Mapping) and "answer" in decoded:
        decoded = decoded["answer"]
    aliases = [reference, *list(spec.get("answer_aliases", []))]
    if spec.get("normalization") == "qa":
        expected_values = [_normalise_qa_answer(value) for value in aliases]
        actual = _normalise_qa_answer(decoded)
    else:
        case_sensitive = bool(spec.get("case_sensitive", False))
        expected_values = [
            _normalise_text(value, case_sensitive=case_sensitive) for value in aliases
        ]
        actual = _normalise_text(decoded, case_sensitive=case_sensitive)
    return _result(
        float(actual in expected_values),
        details={"expected": expected_values, "actual": actual},
    )


def grade_numeric(reference: Any, prediction: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    try:
        expected = _extract_number(reference)
        actual = _extract_number(_decode_prediction(prediction))
    except ValueError as exc:
        return _result(0.0, details={"error": str(exc)})
    abs_tol = float(spec.get("absolute_tolerance", 1e-9))
    rel_tol = float(spec.get("relative_tolerance", 1e-9))
    correct = math.isclose(expected, actual, abs_tol=abs_tol, rel_tol=rel_tol)
    return _result(
        float(correct),
        details={"expected": expected, "actual": actual, "absolute_error": abs(expected - actual)},
    )


def _field_score(expected: Any, actual: Any, comparator: str, spec: Mapping[str, Any]) -> float:
    if comparator == "numeric":
        return grade_numeric(expected, actual, spec)["score"]
    if comparator == "set":
        expected_set = {_normalise_text(item) for item in expected}
        actual_set = {_normalise_text(item) for item in actual}
        return float(expected_set == actual_set)
    if comparator == "ordered_list":
        return float(list(expected) == list(actual))
    return grade_exact(expected, actual, spec)["score"]


def grade_json_fields(reference: Any, prediction: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    decoded = _decode_prediction(prediction)
    if not isinstance(reference, Mapping) or not isinstance(decoded, Mapping):
        return _result(0.0, details={"error": "reference and prediction must be JSON objects"})
    field_specs = spec.get("fields") or {key: {"comparator": "exact"} for key in reference}
    scores: dict[str, float] = {}
    weights: dict[str, float] = {}
    for field, field_spec in field_specs.items():
        field_spec = field_spec or {}
        weights[field] = float(field_spec.get("weight", 1.0))
        if field not in reference or field not in decoded:
            scores[field] = 0.0
            continue
        scores[field] = _field_score(
            reference[field],
            decoded[field],
            str(field_spec.get("comparator", "exact")),
            field_spec,
        )
    total_weight = sum(weights.values()) or 1.0
    score = sum(scores[field] * weights[field] for field in scores) / total_weight
    return _result(score, details={"field_scores": scores})


def grade_evidence(reference: Any, prediction: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    decoded = _decode_prediction(prediction)
    if not isinstance(reference, Mapping) or not isinstance(decoded, Mapping):
        return _result(0.0, details={"error": "evidence grader expects JSON objects"})
    expected_answers = [reference.get("answer", ""), *spec.get("answer_aliases", [])]
    answer_score = max(
        grade_exact(expected, decoded.get("answer", ""), spec)["score"] for expected in expected_answers
    )
    expected = {_normalise_text(item) for item in reference.get("evidence", [])}
    actual = {_normalise_text(item) for item in decoded.get("evidence", [])}
    if not expected and not actual:
        evidence_f1 = 1.0
    elif not expected or not actual:
        evidence_f1 = 0.0
    else:
        overlap = len(expected & actual)
        precision = overlap / len(actual)
        recall = overlap / len(expected)
        evidence_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    answer_weight = float(spec.get("answer_weight", 0.7))
    score = answer_weight * answer_score + (1.0 - answer_weight) * evidence_f1
    return _result(
        score,
        details={"answer_score": answer_score, "evidence_f1": evidence_f1},
    )


def _code_from_prediction(prediction: Any) -> str:
    decoded = _decode_prediction(prediction)
    if isinstance(decoded, Mapping) and "code" in decoded:
        return str(decoded["code"])
    text = str(decoded)
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text


def grade_code_tests(
    reference: Any,
    prediction: Any,
    spec: Mapping[str, Any],
    *,
    allow_code_execution: bool,
) -> dict[str, Any]:
    if not allow_code_execution:
        return _result(0.0, details={"skipped": True, "reason": "code execution not enabled"})
    code = _code_from_prediction(prediction)
    tests = spec.get("tests", [])
    setup = str(spec.get("setup", ""))
    if not isinstance(tests, Sequence) or isinstance(tests, (str, bytes)):
        return _result(0.0, details={"error": "grader tests must be a list"})
    runner = "\n".join([setup, code, *map(str, tests), "print('BUDGETBENCH_TESTS_OK')"])
    timeout_s = float(spec.get("timeout_s", 5.0))
    with tempfile.TemporaryDirectory(prefix="budgetbench-code-") as temp_dir:
        path = Path(temp_dir) / "candidate_test.py"
        path.write_text(runner, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _result(0.0, details={"error": "timeout", "timeout_s": timeout_s})
    passed = completed.returncode == 0 and "BUDGETBENCH_TESTS_OK" in completed.stdout
    return _result(
        float(passed),
        details={
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        },
    )


def _sqlite_authorizer(action: int, _arg1: str | None, _arg2: str | None, _db: str | None, _trigger: str | None) -> int:
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def grade_sql_result(
    reference: Any,
    prediction: Any,
    spec: Mapping[str, Any],
    *,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    decoded = _decode_prediction(prediction)
    if isinstance(decoded, Mapping) and "sql" in decoded:
        sql = str(decoded["sql"])
    else:
        sql = str(decoded)
    tables = spec.get("tables", [])
    database_path = spec.get("database_path")
    if database_path:
        root = Path(asset_root or ".").resolve()
        resolved = (root / str(database_path)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return _result(0.0, details={"error": "database asset path escapes the release root"})
        if not resolved.exists():
            return _result(0.0, details={"error": f"database asset not found: {resolved}"})
        connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(":memory:")
    try:
        if not database_path:
            for table in tables:
                name = str(table["name"])
                columns = list(table["columns"])
                rows = list(table["rows"])
                declarations = ", ".join(f'"{column["name"]}" {column.get("type", "TEXT")}' for column in columns)
                connection.execute(f'CREATE TABLE "{name}" ({declarations})')
                placeholders = ",".join("?" for _ in columns)
                connection.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', rows)
        connection.set_authorizer(_sqlite_authorizer)
        deadline = time.monotonic() + float(spec.get("timeout_s", 5.0))
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        actual_rows = [list(row) for row in connection.execute(sql).fetchall()]
    except (sqlite3.Error, KeyError, TypeError) as exc:
        return _result(0.0, details={"error": str(exc)})
    finally:
        connection.close()
    expected_rows = reference if isinstance(reference, list) else [[reference]]
    if not bool(spec.get("ordered", False)):
        actual_rows = sorted(actual_rows, key=repr)
        expected_rows = sorted(expected_rows, key=repr)
    return _result(float(actual_rows == expected_rows), details={"expected": expected_rows, "actual": actual_rows})


def grade_task(
    task: Mapping[str, Any],
    prediction: Any,
    *,
    allow_code_execution: bool = False,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Grade a prediction using the task's declared deterministic grader."""

    grader = task["grader"]
    grader_type = grader["type"]
    spec = grader.get("spec", {})
    reference = task["reference"]["answer"]
    if grader_type == "exact_match":
        return grade_exact(reference, prediction, spec)
    if grader_type == "numeric":
        return grade_numeric(reference, prediction, spec)
    if grader_type == "json_fields":
        return grade_json_fields(reference, prediction, spec)
    if grader_type == "evidence":
        return grade_evidence(reference, prediction, spec)
    if grader_type == "code_tests":
        return grade_code_tests(reference, prediction, spec, allow_code_execution=allow_code_execution)
    if grader_type == "sql_result":
        return grade_sql_result(reference, prediction, spec, asset_root=asset_root)
    raise ValueError(f"unsupported grader type: {grader_type}")
