import ast
import re

"Answer parsing and native dataset scoring used by baseline implementations."
from . import core

VERSION = "native_success"


def score(text, item):
    if item.get("evaluator") in {"bbh", "livecodebench"} or item["dataset"] in {
        "bbh_logical_deduction",
        "livecodebench",
    }:
        from evorouter.ood import score as ood_score

        return ood_score(text, item)
    from MAR.Benchmarks.budgetbench import grade_budgetbench_prediction

    prediction = core.extract(text)
    result = grade_budgetbench_prediction(
        item["budgetbench_task"], prediction, item["budgetbench_root"]
    )
    correct = bool(result["success"])
    return dict(
        correct=correct,
        score_value=float(correct),
        parsed_prediction=prediction,
        metric_name="native_budgetbench_success",
    )


def failure_score(item):
    return dict(
        correct=False,
        score_value=0.0,
        parsed_prediction="",
        metric_name="native_success",
    )


def _safe_arithmetic(answer):
    """Evaluate a final arithmetic expression, never free-form model text."""
    candidate = answer.strip().replace(",", "")
    if not re.fullmatch("[\\d\\s+\\-*/().]+", candidate) or not re.search(
        "[+*/]", candidate
    ):
        return None
    try:
        tree = ast.parse(candidate, mode="eval")
    except (SyntaxError, ValueError):
        return None
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.USub,
        ast.UAdd,
    )
    if any(
        (
            not isinstance(node, allowed)
            or (
                isinstance(node, ast.Constant)
                and (not isinstance(node.value, (int, float)))
            )
            for node in ast.walk(tree)
        )
    ):
        return None
    try:
        return core.numeric(
            str(eval(compile(tree, "<final-answer>", "eval"), {"__builtins__": {}}, {}))
        )
    except (ArithmeticError, ValueError):
        return None


def prediction_key(text, item):
    if item["dataset"] in {"bbh_logical_deduction", "livecodebench"}:
        from evorouter.ood import prediction_key as ood_key

        return ood_key(text, item)
    key = core.prediction_key(text, item)
    if key:
        return key
    answer = core.extract(text).strip().strip("* ")
    if item["kind"] == "mc":
        match = re.fullmatch(
            "(?i)(?:option|choice|answer)\\s*[:\\-]?\\s*([A-Z0-9])[.)]?", answer
        )
        return match[1].upper() if match and match[1].upper() in item["choices"] else ""
    if item["kind"] != "numeric":
        return key
    primary = re.split(
        "\\s*\\(\\s*(?i:or|equivalently|approximately)\\b", answer, maxsplit=1
    )[0].strip()
    match = re.fullmatch(
        "[$£€]?\\s*([+-]?(?:\\d[\\d,]*(?:\\.\\d+)?|\\.\\d+))\\s*([A-Za-z%][A-Za-z% /().-]*)",
        primary,
    )
    if match:
        value = core.numeric(match[1])
        if value is not None:
            return str(value.normalize())
    value = _safe_arithmetic(primary)
    if value is not None:
        return str(value.normalize())
    return ""
