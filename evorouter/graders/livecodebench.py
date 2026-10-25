"""Execute a LiveCodeBench solution with the supplied test cases."""

import argparse
import ast
import base64
import json
import os
import pickle
import re
import sys
import time
import types
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grade one LiveCodeBench Python solution."
    )
    parser.add_argument("--source_jsonl", type=Path, required=True)
    parser.add_argument("--byte_offset", type=int, required=True)
    parser.add_argument("--official_repo", type=Path, required=True)
    parser.add_argument("--test_timeout", type=int, default=6)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_source_row(path: Path, byte_offset: int) -> Dict[str, Any]:
    path = resolve_path(path)
    with open(path, "rb") as file:
        file.seek(byte_offset)
        line = file.readline()
    if not line:
        raise ValueError(f"No LiveCodeBench row at byte offset {byte_offset} in {path}")
    return json.loads(line.decode("utf-8"))


def decode_cases(value: str) -> List[Dict[str, Any]]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        payload = zlib.decompress(base64.b64decode(str(value).encode("utf-8")))
        decoded = pickle.loads(payload)
        if isinstance(decoded, bytes):
            decoded = decoded.decode("utf-8")
        if isinstance(decoded, str):
            decoded = json.loads(decoded)
    if not isinstance(decoded, list):
        raise TypeError(
            f"Expected a list of LiveCodeBench test cases, got {type(decoded).__name__}"
        )
    return decoded


def evaluation_sample(row: Dict[str, Any]) -> Dict[str, str]:
    public_cases = decode_cases(row["public_test_cases"])
    private_cases = decode_cases(row["private_test_cases"])
    metadata = json.loads(row.get("metadata") or "{}")
    cases = public_cases + private_cases
    return {
        "input_output": json.dumps(
            {
                "inputs": [case["input"] for case in cases],
                "outputs": [case["output"] for case in cases],
                "fn_name": metadata.get("func_name"),
            }
        )
    }


def _abc333_e_events(case_input: str) -> List[Tuple[int, int]]:
    values = [int(token) for token in str(case_input).split()]
    if not values:
        raise ValueError("abc333_e test input is empty")
    count = values[0]
    if count < 0 or len(values) != 1 + 2 * count:
        raise ValueError(
            f"abc333_e test input has {len(values)} integers for N={count}"
        )
    events = [(values[index], values[index + 1]) for index in range(1, len(values), 2)]
    if any(event_type not in {1, 2} for event_type, _ in events):
        raise ValueError("abc333_e test input contains an invalid event type")
    return events


def _abc333_e_simulate(
    events: List[Tuple[int, int]], actions: List[int]
) -> Tuple[bool, int, str]:
    inventory: Dict[int, int] = {}
    action_index = 0
    held = 0
    peak = 0
    for event_type, potion_type in events:
        if event_type == 1:
            action = actions[action_index]
            action_index += 1
            if action:
                inventory[potion_type] = inventory.get(potion_type, 0) + 1
                held += 1
                peak = max(peak, held)
            continue
        if inventory.get(potion_type, 0) <= 0:
            return False, peak, f"monster type {potion_type} cannot be defeated"
        inventory[potion_type] -= 1
        held -= 1
    return True, peak, ""


def _abc333_e_optimal_peak(events: List[Tuple[int, int]]) -> int:
    pickup_indices: Dict[int, int] = {}
    pickup_count = 0
    for event_index, (event_type, _) in enumerate(events):
        if event_type == 1:
            pickup_indices[event_index] = pickup_count
            pickup_count += 1

    required: Dict[int, int] = {}
    actions = [0] * pickup_count
    for event_index in range(len(events) - 1, -1, -1):
        event_type, potion_type = events[event_index]
        if event_type == 2:
            required[potion_type] = required.get(potion_type, 0) + 1
        elif required.get(potion_type, 0) > 0:
            actions[pickup_indices[event_index]] = 1
            required[potion_type] -= 1
    if any(count > 0 for count in required.values()):
        return -1

    feasible, peak, reason = _abc333_e_simulate(events, actions)
    if not feasible:
        raise RuntimeError(f"abc333_e reverse-greedy invariant failed: {reason}")
    return peak


def check_abc333_e_output(
    case_input: str, candidate_output: str
) -> Tuple[bool, Dict[str, Any]]:
    events = _abc333_e_events(case_input)
    optimal_peak = _abc333_e_optimal_peak(events)
    tokens = str(candidate_output or "").split()
    if optimal_peak < 0:
        passed = tokens == ["-1"]
        return passed, {
            "reason": (
                "accepted impossible marker"
                if passed
                else "instance is impossible; expected only -1"
            ),
            "optimal_peak": -1,
        }
    if not tokens:
        return False, {"reason": "empty output", "optimal_peak": optimal_peak}
    try:
        values = [int(token) for token in tokens]
    except ValueError:
        return False, {
            "reason": "output contains a non-integer token",
            "optimal_peak": optimal_peak,
        }

    pickup_count = sum(event_type == 1 for event_type, _ in events)
    if len(values) != pickup_count + 1:
        return False, {
            "reason": f"expected K plus {pickup_count} actions, got {len(values)} integers",
            "optimal_peak": optimal_peak,
        }
    declared_peak, actions = values[0], values[1:]
    if any(action not in {0, 1} for action in actions):
        return False, {"reason": "actions must be 0 or 1", "optimal_peak": optimal_peak}

    feasible, actual_peak, reason = _abc333_e_simulate(events, actions)
    if not feasible:
        return False, {
            "reason": reason,
            "declared_peak": declared_peak,
            "actual_peak": actual_peak,
            "optimal_peak": optimal_peak,
        }
    if declared_peak != actual_peak:
        return False, {
            "reason": "declared K does not equal the action sequence's actual peak",
            "declared_peak": declared_peak,
            "actual_peak": actual_peak,
            "optimal_peak": optimal_peak,
        }
    if actual_peak != optimal_peak:
        return False, {
            "reason": "action sequence is feasible but does not minimize K",
            "declared_peak": declared_peak,
            "actual_peak": actual_peak,
            "optimal_peak": optimal_peak,
        }
    return True, {
        "reason": "feasible action sequence with optimal peak",
        "declared_peak": declared_peak,
        "actual_peak": actual_peak,
        "optimal_peak": optimal_peak,
    }


SPECIAL_STDIO_CHECKERS: Dict[str, Callable[[str, str], Tuple[bool, Dict[str, Any]]]] = {
    "abc333_e": check_abc333_e_output,
}


def extract_code(prediction: Any) -> str:
    text = str(prediction or "").strip()
    answer_match = re.search(
        r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL
    )
    if answer_match:
        text = answer_match.group(1).strip()
    fence_match = re.search(
        r"```(?:python|py|python3)?\s*\n?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        text = fence_match.group(1).strip()
    if text.lower().startswith("python\n"):
        text = text.split("\n", 1)[1]
    return text.strip()


def _is_main_guard(node: ast.AST) -> bool:
    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or len(node.comparators) != 1
    ):
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False

    def value(candidate):
        if isinstance(candidate, ast.Name):
            return candidate.id
        if isinstance(candidate, ast.Constant):
            return candidate.value
        if isinstance(candidate, ast.Str):
            return candidate.s
        return None

    return {value(node.left), value(node.comparators[0])} == {"__name__", "__main__"}


def clean_main_guard(code: str) -> str:
    """Inline a terminal main guard before LiveCodeBench wraps stdin programs."""
    try:
        tree = ast.parse(code)
        last_statement = tree.body[-1]
        if isinstance(last_statement, ast.If) and _is_main_guard(last_statement.test):
            tree.body = tree.body[:-1] + last_statement.body
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
    except Exception:
        pass
    return code


def load_official_testing_util(official_repo: Path):
    util_path = (
        resolve_path(official_repo) / "lcb_runner" / "evaluation" / "testing_util.py"
    )
    if not util_path.exists():
        raise FileNotFoundError(
            f"LiveCodeBench testing utility not found at {util_path}. "
            "Clone https://github.com/LiveCodeBench/LiveCodeBench and set LIVECODEBENCH_REPO."
        )
    source = util_path.read_text(encoding="utf-8")
    module = types.ModuleType("livecodebench_official_testing_util")
    module.__file__ = str(util_path)
    exec(
        compile(
            "from __future__ import annotations\n" + source, str(util_path), "exec"
        ),
        module.__dict__,
    )
    module.clean_if_name = clean_main_guard
    return module


def run_stdio_with_special_checker(
    testing_util: Any,
    prediction: str,
    cases: List[Dict[str, Any]],
    checker_name: str,
    checker: Callable[[str, str], Tuple[bool, Dict[str, Any]]],
    timeout: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    testing_util.signal.signal(
        testing_util.signal.SIGALRM, testing_util.timeout_handler
    )
    testing_util.reliability_guard()
    code = testing_util.make_function(testing_util.clean_if_name(prediction))
    try:
        compiled_solution = testing_util.compile_code(code, timeout)
        method = testing_util.get_function(compiled_solution, "wrapped_function")
    except Exception as exc:
        return [-4], {
            "error_code": -4,
            "error_message": f"Error while compiling solution: {exc}",
            "special_checker": checker_name,
        }
    if method is None:
        return [-4], {
            "error_code": -4,
            "error_message": "Could not load wrapped solution",
            "special_checker": checker_name,
        }

    results: List[Any] = []
    total_execution_time = 0.0
    for case_index, case in enumerate(cases):
        case_input = str(case["input"])
        testing_util.signal.alarm(timeout)
        testing_util.faulthandler.enable()
        with testing_util.Capturing() as captured_output:
            try:
                started = time.perf_counter()
                testing_util.call_method(method, case_input)
                total_execution_time += time.perf_counter() - started
            except Exception as exc:
                testing_util.signal.alarm(0)
                error_text = repr(exc)
                error_code = -3 if "timeoutexception" in error_text.lower() else -4
                results.append(error_code)
                return results, {
                    "error": error_text,
                    "error_code": error_code,
                    "error_message": (
                        "Time Limit Exceeded" if error_code == -3 else "Runtime Error"
                    ),
                    "case_index": case_index,
                    "special_checker": checker_name,
                }
            finally:
                testing_util.signal.alarm(0)
                testing_util.faulthandler.disable()

        try:
            passed, checker_metadata = checker(case_input, captured_output[0])
        except Exception as exc:
            raise RuntimeError(
                f"Special checker {checker_name} failed on case {case_index}: {exc}"
            ) from exc
        if not passed:
            results.append(-2)
            return results, {
                "error_code": -2,
                "error_message": "Wrong Answer under special checker",
                "case_index": case_index,
                "output": testing_util.truncatefn(captured_output[0]),
                "checker_metadata": checker_metadata,
                "special_checker": checker_name,
            }
        results.append(True)

    return results, {
        "execution time": total_execution_time,
        "special_checker": checker_name,
        "checked_cases": len(cases),
    }


def main():
    args = parse_args()
    request = json.load(sys.stdin)
    prediction = extract_code(request.get("prediction"))
    if not prediction:
        print(
            json.dumps(
                {"passed": False, "results": [], "metadata": {"error": "empty code"}}
            )
        )
        return

    row = load_source_row(args.source_jsonl, args.byte_offset)
    testing_util = load_official_testing_util(args.official_repo)
    question_id = str(row.get("question_id") or "")
    special_checker = SPECIAL_STDIO_CHECKERS.get(question_id)
    if special_checker is None:
        sample = evaluation_sample(row)
        results, metadata = testing_util.run_test(
            sample,
            test=prediction,
            timeout=max(1, int(args.test_timeout)),
        )
    else:
        source_metadata = json.loads(row.get("metadata") or "{}")
        if source_metadata.get("func_name") is not None:
            raise ValueError(
                f"Special checker {question_id} only supports standard-input tasks"
            )
        cases = decode_cases(row["public_test_cases"]) + decode_cases(
            row["private_test_cases"]
        )
        results, metadata = run_stdio_with_special_checker(
            testing_util,
            prediction,
            cases,
            question_id,
            special_checker,
            max(1, int(args.test_timeout)),
        )
    results = list(results or [])
    passed = bool(results) and all(
        result is True or result == True for result in results
    )
    print(
        json.dumps(
            {
                "passed": bool(passed),
                "results": [
                    bool(result) if result is True or result == True else int(result)
                    for result in results
                ],
                "metadata": metadata,
                "question_id": row.get("question_id"),
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
