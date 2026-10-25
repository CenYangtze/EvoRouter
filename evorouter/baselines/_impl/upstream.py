from __future__ import annotations
import ast
from pathlib import Path


def load_definitions(path, names, namespace):
    source = Path(path).read_text()
    tree = ast.parse(source)
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    if {n.name for n in nodes} != set(names):
        raise ValueError("Missing upstream definitions")
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace


def current_pool():
    from MAR.MasRouter.llm_coordinator_router import MODEL_POOL
    from .core import MODELS

    return [entry for entry in MODEL_POOL if entry["Name"] in MODELS]
