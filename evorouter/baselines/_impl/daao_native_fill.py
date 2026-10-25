from __future__ import annotations
import ast, re, runpy, sys
from typing import Any, Dict, Optional, Union
from .prepare import ROOT

UP = ROOT / "upstream/DAAO"


def original_fill_node():
    sys.path.insert(0, str(ROOT / "deps"))
    sanitize = runpy.run_path(str(UP / "daao/utils/sanitize.py"))["sanitize"]
    path = UP / "daao/actions/action_node.py"
    tree = ast.parse(path.read_text())
    cls = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActionNode")
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in ["single_fill", "xml_fill", "xml_compile", "code_fill"]
    ]
    node = ast.ClassDef(
        name="OriginalFillNode", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    ast.fix_missing_locations(node)
    ns = dict(
        re=re,
        Optional=Optional,
        Union=Union,
        Dict=Dict,
        Any=Any,
        USE_CONFIG_TIMEOUT=600,
        sanitize=sanitize,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["OriginalFillNode"]


def build_fill(api, current):
    NativeNode = original_fill_node()

    async def fill(self, op_class, prompt, mode=None, **kw):
        fields = op_class.model_fields
        key = next(iter(fields))
        node = NativeNode()
        node.get_field_name = lambda: key
        node.get_field_names = lambda: list(fields)
        node.get_field_types = lambda: {k: v.annotation for k, v in fields.items()}

        class Transport:

            async def aask(_, text, **kwargs):
                return api.solve(
                    current.get(),
                    self.llm.model,
                    text,
                    system="You are a helpful assistant.",
                )

        node.llm = Transport()
        if mode == "single_fill":
            return await node.single_fill(prompt)
        if mode == "code_fill":
            return await node.code_fill(prompt, function_name=kw.get("function_name"))
        if mode == "xml_fill":
            formatted = node.xml_compile(prompt)
            for attempt in range(3):
                result = await node.xml_fill(formatted)
                if result.get(key):
                    return result
                if attempt == 2:
                    raise ValueError("DAAO XML format failed after two corrections")
                formatted += "\nThe previous output missed the required XML tag. Follow the displayed XML format."
        raise ValueError("Unsupported original DAAO fill mode: " + str(mode))

    return fill
