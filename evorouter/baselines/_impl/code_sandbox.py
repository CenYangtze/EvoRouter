import ast, json, resource, sys


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    code = json.load(sys.stdin)["code"]
    tree = ast.parse(code)
    modules = {"math", "fractions", "decimal", "statistics", "itertools"}
    banned = {
        "open",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "type",
        "object",
        "super",
        "help",
        "breakpoint",
        "input",
        "__import__",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attribute prohibited")
        if isinstance(node, ast.Name) and (
            node.id.startswith("__") and node.id != "__name__" or node.id in banned
        ):
            raise ValueError("Unsafe name")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
            )
            if any((n not in modules for n in names)):
                raise ValueError("Only arithmetic standard-library imports supported")
    import builtins

    allowed = [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "print",
        "ValueError",
        "Exception",
    ]

    def safe_import(name, *args, **kwargs):
        if name not in modules:
            raise ValueError("Disallowed import")
        return builtins.__import__(name, *args, **kwargs)

    ns = {
        "__name__": "sandbox",
        "__builtins__": {
            **{n: getattr(builtins, n) for n in allowed},
            "__import__": safe_import,
        },
    }
    exec(compile(tree, "<generated>", "exec"), ns)
    value = ns["solve"]()
    print(json.dumps({"status": "Success", "output": str(value)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"status": "Error", "output": str(e)}))
