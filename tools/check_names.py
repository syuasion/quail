"""Check that the code uses one name for each concept and defines it once.

The check parses every module and reports four kinds of problem. Each
is a place where a reader meets two names for one thing, or one name
for two things.

- duplicate: two functions with the same body, in any modules. One
  should import the other (DataFusion keeps one `arrow_cast`; Spark
  keeps one `TreeNode.transform`).
- renamed: a parameter passed on under another keyword, as in
  `g(shared=resident)`. The caller and the callee should agree on
  the name, as Spark's `SparkSession.sql(sqlText)` is `sqlText` down
  to the parser.
- synonym: two identifiers that differ only by a word from one synonym
  group, such as `n_docs` and `num_documents`. The words below are
  the chosen ones; the others are not used.
- mistyped: a variable named for a type it does not hold, such as
  `table = f()` where `f` returns Rows.

    uv run python tools/check_names.py [paths...]

Exits 1 when any problem is found. `--inventory` instead prints what
the code defines, for reading by hand: each name with the modules
that define it (a name defined in several places may mean several
things), each word the identifiers use with how often (a word used
once is often a synonym of a common one), and each module-level
function with how many places call it (one caller is a candidate to
inline).
"""

from __future__ import annotations

import argparse
import ast
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_PATHS = ("calibrate", "quail", "bench", "tools")

# chosen word -> the words it replaces; an identifier's words are its
# snake_case parts, so `doc_ids` and `document_ids` are one synonym pair
SYNONYMS = {
    "document": ("doc",),
    "documents": ("docs",),
    "index": ("idx",),
    "position": ("pos",),
    "num": ("n", "number"),
    "config": ("cfg", "configuration"),
    "seconds": ("secs", "sec"),
    "token": ("tok",),
    "tokens": ("toks",),
    "estimate": ("est",),
    "count": ("cnt",),
    "gpus": ("gpu_count",),
    "survival": ("expected_survival",),
    "ports": ("frontier",),
    "run": ("arm",),
}
# whole identifiers that name a concept the project calls something else
RENAMED = {
    "live": "expected_rows",
    "expected_inputs": "expected_rows",
    "expected_docs": "expected_rows",
    "expected_tuples": "expected_rows",
    "written_pos": "written_position",
    "exec_idx": "exec_index",
}
# a count is `num_<things>`, never `<thing>_count`
COUNT_SUFFIX = re.compile(r"^(_*)(\w+)_count$")

# a type's name -> the variable names that may hold it
TYPE_NAMES = {
    "Rows": {"rows", "scored", "labels", "answers", "prior"},
    "Table": {"table", "batch", "relation"},
}
# method names Arrow tables share with Rows, so a call says nothing
GENERIC_METHODS = {"filter", "take", "concat", "table", "select", "slice"}

IGNORED_NAMES = {
    "__init__", "__post_init__", "__repr__", "__eq__", "__hash__", "__len__",
    "__iter__", "__call__", "__enter__", "__exit__", "execute", "run", "main",
    "close", "name", "kind", "type_name", "explain_fields", "to_dict",
    "from_dict", "node", "nodes", "spec",
}


def _modules(paths):
    for path in paths:
        root = Path(path)
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for file in files:
            yield file, ast.parse(file.read_text(), filename=str(file))


def _functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _body_key(function) -> str:
    """The function's body with its docstring removed, as a string."""
    body = function.body
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return "\n".join(ast.dump(statement) for statement in body)


def duplicates(modules) -> list[str]:
    """Report functions whose bodies are the same statement for statement."""
    bodies = defaultdict(list)
    for file, tree in modules:
        for function in _functions(tree):
            key = _body_key(function)
            if len(key) > 400 and "NotImplementedError" not in key:
                bodies[key].append(f"{file}:{function.lineno} {function.name}")
    return [f"duplicate: {' and '.join(places)}"
            for places in bodies.values() if len(places) > 1]


def _defined(modules) -> set[str]:
    """The function names this code defines, apart from names too common to resolve."""
    return {function.name for _, tree in modules
            for function in _functions(tree)} - IGNORED_NAMES


def _callee(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    return getattr(function, "id", None)


def renamed(modules) -> list[str]:
    """Report a parameter passed on under another keyword to a function here."""
    defined = _defined(modules)
    out = []
    for file, tree in modules:
        for function in _functions(tree):
            parameters = {argument.arg for argument in itertools.chain(
                function.args.posonlyargs, function.args.args,
                function.args.kwonlyargs)}
            passed = defaultdict(set)
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or _callee(node) not in defined:
                    continue
                for keyword in node.keywords:
                    value = keyword.value
                    if (isinstance(value, ast.Name) and value.id in parameters
                            and keyword.arg is not None
                            and keyword.arg != value.id
                            and value.id not in ("self", "cls")):
                        passed[value.id].add((node.lineno, keyword.arg))
            for parameter, places in passed.items():
                if len({keyword for _, keyword in places}) > 1:
                    continue
                for lineno, keyword in sorted(places):
                    out.append(f"renamed: {file}:{lineno} {function.name} "
                               f"passes {parameter} as {keyword}")
    return out


def _canonical(identifier: str) -> str:
    if identifier in RENAMED:
        return RENAMED[identifier]
    counted = COUNT_SUFFIX.match(identifier)
    if counted:
        return f"{counted.group(1)}num_{counted.group(2)}"
    prefix = identifier[:len(identifier) - len(identifier.lstrip("_"))]
    suffix = identifier[len(identifier.rstrip("_")):]
    words = identifier.strip("_").split("_")
    if len(words) == 1:
        return identifier
    replaced = []
    for word in words:
        for chosen, others in SYNONYMS.items():
            if word in others:
                word = chosen
                break
        replaced.append(word)
    return prefix + "_".join(replaced) + suffix


def _identifiers(tree, defined=frozenset()):
    """The names the module binds, and the keywords of calls to functions here."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node.lineno
        elif isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id, node.lineno
        elif isinstance(node, ast.Call) and _callee(node) in defined:
            for keyword in node.keywords:
                if keyword.arg:
                    yield keyword.arg, node.lineno
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            yield node.attr, node.lineno


# names SQL fixes, kept as SQL spells them
SQL_NAMES = frozenset({"row_number"})


def synonyms(modules) -> list[str]:
    """Report identifiers that use a word the project does not use."""
    defined = _defined(modules)
    out = []
    for file, tree in modules:
        seen = set()
        for identifier, lineno in _identifiers(tree, defined):
            if identifier in seen or identifier.startswith("__") \
                    or identifier in SQL_NAMES:
                continue
            seen.add(identifier)
            canonical = _canonical(identifier)
            if canonical != identifier:
                out.append(f"synonym: {file}:{lineno} {identifier} "
                           f"(the project says {canonical})")
    return out


def _returns(modules) -> dict[str, str]:
    """Function name -> the class its return annotation names, when one."""
    out = {}
    for _, tree in modules:
        for function in _functions(tree):
            annotation = function.returns
            if isinstance(annotation, ast.Attribute):
                annotation = ast.Name(annotation.attr)
            if isinstance(annotation, ast.Name) and annotation.id in TYPE_NAMES \
                    and function.name not in GENERIC_METHODS:
                out[function.name] = annotation.id
    return out


def mistyped(modules) -> list[str]:
    """Report a variable whose name claims a type its value does not have."""
    returns = _returns(modules)
    allowed = {name for names in TYPE_NAMES.values() for name in names}
    out = []
    for file, tree in modules:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)):
                continue
            name = _callee(node.value)
            variable = node.targets[0].id
            if name in returns and variable in allowed \
                    and variable not in TYPE_NAMES[returns[name]]:
                out.append(f"mistyped: {file}:{node.lineno} {variable} holds "
                           f"the {returns[name]} {name}() returns")
    return out


def inventory(modules) -> None:
    """Print the names, the words, and the call counts, for reading by hand."""
    places = defaultdict(list)
    words = defaultdict(int)
    callers = defaultdict(set)
    for file, tree in modules:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                places[node.name].append(f"{file}:{node.lineno}")
            elif isinstance(node, ast.Call) and _callee(node):
                callers[_callee(node)].add(str(file))
        for identifier, _ in _identifiers(tree):
            for word in identifier.strip("_").split("_"):
                if word:
                    words[word] += 1
    print("# names defined in more than one module")
    for name, where in sorted(places.items()):
        files = {place.rsplit(":", 1)[0] for place in where}
        if len(files) > 1 and name not in IGNORED_NAMES \
                and not name.startswith("__"):
            print(f"{name}: {' '.join(where)}")
    print("\n# words by frequency, rarest first")
    for word, count in sorted(words.items(), key=lambda item: (item[1], item[0])):
        print(f"{count:5d} {word}")
    print("\n# module-level functions by number of calling modules")
    top_level = {function.name: f"{file}:{function.lineno}"
                 for file, tree in modules for function in tree.body
                 if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name, place in sorted(top_level.items(),
                              key=lambda item: len(callers[item[0]])):
        print(f"{len(callers[name]):3d} {name} {place}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS))
    parser.add_argument("--inventory", action="store_true",
                        help="print the symbols instead of checking them")
    arguments = parser.parse_args(argv)
    modules = list(_modules(arguments.paths))
    if arguments.inventory:
        inventory(modules)
        return 0
    problems = [*duplicates(modules), *renamed(modules), *synonyms(modules),
                *mistyped(modules)]
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} naming problems", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
