"""Flag long string literals that are not clearly content.

Ruff's line-length rule sees one line at a time, so a message built
from several adjacent literals, or a triple-quoted block, slips past
it. This check measures the whole literal after the parser joins
adjacent pieces. A literal over the limit is allowed only when it is
clearly content rather than code:

- it is the value (or inside the list, tuple, or dict value) of an
  assignment to a name containing PROMPT, TEMPLATE, SQL, QUERY, HTML,
  or TEXT, in any case;
- it lives in a prompts module or folder: any part of the file path
  contains "prompt" (experiments/prompts/x.py);
- it looks like HTML (starts with "<" and ends with ">") or SQL (starts
  with SELECT, WITH, INSERT, CREATE, or UPDATE).

Docstrings are exempt. An f-string counts only its literal text.

    uv run python tools/check_long_strings.py [--limit 200] [paths...]

Exits 1 when any other literal is over the limit. Long messages and
log lines should be shortened; long content should be named for what
it is.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

DEFAULT_PATHS = ("calibrate", "quail", "tests", "experiments", "tools")
DEFAULT_LIMIT = 200
CONTENT_NAME = re.compile(r"prompt|template|sql|query|html|text", re.I)
SQL_START = re.compile(r"^\s*(select|with|insert|create|update)\b", re.I)


def _docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            ids.add(id(body[0].value))
    return ids


def _target_names(node):
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
            elif isinstance(sub, ast.Attribute):
                names.append(sub.attr)
    return names


def _content_ids(tree):
    """Ids of literals assigned to a name that says they are content."""
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        if node.value is None:
            continue
        if not any(CONTENT_NAME.search(name) for name in _target_names(node)):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, (ast.Constant, ast.JoinedStr)):
                ids.add(id(sub))
    return ids


def _looks_like_content(text):
    stripped = text.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    return bool(SQL_START.match(stripped))


def long_strings(path, limit):
    """Yield (line, length) for every over-limit literal in one file."""
    if any("prompt" in part.lower() for part in path.parts):
        return
    tree = ast.parse(path.read_text(), filename=str(path))
    exempt = _docstring_ids(tree) | _content_ids(tree)
    for node in ast.walk(tree):
        if id(node) in exempt:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(part.value for part in node.values
                           if isinstance(part, ast.Constant))
        else:
            continue
        if len(text) > limit and not _looks_like_content(text):
            yield node.lineno, len(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="longest allowed literal, in characters")
    args = parser.parse_args(argv)
    files = []
    for entry in args.paths:
        entry = Path(entry)
        files.extend(sorted(entry.rglob("*.py")) if entry.is_dir()
                     else [entry])
    findings = 0
    for path in files:
        for line, length in long_strings(path, args.limit):
            findings += 1
            print(f"{path}:{line}: string literal of {length} characters "
                  f"(limit {args.limit}) is not named as content")
    if findings:
        print(f"{findings} string literals over {args.limit} characters",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
