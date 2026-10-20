"""Pull every function out of a set of repos (read-only) using tree-sitter."""
import hashlib
import json
import os
import re
import sys

from tree_sitter_language_pack import get_parser

LANGS = {".py": "python", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".jsx": "javascript", ".rs": "rust"}
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "dist", "build", "out", "target", "coverage", "vendor", "__pycache__", ".next"}
FUNC_NODES = {"function_declaration", "function_definition", "method_definition", "function_item", "generator_function_declaration"}
MIN_LINES, MAX_CHARS = 5, 6000

def _exclude_patterns():
    """Paths never to index or show: DEJAVU_EXCLUDE (comma-separated) plus lines in $DEJAVU_HOME/exclude."""
    patterns = os.environ.get("DEJAVU_EXCLUDE", "").split(",")
    try:
        patterns += open(os.path.join(os.path.expanduser(os.environ.get("DEJAVU_HOME", "~/.dejavu")), "exclude")).read().splitlines()
    except OSError:
        pass
    return [p.strip().lower() for p in patterns if p.strip()]


EXCLUDE = _exclude_patterns()


def excluded(path):
    return any(p in path.lower() for p in EXCLUDE)


def source_files(repo):
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith((".", "dist"))]
        names = set(files)
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext not in LANGS or f.endswith((".d.ts", ".min.js")):
                continue
            # compiled output: foo.js next to foo.ts
            if ext == ".js" and (stem + ".ts" in names or stem + ".tsx" in names):
                continue
            path = os.path.join(root, f)
            if excluded(path):
                continue
            if os.path.getsize(path) < 300_000:
                yield path, LANGS[ext]


def name_of(node, src):
    n = node.child_by_field_name("name")
    if n is None and node.parent is not None and node.parent.type == "variable_declarator":
        n = node.parent.child_by_field_name("name")
    return src[n.start_byte:n.end_byte].decode(errors="replace") if n is not None else None


def normalize(code, lang):
    """Strip comments and whitespace so verbatim copies hash the same."""
    code = re.sub(r"#.*" if lang == "python" else r"//.*|/\*[\s\S]*?\*/", "", code)
    return re.sub(r"\s+", "", code)


def functions(path, lang):
    src = open(path, "rb").read()
    tree = get_parser(lang).parse(src)
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        is_func = node.type in FUNC_NODES or (
            node.type in ("arrow_function", "function_expression", "function")
            and node.parent is not None and node.parent.type == "variable_declarator")
        if not is_func:
            continue
        lines = node.end_point[0] - node.start_point[0] + 1
        name = name_of(node, src)
        if lines < MIN_LINES or not name:
            continue
        code = src[node.start_byte:node.end_byte].decode(errors="replace")[:MAX_CHARS]
        yield {"name": name, "line": node.start_point[0] + 1, "lines": lines, "lang": lang, "code": code,
               "hash": hashlib.sha1(normalize(code, lang).encode()).hexdigest()}


def main(repos, out):
    n = 0
    with open(out, "w") as f:
        for repo in repos:
            repo = os.path.abspath(os.path.expanduser(repo))
            for path, lang in source_files(repo):
                try:
                    for fn in functions(path, lang):
                        fn.update(repo=os.path.basename(repo), file=os.path.relpath(path, repo))
                        f.write(json.dumps(fn) + "\n")
                        n += 1
                except Exception as e:  # one bad file shouldn't stop the scan
                    print("skip", path, e, file=sys.stderr)
    print(n, "functions ->", out)


if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
