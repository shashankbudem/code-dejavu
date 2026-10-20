"""Code Déjà Vu: warns when the function you're about to commit already exists somewhere in your repos.

    dejavu index ~/code/app ~/code/lib     add repos and (re)build the index
    dejavu index                           refresh the index
    dejavu search "show a file size in KB"  find a function by describing it
    dejavu check                           what the pre-commit hook runs
    dejavu install-hook                    add the hook to the current repo

Everything runs locally with EmbeddingGemma 2; no code leaves the machine.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

import numpy as np

from extract import LANGS, excluded, functions, functions_in, source_files

HOME = os.path.expanduser(os.environ.get("DEJAVU_HOME", "~/.dejavu"))
MODEL = "google/embeddinggemma-2"
# Cutoffs come from eval/feasibility.py and only hold with these exact settings:
# "sentence similarity" prompt, first 256 dimensions re-normalised, fp32, 1024-token limit, 5+ line functions.
SIM_PREFIX = "task: sentence similarity | query: "
QUERY_PREFIX = "task: code retrieval | query: "
DIM, WARN, HINT = 256, 0.946, 0.91
TEST = re.compile(r"(^|/)(tests?|__tests__|e2e)/|\.test\.|\.spec\.|_test\.py")
HOOK_MARK = "# code-dejavu"
QUESTION = ("Do function A and function B implement the same behaviour, "
            "so that one is a copy, variant or rewrite of the other?")

_model = None


def embed(texts, prefix=""):
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu",
                                     config_kwargs={"vision_config": None, "audio_config": None},
                                     model_kwargs={"torch_dtype": torch.float32})
        _model.max_seq_length = 1024
    e = _model.encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=16,
                      show_progress_bar=len(texts) > 200)[:, :DIM]
    return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)


# ---- index: functions.jsonl + sim.npy (duplicate check) + doc.npy (search), rows aligned ----

def load_index():
    if not os.path.exists(f"{HOME}/functions.jsonl"):
        return [], np.zeros((0, DIM), np.float32), np.zeros((0, DIM), np.float32)
    meta = [json.loads(l) for l in open(f"{HOME}/functions.jsonl")]
    return meta, np.load(f"{HOME}/sim.npy"), np.load(f"{HOME}/doc.npy")


def build_index(repos):
    old_meta, old_sim, old_doc = load_index()
    sim_cache = {m["hash"]: old_sim[i] for i, m in enumerate(old_meta)}
    doc_cache = {(m["hash"], m["file"]): old_doc[i] for i, m in enumerate(old_meta)}  # title includes the path
    meta = []
    for repo in repos:
        for path, lang in source_files(repo):
            rel = os.path.relpath(path, repo)
            if TEST.search(rel):  # test stubs look like the real thing; they were most false alarms
                continue
            try:
                found = list(functions(path, lang))
            except Exception:
                continue
            meta += [dict(repo=repo, file=rel, name=f["name"], line=f["line"], hash=f["hash"], code=f["code"]) for f in found]
    new = {m["hash"]: m["code"] for m in meta if m["hash"] not in sim_cache}
    if new:
        print(f"embedding {len(new)} new functions for duplicate checks")
        sim_cache.update(zip(new, embed(list(new.values()), SIM_PREFIX)))
    new_doc = {(m["hash"], m["file"]): f"title: {m['file']} | text: {m['code']}" for m in meta
               if (m["hash"], m["file"]) not in doc_cache}
    if new_doc:
        print(f"embedding {len(new_doc)} functions for search")
        doc_cache.update(zip(new_doc, embed(list(new_doc.values()))))
    os.makedirs(HOME, exist_ok=True)
    stack = lambda vecs: np.stack(vecs) if vecs else np.zeros((0, DIM), np.float32)
    with open(f"{HOME}/functions.tmp.jsonl", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in meta)
    np.save(f"{HOME}/sim.tmp.npy", stack([sim_cache[m["hash"]] for m in meta]))
    np.save(f"{HOME}/doc.tmp.npy", stack([doc_cache[(m["hash"], m["file"])] for m in meta]))
    for name in ("functions.jsonl", "sim.npy", "doc.npy"):  # swap in together so the rows stay aligned
        stem, ext = os.path.splitext(name)
        os.replace(f"{HOME}/{stem}.tmp{ext}", f"{HOME}/{name}")
    print(f"indexed {len(meta)} functions from {len(repos)} repos -> {HOME}")


def registered_repos(add=()):
    path = f"{HOME}/repos.json"
    repos = json.load(open(path)) if os.path.exists(path) else []
    for r in add:
        r = os.path.realpath(os.path.expanduser(r))
        if r not in repos:
            repos.append(r)
    os.makedirs(HOME, exist_ok=True)
    json.dump(repos, open(path, "w"), indent=1)
    return repos


def where(m):
    return f"{os.path.basename(m['repo'])}/{m['file']}:{m['line']}"


def search(query, k=5):
    meta, _, doc = load_index()
    if not meta:
        return print("no index yet: run `dejavu index <repo> ...` first")
    scores = doc @ embed([query], QUERY_PREFIX)[0]
    for i in np.argsort(-scores)[:k]:
        print(f"{scores[i]:.2f}  {meta[i]['name']:<28} {where(meta[i])}")


# ---- check: compare the functions a commit adds or changes against the index ----

def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def staged(root):
    """(status, old_path, new_path) for every staged change; renames keep both paths."""
    parts, out, i = git(root, "diff", "--cached", "--name-status", "-M", "-z").decode().split("\0")[:-1], [], 0
    while i < len(parts):
        status = parts[i][0]
        if status in "RC":
            out.append((status, parts[i + 1], parts[i + 2]))
            i += 3
        else:
            out.append((status, parts[i + 1], parts[i + 1]))
            i += 2
    return out


def code_functions(root, spec, path):
    lang = LANGS.get(os.path.splitext(path)[1])
    if not lang or TEST.search(path) or excluded(path):
        return []
    try:
        return list(functions_in(git(root, "show", spec), lang))
    except subprocess.CalledProcessError:  # e.g. no HEAD yet
        return []


def second_opinion(a, b):
    """P(same) from a local decision model behind Ollama's /v1/systemone, or None if none answers."""
    url = os.environ.get("DEJAVU_DECIDER_URL", "http://127.0.0.1:11434") + "/v1/systemone"
    body = {"model": os.environ.get("DEJAVU_DECIDER_MODEL", "pplx-decider-flash"),
            "state": f"Function A:\n```\n{a[:3000]}\n```\n\nFunction B:\n```\n{b[:3000]}\n```",
            "questions": {"same": {"type": "noul", "instructions": QUESTION}}}
    try:
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return float(json.load(urllib.request.urlopen(req, timeout=60))["answers"]["same"]["noul"])
    except Exception:
        return None


def check(root, use_second_opinion=False):
    root = os.path.realpath(git(root, "rev-parse", "--show-toplevel").decode().strip())
    before, after, touched = set(), [], set()
    for status, old, new in staged(root):
        touched |= {os.path.join(root, old), os.path.join(root, new)}
        if status != "A":
            before |= {f["hash"] for f in code_functions(root, f"HEAD:{old}", old)}
        if status != "D":
            after += [dict(f, file=new) for f in code_functions(root, f":{new}", new)]
    # Unchanged and moved functions keep their hash; only genuinely new or edited code is checked.
    candidates = [f for f in after if f["hash"] not in before]
    if not candidates:
        return []
    meta, sim, _ = load_index()
    # The index's copies of files in this commit are being replaced, so they can't count as "already exists".
    keep = np.array([os.path.join(m["repo"], m["file"]) not in touched for m in meta], bool)
    rows = np.flatnonzero(keep) if len(meta) else np.zeros(0, int)
    exact = {meta[i]["hash"]: i for i in rows}
    findings, rest = [], []
    for f in candidates:
        if f["hash"] in exact:  # verbatim copy (ignoring comments and whitespace): no model needed
            findings.append(dict(f, level="warn", score=1.0, match=meta[exact[f["hash"]]], exact=True))
        else:
            rest.append(f)
    if rest and len(rows):
        scores = embed([f["code"] for f in rest], SIM_PREFIX) @ sim[rows].T
        for f, row in zip(rest, scores):
            j = int(np.argmax(row))
            m, s = meta[rows[j]], float(row[j])
            if s >= WARN:
                findings.append(dict(f, level="warn", score=s, match=m, exact=False))
            elif s >= HINT:
                p = second_opinion(f["code"], m["code"]) if use_second_opinion else None
                # The second opinion may only promote a hint: as a veto it rejected 13 of 19 real copies.
                findings.append(dict(f, level="warn" if p is not None and p >= 0.5 else "hint",
                                     score=s, match=m, exact=False, opinion=p))
    return findings


def report(findings):
    for f in findings:
        head = "déjà vu" if f["level"] == "warn" else "similar"
        how = "identical apart from comments/whitespace" if f.get("exact") else f"similarity {f['score']:.2f}"
        extra = f", second opinion {f['opinion']:.0%} same" if f.get("opinion") is not None else ""
        print(f"{head}: {f['name']} ({f['file']}:{f['line']})\n"
              f"   already in {where(f['match'])} as {f['match']['name']} ({how}{extra})")


def install_hook(root):
    root = git(root, "rev-parse", "--show-toplevel").decode().strip()
    hooks = os.path.join(root, git(root, "rev-parse", "--git-path", "hooks").decode().strip())  # honours core.hooksPath
    path = os.path.join(hooks, "pre-commit")
    line = f'"{sys.executable}" "{os.path.abspath(__file__)}" check || true'
    if os.path.exists(path) and HOOK_MARK not in open(path).read():
        return print(f"{path} already exists; add this line to it yourself:\n  {line}")
    os.makedirs(hooks, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"#!/bin/sh\n{HOOK_MARK}: never blocks a commit, only warns\n{line}\n")
    os.chmod(path, 0o755)
    print(f"installed {path}")


def main():
    p = argparse.ArgumentParser(prog="dejavu", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index").add_argument("repos", nargs="*")
    sub.add_parser("search").add_argument("query", nargs="+")
    c = sub.add_parser("check")
    c.add_argument("--second-opinion", action="store_true",
                   help="ask a local decision model (Ollama /v1/systemone) about borderline matches")
    sub.add_parser("install-hook")
    a = p.parse_args()
    if a.cmd == "index":
        repos = registered_repos(a.repos)
        if not repos:
            sys.exit("give at least one repo: dejavu index ~/code/my-repo")
        build_index(repos)
    elif a.cmd == "search":
        search(" ".join(a.query))
    elif a.cmd == "install-hook":
        install_hook(os.getcwd())
    else:
        try:  # a pre-commit hook that ever blocks gets uninstalled, so failures only print
            report(check(os.getcwd(), a.second_opinion or os.environ.get("DEJAVU_SECOND_OPINION") == "1"))
        except Exception as e:
            print(f"dejavu: skipped ({e})")


if __name__ == "__main__":
    main()
