"""Describe-a-function search: does EmbeddingGemma 2 find the right function from plain English?

Queries avoid the function's own name. A hit is any function whose name is in the answer set.
Baseline: keyword overlap weighted by rarity (roughly what grep gets you).
"""
import json
import math
import os
import re
from collections import Counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
QUERIES = [
    ("show a file size like 12 KB or 3 MB", {"formatBytes", "formatByteSize"}),
    ("make user text safe to put inside a web page by replacing angle brackets, ampersands and quotes", {"escapeHtml"}),
    ("when one cancellation signal fires, cancel another controller too, and allow unhooking later", {"linkAbortSignal"}),
    ("make the canvas match the window size on high-DPI screens", {"resize"}),
    ("ask a yes or no question in the terminal and wait for the answer", {"confirm"}),
    ("pick a server name that is not taken yet by adding a number at the end", {"uniqueMcpServerName"}),
    ("which permission does a given approval request need", {"capabilityForApprovalAction"}),
    ("keep a running total of prompt and completion tokens across requests", {"recordRunUsage"}),
    ("move big pasted images out of the chat history into files named by their hash", {"externalizeAttachments"}),
    ("merge a partial settings update into the saved app settings", {"applyConfigPatch"}),
    ("sort states by an indicator, respecting whether lower values are better", {"ranked"}),
    ("generate a deterministic fake number for a region and year for the demo", {"sampleValue"}),
    ("verify an email address with retries and measure how long it took", {"check_email"}),
    ("start error monitoring using a DSN from the environment", {"setup_sentry"}),
    ("draw a filled polygon into a grayscale image and optionally grow it outward", {"_poly_mask", "_mask", "mask"}),
    ("pack PNG renders of several sizes into a Windows cursor file", {"build_cur"}),
    ("split a comma separated list of foods into quantity and name", {"parseFoods"}),
    ("next date for a weekday and start time, as an ISO timestamp", {"isoFromSlot"}),
    ("load a user's profile and fall back to their sign-up details if it is missing", {"fetchProfile"}),
    ("detect whether some text contains an API key or password", {"findSecret"}),
    ("turn a stored note into a one-line header with its tags and a shortened body", {"formatMemory"}),
    ("convert a database row back into a note object, tolerating broken tag data", {"rowToMemory"}),
]

funcs = [json.loads(l) for l in open(os.path.join(DATA, "functions.jsonl"))]
unique = {}
for f in funcs:
    unique.setdefault(f["hash"], f)
corpus = list(unique.values())
is_test = lambda f: bool(re.search(r"(^|/)(tests?|__tests__|e2e)/|\.test\.|\.spec\.|_test\.py", f["file"]))
keep = np.array([not is_test(f) and not any(x and x in (f["repo"] + f["file"]).lower() for x in os.environ.get("DEJAVU_EXCLUDE", "").lower().split(",")) for f in corpus])
names = [f["name"] for f in corpus]
print(f"searching {keep.sum()} functions (tests excluded)")


def report(label, ranked_lists):
    ranks = []
    for (q, answers), order in zip(QUERIES, ranked_lists):
        r = next((k + 1 for k, i in enumerate(order) if names[i] in answers), None)
        ranks.append(r)
    hit = lambda n: sum(r is not None and r <= n for r in ranks)
    mrr = np.mean([1 / r if r else 0 for r in ranks])
    print(f"{label:<28} top1 {hit(1):>2}/{len(QUERIES)}  top5 {hit(5):>2}/{len(QUERIES)}  top10 {hit(10):>2}/{len(QUERIES)}  MRR {mrr:.2f}")
    return ranks


# keyword baseline
tok = lambda s: set(t.lower() for t in re.findall(r"[A-Za-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", s)) if len(t) > 2)
docs = [tok(f["name"] + " " + f["code"]) for f in corpus]
df = Counter(t for d in docs for t in d)
idf = {t: math.log(len(docs) / c) for t, c in df.items()}
kw = []
for q, _ in QUERIES:
    qt = tok(q)
    scores = np.array([sum(idf[t] for t in qt & d) for d in docs])
    scores[~keep] = -1
    kw.append(np.argsort(-scores, kind="stable"))
kw_ranks = report("keyword match (grep-like)", kw)

model = SentenceTransformer("google/embeddinggemma-2", device="mps",
                            config_kwargs={"vision_config": None, "audio_config": None},
                            model_kwargs={"torch_dtype": torch.float32})
E = np.load(os.path.join(DATA, "emb_titled-doc.npy"))  # corpus encoded as "title: {file} | text: {code}"
Q = model.encode([f"task: code retrieval | query: {q}" for q, _ in QUERIES], normalize_embeddings=True)
for dim in (768, 256):
    Ed = E[:, :dim] / np.linalg.norm(E[:, :dim], axis=1, keepdims=True)
    Qd = Q[:, :dim] / np.linalg.norm(Q[:, :dim], axis=1, keepdims=True)
    S = Qd @ Ed.T
    S[:, ~keep] = -1
    ranks = report(f"EmbeddingGemma 2 ({dim}d)", [np.argsort(-s) for s in S])
    if dim == 256:
        for (q, ans), r, kr, s in zip(QUERIES, ranks, kw_ranks, S):
            top = names[int(np.argmax(s))]
            print(f"  rank {str(r):>4} (grep {str(kr):>4})  {q[:62]:<62} -> top: {top}")
