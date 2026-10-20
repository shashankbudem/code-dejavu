"""Feasibility test: can EmbeddingGemma 2 (+ optionally Laya) tell real duplicates from look-alikes?

Pass rule, fixed before running: some cutoff gives >=90% precision on labelled positives vs hard negatives
with >=60% recall, and that cutoff sits above the 99th percentile of random pairs.
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
HERE = os.path.dirname(__file__)
random.seed(0)

funcs = [json.loads(l) for l in open(os.path.join(DATA, "functions.jsonl"))]
unique = {}
for f in funcs:
    unique.setdefault(f["hash"], f)
corpus = list(unique.values())
print(f"stage 0: {len(funcs)} functions, {len(corpus)} unique bodies, "
      f"{len(funcs) - len(corpus)} exact copies caught by hashing alone (no model needed)")

labels = json.load(open(os.path.join(DATA, "labels.json")))
rewrites = json.load(open(os.path.join(HERE, "rewrites.json")))
key = lambda f: (f["repo"], f["file"], f["name"], f["line"])
index = {key(f): i for i, f in enumerate(corpus)}
by_target = {(f["name"], f["file"]): f for f in corpus}
for r in rewrites:
    r["orig"] = by_target[tuple(r["target"])]

random_pairs = []
while len(random_pairs) < 500:
    a, b = random.sample(corpus, 2)
    if a["name"] != b["name"]:
        random_pairs.append((a, b))

model = SentenceTransformer("google/embeddinggemma-2", device="mps",
                            config_kwargs={"vision_config": None, "audio_config": None},
                            model_kwargs={"torch_dtype": torch.float32})
model.max_seq_length = 1024

VARIANTS = {
    "similarity": lambda f: dict(prompt_name="SentenceSimilarity"),
    "titled-doc": lambda f: dict(prompt=f"title: {f.get('file', f.get('name', 'none'))} | text: "),
}


def embed(texts_and_meta, variant):
    # Prompts differ per item only for titled-doc, so encode one by one there; batch otherwise.
    if variant == "similarity":
        return model.encode([t["code"] for t in texts_and_meta], prompt_name="SentenceSimilarity",
                            normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    return np.stack([model.encode(t["code"], prompt=f"title: {t.get('file', 'none')} | text: ",
                                  normalize_embeddings=True) for t in texts_and_meta])


def shrink(e, dim):
    e = e[:, :dim]
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def cutoff_report(pos, neg, rand):
    """Best cutoff under the pass rule; returns (cutoff, precision, recall, passed)."""
    p99 = float(np.percentile(rand, 99)) if len(rand) else -1
    best = None
    for t in sorted(set(pos) | set(neg)):
        if t <= p99:
            continue
        tp = sum(s >= t for s in pos)
        fp = sum(s >= t for s in neg)
        if tp == 0:
            continue
        prec, rec = tp / (tp + fp), tp / len(pos)
        if prec >= 0.9 and (best is None or rec > best[2]):
            best = (t, prec, rec)
    if best is None:
        return p99, None, None, False
    return best[0], best[1], best[2], best[2] >= 0.6


def describe(name, xs):
    xs = np.array(xs)
    return f"{name:<16} n={len(xs):<4} min={xs.min():.3f} p10={np.percentile(xs,10):.3f} median={np.median(xs):.3f} p90={np.percentile(xs,90):.3f} max={xs.max():.3f}"


results = {}
for variant in VARIANTS:
    t0 = time.time()
    cache = os.path.join(DATA, f"emb_{variant}.npy")
    if os.path.exists(cache):
        E = np.load(cache)
    else:
        E = embed(corpus, variant)
        np.save(cache, E)
    R = embed([{"code": r["code"], "file": "untitled"} for r in rewrites], variant)
    print(f"\n=== variant {variant} (embedded {len(corpus)} functions in {time.time()-t0:.0f}s)")
    for dim in (768, 256):
        Ed, Rd = shrink(E, dim), shrink(R, dim)
        sim = lambda a, b: float(Ed[index[key(a)]] @ Ed[index[key(b)]])
        drift_pos = [sim(l["a"], l["b"]) for l in labels if l["label"] == "P"]
        hard_neg = [sim(l["a"], l["b"]) for l in labels if l["label"] == "N"]
        rw = [float(Rd[i] @ Ed[index[key(r["orig"])]]) for i, r in enumerate(rewrites)]
        rand = [sim(a, b) for a, b in random_pairs]
        # retrieval: where does the original rank when the rewrite searches the whole corpus?
        ranks = []
        for i, r in enumerate(rewrites):
            scores = Ed @ Rd[i]
            ranks.append(int((scores > scores[index[key(r['orig'])]]).sum()) + 1)
        t, prec, rec, ok = cutoff_report(drift_pos + rw, hard_neg, rand)
        results[(variant, dim)] = dict(t=t, prec=prec, rec=rec, ok=ok, pos=drift_pos + rw, neg=hard_neg, rand=rand)
        print(f"-- dim {dim}")
        for n, xs in (("drift copies", drift_pos), ("rewrites", rw), ("hard negatives", hard_neg), ("random pairs", rand)):
            print("  " + describe(n, xs))
        print(f"  rewrite -> original rank among {len(corpus)}: {ranks}  (top1 {sum(r==1 for r in ranks)}/{len(ranks)}, top5 {sum(r<=5 for r in ranks)}/{len(ranks)})")
        print(f"  cutoff {t:.3f}: precision {prec if prec is None else round(prec,3)}, recall {rec if rec is None else round(rec,3)} -> {'PASS' if ok else 'fail'}")
        missed = [l["a"]["name"] for l in labels if l["label"] == "P" and sim(l["a"], l["b"]) < t] + \
                 [r["target"][0] + " (rewrite)" for i, r in enumerate(rewrites) if rw[i] < t]
        caught = [l["a"]["name"] for l in labels if l["label"] == "N" and sim(l["a"], l["b"]) >= t]
        print(f"  missed positives: {missed}\n  false alarms: {caught}")

# ---- Laya as a second-stage judge on the same pairs ----
if "--laya" in sys.argv:
    from laya import Router
    router = Router(device="mps")
    q = {"same": {"type": "noul", "instructions": "Do function A and function B implement the same behaviour, "
                  "so that one is a copy, variant or rewrite of the other?"}}

    def judge(a, b):
        state = f"Function A:\n{a[:2500]}\n\nFunction B:\n{b[:2500]}"
        return float(router.predict(state, q, model="multilingual", max_len=2048)["answers"]["same"]["noul"])

    t0 = time.time()
    lp = [judge(l["a"]["code"], l["b"]["code"]) for l in labels if l["label"] == "P"] + \
         [judge(r["code"], r["orig"]["code"]) for r in rewrites]
    ln = [judge(l["a"]["code"], l["b"]["code"]) for l in labels if l["label"] == "N"]
    lr = [judge(a["code"], b["code"]) for a, b in random_pairs[:60]]
    n = len(lp) + len(ln) + len(lr)
    print(f"\n=== Laya (multilingual, noul) on {n} pairs, {(time.time()-t0)/n:.2f}s per pair")
    for name, xs in (("positives", lp), ("hard negatives", ln), ("random pairs", lr)):
        print("  " + describe(name, xs))
    t, prec, rec, ok = cutoff_report(lp, ln, lr)
    print(f"  Laya alone: cutoff {t:.3f} precision {prec} recall {rec} -> {'PASS' if ok else 'fail'}")
    # AUC of each signal on positives vs hard negatives
    auc = lambda P, N: np.mean([p > n_ for p in P for n_ in N]) + 0.5 * np.mean([p == n_ for p in P for n_ in N])
    best = max(results, key=lambda k: (results[k]["ok"], results[k]["rec"] or 0))
    print(f"  AUC pos vs hard-neg: Laya {auc(lp, ln):.3f} | embedding {best} {auc(results[best]['pos'], results[best]['neg']):.3f}")

passed = [k for k, v in results.items() if v["ok"]]
print("\nGO" if passed else "\nNO-GO", "- passing configs:", passed or "none")
