"""Judge bake-off: does a second model beat EmbeddingGemma 2 alone at "is this the same function?"

Run one judge per process so memory is freed between them, then report:
    python eval/judges.py pairs            # build the shared pair set + embedding scores
    python eval/judges.py laya|coreb|clef|pplx
    python eval/judges.py report

Every judge sees the same pairs and, where it takes one, the same question. Thresholds are the judge's own
natural point (P(yes) = 0.5), never tuned on these pairs.

Keep rule (written before running): a judge is kept only if
  (a) "embedding >= 0.91, then judge says yes" has fewer false alarms than embedding alone at 0.946,
      with equal or better recall,
  (b) "embedding >= 0.946 and judge says yes" loses no true drift copy, and
  (c) it takes <= 2 s per pair on this laptop.
"""
import json
import math
import os
import random
import re
import statistics
import sys
import time

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
HERE = os.path.dirname(__file__)
PAIRS = os.path.join(DATA, "judge_pairs.json")
QUESTION = ("Do function A and function B implement the same behaviour, "
            "so that one is a copy, variant or rewrite of the other?")
EMB_WARN, EMB_HINT = 0.946, 0.91  # from the feasibility test (similarity prompt, 256d)


def state(a, b):
    return f"Function A:\n```\n{a[:3000]}\n```\n\nFunction B:\n```\n{b[:3000]}\n```"


def build_pairs():
    import torch
    from sentence_transformers import SentenceTransformer
    funcs = [json.loads(l) for l in open(os.path.join(DATA, "functions.jsonl"))]
    unique = {}
    for f in funcs:
        unique.setdefault(f["hash"], f)
    corpus = list(unique.values())
    index = {(f["repo"], f["file"], f["name"], f["line"]): i for i, f in enumerate(corpus)}
    E = np.load(os.path.join(DATA, "emb_similarity.npy"))[:, :256]
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    cos = lambda a, b: float(E[index[(a["repo"], a["file"], a["name"], a["line"])]] @
                             E[index[(b["repo"], b["file"], b["name"], b["line"])]])
    pairs = []
    for l in json.load(open(os.path.join(DATA, "labels.json"))):
        pairs.append(dict(kind="drift" if l["label"] == "P" else "hard_neg", label=int(l["label"] == "P"),
                          a=l["a"]["code"], b=l["b"]["code"], name=l["a"]["name"], cos=cos(l["a"], l["b"])))
    model = SentenceTransformer("google/embeddinggemma-2", device="mps",
                                config_kwargs={"vision_config": None, "audio_config": None},
                                model_kwargs={"torch_dtype": torch.float32})
    by_target = {(f["name"], f["file"]): f for f in corpus}
    rewrites = json.load(open(os.path.join(HERE, "rewrites.json")))
    R = model.encode([r["code"] for r in rewrites], prompt_name="SentenceSimilarity", normalize_embeddings=True)[:, :256]
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    for r, v in zip(rewrites, R):
        orig = by_target[tuple(r["target"])]
        i = index[(orig["repo"], orig["file"], orig["name"], orig["line"])]
        pairs.append(dict(kind="rewrite", label=1, a=r["code"], b=orig["code"], name=r["target"][0], cos=float(v @ E[i])))
    random.seed(42)
    pool = [f for f in corpus if not any(x and x in (f["repo"] + f["file"]).lower() for x in os.environ.get("DEJAVU_EXCLUDE", "").lower().split(","))]
    while sum(p["kind"] == "random" for p in pairs) < 60:
        a, b = random.sample(pool, 2)
        if a["name"] != b["name"]:
            pairs.append(dict(kind="random", label=0, a=a["code"], b=b["code"], name=f"{a['name']}~{b['name']}", cos=cos(a, b)))
    json.dump(pairs, open(PAIRS, "w"))
    print({k: sum(p["kind"] == k for p in pairs) for k in ("drift", "rewrite", "hard_neg", "random")})


# ---- judges: each returns a list of (score, p_yes) per pair; score is any monotone "same-ness" value ----

def judge_laya(pairs):
    from laya import Router
    router = Router(device="mps")
    q = {"same": {"type": "noul", "instructions": QUESTION}}
    out = []
    for p in pairs:
        y = float(router.predict(state(p["a"], p["b"]), q, model="multilingual", max_len=2048)["answers"]["same"]["noul"])
        out.append((logit(y), y))
    return out


def judge_coreb(pairs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "hq-bench/coreb-code-reranker"
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, trust_remote_code=True).to("mps").eval()
    prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct '
              'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    # The CoREB card omits the empty think block, but its base (Qwen3-Reranker) needs it: without it the next token
    # is "<think>" and the yes/no margin is noise (add() vs an unrelated function scored P(yes)=0.998).
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    task = "Given a code snippet, retrieve code that is semantically equivalent or solves the same task."
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    out = []
    for p in pairs:
        prompt = f"{prefix}<Instruct>: {task}\n<Query>: {p['a'][:3000]}\n<Document>: {p['b'][:3000]}{suffix}"
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to("mps")
        with torch.no_grad():
            lg = model(**ids).logits[0, -1, :].float()
        m = float(lg[yes] - lg[no])
        out.append((m, 1 / (1 + math.exp(-m))))
    return out


def judge_clef(pairs):
    import urllib.request
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11435") + "/v1/systemone"
    out = []
    for p in pairs:
        body = json.dumps({"model": "clef-flash", "state": state(p["a"], p["b"]), "keep_alive": "10m",
                           "questions": {"same": {"type": "noul", "instructions": QUESTION}}}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        y = float(json.load(urllib.request.urlopen(req, timeout=300))["answers"]["same"]["noul"])
        out.append((logit(y), y))
    return out


def judge_pplx(pairs):
    import torch
    from huggingface_hub import snapshot_download
    path = snapshot_download("perplexity-ai/pplx-decider-flash-v1.2")
    sys.path.insert(0, os.path.join(DATA, "pplx_source", "src"))  # patched copy, see report notes
    from pplx_decider import Decider
    decider = Decider.from_pretrained(path, device="mps")
    out = []
    for p in pairs:
        req = {"state": state(p["a"], p["b"]), "questions": {"same": {"type": "noul", "instructions": QUESTION}}}
        false, true = decider.logits([req])[0]["same"]  # noul options are ordered [false, true]
        m = float(true - false)  # the card recommends this margin for ranking; P(yes) = 0.5 exactly at m = 0
        out.append((m, 1 / (1 + math.exp(-m))))
    return out


def logit(y, eps=1e-6):
    y = min(max(y, eps), 1 - eps)
    return math.log(y / (1 - y))


JUDGES = {"laya": judge_laya, "coreb": judge_coreb, "clef": judge_clef, "pplx": judge_pplx}


SANITY = [
    ("identical", "function add(a, b) {\n  return a + b;\n}", "function add(a, b) {\n  return a + b;\n}"),
    ("unrelated", "function add(a, b) {\n  return a + b;\n}",
     "async function readConfig(path) {\n  const raw = await fs.readFile(path, 'utf8');\n  return JSON.parse(raw);\n}"),
]


def run(name):
    pairs = json.load(open(PAIRS))
    drift = next(p for p in pairs if p["kind"] == "drift")
    checks = [dict(a=a, b=b) for _, a, b in SANITY] + [drift]
    res = JUDGES[name](checks)  # also warms the model up, excluded from timing
    for (label, *_), (score, y) in zip(SANITY + [("drift copy",)], res):
        print(f"  sanity {label:<11} score {score:8.3f}  P(yes) {y:.4f}")
    if any(math.isnan(x[0]) for x in res) or len({round(x[0], 4) for x in res}) == 1 or res[0][1] < 0.5 or res[1][1] >= 0.5:
        sys.exit(f"{name}: failed the sanity check, not running the full set")
    t0 = time.time()
    res = JUDGES[name](pairs)
    per = (time.time() - t0) / len(pairs)
    json.dump({"per_pair_s": per, "scores": res}, open(os.path.join(DATA, f"judge_{name}.json"), "w"))
    print(f"{name}: {len(pairs)} pairs, {per:.2f}s per pair")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def report():
    pairs = json.load(open(PAIRS))
    lab = np.array([p["label"] for p in pairs])
    kind = np.array([p["kind"] for p in pairs])
    cos = np.array([p["cos"] for p in pairs])
    labelled = kind != "random"
    band = labelled & (cos >= 0.90) & (cos < 0.955)

    def rule(warn):
        tp = int((warn & labelled & (lab == 1)).sum())
        fp = int((warn & labelled & (lab == 0)).sum())
        rnd = int((warn & (kind == "random")).sum())
        P = int((labelled & (lab == 1)).sum())
        return f"precision {tp / max(tp + fp, 1):.2f}  recall {tp / P:.2f}  ({tp} caught, {fp} false alarms, {rnd}/60 random flagged)"

    print(f"pairs: {int((lab[labelled] == 1).sum())} positives (drift+rewrites), {int((lab[labelled] == 0).sum())} hard negatives, "
          f"60 random; borderline band 0.90-0.955: {int((band & (lab == 1)).sum())} pos / {int((band & (lab == 0)).sum())} neg\n")
    rng = np.random.default_rng(0)

    def band_ci(score):
        pos, neg = score[band & (lab == 1)], score[band & (lab == 0)]
        boots = [auc(list(rng.choice(pos, len(pos))), list(rng.choice(neg, len(neg)))) for _ in range(1000)]
        return np.percentile(boots, [2.5, 97.5])

    print(f"{'signal':<22}{'AUC all':>9}{'AUC band':>10}{'band 95% CI':>15}{'random says same':>18}{'ties':>6}{'s/pair':>8}")
    lo, hi = band_ci(cos)
    print(f"{'EmbeddingGemma 2':<22}{auc(list(cos[labelled & (lab == 1)]), list(cos[labelled & (lab == 0)])):9.3f}"
          f"{auc(list(cos[band & (lab == 1)]), list(cos[band & (lab == 0)])):10.3f}{f'{lo:.2f}-{hi:.2f}':>15}{'n/a':>18}{'':>6}{'~0.02':>8}")
    rows = {}
    for name in JUDGES:
        f = os.path.join(DATA, f"judge_{name}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        s = np.array([x[0] for x in d["scores"]])
        y = np.array([x[1] for x in d["scores"]])
        rows[name] = y
        lo, hi = band_ci(s)
        ties = int(labelled.sum() - len(set(np.round(s[labelled], 4))))  # tied scores among labelled pairs blur AUC
        print(f"{name:<22}{auc(list(s[labelled & (lab == 1)]), list(s[labelled & (lab == 0)])):9.3f}"
              f"{auc(list(s[band & (lab == 1)]), list(s[band & (lab == 0)])):10.3f}{f'{lo:.2f}-{hi:.2f}':>15}"
              f"{(y[kind == 'random'] >= 0.5).mean():>18.0%}{ties:>6}{d['per_pair_s']:8.2f}")
    print("\nhook rules (judge thresholds fixed at P(yes)=0.5, not tuned; the 0.91 gate alone blocks all random pairs):")
    print(f"  {'embedding alone >= 0.946 (current)':<46}{rule(cos >= EMB_WARN)}")
    print(f"  {'embedding alone >= 0.91 (judge shortlist)':<46}{rule(cos >= EMB_HINT)}")
    drift_ok = (kind == "drift") & (cos >= EMB_WARN)
    for name, y in rows.items():
        print(f"  {f'embedding >= 0.91, then {name} says yes':<46}{rule((cos >= EMB_HINT) & (y >= 0.5))}")
        print(f"  {f'embedding >= 0.946 and {name} says yes':<46}{rule((cos >= EMB_WARN) & (y >= 0.5))}"
              f"  [drift copies vetoed: {int((drift_ok & (y < 0.5)).sum())}]")


def crossval():
    """Leave-one-out: pick each cutoff on the other 52 labelled pairs (best F1), score the held-out pair.
    Applies equally to the embedding's own cutoff, which was also chosen on these pairs."""
    pairs = [p for p in json.load(open(PAIRS))]
    keep = [i for i, p in enumerate(pairs) if p["kind"] != "random"]
    lab = np.array([pairs[i]["label"] for i in keep])
    cos = np.array([pairs[i]["cos"] for i in keep])

    def best_cut(score, y, gate):
        cands = sorted(set(score[gate]))
        f1 = lambda t: (lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1))(
            ((score >= t) & gate & (y == 1)).sum(), ((score >= t) & gate & (y == 0)).sum(), (~((score >= t) & gate) & (y == 1)).sum())
        return max(cands, key=f1) if cands else math.inf

    def loo(score, gate):
        pred = np.zeros(len(lab), bool)
        for i in range(len(lab)):
            m = np.arange(len(lab)) != i
            pred[i] = gate[i] and score[i] >= best_cut(score[m], lab[m], gate[m])
        tp, fp = int((pred & (lab == 1)).sum()), int((pred & (lab == 0)).sum())
        return f"precision {tp / max(tp + fp, 1):.2f}  recall {tp / (lab == 1).sum():.2f}  ({tp} caught, {fp} false alarms)"

    print("\nleave-one-out cutoffs (honest estimate when a cutoff has to be tuned):")
    print(f"  {'embedding alone, cutoff tuned':<46}{loo(cos, np.ones(len(lab), bool))}")
    for name in JUDGES:
        f = os.path.join(DATA, f"judge_{name}.json")
        if os.path.exists(f):
            s = np.array([x[0] for x in json.load(open(f))["scores"]])[keep]
            print(f"  {f'embedding >= 0.91, then {name}, cutoff tuned':<46}{loo(s, cos >= EMB_HINT)}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(), crossval()
        sys.exit()
    cmd = sys.argv[1]
    build_pairs() if cmd == "pairs" else report() if cmd == "report" else run(cmd)
