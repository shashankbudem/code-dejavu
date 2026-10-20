"""Pull official MTEB(Code) per-task results for candidate embedders from embeddings-benchmark/results."""
import json, subprocess, sys
TASKS = ["AppsRetrieval","CodeEditSearchRetrieval","CodeFeedbackMT","CodeFeedbackST","CodeSearchNetCCRetrieval","CodeSearchNetRetrieval",
         "CodeTransOceanContest","CodeTransOceanDL","CosQA","COIRCodeSearchNetRetrieval","StackOverflowQA","SyntheticText2SQL"]
MODELS = {"EmbeddingGemma 2 (0.3B text)":"google__embeddinggemma-2","EmbeddingGemma 1 (0.3B)":"google__embeddinggemma-300m",
          "pplx-embed-v1 0.6B":"perplexity-ai__pplx-embed-v1-0.6b","pplx-embed-v1 4B":"perplexity-ai__pplx-embed-v1-4b",
          "Qwen3-Embedding 0.6B":"Qwen__Qwen3-Embedding-0.6B","Qwen3-Embedding 4B":"Qwen__Qwen3-Embedding-4B"}
def gh(path):
    r = subprocess.run(["gh","api",path], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None
out = {}
for label, d in MODELS.items():
    revs = gh(f"repos/embeddings-benchmark/results/contents/results/{d}") or []
    scores = {}
    for rev in revs:
        files = {f["name"] for f in gh(f"repos/embeddings-benchmark/results/contents/results/{d}/{rev['name']}") or []}
        for t in TASKS:
            if t in scores or f"{t}.json" not in files: continue
            raw = subprocess.run(["curl","-s","-m","20",f"https://raw.githubusercontent.com/embeddings-benchmark/results/main/results/{d}/{rev['name']}/{t}.json"],capture_output=True,text=True).stdout
            try:
                j = json.loads(raw); split = j["scores"].get("test") or next(iter(j["scores"].values()))
                scores[t] = sum(s["main_score"] for s in split)/len(split)
            except Exception as e: print("parse fail", d, t, e, file=sys.stderr)
    out[label] = scores
json.dump(out, open("data/mteb/code_scores.json","w"), indent=1)
