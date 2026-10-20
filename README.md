# Code Déjà Vu

**The laziest code is code you've already written.**

Code Déjà Vu is a git pre-commit hook that notices when the function you're about to commit already exists somewhere in
your repos, under any name, in any of them. It also lets you find your own code by describing it. Everything runs on your
laptop with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), an open-weight model. Your code never
leaves the machine.

```
$ git commit -m "add helpers"
déjà vu: forwardCancellation (src/abort.ts:1)
   already in arivu-agent/src/agent/OpenAICompatibleChatClient.ts:979 as linkAbortSignal (similarity 0.93, second opinion 97% same)
déjà vu: humanFileSize (src/files.ts:1)
   already in arivu-agent/src/tools/registry.ts:1567 as formatByteSize (similarity 0.96)
[main 7db292a] add helpers
 3 files changed, 24 insertions(+)
```

The third new function in that commit, `isLeapYear`, really was new, so it stayed silent.

```
$ dejavu search "sort states by an indicator, lower is better"
0.88  ranked                     Bharath Choropleth/examples/observatory/react/src/App.tsx:109
0.85  ranked                     Bharath Choropleth/examples/observatory/js/app.js:75
0.79  trend                      Bharath Choropleth/examples/soc-dashboard/src/feeds.ts:260
...
```

## Install

Python 3.10+ on macOS (Apple GPU) or Linux. The first run downloads EmbeddingGemma 2 (about 1.5 GB).

```bash
git clone <this repo> && cd code-dejavu
pip install -e .
```

## Use

```bash
dejavu index ~/code/app ~/code/lib     # add repos and build the index (read-only; stored in ~/.dejavu)
dejavu index                           # refresh after you've written more code
dejavu search "retry with exponential backoff"
cd ~/code/app && dejavu install-hook   # warn on every commit in this repo
dejavu serve &                         # optional: keep the model loaded
```

Loading PyTorch and the model takes about 13 s, so `dejavu serve` keeps them warm in the background (local only,
port 7878). With it running, a commit that adds functions is checked in about 0.3 s, and a search takes about 0.2 s.
Without it, everything still works, just slower. Commits that don't add or change functions never need the model:
a docs-only commit takes about 0.2 s either way.

The hook **never blocks a commit**. It prints a warning and gets out of the way, and if anything goes wrong it prints one
line and exits cleanly.

## How it works

1. **Find functions.** tree-sitter pulls every function of 5+ lines out of TypeScript, JavaScript, Python and Rust files.
   Test files are skipped: test stubs look like the real implementation and caused most false alarms.
2. **Exact copies, no AI needed.** Each function's body is fingerprinted with comments and whitespace stripped. In my own
   10 repos, 2,451 of 5,094 functions were verbatim copies, and a hash catches every one of them instantly.
3. **Near-copies and rewrites.** New or changed functions are embedded with EmbeddingGemma 2 (sentence-similarity
   prompt, first 256 Matryoshka dimensions) and compared with everything in the index.
   - similarity ≥ 0.946: **déjà vu** warning
   - 0.91 to 0.946: a gentler **similar** hint
4. **Only what changed.** Functions whose fingerprint matches the last commit (unchanged or moved) are skipped, and the
   files you're committing are excluded from the comparison, so editing a function never matches its own old version.
   A docs-only commit never loads the model.

### Optional: a second opinion from a decision model

`dejavu check --second-opinion` (or `DEJAVU_SECOND_OPINION=1`) asks
[pplx-decider-flash](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2), an open-weight decision model, about
borderline hints, through Ollama's `/v1/systemone` endpoint. It can only **upgrade** a hint to a warning, never cancel
one. Needs Ollama 0.40.7 or later with the model imported (`DEJAVU_DECIDER_URL`, `DEJAVU_DECIDER_MODEL` to point
elsewhere). If no compatible server answers, the hint stays a hint.

## How well does it work?

Measured on my own repos (TypeScript-heavy, 2,643 unique functions). The scripts are in [`eval/`](eval).

| Test | Result |
|---|---|
| Copies that drifted apart (19 hand-labelled pairs) | 18 warnings, 1 hint (just under the line) |
| From-scratch rewrites (9) | 5 warnings, 4 hints; a function doing the same job ranked first 9/9 times out of 2,643 |
| Look-alikes that do different things (25 hand-labelled pairs) | 2 false warnings, 11 hints |
| …with `--second-opinion` | all 4 hinted rewrites upgraded to warnings, no look-alike upgraded (one test mock was, but test files aren't indexed) |
| Simulated hook over 381 functions from standalone repos | 1 warning, and it was real |
| Describe-a-function search (22 queries) | right function first 18–19/22 times; grep-style keyword matching: 2/22 |

**Caveats:** the cutoffs were chosen on the same small labelled set they were scored on, the upgrade-only second
opinion was designed after seeing the bake-off below, the rewrites were written by me while looking at the originals,
and every repo is mine. Treat the numbers as a careful first measurement, not a
benchmark.

### Why not ask an AI judge about every match?

I tested four open models as a second-stage judge on the same pairs, with the cutoff fixed before running:
Laya, CoREB (a code-search reranker), Clef Flash and pplx-decider-flash. None beat the embedding alone under that rule
([`eval/judge_report.txt`](eval/judge_report.txt)). pplx-decider-flash was by far the best at the borderline cases,
which is why it survives as the optional upgrade-only second opinion above. Two lessons from the bake-off:

- **Decision models answer the question you literally ask.** "Do these implement the same behaviour?" got a *no* for a
  copy that had gained two extra options. Used as a veto, that rejected 13 of 19 real copies.
- **Sanity-check every judge first.** CoREB's model card leaves out the empty `<think></think>` block its Qwen3-Reranker
  base expects. Without it, `add(a, b)` versus an unrelated config reader scored 99.8% "same".

## Why open models

This only works because the model runs locally. Most people can't send their employer's code, or their private repos,
to a cloud API just to ask "have I written this before?". An open-weight embedding model on a laptop can look at all
of it, offline, for free, as often as you commit.

## Built for

The [DEV Hacktoberfest 2026 Open-Source AI Challenge](https://dev.to/challenges/hacktoberfest-week3-2026-10-19)
(Week 3, "Slow and Steady"), October 21–25, 2026. Any commits after the deadline will be listed here.

## License

Apache 2.0
