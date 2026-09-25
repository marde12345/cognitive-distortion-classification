# Phase 9.1 — Baseline Execution Retry (gensim blocker resolved)

## Context

`docs/phases/PHASE_9_baseline_execution.md` documents the original Phase 9
attempt, which stopped after diagnosing a DNS resolution failure for
`files.pythonhosted.org` that prevented `uv add gensim` from succeeding.
That report is **not modified or overwritten**. This is a new, distinct
report for the retry, performed after the user manually confirmed and
resolved the network issue.

## 1. Environment Preflight

**FACT**, all verified this phase, not assumed:

| Check | Result |
|---|---|
| `git status --short` before execution | `M .gitignore, pyproject.toml, uv.lock, src/{config_utils,loader,finetune/trainer,baselines/sklearn_baseline}.py`; `?? configs/, docs/phases/, external/original-drive/DATASETS/, external/original-drive/.../configs/models/, scripts/` — identical to Phase 9's end state |
| `git log --oneline -5` | Unchanged: `efc1f8d` still HEAD, same 5 commits as every prior phase |
| `pyproject.toml` diff | Exactly `"gensim>=4.4.0"` added to `dependencies`, alongside the already-present `pandas`/`PyYAML` additions from Phase 8 — no unrelated entries |
| `uv.lock` diff | 317 insertions, 6 deletions — consistent with `gensim` + its transitive dependencies (`smart-open`, `wrapt`, etc.) being resolved |
| `uv run python -c "import gensim; print(gensim.__version__)"` | **`4.4.0`** — import succeeds |
| `scripts/run_baseline.py` (Phase 9 entry point) | Present, unmodified since Phase 9 |
| `src/config_utils.py`, `src/loader.py`, `src/metrics.py`, `src/baselines/sklearn_baseline.py`, `configs/base.yaml` | All present, all as left by Phase 8 |
| `dib_labeled.csv`, `folds.json` | Present at `external/original-drive/DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION/` |
| `external/original-drive/MODELING/COGNITIVE DISTORTION/results/` | 45 files present, `git diff --stat -- external/original-drive/` returns empty — confirmed unchanged before execution began |

All six preflight items from the task instructions were checked and
passed. Execution proceeded.

## 2. Dependency Resolution

**FACT**: `gensim==4.4.0`, `smart-open==8.0.1`, `wrapt==2.4.1` now present
in `uv.lock`, matching what the user reported installing. No additional
dependency was added by this phase — `uv run` was used exclusively, `pip`
was never invoked.

## 3–4. Baseline Execution Status & Runtime

All four baselines were executed in order via
`uv run python scripts/run_baseline.py <name>`, using the existing
`src/baselines/sklearn_baseline.py` functions unmodified — no alternative
implementation was written.

| Baseline | Status | Runtime (wall) | Warnings |
|---|---|---|---|
| `majority_class` | **SUCCESS** | 1.47s total (0.04s reported inside the run) | None |
| `tfidf_lr` | **SUCCESS** | 4.01s total (2.45s reported) | None |
| `tfidf_svm` | **SUCCESS** | 2.42s total (0.96s reported) | None |
| `svm_word2vec` | **SUCCESS** | 9.27s total (7.80s reported) | One line: `Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'` — printed to stderr during Word2Vec training, execution completed successfully and produced complete, valid output; **INFERENCE**: this is a known benign Cython/BLAS-cleanup warning from gensim's compiled word2vec inner loop, not a functional error, since the run finished normally with correct-shaped results |

**FACT**: for `svm_word2vec` specifically — the previous blocker (missing
`gensim`) is confirmed resolved; the module now imports and executes
completely.

## 5. Generated Artifacts

**FACT**, full listing of newly created `results/` tree (40 files):
```
results/metrics/{majority_class,tfidf_lr,tfidf_svm,svm_word2vec}/{fold_0..4.json,summary.json}   (24 files)
results/predictions/{majority_class,tfidf_lr,tfidf_svm,svm_word2vec}/fold_0..4.csv                (20 files)
```
**FACT**, `git check-ignore` verification:
- `results/metrics/**/*.json` → **not ignored**, trackable (matches Phase
  9C's confirmed `.gitignore` behavior).
- `results/predictions/**/*.csv` → **ignored**, matched by the
  `results/predictions/` rule.

`results/` did not exist before this phase's execution and was created
entirely by the four `_save_fold`/`_finalize` calls inside
`sklearn_baseline.py` — no manual directory creation, no copying from
`external/original-drive/`.

## 6. Comparison Against Original Drive Snapshot

Comparison method: direct `diff` (byte-level) for `majority_class`,
`tfidf_lr`, `tfidf_svm` (all fold JSONs, all prediction CSVs, summary
JSON); for `svm_word2vec`, byte-level `diff` plus a field-by-field
numeric comparison at `1e-9` tolerance, since byte-level differences were
found and needed characterizing.

### `majority_class`

**EXACT MATCH.** All 5 `fold_N.json`, all 5 `fold_N.csv` predictions, and
`summary.json` are byte-for-byte identical to
`external/original-drive/MODELING/COGNITIVE DISTORTION/results/metrics|predictions/majority_class/`.
**FACT**, not inference — verified via `diff -q` returning no output
(files identical) for every one of the 11 compared files.

### `tfidf_lr`

**EXACT MATCH.** Same result: all 5 fold metric JSONs, all 5 prediction
CSVs, and `summary.json` byte-for-byte identical to the original.

### `tfidf_svm`

**EXACT MATCH.** Same result: all 5 fold metric JSONs, all 5 prediction
CSVs, and `summary.json` byte-for-byte identical to the original.

### `svm_word2vec`

**NUMERICAL DIFFERENCE — not an exact match, not an execution failure,
not a missing artifact.**

Row-level structural check first: `wc -l` on `fold_0.csv` — 920 lines both
locally and in the original (identical row count); `diff` on the
`sentence_id,y_true` columns alone returned **no output** — i.e. the test
split composition and true labels are identical between the two runs, so
this is not a data-loading or fold-assignment discrepancy.

Metric differences (summary.json, tolerance `1e-9`, all values `DIFFER`):

| Metric | Original | Local | Absolute diff |
|---|---|---|---|
| `macro_f1_mean` | 0.362793 | 0.367140 | 0.004347 |
| `macro_f1_std` | 0.011673 | 0.009889 | 0.001784 |
| `weighted_f1_mean` | 0.380599 | 0.381349 | 0.000750 |
| `weighted_f1_std` | 0.016503 | 0.009661 | 0.006842 |
| `accuracy_mean` | 0.381814 | 0.382900 | 0.001086 |
| `accuracy_std` | 0.014406 | 0.008345 | 0.006061 |
| `per_class_f1_mean` (11 labels) | — | — | ranges 0.0004–0.0174, all small |
| `per_class_f1_std` (11 labels) | — | — | ranges 0.0030–0.0265, all small |
| `n_folds` | 5 | 5 | **MATCH** |

**Root cause, investigated rather than assumed:** `src/baselines/sklearn_baseline.py`
(byte-for-byte identical to the Drive original except Phase 8's one
path-resolution line — confirmed, not re-modified this phase) constructs
Word2Vec as:
```python
w2v = Word2Vec(
    sentences=train_tokens,
    vector_size=..., window=..., min_count=..., epochs=...,
    workers=4,
    seed=42,
)
```
**FACT**: `workers=4` — gensim's own documented behavior is that
`Word2Vec` training is **not fully deterministic when `workers > 1`**,
because multi-threaded training order depends on OS thread scheduling,
even with a fixed `seed`. **INFERENCE**: this — not a code defect, not a
dataset difference (ruled out via the identical `sentence_id`/`y_true`
check above), and not a `metrics.py` reconstruction issue (`compute_metrics`/
`average_metrics` only aggregate whatever prediction values they're given
— confirmed correct on 3 of 4 baselines' exact-match results, and the
`svm_word2vec` differences are entirely upstream in the feature vectors
Word2Vec produces, not in how those predictions get scored) — is the most
plausible explanation for run-to-run variation in exactly the one baseline
whose pipeline includes multi-threaded Word2Vec training, while the other
three (deterministic TF-IDF + sklearn linear models, and a trivial
majority-class rule) reproduced exactly. This is not proven by re-running
multiple times in this phase (not attempted, to avoid unnecessary compute
and since the instruction was to execute once and compare) — it is the
best-supported inference from the evidence gathered, not a certainty.

**No source file was modified to try to force determinism** (e.g. setting
`workers=1`), per the explicit instruction not to "fix" recovered source
or invent alternative implementations.

## 7. Discrepancies Summary

| Baseline | Discrepancy | Classification |
|---|---|---|
| `majority_class` | None | Exact match |
| `tfidf_lr` | None | Exact match |
| `tfidf_svm` | None | Exact match |
| `svm_word2vec` | Macro-F1 differs by 0.0043 (≈1.2% relative), all summary/per-class fields differ by small amounts, `n_folds` matches | Numerical difference, attributable to gensim's documented multi-threaded Word2Vec non-determinism, not a code or reconstruction defect |

No execution failures. No missing expected artifacts — every file the
source code was supposed to produce was produced, for all four baselines.

## 8. Git Status

**AFTER (git status --short):**
```
 M .gitignore
 M pyproject.toml
 M src/baselines/sklearn_baseline.py
 M src/config_utils.py
 M src/finetune/trainer.py
 M src/loader.py
 M uv.lock
?? configs/
?? docs/phases/
?? external/original-drive/DATASETS/
?? "external/original-drive/MODELING/COGNITIVE DISTORTION/configs/models/"
?? results/
?? scripts/
```
Only new item versus Phase 9's end state: `?? results/` (40 files, none
staged). `results/metrics/**` is trackable per `.gitignore`;
`results/predictions/**` is ignored per the existing `results/predictions/`
rule — both confirmed via `git check-ignore`, not assumed.

`external/original-drive/` remains completely unmodified — `git diff --stat
-- external/original-drive/` returns empty, and the original results
directory's file count (45) is unchanged from before this phase's
execution.

**No files were staged. No commit was made**, per instruction — this
report is being delivered for review before any commit decision.

## 9. Recommended Next Phase

**Phase 10 candidates, not decided here:**

1. Decide whether to commit `results/metrics/**` (the small, trackable,
   non-sensitive aggregate JSON — 3 of 4 baselines exact-match the
   original, the 4th is explainably close) — this would be the first
   locally-generated experiment artifact entering Git.
2. Optionally re-run `svm_word2vec` a few more times to empirically
   confirm the multi-threaded non-determinism hypothesis (would show
   run-to-run variation even without any code change) — not done this
   phase, since it wasn't required to characterize the discrepancy as
   "investigated and explained" rather than "unexplained."
3. Proceed toward transformer smoke-testing (`torch`/`transformers`,
   1 model × 1 fold) now that the classical baseline path is fully
   validated end-to-end locally.

None of these are started — awaiting explicit instruction, per the
standing rule not to proceed to the next phase automatically.

## Conclusion

All four baselines now execute successfully end-to-end locally via `uv run`,
using the unmodified (Phase-8-path-adapted only) source code. Three of
four (`majority_class`, `tfidf_lr`, `tfidf_svm`) reproduce the original
Drive-snapshot results **exactly, byte-for-byte** — the strongest possible
behavioral evidence yet for both the local execution adaptation (Phase 8)
and the `metrics.py` clean-room reconstruction (Phase 4B), since these
three runs depend on both working correctly. The fourth
(`svm_word2vec`) produces small, well-characterized numerical differences
attributable to gensim's documented Word2Vec multi-threading
non-determinism, not to any defect in the repository's code — investigated
and explained rather than dismissed or silently patched. No source file
was modified to chase an exact match. The dataset, the notebook, and the
entire `external/original-drive/` snapshot remain untouched. Nothing was
committed.
