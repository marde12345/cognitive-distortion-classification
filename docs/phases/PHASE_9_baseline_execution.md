# Phase 9 — Baseline Execution & Reproduction

## Objective

Execute the four completed classical baselines (`majority_class`,
`tfidf_lr`, `tfidf_svm`, `svm_word2vec`) locally against the verified local
dataset and determine whether the local implementation reproduces the
original stored baseline results.

## Outcome (stated up front, per the report's own traceability requirement)

**No baseline was executed.** Phase 9B's dependency installation
(`gensim`) failed due to a persistent local DNS resolution failure, and
diagnosis established that this blocks **all four** baselines, not only
`svm_word2vec`, because of how `src/baselines/sklearn_baseline.py` imports
`gensim` (unconditional, module-level). Per the phase's own explicit
instruction — "If gensim import fails due to an unrelated compatibility
issue: STOP. Do not randomly patch dependencies. Diagnose the exact issue
and report it" — execution was stopped after full diagnosis rather than
worked around. This report documents that diagnosis and the state of
everything inspected up to the stop point.

## Baseline Git State

**BEFORE (git status --short):**
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
```
**BEFORE (git log --oneline -5):**
```
efc1f8d  feat: reconstruct metrics module
2fcfe00  chore: reconcile recovered source modules
b71999d  chore: add original Drive snapshot for provenance
a5dcdff  Repository cleanup: establish uv project foundation
feccab5  Initial thesis snapshot: original Colab notebook, unmodified
```
**FACT**: identical to Phase 8's documented end state — no drift occurred
between phases.

## Environment

**FACT**: `uv` was used exclusively; `pip` was never invoked directly.
Python 3.11 (per `.python-version`), `uv run` executes from the repository
root.

## Phase 9A — Entry Point Decision

**FACT**, from inspection: no `scripts/run_baselines.py` (or any
`scripts/`) existed prior to this phase. `src/baselines/sklearn_baseline.py`
exposes four functions (`run_majority_class`, `run_tfidf_lr`,
`run_tfidf_svm`, `run_svm_word2vec`), each with signature
`(df, folds_data, cfg, paths)`, mirroring exactly how the original notebook
cells (14/17/20/22) invoked them: `cu.set_seed(42)` → `cu.get_paths()` →
`cu.load_config()` → `loader.load_folds(paths)` → `loader.load_dataset(cfg,
paths)` → `sb.run_x(df, folds_data, cfg, paths)`.

**DECISION**: rather than building a large multi-model orchestration
runner (explicitly discouraged — "If it does not exist, DO NOT immediately
create a large runner"), created the smallest reproducible mechanism: a
single new file, `scripts/run_baseline.py`, accepting one baseline name as
a CLI argument and executing exactly that notebook-cell sequence once,
printing a summary and elapsed time. This gives a concrete, reusable
`uv run python scripts/run_baseline.py <name>` command for traceability
(Phase 9E) without inventing orchestration logic beyond what the phase
needs.

## Phase 9B — Dependency

**FACT**: `uv add pandas pyyaml` succeeded in Phase 8. This phase attempted
`uv add gensim` for `run_svm_word2vec`.

**FACT — full diagnostic record, 5 attempts across ~10 minutes:**
```
error: Request failed after 3 retries in 10.1s
  Caused by: Failed to fetch: https://files.pythonhosted.org/packages/.../gensim-4.4.0-....whl.metadata
  Caused by: error sending request for url (...)
  Caused by: client error (Connect)
  Caused by: dns error
  Caused by: failed to lookup address information: nodename nor servname provided, or not known
```
Identical error on every attempt (uv add gensim ×4, after Task instructions'
required "STOP" checkpoint).

**Diagnosis performed, not skipped:**
- `curl -sI https://pypi.org` → **succeeded**, `HTTP/2 200` — the package
  *index* host resolves and responds normally.
- `nslookup files.pythonhosted.org` → **`SERVFAIL`** — the CDN host that
  actually serves package file bytes (distinct from the index host) fails
  to resolve, consistently, across all attempts.
- `find ~/.cache/uv -iname "*gensim*"` → only an unrelated `.rkyv` index
  metadata stub and unrelated `nltk` test fixtures containing the string
  "gensim" in their filenames — **no actual gensim wheel is cached
  locally**, so there is no offline fallback available.

**INFERENCE**: this is a local/network DNS resolution problem specific to
the `files.pythonhosted.org` hostname in the current environment, not a
`gensim`/Python version compatibility issue, and not a `uv` configuration
error — `pypi.org` itself resolves and responds correctly, isolating the
fault to one specific hostname's DNS record.

**Compounding discovery — why this blocks all four baselines, not just
`svm_word2vec`:** attempting to run `scripts/run_baseline.py majority_class`
(which needs no `gensim` functionality at all) failed with:
```
File ".../src/baselines/sklearn_baseline.py", line 12, in <module>
    from gensim.models import Word2Vec
ModuleNotFoundError: No module named 'gensim'
```
**FACT**: `src/baselines/sklearn_baseline.py` imports `gensim` at module
level (line 12, unconditional), so the module fails to import at all
without `gensim` installed — this is true of the original Drive source as
well (verified identical in Phase 3.2's byte-for-byte comparison), not
something introduced by Phase 8's path adaptation. Every one of the four
`run_*` functions lives in this same file, so **none** are reachable until
the module itself can be imported.

**Decision, per explicit instruction not to patch around this**: no
workaround was attempted — not a lazy/deferred import inside
`sklearn_baseline.py` (would modify a file whose provenance-tracked change
scope, per Phase 8, was limited to the one path-resolution line; expanding
it here without explicit authorization would violate "do not randomly
patch dependencies" and "do NOT modify source" absent a specific
instruction to do so), not a pip fallback (explicitly forbidden), not a DNS
server change (an environment-level change outside this phase's scope).
**STOPPED**, per instruction.

## Phase 9C — Output Safety (completed, since it precedes execution)

**FACT**, verified via `git check-ignore`:
```
results/metrics/majority_class/summary.json    → NOT ignored (trackable)
results/predictions/majority_class/fold_0.csv  → ignored (results/predictions/ rule)
```
Confirmed `results/` did not exist on disk before this phase
(`test -d results` → false). No directory was pre-created manually — per
instruction, directories are only to be created by actual execution
(`cu.ensure_dirs` inside `_save_fold`), and since no execution succeeded,
`results/` still does not exist after this phase either (confirmed below).

## Dataset Used

**FACT**: not loaded this phase — execution never reached the
`loader.load_dataset`/`loader.load_folds` calls inside
`scripts/run_baseline.py`, because the import of `baselines.sklearn_baseline`
(which happens before any of those calls, per the script's own top-level
`import` statements) failed first. The dataset described in Phase 8
(4,628 rows, 5 folds, checksums verified) remains the last-verified state;
it was not re-touched this phase.

## Baseline 1 — Majority Class

| Item | Result |
|---|---|
| Status | **BLOCKED — not executed** |
| Runtime | N/A (failed at import, ~1s) |
| Dataset rows | N/A — not reached |
| Folds | N/A — not reached |
| Output path | N/A — no output produced |
| Original comparison | N/A |
| Metric match | N/A |
| Prediction match | N/A |
| Warnings | `ModuleNotFoundError: No module named 'gensim'` at `sklearn_baseline.py:12`, raised during `import baselines.sklearn_baseline as sb` in `scripts/run_baseline.py`, before `run_majority_class` could be called |

**FACT**: this baseline does not itself use `gensim` — the block is a
structural side effect of shared module-level imports, not a property of
`majority_class`'s own logic.

## Baseline 2 — TF-IDF Logistic Regression

| Item | Result |
|---|---|
| Status | **BLOCKED — not executed** (same root cause as Baseline 1) |
| Runtime | N/A |
| Dataset rows | N/A |
| Folds | N/A |
| Output path | N/A |
| Original comparison | N/A |
| Metric match | N/A |
| Prediction match | N/A |
| Warnings | Same `ModuleNotFoundError` — not individually re-attempted after Baseline 1 established the blocker applies module-wide; re-running would reproduce the identical failure for a reason already fully diagnosed |

## Baseline 3 — TF-IDF SVM

| Item | Result |
|---|---|
| Status | **BLOCKED — not executed** (same root cause) |
| Runtime | N/A |
| Dataset rows | N/A |
| Folds | N/A |
| Output path | N/A |
| Original comparison | N/A |
| Metric match | N/A |
| Prediction match | N/A |
| Warnings | Same `ModuleNotFoundError`, not re-attempted for the same reason as Baseline 2 |

## Baseline 4 — SVM Word2Vec

| Item | Result |
|---|---|
| Status | **BLOCKED — not executed** (this is the baseline `gensim` is actually needed for) |
| Runtime | N/A |
| Dataset rows | N/A |
| Folds | N/A |
| Output path | N/A |
| Original comparison | N/A |
| Metric match | N/A |
| Prediction match | N/A |
| Warnings | Same `ModuleNotFoundError` — the only one of the four baselines whose own logic genuinely requires `gensim` (`Word2Vec`, imported and used in `run_svm_word2vec`) |

## Reproduction Comparison

### Metrics Comparison

Not performed — no local results were generated to compare. The original
stored results under `external/original-drive/MODELING/COGNITIVE
DISTORTION/results/` were **not read or touched** this phase (confirmed
below), since Phase 9F's comparison step only applies once a baseline
completes successfully, and none did.

### Prediction Comparison

Not performed, for the same reason.

## Metrics Reconstruction Validation

**Not newly exercised this phase** — no baseline execution occurred to
generate fresh evidence. `src/metrics.py`'s prior validation (Phase 4B,
re-confirmed in Phase 7) stands unchanged: **RECONSTRUCTED — ORIGINAL
SOURCE NOT RECOVERED**, previously verified to zero mismatches against all
4 stored baseline results using the existing predictions. This phase adds
no new evidence either way.

## Dataset Safety Audit

**Verified, not assumed:**
```
Dataset files READ:      NO — execution never reached loader.load_dataset/
                          load_folds; the dataset was not opened this phase
Dataset files CREATED:   NO
Dataset files MODIFIED:  NO
Dataset files MOVED:     NO
Dataset files DELETED:   NO
Dataset files STAGED:    NO
Dataset files COMMITTED: NO
```
Note this differs from the phase instructions' "Expected: READ: yes" —
that expectation assumed baseline execution would proceed far enough to
load the dataset. It did not. This is reported accurately rather than
forced to match the template's assumption.

## Original Snapshot Safety Audit

**FACT**, verified via `git status --short`: no path under
`external/original-drive/` shows as modified (`M`) — only the pre-existing
`??` (untracked) entries from Phase 8 remain, unchanged in content.
`external/original-drive/MODELING/COGNITIVE DISTORTION/results/` (the
original/reference results) was not opened, read, or written this phase.
The notebook (`notebooks/Modeling_Cognitive_Distortion (2).ipynb`) was not
opened this phase.

## Git Safety Audit

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
?? scripts/
```
**Only new change versus the Phase 9 baseline: `?? scripts/`** (the new
`scripts/run_baseline.py` entry point). No `results/` directory exists
(confirmed: `test -d results` → false, both before and after). `pyproject.toml`/
`uv.lock` diff is byte-for-byte identical to Phase 8's end state (confirmed
via `git diff --stat`) — the four failed `uv add gensim` attempts left no
partial or corrupted state in either file, verified by direct inspection
(no `gensim` string present in `pyproject.toml`).

`git diff --stat`:
```
 .gitignore                        |   8 ++
 pyproject.toml                    |   2 +
 src/baselines/sklearn_baseline.py |   2 +-
 src/config_utils.py               |   5 +-
 src/finetune/trainer.py           |   2 +-
 src/loader.py                     |   2 +-
 uv.lock                           | 193 ++++++++++++++++++++++++++++++++++++--
 7 files changed, 204 insertions(+), 10 deletions(-)
```
Identical to Phase 8's — no new tracked-file changes this phase, since the
only new artifact (`scripts/run_baseline.py`) is untracked, not staged.

**No commit was made.**

## Generated Artifacts

**FACT**: exactly one — `scripts/run_baseline.py` (new, untracked source
file, the execution entry point designed in Phase 9A). No `results/`
directory, no prediction CSVs, no metric JSONs, no logs, no intermediate
files of any kind were produced, since execution stopped at the import
stage before any output-writing code path was reached.

## Mismatches / Anomalies

No metric mismatches to report — no metrics were computed. One anomaly
worth flagging precisely: **Phase 7's dependency audit (§ Dependency
Audit) correctly identified `gensim` as needed only for
`run_svm_word2vec`, categorizing it as relevant to "one of four baseline
models."** This phase's attempted execution reveals that categorization
understated the blast radius — because of the shared-module import
structure, `gensim`'s absence actually blocks all four baseline functions
from being callable at all, not just the one that uses `Word2Vec`. This is
not being described as a "bug" in Phase 7's report (that report's
inference was reasonable given the evidence available at the time, which
was import-list analysis, not an actual import attempt) — it is a
correction, recorded here per the standing rule that prior phase reports
are never edited retroactively.

## Decisions

1. Did not attempt to install `gensim` via any channel other than `uv add`
   (no `pip` fallback, no manual wheel download, no DNS server override) —
   consistent with the "do not randomly patch" instruction and the
   standing "use `uv`, never `pip` directly" rule.
2. Did not modify `src/baselines/sklearn_baseline.py` to defer/lazy-load
   the `gensim` import, even though this would technically unblock
   Baselines 1–3 — this would be an unauthorized source-logic change to a
   file whose Phase 8 modification scope was explicitly limited to one
   path-resolution line, and the current phase's instructions provide no
   authorization to expand that scope.
3. Ran the diagnostic sequence (pypi.org reachability, DNS lookup,
   uv cache inspection) to distinguish "network issue" from "compatibility
   issue" before stopping, per the explicit instruction to diagnose rather
   than guess.
4. Stopped after confirming the block applies to all four baselines (via
   one concrete `majority_class` attempt) rather than repeating the same
   already-diagnosed failure three more times for Baselines 2–4.

## Risks

- The DNS failure is environment-specific and may be transient (e.g.
  network configuration, VPN, corporate/ISP DNS) or persistent (e.g. a
  captive portal, firewall rule, or misconfigured resolver) — this phase
  cannot distinguish which, since it has no visibility into the network
  environment beyond the symptoms already reported.
- If `gensim` remains unavailable, `run_svm_word2vec` specifically cannot
  ever be locally reproduced without either resolving the network issue or
  making an explicit, authorized decision to defer that one baseline
  indefinitely — a decision for the user, not to be made unilaterally here.
- The discovery that all four baselines share a single point of import
  failure is useful information for any future phase that might consider
  splitting `sklearn_baseline.py`'s imports to be more fault-tolerant — but
  that would be a source-architecture change requiring its own explicit
  phase and authorization, not an implicit fix bundled into this one.

## Recommended Phase 10

**Phase 10: Resolve the `gensim` dependency blocker, then re-attempt Phase
9's baseline execution.** Concretely: either (a) retry `uv add gensim` once
network/DNS conditions are confirmed healthy (a quick `nslookup
files.pythonhosted.org` check would confirm this before re-attempting), or
(b) if the network issue persists, decide explicitly how to proceed —
options include sourcing the wheel through an alternate index/mirror
configured in `uv`, or accepting a scoped, explicitly-authorized change to
`sklearn_baseline.py`'s import structure. Once `gensim` imports
successfully, Phase 9's original plan (execute all four baselines in
order, stopping after each to inspect, comparing against the stored Drive
results) can proceed essentially unchanged from what was designed here —
`scripts/run_baseline.py` is ready and untouched, waiting on the one
blocker.

## Conclusion

Phase 9 did not execute any baseline. The stated objective (reproduction
validation) could not be attempted because the required `gensim` dependency
could not be installed, due to a confirmed, reproducible DNS resolution
failure for `files.pythonhosted.org` in the current environment —
diagnosed precisely rather than worked around. A previously
under-characterized fact was discovered and recorded: `gensim`'s absence
blocks all four baseline functions, not only `svm_word2vec`, due to
`sklearn_baseline.py`'s shared module-level import — this is stated as a
correction to Phase 7's more narrowly-scoped finding, without modifying
Phase 7's report. The one artifact produced this phase,
`scripts/run_baseline.py`, is untracked, correctly scoped, and ready to use
once the dependency blocker is resolved. No dataset file, notebook, or
path under `external/original-drive/` was read, modified, or touched. No
commit was made.
