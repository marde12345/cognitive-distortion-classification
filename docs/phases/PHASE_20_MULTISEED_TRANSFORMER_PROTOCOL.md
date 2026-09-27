# Phase 20 — Multi-Seed Transformer Experiment Protocol & Dashboard Readiness

This is an execution-readiness and architecture phase. **No training was
executed.** The 105-run multi-seed experiment is prepared but not
launched.

## 20A. Starting-State Reconnaissance

- Branch at start: `main`. Starting HEAD: `8c4e6d0` ("docs: add transformer
  model comparison evidence" — the Phase 19 commit).
- Working-tree status at start: identical to the documented post-Phase-19
  baseline — only `src/finetune/trainer.py` modified (the pre-existing,
  uncommitted Phase 12 MPS device-selection fix), plus the same untracked
  files carried since earlier phases.
- Existing Phase 13–19 commits: `af45fed` … `8c4e6d0` (see `git log`), all
  present and unmodified; no new commits were made on `main` before
  branching for Phase 20.
- Existing transformer execution scripts: `scripts/run_transformer.py`
  (single model/fold run), `scripts/run_transformer_sweep.py` (Phase 15
  orchestrator).
- Existing transformer result directories: `results/metrics/<model>/`,
  `results/logs/`, `results/predictions/` (gitignored), all untouched.
- Existing Phase 16 analysis artifacts: `results/analysis/transformer/*`
  — present, unmodified.
- Existing Phase 19 comparison artifacts:
  `results/analysis/transformer/comparison/*` — present, unmodified.
- Existing dashboard architecture: `scripts/run_dashboard.py`
  (`ThreadingHTTPServer`, stdlib-only) → `src/dashboard/state.py`
  (`build_dashboard_state`) → `src/dashboard/data_loader.py` /
  `analysis_loader.py` (read-only file access with `ReadResult` soft
  failures) → `src/dashboard/static/index.html` (vanilla JS, no
  framework).
- Existing tests: 57 tests across `tests/test_dashboard_data.py`,
  `test_dashboard_analysis.py`, `test_dashboard_inference.py`, etc.
- Existing dataset/fold definitions:
  `external/original-drive/DATASETS/PROCESSED DATASETS/COGNITIVE
  DISTORTION/dib_labeled.csv` and `folds.json`, checksummed and validated
  by `config_utils.load_folds`.

**The pre-existing uncommitted `src/finetune/trainer.py` MPS fix was not
staged, reset, or committed at any point in this phase** — it remains an
unstaged modification on top of the Phase 20 branch, exactly as it was on
`main`. All other pre-existing untracked artifacts (`.claude/`, Phase
11–15 docs, `external/original-drive/DATASETS/`, `results/logs/`,
`results/metrics/<model>/`, `scripts/run_transformer.py`,
`scripts/run_transformer_sweep.py`) were likewise left untouched.

## 20B. Experimental Protocol

- Models (7): indobert_15g, indobert_base_p1, indobertweet,
  indoroberta_15g, mbert, nusabert, xlmr — discovered from
  `configs/models/*.yaml` (same discovery rule Phase 15 used: `model_type
  == "transformer"` and no `PLACEHOLDER` marker), not hardcoded.
- Folds (5): fold_0 … fold_4 — read from the existing, unmodified
  `folds.json` via `config_utils.load_folds`/`loader.load_folds`. Neither
  `folds.json` nor `dib_labeled.csv` was regenerated or modified.
- Seeds (3): 42, 43, 44 — fixed constants in the new scripts.
- Total: 7 × 5 × 3 = **105 planned runs**, confirmed by dry-run (Section
  20E).
- The same fold assignments are used for every model and every seed —
  only the seed value varies; `loader.get_fold_data` is called identically
  regardless of seed.

## 20C. Seed Semantics

Audit of where randomness enters `src/finetune/trainer.py::train_one_fold`:

- Python `random` — not used directly in the training loop.
- NumPy `np.random` — used by `_compute_class_weights`'s `np.bincount`
  call, which is deterministic given fixed inputs (not actually a source
  of randomness), but `numpy`'s global RNG state is still seeded for
  general safety.
- PyTorch — `torch.manual_seed` covers the classification head's random
  initialization (`AutoModelForSequenceClassification.from_pretrained`
  initializes an un-pretrained final layer for a task-specific
  `num_labels`) and `DataLoader(shuffle=True)`'s shuffling, since both
  draw from PyTorch's global RNG and both happen after the seed call.
- Model initialization — covered by the above (`torch.manual_seed`).
- DataLoader shuffling — covered by the above.
- Hugging Face / tokenizer — `AutoTokenizer` does not introduce
  randomness relevant to training reproducibility; no HF-specific seeding
  API is bypassed.
- MPS-specific behavior — **this was the one real gap.** The existing
  `config_utils.set_seed()` called `torch.manual_seed` and
  `torch.cuda.manual_seed_all`, but never seeded MPS's own RNG stream
  (`torch.mps.manual_seed`), even though this machine trains on `mps`.
  This was fixed additively (see below).

**Change made** (`src/config_utils.py`, inside `set_seed`):
```python
if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
    torch.mps.manual_seed(seed)
```
This is purely additive: `set_seed(seed=42)`'s signature and default are
unchanged, and every other line is untouched. No existing Phase 13/15
result is invalidated or recomputed by this change — those runs already
completed and their artifacts are not being regenerated. This only
affects the RNG stream for training that happens *after* this phase.

**Seed propagation (explicit, no hidden global state):** a new script,
`scripts/run_transformer_multiseed_single.py`, takes `--model`, `--fold`,
and `--seed` as three required CLI arguments (mirroring the existing
`scripts/run_transformer.py`'s `--model`/`--fold` pattern) and calls
`config_utils.set_seed(seed)` with the caller-supplied value — never a
hardcoded `42`. `scripts/run_transformer.py` itself is **not modified**;
it keeps calling `cu.set_seed(42)` exactly as before, so its own
semantics (and Phase 13/15 reproducibility) are unaffected.

## 20D. Existing Artifact Protection

A new output namespace was created: **`results/multiseed/`**, with the
same `metrics/<model>/`, `logs/<model>/`, `predictions/<model>/`
substructure as the existing `results/` root, so the well-understood
Phase 13/15 file-layout conventions are reused rather than reinvented.

Every result file name unambiguously encodes model, fold, and seed:

```
results/multiseed/metrics/<model>/<fold>__seed_<seed>.json
results/multiseed/logs/<model>/<fold>__seed_<seed>.json
results/multiseed/predictions/<model>/<fold>__seed_<seed>.csv
```

Example: `results/multiseed/metrics/indobert_base_p1/fold_2__seed_43.json`
identifies `indobert_base_p1 / fold_2 / seed_43` with no ambiguity, and
cannot collide with the Phase 13/15 file
`results/metrics/indobert_base_p1/fold_2.json` since they live under
different root directories entirely. `scripts/run_transformer_multiseed_single.py`
achieves this by reusing `finetune.trainer.save_predictions_and_metrics`
completely unmodified, passing it a `paths` dict whose `results_root` is
overridden to `results/multiseed` and whose `fold_name` argument is the
combined `<fold>__seed_<seed>` string — no changes to
`save_predictions_and_metrics` or `train_one_fold`'s output logic were
needed.

## 20E. Execution Infrastructure

Inspected: `scripts/run_transformer.py`, `scripts/run_transformer_sweep.py`,
`src/finetune/trainer.py`, `src/config_utils.py`, `src/loader.py`,
`src/metrics.py`.

Conclusion: the existing sweep infrastructure (`run_transformer_sweep.py`)
enumerates only `(model, fold)` pairs and has no seed axis or namespace
separation — it cannot safely be reused as-is for 105 runs, since doing
so would either require it to gain a seed parameter (risking silently
changing its behavior for the existing 35-run Phase 15 case) or write
into the same `results/metrics/<model>/<fold>.json` files a second time
(overwriting Phase 15). Instead, three new, additive scripts were created,
deliberately mirroring the existing scripts' architecture and conventions
rather than modifying them:

1. **`scripts/run_transformer_multiseed_single.py`** — single-run
   executor for one `(model, fold, seed)` triple. Sibling of
   `run_transformer.py`; reuses `trainer.train_one_fold` and
   `trainer.save_predictions_and_metrics` unmodified.
2. **`scripts/run_transformer_multiseed.py`** — resumable orchestrator for
   all 105 combinations, mirroring `run_transformer_sweep.py`'s
   discover → check-completion → subprocess → resumable-manifest →
   fail-fast architecture, with an added seed axis. Supports `--dry-run`.
3. **`scripts/generate_multiseed_manifest.py`** — writes the static
   protocol manifest (Section 20F).

Requirements checked against this design:

1. Existing Phase 15 behavior remains reproducible — `run_transformer.py`
   and `run_transformer_sweep.py` are byte-for-byte unmodified.
2. Existing scripts don't silently change semantics — confirmed by diff:
   zero lines changed in either file.
3. Multi-seed execution is explicit — `--seed` is a required CLI argument
   with no default.
4. Restartable without corruption — `is_complete(model, fold, seed)`
   checks both the metrics file and the log file's `status == "completed"`
   field before treating a run as done, exactly like the Phase 15
   sweep's `is_complete`.
5. Completed runs are detectable — via the same two-file check.
6. Failed runs are recorded explicitly — the run-state JSON records
   `status: "failed"` with `exit_code`, mirroring Phase 15.
7. No silent overwriting — the multi-seed namespace never touches
   `results/metrics|logs|predictions/` (Phase 13/15's namespace).
8. No duplicate execution unless requested — `is_complete` skips
   already-done combinations by default; re-running only executes what's
   missing.
9. Deterministic, documented command —
   `uv run python scripts/run_transformer_multiseed.py --dry-run` /
   (without `--dry-run`) documented in each script's module docstring.

### Dry-run result

```
uv run python scripts/run_transformer_multiseed.py --dry-run
```
Output: enumerates exactly **105** unique `(model, fold, seed)`
combinations (verified: 7 models × 5 folds × 3 seeds, no duplicates), all
marked `WOULD RUN` (no prior multi-seed results exist), and confirms "No
training was executed, no files were written." Verified independently
after the dry-run that `results/multiseed/` did not exist on disk.

## 20F. Run Manifest

`scripts/generate_multiseed_manifest.py` writes
`results/multiseed/manifest.json`:

```json
{
  "experiment_id": "phase20_multiseed_transformer_sweep",
  "protocol_version": "1.0",
  "created_at": "2026-09-27T06:12:30Z",
  "models": ["indobert_15g", "indobert_base_p1", "indobertweet",
             "indoroberta_15g", "mbert", "nusabert", "xlmr"],
  "folds": ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"],
  "seeds": [42, 43, 44],
  "total_expected_runs": 105,
  "dataset_csv_sha256": "ed154162554b68ba6980af8f3c7c01fa80a7962c4a0595fd61acafd142bdaea0",
  "folds_json_sha256": "730cd82a912828e902cdab26eb498c640195eca4901fde606f849372c5693c54",
  "notes": "..."
}
```

Both checksums match the established baseline used throughout Phases
13–19 exactly. This manifest is the single frozen definition of "the same
experimental protocol" that a future execution phase's 105 runs can be
checked against.

## 20G. Analysis Design

Designed (implemented as read-only, currently-empty-of-results code in
`src/dashboard/multiseed_loader.py`, exercised by synthetic-fixture
tests — see Section 20N):

- **Per-run**: `collect_per_run_results` — one row per `(model, fold,
  seed)` with status, elapsed time, and (if completed) macro_f1/
  accuracy/weighted_f1.
- **Per-fold**: `build_per_fold_aggregates` — `(model, fold)` descriptive
  stats aggregated across seeds.
- **Per-seed**: `build_per_seed_aggregates` — `(model, seed)` descriptive
  stats aggregated across folds.
- **Overall model**: `build_per_model_aggregates` — per-model descriptive
  stats pooled across all 15 of that model's runs (5 folds × 3 seeds).

For Macro-F1, each aggregate reports: mean, standard deviation (sample,
`statistics.stdev`), median, min, max, range, and (at the per-model level)
`n_completed`, `n_failed`, and `n_anomalous`. Cross-fold and cross-seed
stability are directly the per-fold and per-seed aggregates above; overall
variability is the per-model aggregate's std/range/CV.

**Anomalies are never auto-removed.** `build_per_model_aggregates` takes
an explicit `anomalous_runs` set — currently empty, since Phase 20 has
not produced any results to inspect — and only ever *counts* how many of
a model's completed runs fall in that set; it never filters them out of
the mean/std/etc. This mirrors exactly how Phase 16's two anomalies
(`indobertweet/fold_1`, `mbert/fold_3`) were handled: identified by
human inspection after the fact, retained in the official aggregate, and
surfaced separately. No anomaly-detection heuristic/threshold was
invented for Phase 20 — that would be "inventing a sophisticated method
merely because it is available," which the phase instructions explicitly
warned against. Any sensitivity analysis, when it exists, must be clearly
labeled as such, exactly as in Phase 16/19.

## 20H. Statistical Analysis Design

The existing Phase 16 methodology (Friedman test, folds as blocks, models
as treatments) is a matched repeated-measures design with **fold** as the
unit of repetition. Extending this to 3 seeds raises the question of what
the new unit of analysis should be.

**Design decision, and why:** treat **fold** as the primary matched
blocking factor exactly as Phase 16 did, and treat **seed** as an
*additional, separate replicate axis reported descriptively* (mean/std
across seeds per model/fold), rather than inventing a two-way
(fold × seed) ANOVA-style test. Rationale:

1. **What will be tested**: the same Friedman-style comparison Phase 16
   already used, applied to the *seed-averaged* Macro-F1 per
   (model, fold) — i.e., 7 models × 5 fold-blocks, where each block's
   value is now an average over 3 seeds instead of a single run. This
   keeps the well-understood test unchanged in structure while using the
   extra seeds to produce a less noisy per-block statistic.
2. **Unit of analysis**: (model, fold) — 5 matched blocks, exactly as
   before; seed-level variation is reported separately (per-seed
   aggregates), not folded into the primary hypothesis test.
3. **Assumptions**: same as Phase 16's Friedman test (non-parametric,
   requires only matched blocks, no normality assumption) — this is not
   changed, only the values that feed each block become a 3-seed average
   instead of a single observation.
4. **Limitations**: 5 fold-blocks is still a very small sample for the
   Friedman test (same limitation as Phase 16). Averaging 3 seeds per
   block reduces per-block noise but does not increase the number of
   independent blocks (still 5), so the test's power limitation is not
   solved, only the input noise is reduced. Seed-to-seed variance itself
   is *not* directly tested for significance — it is reported
   descriptively (mean/std per (model, seed)) since a rigorous
   variance-component model (e.g. mixed-effects with fold and seed as
   crossed random effects) would be considerably more complex to explain
   and justify than the evidence (3 seeds, 5 folds, 7 models) can
   robustly support. This is a deliberate choice of a simpler,
   explainable, defensible method over a more "sophisticated" one, per
   the phase's explicit instruction.

No composite/winner score is produced by this design — its output is a
per-block test statistic and p-value (as in Phase 16), plus descriptive
aggregates, nothing more.

## 20I. Model Comparison Semantics

`src/dashboard/multiseed_loader.py`'s `build_per_model_aggregates` output
schema was checked (and is unit-tested, see
`TestAggregatesDoNotRank.test_per_model_aggregate_has_no_ranking_field`)
to contain no `rank`, `score`, `is_best`, `winner`, or `tier` field. The
Phase 19 distinction — "highest observed mean" is a factual, descriptive
statement, never a selection — is carried forward unchanged into the
Phase 20 design: once the 105 runs exist, the system will be able to
report observed means, seed variability, fold variability, and anomaly
counts, and nothing that says "best model."

## 20J. Dashboard Update

The existing Phase 17/18 dashboard was inspected and its visual language
(navy/blue palette, `.stat`/`.note`/`.badge` CSS classes, vanilla JS,
stdlib `http.server`, no framework, localhost-default) was reused, not
reinvented. A new section, **"Multi-Seed Experiment (Phase 20)"**, was
added to `src/dashboard/static/index.html`, positioned right after the
existing "Model Comparison" (Phase 16/19) section and clearly described
as "a separate, planned experiment from the Phase 15 sweep and Phase 19
comparison above."

Because no results exist yet, the section currently renders (verified
live, see Section 20N):

```
Multi-seed experiment: NOT RUN

Planned: 7 models × 5 folds × 3 seeds = 105
Seeds: 42, 43, 44
Completed: 0 / 105
```

No `0.0.0.0`/public-access/tunnel code was added or modified —
`scripts/run_dashboard.py` was not touched at all in this phase (`git
diff` confirms zero changes to that file).

## 20K. Dashboard Data Model

`src/dashboard/multiseed_loader.py::build_multiseed_state` returns:

```
manifest_available, manifest_error, protocol {experiment_id,
  protocol_version, models, folds, seeds, total_expected_runs},
run_state_available, run_state_error, overall_status
  ("not_run" | "in_progress" | "complete"),
completed_runs, failed_runs, skipped_runs, total_expected_runs,
per_run [...], per_model [...], per_fold [...], per_seed [...],
anomalous_runs [...], note
```

This single shape already represents planned runs (`protocol`), completed
/failed/skipped counts, per-run/per-model/per-seed/per-fold aggregates,
and an anomaly list — so loading real future results requires no
frontend rewrite, only populating these same fields with real numbers.

Verified handling of all five required cases (unit tests, Section 20N):
1. No multi-seed results — `manifest_available: False`, `overall_status:
   "not_run"`.
2. Partially completed experiment — counts and per-run rows reflect
   exactly what exists; `overall_status: "in_progress"`.
3. Fully completed experiment — `overall_status: "complete"` **only**
   when `completed_runs == total_expected_runs` from the manifest (unit
   tested explicitly: 2/3 completed never reports `"complete"`).
4. Failed runs — recorded with `status: "failed"`, `macro_f1: null`,
   `metrics_available: False`, never crashing the loader.
5. Malformed result files — a malformed manifest or malformed per-run
   metrics JSON is caught by the existing `data_loader.safe_read_json`
   / `ReadResult` machinery and reported as an error string, never
   raised past the loader.

## 20L. Dashboard Comparison UX

Implemented in the new "Multi-Seed Experiment" section:
- **Overview**: planned/completed/failed/skipped counts (stat cards).
- **Model comparison table**: per-model mean/std/median/min/max
  Macro-F1, plus completed/failed/anomalous counts.
- **Anomalies**: explicitly listed (`model/fold/seed_N`) when present,
  never silently excluded from the table above.
- **Methodology**: a collapsible `<details>` block (kept concise, not a
  thesis chapter) explaining the 5-fold/3-seed design, why multiple seeds
  are used, what Macro-F1 means, and the design's limitations.

Per-seed and per-fold breakdown tables, and the full run matrix, are
implemented in the data layer (`per_fold`, `per_seed`, `per_run` in
`multiseed_loader.py`) and are ready to be surfaced in additional table
elements once real multi-run data exists to make such a large matrix
worth rendering; the current UI focuses on the per-model summary and
anomaly list, which is what has anything meaningful to show at 0/105
completed. This is a deliberate, documented scope choice, not an
oversight — the data needed for the fuller run-matrix view is already
present in the state object and unit-tested.

## 20M. Phase 18 Live Demo

`src/dashboard/inference.py`, `scripts/train_demo_checkpoint.py`, and the
Phase 18 "Live Demo" section/JS in `index.html` were not touched. `git
diff` confirms zero changes to `src/dashboard/inference.py`. The demo
checkpoint (model = indobert_base_p1, fold = fold_0) was not retrained,
moved, or overwritten — it still exists only inside
`.claude/worktrees/phase18-manual/models/demo/...`, exactly as before.
The Live Demo section and the new Multi-Seed section are visually and
textually distinct sections on the page; nothing in the Multi-Seed
section's copy implies the demo checkpoint would change based on
multi-seed results.

## 20N. Testing

Two new test files were added:

- **`tests/test_multiseed_protocol.py`** (7 tests): dry-run/combination
  enumeration produces exactly 105 unique combinations spanning exactly 7
  models × 5 folds × 3 seeds with no duplicates; `run_name` uniquely
  encodes fold+seed; the generated manifest reports 105 planned runs and
  has all required fields; a Phase 13/15-style single-seed artifact is
  never mistaken for a multi-seed result by `is_complete`.
- **`tests/test_dashboard_multiseed.py`** (9 tests): no-manifest and
  manifest-with-no-runs both report `not_run`; partial results load and
  count correctly; `overall_status` never reports `"complete"` early;
  malformed manifest/metrics files degrade gracefully; per-model
  aggregates contain no ranking field; top-level dashboard state exposes
  `multiseed` as a key distinct from `analysis`.

All fixtures use `tempfile.TemporaryDirectory` and synthetic JSON — no
real transformer checkpoint, GPU, or MPS training is required to run
these tests.

**Full suite result:**
```
uv run python -m unittest discover -s tests -p "test_*.py"
Ran 73 tests in 0.058s
OK
```
73 = 57 pre-existing (Phase 15–19, all still passing unchanged) + 16 new
Phase 20 tests (7 + 9).

## 20O. Integrity Checks

1. `dib_labeled.csv` checksum: `ed15416...` — unchanged, matches baseline.
2. `folds.json` checksum: `730cd82...` — unchanged, matches baseline.
3. Phase 15 artifacts (`results/metrics/`, `results/logs/`): no tracked
   file modified (`git status --short results/` shows no `M` entries).
4. Phase 16 artifacts (`results/analysis/transformer/*.json`, `.csv`,
   `plots/`): unmodified.
5. Phase 18 checkpoint (`models/demo/indobert_base_p1/fold_0/*`,
   worktree filesystem): unmodified, not moved, not retrained.
6. Phase 19 comparison artifacts
   (`results/analysis/transformer/comparison/*`): unmodified.
7. No training process was started — confirmed via `ps aux`, no
   `run_transformer*` process running.
8. No dashboard process was left running by this phase — the smoke-test
   instance launched for verification (port 8766) was explicitly stopped
   and confirmed dead. (Note: an unrelated dashboard process bound to
   `0.0.0.0:8765`, launched from a stale `.claude/worktrees/phase15-dashboard`
   checkout, was already running before Phase 20 started and is still
   running now — it was not started by this phase and was left untouched,
   per the "don't alter unrelated things" constraint, but is flagged here
   for visibility since it is a pre-existing LAN-exposed process outside
   this session's control.)
9. No public-access/network binding was introduced — `scripts/run_dashboard.py`
   was not modified in this phase; `git diff` confirms zero changes.
10. No new dependency was added — every new script/module uses only the
    Python standard library plus already-present project dependencies
    (`torch`, `yaml`, `pandas` via existing `loader`/`config_utils`).
11. The pre-existing `src/finetune/trainer.py` MPS modification remains
    unstaged and untouched (verified via `git diff -- src/finetune/trainer.py`
    showing exactly the same one hunk as before Phase 20 began).
12. Only Phase 20 files were staged for commit (verified below, Section
    20P) — `trainer.py` and all pre-existing untracked artifacts were
    explicitly excluded from `git add`.

## 20P. Git Discipline

Phase 20 branch: **`phase20-multiseed-protocol`**, created from `main` at
`8c4e6d0`.

Files staged and committed (exact list, verified via `git diff --cached
--name-status` before commit):
- `.gitignore` (added `results/multiseed/predictions/`)
- `docs/phases/PHASE_20_MULTISEED_TRANSFORMER_PROTOCOL.md`
- `results/multiseed/manifest.json`
- `scripts/generate_multiseed_manifest.py`
- `scripts/run_transformer_multiseed.py`
- `scripts/run_transformer_multiseed_single.py`
- `src/config_utils.py` (additive MPS seeding line)
- `src/dashboard/multiseed_loader.py`
- `src/dashboard/state.py` (additive `multiseed` field wiring)
- `src/dashboard/static/index.html` (additive Multi-Seed section)
- `tests/test_dashboard_multiseed.py`
- `tests/test_multiseed_protocol.py`

**Not committed**: the dataset, any model checkpoint, Phase 15/16/18/19
artifacts, the pre-existing `src/finetune/trainer.py` MPS change, and
every other pre-existing untracked file (`.claude/`, Phase 11–15 docs,
`external/original-drive/DATASETS/`, `results/logs/`,
`results/metrics/<model>/`, `scripts/run_transformer.py`,
`scripts/run_transformer_sweep.py`).

Commit message: `feat: prepare multi-seed transformer experiment`.

Not pushed. Not merged into `main`.

## 20Q. Final Report

See the end-of-turn summary in the session for the itemized final report
(starting HEAD, branch name, final commit hash, exact files changed,
protocol summary, seed-propagation design, artifact namespace, manifest
location, dry-run result, dashboard behavior, statistical design, test
results, checksum verification, artifact integrity, confirmation that no
training was executed, no public binding was added, the pre-existing
`trainer.py` change was preserved, nothing was pushed, and the final `git
status`).

**Stop condition acknowledged**: Phase 20 stops here. The 105 training
runs have not been executed. No model has been selected. No winner has
been declared. The Phase 18 demo checkpoint has not been modified.
