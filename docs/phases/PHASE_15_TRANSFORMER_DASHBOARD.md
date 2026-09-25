# Phase 15 — Transformer Experiment Dashboard

## 1. Objective

Build a lightweight, local, read-only dashboard for monitoring and
inspecting the 7-model × 5-fold (35-run) transformer sweep while a
separate agent runs it in the background, without starting, stopping, or
otherwise touching the sweep process or its artifacts. The already-completed
`indobert_15g` 5-fold results serve as the first real-data validation
target.

## 2. Repository State Before Changes

**FACT**, `git status --short` at the start of this phase (main checkout):
```
 M pyproject.toml
 M src/finetune/trainer.py
 M uv.lock
?? docs/phases/PHASE_11_TRANSFORMER_EXECUTION_READINESS.md
?? docs/phases/PHASE_12_MPS_AND_EXECUTION_INFRASTRUCTURE.md
?? docs/phases/PHASE_13_FULL_DATA_SINGLE_FOLD_PILOT.md
?? docs/phases/PHASE_14_TRANSFORMER_FULL_SWEEP_INFRASTRUCTURE.md
?? external/original-drive/DATASETS/
?? results/logs/
?? results/metrics/indobert_15g/
?? results/metrics/indobert_base_p1/
?? scripts/run_transformer.py
?? scripts/run_transformer_sweep.py
```
HEAD unchanged since Phase 10B (`fd68470`). No dashboard code, `results/`
data, or test suite existed in the repository before this phase.

**FACT**, at reconnaissance time `scripts/run_transformer_sweep.py`
(PID confirmed via `ps aux`) was actively running in the background,
launched by another agent. It was still running, unmodified and
uninterrupted, at the end of this phase — confirmed by re-checking the
process table and observing `completed_runs` advance in the manifest
between the start and end of this phase's work (5 → 6 completed runs while
this phase was in progress).

**FACT**, no dashboard or web framework (Flask, FastAPI, Streamlit, etc.)
was present in `pyproject.toml`'s dependency list, which was:
`gensim, numpy, pandas, pyyaml, scikit-learn, torch, transformers`. No
`pytest` was installed either (`uv run python -c "import pytest"` failed
with `ModuleNotFoundError`).

## 3. Reconnaissance

**FACT**, inspected directly before writing any code:
- `results/logs/transformer_sweep.json` — the sweep manifest, containing
  `created_at`, `device`, `models[]`, `folds[]`, `total_runs`,
  `completed_runs`, `failed_runs`, `skipped_runs`, and a `runs[]` array of
  `{model, fold, status, start_time, end_time, elapsed_seconds, exit_code}`.
  At first read: 5 completed, 1 skipped (`indobert_base_p1/fold_0` —
  pre-existing from an earlier phase's pilot run, per Phase 14's own
  report), 0 failed.
- `results/logs/indobert_15g/fold_0.json` — per-run execution metadata:
  `model_name`, `hf_model_id`, `fold`, `device`, `configured_epochs`,
  `batch_size`, `max_seq_length`, `status`, `start_time`, `end_time`,
  `elapsed_seconds`, `completed_epochs`, and a `history[]` array of
  `{epoch, train_loss, val_macro_f1}`.
- `results/metrics/indobert_15g/fold_0.json` — per-run evaluation metrics:
  `macro_f1`, `weighted_f1`, `accuracy`, `per_class{}`, and the same
  `history[]` array duplicated from the log.
- `results/predictions/<model>/<fold>.csv` — per-run prediction files,
  confirmed present for `indobert_15g` but explicitly not used to derive
  sweep status, per instruction.
- No `tests/` directory existed anywhere in the repository.

**OBSERVATION**: the manifest's `status` values seen were `"completed"`
and `"skipped"`; `"failed"`, `"running"`, and `"pending"` (pair absent from
`runs[]`) were not observed live and were validated only through
synthetic fixtures (Section 11).

## 4. Technology Selection

**Chosen**: Python standard library only — `http.server`
(`ThreadingHTTPServer`) for the backend, a single static HTML page with
inline CSS/vanilla JS (polling `fetch`) for the frontend, `unittest` for
tests.

**Reason**: no web/dashboard framework was already a project dependency
(Section 2/3), and the instructions explicitly prefer the smallest
appropriate technology and forbid adding a large frontend framework.
Adding Flask/FastAPI/Streamlit would have required modifying
`pyproject.toml`/`uv.lock` for a task that a few hundred lines of stdlib
code fully cover: reading a handful of small JSON files on a timer and
rendering tables. `pyproject.toml` and `uv.lock` were therefore **not**
modified by this phase at all.

## 5. Architecture

```
results/ (read-only)
   |
   +-- src/dashboard/data_loader.py   -- safe_read_json, load_manifest,
   |                                     load_run_log, load_run_metrics,
   |                                     prediction_csv_exists
   |                                     (never raises; returns ReadResult
   |                                     with data=None + error string on
   |                                     any failure)
   |
   +-- src/dashboard/state.py         -- build_dashboard_state (top-level),
   |                                     build_overview, build_matrix,
   |                                     build_completed_runs_table,
   |                                     build_model_detail, build_run_detail
   |                                     (pure functions, no I/O side
   |                                     effects beyond data_loader reads,
   |                                     no HTTP/HTML knowledge)
   |
   +-- scripts/run_dashboard.py       -- StateStore (holds last-good state
   |                                     behind a lock, refreshed by a
   |                                     daemon thread on --poll-interval),
   |                                     ThreadingHTTPServer routes:
   |                                     GET / -> static/index.html
   |                                     GET /api/state
   |                                     GET /api/model?model=...
   |                                     GET /api/run?model=...&fold=...
   |
   +-- src/dashboard/static/index.html -- presentation only; polls
                                           /api/state every 5s client-side
                                           (plus a server-side background
                                           refresh thread independent of
                                           that)
```

Data access/state-derivation is fully separated from presentation:
`data_loader.py` and `state.py` contain no HTTP or HTML code and are
tested directly via `unittest`; `run_dashboard.py` only wires them to
HTTP routes; `index.html` only renders whatever JSON those routes return.

## 6. Dashboard Features

- **Sweep Overview**: `N / 35 completed` headline, plus completed /
  running / failed / skipped / pending counts, device, total models,
  total folds, and average runtime of completed runs — all clearly
  sourced from the manifest, with a note distinguishing the measured sum
  of completed-run elapsed time from any notion of total sweep wall-clock.
  No ETA is shown (none was derivable with confidence from a single
  device/manifest without over-claiming, so none was fabricated).
- **Model × Fold Matrix**: dynamically built from `manifest["models"]` and
  `manifest["folds"]` (never hardcoded), with a compact ✓/✕/…/·/— cell per
  status and a color-only legend (completed/running/failed/pending/skipped).
- **Completed Runs table**: model, fold, runtime, accuracy, macro-F1,
  weighted-F1, status — in manifest/execution order, explicitly not
  sorted by metric, with a note stating it is not a leaderboard.
- **Model Detail** (indobert_15g first, but works for any model in the
  manifest): per-fold runtime/accuracy/macro-F1/weighted-F1 table, a
  "Cross-fold descriptive statistics" block (mean/std, explicitly labeled
  descriptive, not comparative), and a fold selector showing real
  per-epoch training history (train loss, val macro-F1) from the log/metric
  files.
- **Run Detail panel**: clicking a completed run shows model, fold,
  status, device, configured/completed epochs, batch size, max sequence
  length, runtime, accuracy/macro-F1/weighted-F1, and whether a
  predictions file exists — factual fields only, no qualitative labels.

## 7. Live Refresh

Two independent polling loops, both simple interval-based reads (no
websockets/queues/DB):
1. **Server-side**: a daemon thread in `run_dashboard.py` calls
   `state.build_dashboard_state()` every `--poll-interval` seconds
   (default 5s) and stores the result behind a lock.
2. **Client-side**: the static page calls `GET /api/state` every 5 seconds
   via `setInterval`/`fetch`, and re-renders the overview, matrix, and
   completed-runs table on each tick; the model-detail panel refetches
   `/api/model` only when the model list changes or the user changes the
   selection.

The dashboard was left running against the real, still-active sweep
during validation and was observed to pick up the manifest's progression
(5 → 6 completed runs) without restart.

## 8. Partial JSON Handling

`data_loader.safe_read_json` never raises: a missing file yields
`ReadResult(None, "missing")`; a `json.JSONDecodeError` yields
`ReadResult(None, "malformed: <reason>")`. `state.build_dashboard_state`
takes an optional `previous_state`; when the manifest read fails, it
returns a shallow copy of `previous_state` with `manifest_stale=True` and
`manifest_error` set, instead of raising or rendering empty/broken data.
If there is no previous state yet, it returns an explicit "waiting for
manifest" placeholder (`overview: None`, empty matrix). The frontend shows
a small "Updating…" banner whenever `manifest_stale` is true. This exact
behavior was validated live (Section 12) by truncating a copy of the real
manifest mid-JSON and observing the dashboard retain its last good state.

## 9. Sweep State Detection

`state.build_matrix` looks up each `(model, fold)` pair (from
`manifest["models"] x manifest["folds"]`, built dynamically) against the
manifest's `runs[]` list. If present, the cell's status is the run's own
`status` field, taken verbatim (so `"completed"`, `"failed"`, `"running"`,
`"skipped"`, or any other value the sweep runner writes is surfaced
as-is). If a pair is **absent** from `runs[]`, it is classified
`"pending"` — the sweep has not reached it yet. `completed_runs`,
`failed_runs`, and `skipped_runs` counters in the overview are taken
directly from the manifest's own top-level fields (measured); `pending`
and `running` counts are derived by scanning `runs[]`.

## 10. Real indobert_15g Validation

**FACT**, real per-fold values read from
`results/metrics/indobert_15g/fold_{0..4}.json` and displayed by the
dashboard's Model Detail view:

| fold | runtime (from log) | accuracy | macro-F1 | weighted-F1 |
|------|--------------------|----------|----------|-------------|
| fold_0 | 635.66s | 0.6605 | 0.5665 | 0.6679 |
| fold_1 | 510.74s | 0.6537 | 0.5557 | 0.6586 |
| fold_2 | 513.08s | 0.6306 | 0.5527 | 0.6394 |
| fold_3 | 510.12s | 0.6315 | 0.5709 | 0.6339 |
| fold_4 | 516.90s | 0.6576 | 0.5739 | 0.6602 |

Cross-fold descriptive statistics (mean, std), as computed by
`state._mean_std` and shown in the dashboard's "Cross-fold descriptive
statistics" block: accuracy mean 0.6468 / std 0.0146; macro-F1 mean
0.5639 / std 0.0093; weighted-F1 mean 0.6520 / std 0.0146. These are
descriptive only — the dashboard does not compare `indobert_15g` against
any other model or declare it superior.

## 11. Testing

`tests/test_dashboard_data.py`, `unittest`, all fixtures written to
`tempfile.TemporaryDirectory()` — nothing under `results/` was created,
read-as-fixture, or modified by the test suite. 16 tests, covering every
case listed in the phase instructions:

1. no manifest (`test_no_manifest_no_previous_state`, `test_missing_manifest`)
2. empty sweep (`test_empty_sweep_all_pending`)
3. partially completed sweep (`test_partially_completed_sweep`)
4. completed run (`test_completed_run_reads_metrics`)
5. failed run (`test_failed_run`)
6. currently running run (`test_running_run`)
7. pending run (`test_pending_run_not_in_manifest_runs_list`)
8. skipped run (`test_skipped_run`)
9. malformed/incomplete JSON (`test_malformed_manifest_retains_error`,
   `test_malformed_manifest_retains_previous_state`)
10. missing metric file (`test_completed_run_missing_metrics_file`,
    `test_missing_metric_file_for_model_detail`)
11. missing execution log (`test_missing_execution_log_for_model_detail`)
12. all 35 (generalized: all runs in a manifest) completed
    (`test_all_runs_completed_cross_fold_stats`)

**FACT**, actual command and outcome:
```
$ uv run python -m unittest tests.test_dashboard_data -v
...
Ran 16 tests in 0.010s
OK
```

## 12. Validation

**FACT**, dashboard started successfully against the real repository:
```
uv run python scripts/run_dashboard.py --results-dir /Users/MAC/Projects/Tesis-Mbak-Ai/results --port 8765
```
`GET /api/state` returned `manifest_stale: false`, `completed_runs: 6`
(the sweep had advanced from 5 to 6 completed runs between reconnaissance
and this check — confirming a live, unmodified, still-running process),
matching `results/logs/transformer_sweep.json` read directly. `GET
/api/model?model=indobert_15g` and `GET
/api/run?model=indobert_15g&fold=fold_0` returned the real per-fold
metrics and log fields shown in Section 10, sourced from
`results/metrics/indobert_15g/` and `results/logs/indobert_15g/`
respectively. `GET /` served the static dashboard page.

**FACT**, partial-sweep behavior: the real manifest at validation time was
`6/35` (indobert_15g's 5 folds + 1 pre-existing skipped run), with the
remaining 29 `(model, fold)` pairs correctly classified `"pending"` in the
matrix, none hardcoded.

**FACT**, malformed-manifest handling was validated using a temporary
fixture directory (`mkdir`-created under the job's own tmp path, **not**
under `results/`): a copy of the real manifest was read successfully
(`manifest_stale: false`, `completed: 6`), then overwritten with a
truncated `{"models": ["a",` and re-polled — the dashboard returned
`manifest_stale: true`, `manifest_error: "malformed: Expecting value:
line 1 column 17 (char 16)"`, and **retained** `completed: 6` from the
last good read. Restoring the fixture's manifest caused the dashboard to
recover (`manifest_stale: false`) on the next poll. The real
`results/logs/transformer_sweep.json` was never modified during this
test — only the isolated fixture copy was.

**FACT**, live refresh was validated by starting the dashboard against the
real `results/` directory a second time later in this phase and observing
`completed_runs` in `/api/state` match the manifest's then-current value
(6), confirming the dashboard reflects the sweep's progress without a
server restart.

## 13. Dataset Safety

**FACT**, verified via `git status --short` and `git diff --stat` in the
worktree used for this phase (Section 14 explains why a worktree was
still meaningful here): the dashboard implementation added only new files
under `scripts/run_dashboard.py`, `src/dashboard/`, and `tests/`. No
command in this phase wrote to, renamed, or deleted anything under
`external/original-drive/DATASETS/`. The dashboard code only ever opens
files under `results/logs/` and `results/metrics/` for reading (`.read_text()`
calls only — no write/open-for-write calls anywhere in `data_loader.py`
or `state.py`).

## 14. external/original-drive Safety

**FACT**: `external/original-drive/DATASETS/` was not referenced by any
dashboard code path (`data_loader.py` only ever touches `results/logs`,
`results/metrics`, and `results/predictions`), and no shell command in
this phase touched that directory. Confirmed absent from `git status`
output for the entire phase.

## 15. Notebook Safety

**FACT**: `notebooks/` was not referenced anywhere in the dashboard
implementation and does not appear in `git status --short` for this
phase. No notebook file was opened, read, or modified.

## 16. Git Status

**FACT**, work for this phase was done in a git worktree
(`.claude/worktrees/phase15-dashboard`, branch
`worktree-phase15-dashboard`) created specifically to keep edits isolated
from the main checkout, per this session's background-job isolation
policy. Because the worktree branches from `origin/main` and the real
`results/`, `scripts/run_transformer*.py`, `src/finetune/`, `configs/`,
and `external/` changes in the main checkout are uncommitted/untracked,
none of those paths exist inside the worktree at all — they were
structurally impossible for this phase's code to modify from within it.
Read access to the real experiment artifacts for validation (Sections 10,
12) was done via an absolute `--results-dir` path pointing at the main
checkout's `results/` directory, opened strictly read-only.

`git status --short` inside the worktree at the end of this phase:
```
?? scripts/
?? src/
?? tests/
```
(each containing only the new dashboard files listed in Section 17;
`__pycache__/` directories generated during test runs were removed before
finishing).

## 17. Files Changed

Created (all new, none pre-existing):
- `scripts/run_dashboard.py`
- `src/dashboard/__init__.py`
- `src/dashboard/data_loader.py`
- `src/dashboard/state.py`
- `src/dashboard/static/index.html`
- `tests/test_dashboard_data.py`
- `docs/phases/PHASE_15_TRANSFORMER_DASHBOARD.md` (this file)

Modified: none. `pyproject.toml` and `uv.lock` were not touched — no new
dependency was needed.

## 18. Known Limitations

- No ETA/remaining-time projection is shown. A confident estimate would
  need a defensible model-runtime-variance assumption not available from
  a single manifest snapshot; omitting it avoids presenting a speculative
  number as authoritative, per instruction.
- The dashboard has no authentication and binds to `127.0.0.1` only —
  appropriate for a local research tool, not intended for exposure beyond
  localhost.
- `"running"` and `"failed"` status handling was validated only through
  synthetic fixtures (Section 11), not against a real run of those
  statuses, because none occurred live during this phase's window.
- The client-side poll interval (5s) and server-side poll interval
  (default 5s, configurable via `--poll-interval`) are independent; a
  user pointing the server at a much slower filesystem could see a lag up
  to one server poll interval before a client refresh reflects it.
- No pinning/versioning of the dashboard's own dependencies beyond what
  is already in `pyproject.toml`, since none were added.

## 19. How to Start the Dashboard

From the repository root:
```
uv run python scripts/run_dashboard.py --results-dir results --port 8765
```
Then open `http://127.0.0.1:8765/` in a browser. Optional flags:
`--poll-interval <seconds>` (default 5.0) controls how often the server
re-reads the manifest.

## 20. Conclusion

A local, read-only, stdlib-only dashboard was implemented, tested (16/16
`unittest` cases passing), and validated against the real, currently
partial `indobert_15g`-complete transformer sweep (6/35 at validation
time), including live refresh and malformed-manifest resilience using an
isolated fixture. No experiment or training process was started, stopped,
or modified by this phase — the transformer sweep (`scripts/run_transformer_sweep.py`)
was confirmed still running, unmodified, at both the start and end of
this phase's work. No changes were made to `results/`,
`src/finetune/`, `scripts/run_transformer.py`,
`scripts/run_transformer_sweep.py`, `configs/`,
`external/original-drive/`, or `notebooks/`. Nothing was staged or
committed, per instruction; the worktree branch
(`worktree-phase15-dashboard`) is left for review.
