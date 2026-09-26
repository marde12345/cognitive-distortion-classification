# Phase 18 — Transformer Live Inference Demo

## 1. Objective

Build a small, reproducible live-inference capability for the dashboard's
Live Demo section, using ONE explicitly selected transformer checkpoint,
without rerunning the 35-run sweep, without modifying any existing
Phase 13/15/16 artifact, and without claiming the demo model is "the best
model."

## 18A. Reconnaissance (performed before any modification)

**FACT**, recorded at the start of this phase:
```
main HEAD: 116c8bd  merge: integrate Phase 17 dashboard enhancement
```
**OBSERVATION**: `main` had advanced since the end of Phase 17 (which left
`main` at `c2705d4`) — a merge of the Phase 17 dashboard-enhancement
branch had occurred. This was not an action taken by this phase; it is
recorded as the observed starting state.

**FACT**, inspected before writing any code:
- `src/finetune/trainer.py` (committed version, at `HEAD`): device
  selection is still `"cuda" if torch.cuda.is_available() else "cpu"` —
  **the Phase 12 MPS fix has never been committed to `main`**, it exists
  only as an uncommitted working-tree change there. `train_one_fold`
  accepts an explicit `device` argument, though, so a caller can pass
  `device="mps"` directly without needing that fix to be present at all.
  `train_one_fold` returns `(val_metrics, test_metrics, test_preds,
  test_probs, test_labels)` — **it does not return the trained model
  object**, and there is no `torch.save` anywhere in the file. This is
  exactly the "missing piece" the phase instructions asked to be reported
  before proceeding: **a checkpoint cannot be reconstructed from metrics
  alone, and the existing function has no way to hand back the trained
  weights to a caller.**
- `scripts/run_transformer.py` and `scripts/run_transformer_sweep.py`
  (Phase 12/14) are **also not committed to `main`** — they exist only in
  the main checkout's working tree, not in git history, so this worktree
  (created from `HEAD`) does not have them.
- `configs/models/indobert_base_p1.yaml`, `configs/base.yaml`, and all of
  `src/config_utils.py`/`src/loader.py`/`src/metrics.py` **are** committed
  and unchanged — sufficient to reconstruct the model configuration,
  tokenizer id, and dataset/fold access without any gap.
- `find models -type f` (real project directory): empty, confirming no
  checkpoint exists anywhere, consistent with Phase 11 onward.
- The Phase 17 dashboard's `build_live_demo_state` only checked for *any*
  file under `models/` and reported a generic disabled message — it had
  no concept of a specific demo checkpoint's model/fold identity.

**Resolution chosen, documented rather than silently assumed**: since
`train_one_fold` cannot hand back the trained model, and since
`scripts/run_transformer.py` isn't available on this branch to build on,
this phase (a) adds one small, additive, opt-in parameter to
`train_one_fold` (`checkpoint_path=None`, §18C) rather than modifying its
return signature or duplicating its ~120 lines of training logic, and
(b) writes a new, dedicated script (`scripts/train_demo_checkpoint.py`,
Option 1 from §18C) rather than depending on the uncommitted
`run_transformer.py`. Device selection for this new script calls
`train_one_fold(..., device="mps")` explicitly when MPS is available,
requiring **no change at all** to `trainer.py`'s device-selection branch.

## 18B. Explicit Demo Selection

**Explicitly selected**: `model = indobert_base_p1`, `fold = fold_0`.

**Why**: the Phase 16 report names `indobert_base_p1` as one candidate
"for explicit human review" for future checkpoint/live-demo work (mean
macro-F1 0.5724, no anomaly, alongside two comparably-strong alternatives
it also names). `fold_0` is the specific fold Phase 13 already trained
and fully documented as a real, reproducible experiment predating the
full sweep. This combination was named as a starting point in the Phase
18 task description itself.

**This is explicitly NOT an automatic "best model" selection** — Phase
16 never ranked models, and this phase does not either. Throughout this
report and the new code/UI copy, this checkpoint is referred to only as
**"the explicitly selected demo checkpoint,"** never as best/winner/
superior/production model.

## 18C. Existing Experiment Reference (Phase 13)

**FACT**, from `docs/phases/PHASE_13_FULL_DATA_SINGLE_FOLD_PILOT.md`
(read, not re-derived): `indobert_base_p1` / `fold_0`, full dataset
(3345/364/919 rows), 5 configured epochs (none early-stopped), device
`mps`, elapsed 521.30s, `test_macro_f1 = 0.5923`. This phase's demo
training run (§18E) reproduced this experiment's exact per-epoch values
(§18E), confirming the reference experiment's configuration was correctly
reconstructed from the committed repository state alone.

## 18D. Checkpoint Strategy

**Chosen approach**: Option 2 in spirit ("optional checkpoint flag,
defaults OFF"), implemented as an additive function parameter rather than
a CLI flag on a script that isn't committed here.

**Exact `trainer.py` diff** (the only change to this file):
```diff
 def train_one_fold(
     model_name, hf_model_id,
     train_df, val_df, test_df,
     cfg, paths,
     num_labels=11, max_len=128, batch_size=16, learning_rate=2e-5,
     weight_decay=0.01, warmup_ratio=0.1, num_epochs=5, patience=2,
-    fp16=True, device=None, fold_name="fold_0",
+    fp16=True, device=None, fold_name="fold_0", checkpoint_path=None,
 ):
```
plus, immediately before the existing `return` statement, a new `if
checkpoint_path is not None:` block (the only other change) that saves
`model.state_dict()` to `<checkpoint_path>/model.pt` and a metadata dict
to `<checkpoint_path>/checkpoint_meta.json`. **No existing line was
removed or altered.** Every existing call site (none of which exist on
this branch, since the sweep scripts aren't committed here — but the
contract holds regardless) that omits `checkpoint_path` gets `None`,
takes the pre-existing code path unchanged, and returns the exact same
5-tuple as before.

**Checkpoint location**: `models/demo/indobert_base_p1/fold_0/` — a
sibling of, not inside, `results/metrics|logs|predictions/`, matching the
instruction exactly. `scripts/train_demo_checkpoint.py` refuses to
proceed if this directory already exists and is non-empty (checked and
verified: no such conflict existed before this phase ran).

**Checkpoint contents** (`checkpoint_meta.json`, real file written this
phase):
```json
{
  "model_name": "indobert_base_p1",
  "hf_model_id": "indobenchmark/indobert-base-p1",
  "num_labels": 11,
  "max_len": 128,
  "fold_name": "fold_0",
  "seed": 42,
  "device_used": "mps",
  "training_config": {"batch_size": 16, "learning_rate": 2e-05, "weight_decay": 0.01,
                        "warmup_ratio": 0.1, "configured_epochs": 5, "completed_epochs": 5,
                        "patience": 2, "class_weighted_loss": true},
  "val_macro_f1": 0.5799861753387536,
  "test_macro_f1": 0.5923101427509724,
  "test_weighted_f1": 0.679074745383247,
  "test_accuracy": 0.6746463547334058
}
```
plus `model.pt` (475MB `state_dict`). No dataset content of any kind is
stored in either file. Both files are covered by the pre-existing
`.gitignore` rule `models/` (verified via `git check-ignore`) — the
475MB weights file will never be staged or committed.

## 18E. Training

**FACT**, exact command used:
```bash
uv run python scripts/train_demo_checkpoint.py --model indobert_base_p1 --fold fold_0
```
**FACT**, pre-training safety baseline recorded:
```
dib_labeled.csv SHA-256: ed154162554b68ba6980af8f3c7c01fa80a7962c4a0595fd61acafd142bdaea0
folds.json SHA-256:      730cd82a912828e902cdab26eb498c640195eca4901fde606f849372c5693c54
No run_transformer/train_demo process running.
No pre-existing models/demo/ checkpoint (no conflict).
```
Since this worktree's checkout does not include the untracked
`external/original-drive/DATASETS/` (deliberately never committed, per
Phase 6), a local, uncommitted symlink was created inside the worktree
(`external/original-drive/DATASETS -> <main checkout's DATASETS
directory>`) purely so the unmodified, read-only dataset could be read for
training. This symlink is not staged or committed as part of this phase
(verified in §Git below).

**FACT**, training completed successfully:
```
Device: mps
Dataset loaded: 4628 rows. Fold split: train=3345 val=364 test=919
  Epoch 1/5 | loss=1.9462 | val_macro_f1=0.4745
  Epoch 2/5 | loss=0.9794 | val_macro_f1=0.5313
  Epoch 3/5 | loss=0.5947 | val_macro_f1=0.5556
  Epoch 4/5 | loss=0.3517 | val_macro_f1=0.5639
  Epoch 5/5 | loss=0.2468 | val_macro_f1=0.5800
Training completed in 515.13s
val macro_f1  = 0.5800
test macro_f1 = 0.5923
```
**OBSERVATION**: every one of these five per-epoch loss/macro-F1 values,
and the final test macro-F1 (0.5923), are **byte-for-byte identical** to
Phase 13's originally reported values — direct confirmation that
`seed=42` plus the reconstructed configuration reproduced the exact same
experiment. No warnings beyond the standard, already-previously-observed
Hugging Face "unauthenticated requests" notice and the expected
newly-initialized-classifier-head notice (identical to every prior
transformer run in this project, e.g. Phase 13/15). No anomaly occurred.

**Post-training verification, FACT**:
```
dib_labeled.csv SHA-256 (after): ed154162554b68ba6980af8f3c7c01fa80a7962c4a0595fd61acafd142bdaea0  (unchanged)
folds.json SHA-256 (after):      730cd82a912828e902cdab26eb498c640195eca4901fde606f849372c5693c54  (unchanged)
results/metrics/indobert_base_p1/fold_0.json: read, confirmed present and unaltered
                                               (this script never calls
                                               save_predictions_and_metrics,
                                               by design -- see the script's
                                               own closing comment)
results/logs/transformer_sweep.json: completed=34, failed=0, skipped=1 (unchanged, Phase 15's exact numbers)
Only new artifacts created: models/demo/indobert_base_p1/fold_0/{model.pt, checkpoint_meta.json}
```

## 18F. Inference

**New script**: `scripts/run_inference.py`. **New module**:
`src/dashboard/inference.py` (`DemoModel` class + `LABEL_NAMES`), shared
by both the CLI script and the dashboard's `/api/infer` endpoint so the
loading/inference logic exists in exactly one place.

**Input format**: raw text string (CLI: `--text "..."`; HTTP: JSON body
`{"text": "..."}`). Tokenization uses the exact same
`truncation=True, padding="max_length", max_length=<meta.max_len>`
settings the training pipeline's `TextDataset` uses, read from the
checkpoint's own recorded `max_len` rather than hardcoded.

**Output format**: `predicted_label` (int 0-10), `predicted_label_name`
(string), `confidence` (float), `top_k` (list of `{label, label_name,
probability}`), `model_name`, `fold`.

**Label names are not invented.** `LABEL_NAMES` in
`src/dashboard/inference.py` is the exact reverse mapping of the
`label_map` already documented in this project's own preprocessing
provenance
(`external/original-drive/DATASETS/PREPROCESSING/COGNITIVE DISTORTION/
configs/preprocessing.yaml`, discovered and recorded in the Phase 6.1
reconnaissance) — read again directly this phase to confirm it verbatim
(§Reproduction below shows the exact command used to re-verify it). Label
1's three merged source categories ("Jumping to Conclusions", "Mind
Reading", "Fortune-telling") are represented by the project's own
documented merged name, "Jumping to Conclusions."

**Determinism, FACT, verified live**: identical input text sent twice to
`/api/infer` produced byte-identical JSON responses (`diff` on the two
response bodies: empty). Inference runs under `torch.no_grad()` and the
model is in `.eval()` mode; model weights are never modified after
loading (no optimizer, no `.backward()`, no `.step()` anywhere in
`inference.py`).

**Real inference, verified live** (not a fixture):
```
Input: "Saya selalu gagal dalam segala hal, semua orang pasti membenci saya."
-> predicted_label_name: "Labeling", confidence: 0.394

Input: "Semua orang selalu membenci saya, tidak ada yang mau berteman dengan saya." (CLI)
-> predicted_label_name: "Jumping to Conclusions", confidence: 0.815
```

**Invalid/missing-checkpoint handling, verified live**: an empty/
whitespace-only `text` returns HTTP 400 with a clear error, never a
crash or a fabricated prediction. Pointed at a results directory with no
checkpoint, `/api/infer` returns HTTP 503 with `"inference unavailable:
no demo checkpoint available"` — the dashboard never claims a model is
live when it is not.

## 18G/18H. Dashboard Integration, Plain Language, and Safety Scope

**FACT**: the Live Demo section (`src/dashboard/static/index.html`) now
shows, when a checkpoint is present: the exact plain-language framing
requested ("Enter a piece of text below. The model will estimate which of
the 11 cognitive-distortion categories best matches the text."), an
explicit identification of the checkpoint ("Explicitly selected demo
checkpoint: model = indobert_base_p1, fold = fold_0 (test macro-F1 at
training time: 0.5923)... not a claim that this is the best-performing
model among the seven evaluated"), a text box, a Classify button, and a
result panel showing the predicted category, confidence, and top-3
alternatives. A collapsible "How this works / important limitations"
`<details>` block states verbatim: **"This is an experimental research
model, not a clinical, diagnostic, or medically validated tool,"** and
that a prediction "must not be treated as a mental-health assessment,
diagnosis, or professional judgment about the person who wrote the text."
No medical/psychological interpretation beyond the project's own 11
existing research labels appears anywhere in the new UI copy.

**API**: `POST /api/infer` with JSON body `{"text": "..."}`, implemented
as a `do_POST` handler on the existing stdlib `http.server`-based
`Handler` class — no new dependency, no new server, no frontend
framework. The demo model is loaded at most once (lazily, on first
request, cached for the server process's lifetime) rather than per
request.

**Design preserved**: no new CSS framework, no new color/typography
system — the new form uses the existing navy/blue palette variables and
existing `<details>`/`.note` conventions already established in Phase 17.

## Validation Summary

**FACT**, full test suite:
```
uv run python -m unittest discover -s tests -p "test_*.py"
----------------------------------------------------------------------
Ran 57 tests in 0.039s

OK
```
(44 pre-existing Phase 15/17 tests, unmodified and all still passing +
13 new Phase 18 tests in `tests/test_dashboard_inference.py`, covering
checkpoint-metadata loading (valid/missing/malformed), the label-name
mapping's completeness and provenance, device resolution, live-demo
checkpoint detection (none/present/weights-missing/malformed-metadata),
and input-validation logic.)

**Real, non-fixture validation performed live** (not merely unit tests):
checkpoint training to completion; dashboard `/api/state` correctly
reporting the real checkpoint's identity and metrics; two real `/api/infer`
calls producing genuine, different, sensible predictions for two different
Indonesian sentences; a determinism check (identical repeated input ->
identical output); the standalone CLI script producing the same kind of
output independently; empty-input rejection (400); and no-checkpoint
graceful degradation (503) against a separate, empty results directory.

**Regression check**: Phase 17's dashboard sections (overview, matrix,
completed-runs table, model detail, and all Phase 16 analysis
sections/plots) were not modified in this phase beyond the Live Demo
section and the additive `live_demo` state fields — confirmed by the fact
that all of Phase 17's own tests (`test_dashboard_analysis.py`, 28 tests)
still pass unmodified.

## Safety / Integrity

**FACT**:
```
dib_labeled.csv SHA-256 before/after: ed154162554b68ba6980af8f3c7c01fa80a7962c4a0595fd61acafd142bdaea0 / SAME
folds.json SHA-256 before/after:      730cd82a912828e902cdab26eb498c640195eca4901fde606f849372c5693c54 / SAME
results/metrics/**:      unchanged (this phase's training script never writes there)
results/logs/**:          unchanged (transformer_sweep.json still reports completed=34/failed=0/skipped=1)
results/predictions/**:   unchanged
results/analysis/transformer/**: unchanged (not read or written by this phase's new code at all)
No sweep process was started at any point in this phase.
No existing experiment (Phase 13/15) artifact was overwritten.
```

## Limitations

- Single checkpoint, single fold, single seed (42) — this demo
  demonstrates one specific trained run, not a validated or averaged
  model.
- Experimental research status only; explicitly not production-ready,
  not clinically validated, and not a diagnostic tool (stated both in
  this report and in the dashboard's own UI copy).
- No claim is made, here or in the UI, that `indobert_base_p1` is
  superior to the other six evaluated models — Phase 16's descriptive
  statistics named it only as one reasonable candidate among several.
- The 475MB checkpoint is intentionally excluded from git (already
  covered by the pre-existing `models/` ignore rule) — reproducing this
  demo elsewhere requires re-running the training command below, not
  pulling a committed artifact.
- Model loading takes a few seconds on first `/api/infer` request (lazy
  load); this is a one-time cost per server process, not per request.

## Reproduction

Train (or recreate) the demo checkpoint:
```bash
uv run python scripts/train_demo_checkpoint.py --model indobert_base_p1 --fold fold_0
```
Run standalone inference:
```bash
uv run python scripts/run_inference.py \
  --checkpoint models/demo/indobert_base_p1/fold_0 \
  --text "some Indonesian text here"
```
Launch the dashboard (Live Demo becomes active once the checkpoint above exists):
```bash
uv run python scripts/run_dashboard.py --results-dir results --host 127.0.0.1 --port 8765
```
Re-verify the label-name provenance directly from the project's own
preprocessing config (read-only; not modified by this or any phase):
```bash
grep -A 15 "^label_map:" "external/original-drive/DATASETS/PREPROCESSING/COGNITIVE DISTORTION/configs/preprocessing.yaml"
```

## Conclusion

A single, explicitly selected demo checkpoint (`indobert_base_p1` /
`fold_0`) was trained via one small, additive, opt-in change to
`train_one_fold` (never touching its existing behavior for any caller
that omits the new parameter), reproducing Phase 13's exact reference
result byte-for-byte. A shared inference module now backs both a
standalone CLI script and a new dashboard `/api/infer` endpoint, verified
live with real (not fabricated) predictions, confirmed deterministic, and
verified to fail gracefully — never falsely — when no checkpoint exists
or input is invalid. The dashboard's Live Demo section now identifies the
checkpoint explicitly, explains its output in plain language, and states
its experimental/non-clinical scope clearly. All 57 tests pass (44
pre-existing, unmodified, plus 13 new). The dataset, all Phase 13/15/16
artifacts, and `main` itself remain untouched — this work exists only on
the `phase18-live-inference` branch.
