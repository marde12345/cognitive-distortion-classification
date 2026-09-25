# Phase 7 — Execution Readiness Audit

## Objective

Determine whether the repository is ready for controlled local execution of
the thesis pipeline (baselines first, transformers later), and identify
exactly what remains before Phase 8 execution — without training anything,
running the full pipeline, or modifying the notebook or the Drive snapshot.

## Scope

Read-only/audit only. No transformer training. No full cross-validation. No
regeneration of existing baseline results. No model downloads. No Google
Drive access. No modification of `notebooks/Modeling_Cognitive_Distortion (2).ipynb`.
No modification of `external/original-drive/`. One lightweight, read-only
metrics re-validation was performed (reusing Phase 4B's method); no other
execution was attempted beyond confirming import/config-loading failure
modes, which themselves created no files.

## Repository State

**BEFORE (git status --short):**
```
?? docs/phases/
?? external/original-drive/DATASETS/
```
**BEFORE (git log --oneline -5):**
```
efc1f8d  feat: reconstruct metrics module
2fcfe00  chore: reconcile recovered source modules
b71999d  chore: add original Drive snapshot for provenance
a5dcdff  Repository cleanup: establish uv project foundation
feccab5  Initial thesis snapshot: original Colab notebook, unmodified
```
FACT: 66 files currently tracked in Git (`git ls-files | wc -l`).

## Git Safety Audit

FACT: No tracked file exceeds 1MB (`git ls-files -z | xargs -0 du -k` sorted
descending — largest tracked file is the notebook at 116KB, followed by
`uv.lock` at 100KB; every other tracked file is under 12KB).

FACT: `external/original-drive/DATASETS/` remains untracked and unprotected
by `.gitignore` — unchanged since Phase 6/6.1/6.2's findings, still pending
the `.gitignore` decision proposed (not applied) in those phases.

**FACT — new finding this phase, not previously reported:** the existing
`.gitignore` rule `models/` (line 14, no leading slash — matches a directory
named `models` at *any* depth) incorrectly also matches
`external/original-drive/MODELING/COGNITIVE DISTORTION/configs/models/` —
the directory holding all 12 per-model YAML configs (`tfidf_lr.yaml`,
`indobert_base_p1.yaml`, etc.). Verified via
`git check-ignore -v` (matched `.gitignore:14:models/`) and `git ls-files`
on that directory returning empty. This is a **collision between two
different meanings of "models"**: the rule was written to protect model
*checkpoints* (correct intent, per Phase 1's original `.gitignore` design),
but it also silently hides model *configuration* (small YAML files with
real provenance/reproducibility value, already inventoried and read in
Phase 5 §6). Only `configs/base.yaml` is currently tracked; none of the 12
`configs/models/*.yaml` files are. This was not caught in any prior phase's
`.gitignore` audit (Phase 6 audited the `DATASETS/` collision with the same
`data/` rule pattern but did not check the `MODELING/` tree's `configs/`
subdirectory against the `models/` rule).

RECOMMENDATION: narrow the `models/` rule (e.g. anchor it or rename to
target only actual checkpoint directories) before the next commit that
touches this area. Not applied — reported only, per this phase's explicit
instruction not to modify `.gitignore` silently.

FACT: no other large files, checkpoints, or prediction dumps are currently
tracked. The already-committed baseline prediction CSVs (Phase 6.2/`b71999d`)
remain small (12KB each, confirmed above) and were a deliberate prior
decision, not a new finding.

## Local Drive Snapshot Audit

Inventory of `external/original-drive/` relevant to execution (no files
copied, moved, or modified):

- **Datasets**: `DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION/` —
  `dib_labeled.csv` (2,500,687 bytes), `folds.json` (440,419 bytes),
  `dib_groups.json` (86,811 bytes), `dib_labeled.sha256`, `folds.sha256`,
  plus a `reports/` subdirectory of pipeline logs. `DATASETS/RAW DATASETS/`
  and `DATASETS/PREPROCESSING/` also present (mapped in Phase 6.1, unchanged).
- **Configuration**: `MODELING/COGNITIVE DISTORTION/configs/base.yaml` +
  `configs/models/*.yaml` (12 files — see Git Safety Audit above for their
  tracking gap).
- **Source**: `MODELING/COGNITIVE DISTORTION/src/{config_utils.py,loader.py,
  baselines/sklearn_baseline.py,finetune/trainer.py}` — already verified
  byte-identical to the reconciled `src/` copies in Phase 3.2; re-confirmed
  present and unchanged this phase (not re-hashed, since no modification
  event occurred that would require re-verification).
- **Existing results**: `MODELING/COGNITIVE DISTORTION/results/{metrics,
  predictions,figures,tables}/` — metrics and predictions populated for 4
  baselines (`majority_class`, `tfidf_lr`, `tfidf_svm`, `svm_word2vec`);
  `figures/` and `tables/` are empty directories.
- **Model/checkpoint directories**: `MODELING/COGNITIVE DISTORTION/models/`
  — 12 correctly-named subdirectories, confirmed still empty on disk (no
  checkpoint files for any model, transformer or otherwise).
- **Scripts/notebooks/logs** (under `MODELING/COGNITIVE DISTORTION/`):
  `scripts/`, `notebooks/`, `logs/` — all empty, unchanged from Phase 5's
  findings.

## Dataset Inventory

| Path | Size | Type | Raw/intermediate/processed | Referenced by current pipeline? | Required for baseline? | Required for transformer? |
|---|---|---|---|---|---|---|
| `external/original-drive/DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION/dib_labeled.csv` | 2,500,687 B | CSV | Processed (final) | **FACT** — `src/config_utils.py` line 90: `"dataset_csv": os.path.join(processed_dir, "dib_labeled.csv")`, opened via `pd.read_csv` in `validate_dataset`/`loader.load_dataset` | Yes | Yes |
| `.../folds.json` | 440,419 B | JSON | Processed (final) | **FACT** — line 91, opened in `load_folds`, hash-cross-checked against the CSV | Yes | Yes |
| `.../dib_groups.json` | 86,811 B | JSON | Processed (final) | **FACT** — path defined at line 92 (`"groups_json"`), but **grep of all 5 reconciled `src/` files for `groups_json` finds only this one definition site — never opened/read anywhere.** Unchanged from Phase 5's finding. | No (not currently required by any traced code path) | No |
| `.../dib_labeled.sha256`, `folds.sha256` | 64 B each | text | Sidecar checksums | Not directly opened by `src/`; `config_utils.py` computes its own SHA-256 via `_sha256_file()` rather than reading these sidecar files | No | No |
| `external/original-drive/DATASETS/RAW DATASETS/**` | ~805MB combined (Phase 6 finding) | CSV/ZIP | Raw | Not referenced by any file in `src/` (only by the separate preprocessing pipeline mapped in Phase 6.1) | No | No |
| `external/original-drive/DATASETS/PREPROCESSING/**/data/interim/*.csv` | up to 2.6MB each | CSV | Intermediate | Not referenced by `src/` (preprocessing-pipeline-internal only) | No | No |

No dataset file was assumed to be relevant merely by file extension — every
"required" determination above traces to an actual `os.path.join`/`open`/
`pd.read_csv` call in the reconciled source, cross-referenced against Phase
5's original inventory.

## Configuration Path Audit

Full chain traced, `configuration → dataset path → loader → fold loading →
execution → metrics → results`:

```
src/config_utils.py line 11:
  MODELING_ROOT = "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION"
        ↓ (CONFIG_DIR = MODELING_ROOT/configs, used by load_config())
base.yaml  paths.processed_dataset_dir:
  /content/drive/MyDrive/THESIS/DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION
        ↓ (get_paths() → dataset_csv, folds_json)
src/loader.py line 7:
  sys.path.insert(0, "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src")
        ↓
src/finetune/trainer.py line 17: same sys.path.insert pattern
src/baselines/sklearn_baseline.py line 14: same sys.path.insert pattern
        ↓
src/metrics.py: NO hardcoded path found (grep confirms zero matches) —
  the only one of the five reconciled files with no environment dependency,
  consistent with it being a clean-room reconstruction with no filesystem
  assumptions baked in.
```

**FACT**: exactly 4 of the 5 reconciled source files contain at least one
`/content/drive/MyDrive/...` literal (`config_utils.py`, `loader.py`,
`trainer.py`, `sklearn_baseline.py`); `metrics.py` contains none.

**FACT, demonstrated not inferred**: `MODELING_ROOT` is what `load_config()`
actually uses to locate `base.yaml` — not `base.yaml`'s own `paths:` block
(which is only consulted *after* the file is successfully loaded). This
means the single highest-leverage blocker is the `MODELING_ROOT` constant in
`config_utils.py`; fixing it alone would let config loading proceed to the
dataset-path stage, where `base.yaml`'s own `/content/drive/...` paths would
then need the same treatment.

None of these paths were changed this phase, per the explicit instruction to
document rather than modify them.

## Dependency Audit

**FACT** — imports traced directly from the five reconciled source files
(no speculation):

| Package | Required by | Category | Currently in `uv.lock`? |
|---|---|---|---|
| `numpy` | `metrics.py`, `config_utils.py`, `trainer.py`, `sklearn_baseline.py` | A. Setup / B. Baseline / C. Transformer | **Yes** |
| `scikit-learn` | `metrics.py`, `trainer.py`, `sklearn_baseline.py` | A/B/C | **Yes** |
| `pandas` | `loader.py`, `trainer.py`, `sklearn_baseline.py` | B. Baseline (needed now for even a smoke test) | **No** |
| `PyYAML` (`yaml`) | `config_utils.py` | A. Setup (blocks even config loading — demonstrated below) | **No** |
| `torch` | `trainer.py` | C. Transformer only | **No** |
| `transformers` | `trainer.py` | C. Transformer only | **No** |
| `gensim` | `sklearn_baseline.py` (`svm_word2vec` baseline only) | B. Baseline (one of four baseline models) | **No** |
| `streamlit` (proposed, Phase 6.3) | future `src/dashboard/` | D. Dashboard — not needed now | **No** |

**RECOMMENDATION**, not applied: `pandas` and `PyYAML` are the minimum
addition needed to even attempt a config-loading/import smoke test (both A
and B category, needed immediately); `gensim` is needed only once baseline
execution actually begins; `torch`/`transformers` only once transformer
execution begins; `streamlit` only in the future dashboard phase per Phase
6.3's explicit sequencing. No dependency was installed this phase — `uv`
was used only for inspection (`uv pip list`, `uv run python -c ...`), never
`pip` directly, per the standing instruction.

## Smoke Test

**Attempted** (read-only, no data/config/source modification): `uv run
python -c "import sys; sys.path.insert(0,'src'); import config_utils as cu; cu.load_config()"`.

**Result — FACT, not inference**:
```
ModuleNotFoundError: No module named 'yaml'
```
raised at `config_utils.py` line 7 (`import yaml`), *before* even reaching
the `MODELING_ROOT` hardcoded-path issue. This is a stronger, more precise
finding than the path audit alone: the very first blocker on the critical
path is a missing dependency, not the hardcoded path — the hardcoded path
is the *second* blocker, only reachable once `PyYAML` is installed.

No file was created, modified, or left behind by this test — confirmed via
`git status --short` before and after, unchanged (`?? docs/phases/` and `??
external/original-drive/DATASETS/` only, both pre-existing from before this
phase).

A full "smallest possible execution test" (config load → dataset path
resolution → loader → fold loading → `metrics.py` round-trip) **cannot yet
be attempted end-to-end** — it is blocked at step 1 (missing `PyYAML`) and
would then hit step 2 (hardcoded `MODELING_ROOT`/dataset paths) immediately
after. This defines exactly the two changes needed before any smoke test
can proceed further, without speculating about what lies beyond them.

## Existing Baseline Result Provenance

**FACT**: 4 baseline experiments have complete stored results under
`external/original-drive/MODELING/COGNITIVE DISTORTION/results/`:
`majority_class`, `tfidf_lr`, `tfidf_svm`, `svm_word2vec` — each with 5
per-fold prediction CSVs, 5 per-fold metric JSONs, and one `summary.json`.
`results/figures/` and `results/tables/` are empty (no charts/tables were
ever generated from these results in the original Drive project).

**FACT**: these results were not regenerated, overwritten, or modified this
phase — only read, via the metrics re-validation below.

**INFERENCE**: these artifacts are sufficient for later reproduction/
comparison — the `summary.json`/`fold_N.json` schema is fully specified
(Phase 4B) and the raw predictions needed to *re-derive* the metrics
(rather than merely trust the stored numbers) are present alongside them,
which is a stronger provenance position than metrics-only storage would be.

## Metrics Reconstruction Validation

**FACT, freshly re-verified this phase** (not merely cited from Phase 4B):
recomputed `compute_metrics`/`average_metrics` from the actual stored
prediction CSVs for all 4 baselines × 5 folds, compared against the stored
`fold_N.json`/`summary.json` values at `1e-9` tolerance:

```
majority_class: recomputed macro_f1_mean=0.058925 vs stored=0.058925
tfidf_lr:        recomputed macro_f1_mean=0.502212 vs stored=0.502212
tfidf_svm:       recomputed macro_f1_mean=0.510612 vs stored=0.510612
svm_word2vec:    recomputed macro_f1_mean=0.362793 vs stored=0.362793

mismatches: 0
```
**FACT**: `src/metrics.py` remains behaviorally consistent with the stored
baseline results. It remains, as established in Phase 4B and restated per
the standing instruction:

**`src/metrics.py` — RECONSTRUCTED — ORIGINAL SOURCE NOT RECOVERED.** This
phase performed a validation of that reconstruction, not a recovery of the
original file. `src/metrics.py` was not modified.

## Execution Architecture

Distinguishing the five categories per the phase objective, grounded in
what was actually found this phase (extends, does not repeat, Phase 6.3's
architecture proposal):

- **SOURCE**: `src/` (5 reconciled files, 1 reconstructed) — code only,
  fully tracked, no data.
- **DATA**: `external/original-drive/DATASETS/PROCESSED DATASETS/COGNITIVE
  DISTORTION/{dib_labeled.csv,folds.json,dib_groups.json}` — local-only,
  confirmed present, confirmed referenced by actual code (except
  `dib_groups.json`, confirmed unreferenced).
- **CONFIGURATION**: `external/original-drive/MODELING/COGNITIVE
  DISTORTION/configs/{base.yaml, models/*.yaml}` — small, should be tracked,
  currently *partially* untracked due to the `.gitignore` collision found
  this phase.
- **GENERATED ARTIFACTS**: `results/{predictions,metrics}/` — 4 baselines'
  worth already exist and are tracked (small); transformer results don't
  exist yet.
- **REPORTING**: `docs/phases/*.md` (this convention, active since the
  previous phase) + the future dashboard (Phase 6.3, not built yet, not
  triggered by anything found this phase — no structural prerequisite for
  the dashboard was discovered).

The chain `original notebook → reconciled source → local configuration →
local dataset → controlled experiment execution → versioned code +
experiment configuration → generated research artifacts → results/reporting
→ future dashboard/live demo` is **not yet traversable end-to-end** — it
breaks at "local configuration" (missing `PyYAML` + hardcoded
`MODELING_ROOT`), before ever reaching "controlled experiment execution."

## Risks

- The `.gitignore` `models/` collision (Git Safety Audit) means that if
  someone runs `git add -A` today intending to commit newly-added source,
  the 12 model config YAMLs would silently *not* be included even if the
  user believes they staged everything — a future commit could be made
  believing configs are tracked when they are not.
- Fixing `MODELING_ROOT` and the dataset paths touches the same files
  already carefully verified byte-identical to the Drive snapshot in Phase
  3.2 — any future edit here needs its own explicit before/after diff and
  rationale, not a silent patch, to preserve the provenance distinctions
  established across Phases 3–4.
- `dib_groups.json` is present locally and defined in `get_paths()` but
  never read by any traced code path — worth confirming (not assumed) in a
  future phase whether this is truly dead code or whether some
  not-yet-examined execution path needs it before treating it as safe to
  ignore indefinitely.

## Decisions

None were made this phase requiring irreversible action — this was an
audit. The one implicit decision made was to run a read-only metrics
re-validation (justified under the phase's own "safe smoke-test planning"
and "metrics reconstruction status" scope items) rather than skip it,
since it required no file creation and directly fulfilled two explicit
scope requirements.

## Unresolved Issues

1. Whether to fix the `.gitignore` `models/` collision now or in a
   dedicated future phase — not decided here, reported only.
2. Whether `MODELING_ROOT`/dataset-path normalization happens via the
   `DATA_ROOT` mechanism proposed in Phase 6.3 §3, or some simpler
   MacBook-only interim fix — not decided here.
3. Whether `dib_groups.json` is genuinely unused or represents a gap in
   the currently-recovered source — not resolved.
4. Exact target for `DATA_ROOT` (`external/original-drive/DATASETS` vs. a
   separate `data/` directory) — flagged as open in Phase 6.3 §18,
   still open.

## Required Changes Before Execution

In dependency order (RECOMMENDATION, none applied):

1. Add `PyYAML` and `pandas` to `pyproject.toml` via `uv add` — minimum
   needed to even re-attempt the smoke test past its current failure point.
2. Resolve the path-normalization mechanism (§ Unresolved Issues #2) and
   apply it to `src/config_utils.py`'s `MODELING_ROOT` and `base.yaml`'s
   `paths:` block, with an explicit before/after diff in whatever phase
   performs it.
3. Fix or narrow the `.gitignore` `models/` rule so `configs/models/*.yaml`
   becomes trackable without also un-protecting actual checkpoint
   directories.
4. Only after 1–3: re-attempt the smoke test (config load → dataset
   resolution → fold loading) to discover any further blockers not yet
   visible because execution currently halts at step 1.

## Recommended Phase 8

**Phase 8: Path normalization + minimal dependency addition**, scoped
narrowly to items 1–3 under "Required Changes Before Execution" above —
add `PyYAML`/`pandas`, normalize the hardcoded Drive paths in
`config_utils.py`/`base.yaml` via an explicit, diffed change (not a silent
patch), and fix the `.gitignore` collision — followed by a re-run of the
smoke test to discover the next layer of blockers, if any. Transformer-
specific work (`torch`/`transformers`/`gensim`, GPU considerations) remains
out of scope until baseline execution is confirmed working end-to-end
locally.

## Conclusion

The repository is **not yet ready** for local execution, but the distance
to readiness is now precisely characterized rather than assumed: exactly
two blockers stand between the current state and a successful config-load
smoke test (missing `PyYAML`, hardcoded `MODELING_ROOT`), `metrics.py`'s
reconstruction remains verified-consistent against all 4 stored baseline
results, and one previously-unreported `.gitignore` defect was found and
documented (not fixed) — the model-config YAMLs have been silently
untracked since the collision was introduced, and no prior phase's git
audit caught it because none checked the `MODELING/` tree's `configs/`
subdirectory against the `models/` rule specifically. No source, data, or
notebook files were modified this phase.
