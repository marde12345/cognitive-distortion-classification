# Phase 8 — Local Execution Adaptation

## Objective

Make the repository capable of a minimal local execution smoke test
(environment → config loading → local dataset resolution → dataset loading
→ fold loading → basic metrics validation) without depending on Google
Colab/Google Drive paths — without training any model, running full
cross-validation, or regenerating existing baseline results.

## Baseline Git State

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
**BEFORE — git blob hashes of the four previously byte-identical files**
(captured via `git hash-object` prior to any edit, establishing the
before-state per Task 2):
```
src/config_utils.py               f559470...
src/loader.py                      860ebd1...
src/finetune/trainer.py             acde889...
src/baselines/sklearn_baseline.py   1245365...
```

## Proposed Local Path Strategy

**FACT**, from inspection: `config_utils.py`'s `load_config()` locates
`base.yaml` via the module-level constant `MODELING_ROOT` — not via
anything in `base.yaml` itself (that file is only consulted *after* it's
found). Separately, `base.yaml`'s own `paths:` block (`processed_dataset_dir`,
`models_root`, `results_root`) is consumed as plain strings by `get_paths()`
with no relative-path resolution logic anywhere in `config_utils.py`.

**DESIGN (documented before implementation, per Task 1):**

1. `MODELING_ROOT` becomes a repository-relative path, resolved once via
   `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (i.e. the
   parent of `src/`, which is the repository root when `config_utils.py`
   lives at `<repo>/src/config_utils.py`), with an optional
   `COGNITIVE_DISTORTION_PROJECT_ROOT` environment-variable override for
   any future non-standard layout (e.g. Colab). This avoids hardcoding
   `/Users/MAC/...` (computed from `__file__`, not typed literally) and
   avoids hardcoding `/content/drive/...` (only used as a fallback string
   nowhere anymore).
2. `CONFIG_DIR`/`MODELS_CONFIG_DIR` then naturally resolve to a **new,
   local, top-level `configs/`** directory — not to
   `external/original-drive/...`. This directly satisfies the standing
   principle (Phase 6.2/6.3) that the repository should not depend on
   `external/original-drive/` as its primary runtime source/config
   location; that tree remains provenance-only.
3. `base.yaml`'s dataset/model/results paths are written as **plain paths
   relative to the process's working directory** (`external/original-drive/
   DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION`, `models`, `results`),
   which resolve correctly because `uv run` executes from the project root
   by default. This required **zero changes** to `get_paths()`'s code —
   only to the YAML content of the new local config file.
4. The three `sys.path.insert(0, "/content/drive/...")` lines (in
   `loader.py`, `trainer.py`, `sklearn_baseline.py`) are replaced with
   `__file__`-relative equivalents, preserving the exact same flat/
   unqualified import style (`import config_utils as cu`,
   `from loader import ...`, `from metrics import ...`) rather than
   restructuring `src/` into a formal installable package. This was
   evaluated against reworking `src/` into a proper package with
   `src/__init__.py` and qualified imports, and rejected for this phase as
   a larger change than necessary — Task 5 explicitly asks for the minimal
   fix, not a redesign.

This design was decided *before* editing any file, per Task 1's
instruction, and is unchanged from what was implemented below.

## Changes Made

### 1. `src/config_utils.py` (previously byte-identical to Drive)

**BEFORE:**
```python
MODELING_ROOT = "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION"
```
**AFTER:**
```python
MODELING_ROOT = os.environ.get(
    "COGNITIVE_DISTORTION_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
```
**Why necessary:** this single constant is what makes `load_config()`
locate `base.yaml` at all; the hardcoded Colab path does not exist on the
MacBook, so `load_config()` would fail with `FileNotFoundError` before ever
reaching dataset resolution (confirmed by Phase 7's smoke test, which
failed one step earlier on a missing `yaml` import — this path issue is
the *next* blocker Phase 7 predicted, now addressed).
**Behavior change:** environment/path resolution only. No change to
`load_config`, `get_paths`, `validate_dataset`, `load_folds`,
`get_fold_data`, `set_seed`, or any other function — none were renamed,
refactored, or had their logic altered. `CONFIG_DIR`/`MODELS_CONFIG_DIR`
(derived from `MODELING_ROOT`) are unchanged as expressions.

### 2. `src/loader.py` (previously byte-identical to Drive)

**BEFORE:**
```python
sys.path.insert(0, "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src")
```
**AFTER:**
```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```
**Why necessary:** makes `import config_utils as cu` (the next line,
unchanged) resolve locally without requiring the caller to have manually
pre-inserted `src/` onto `sys.path`.
**Behavior change:** environment/path resolution only. `load_dataset`,
`load_folds`, `get_fold_data`, `get_texts_labels` are byte-for-byte
unchanged.

### 3. `src/finetune/trainer.py` (previously byte-identical to Drive)

**BEFORE:**
```python
sys.path.insert(0, "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src")
```
**AFTER:**
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
(One extra `dirname` versus `loader.py` because `trainer.py` lives one
level deeper, at `src/finetune/trainer.py`.)
**Why necessary:** same reasoning as `loader.py`.
**Behavior change:** environment/path resolution only. `TextDataset`,
`_compute_class_weights`, `train_one_fold`, `save_predictions_and_metrics`,
`finalize` are byte-for-byte unchanged. This phase did **not** exercise
`trainer.py`'s training path — no transformer code was executed.

### 4. `src/baselines/sklearn_baseline.py` (previously byte-identical to Drive)

**BEFORE:**
```python
sys.path.insert(0, "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src")
```
**AFTER:**
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
**Why necessary:** same reasoning as above.
**Behavior change:** environment/path resolution only. `_save_fold`,
`_finalize`, `run_majority_class`, `_run_tfidf`, `run_tfidf_lr`,
`run_tfidf_svm`, `run_svm_word2vec` are byte-for-byte unchanged. This phase
did **not** call any of these functions — no baseline experiment was run
or regenerated.

### 5. New file: `configs/base.yaml` (NOT a modification of the Drive snapshot)

A new, local, top-level file — the Drive original at
`external/original-drive/MODELING/COGNITIVE DISTORTION/configs/base.yaml`
was read but never written to. Only the `paths:` block differs from that
original; `data:`, `training:`, and `eval:` blocks are copied verbatim
(confirmed identical by direct comparison during authoring):

**Drive original `paths:` block:**
```yaml
paths:
  processed_dataset_dir: /content/drive/MyDrive/THESIS/DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION
  models_root: /content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/models
  results_root: /content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/results
```
**New local `configs/base.yaml` `paths:` block:**
```yaml
paths:
  processed_dataset_dir: external/original-drive/DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION
  models_root: models
  results_root: results
```
**Why necessary:** required for `get_paths()` to resolve real, existing
local files rather than a nonexistent Colab mount. `models_root`/
`results_root` point at not-yet-existing local top-level `models/`/
`results/` directories (already covered by existing `.gitignore` rules);
neither was created this phase, since nothing in the smoke test writes to
them.

### 6. New files: `configs/models/*.yaml` (12 files) — verbatim copies

Copied byte-for-byte from
`external/original-drive/MODELING/COGNITIVE DISTORTION/configs/models/`
into the new local `configs/models/`. **Verified via `diff -rq`: identical,
zero differences.** These required no path editing — only the two DAPT
placeholder configs (`indobert_dapt.yaml`, `indobert_dapt_tapt.yaml`)
contain any `/content/drive/...` reference (their still-unresolved
placeholder `dapt_model_dir`/`tapt_model_dir_template` fields, unchanged
from the original reconnaissance), and those remain untouched, unresolved
placeholders in this copy exactly as they are in the Drive original — not
addressed this phase, as DAPT is out of scope (no experiment currently
depends on it).

### 7. `.gitignore`

**BEFORE:**
```gitignore
# External data / large artifacts
data/
models/
results/predictions/
```
**AFTER:**
```gitignore
# External data / large artifacts
data/
models/
results/predictions/

# Exception: model *configuration* YAML files are small, tracked
# provenance/reproducibility artifacts, not checkpoints. The `models/`
# rule above would otherwise also match any `configs/models/` directory
# (e.g. external/original-drive/MODELING/COGNITIVE DISTORTION/configs/models/,
# configs/models/) purely because its last path segment is "models".
!**/configs/models/
!**/configs/models/**
```
**Why necessary:** Phase 7 found the bare `models/` rule (intended for
checkpoints) was also silently hiding all 12 model config YAMLs from Git,
under both the Drive snapshot path and the new local `configs/` path. This
is the smallest fix that preserves the original rule's intent for actual
checkpoints while un-hiding configuration.

## Provenance Impact

Per the standing requirement, explicit provenance status for each modified,
previously-byte-identical file:

| File | Original provenance | Current status | Reason |
|---|---|---|---|
| `src/config_utils.py` | Drive/notebook byte-identical source (verified Phase 3.2) | **Locally adapted copy** — one constant's value changed, all logic unchanged | Required for local execution (path resolution) |
| `src/loader.py` | Drive/notebook byte-identical source (verified Phase 3.2) | **Locally adapted copy** — one line changed | Required for local execution |
| `src/finetune/trainer.py` | Drive/notebook byte-identical source (verified Phase 3.2) | **Locally adapted copy** — one line changed | Required for local execution |
| `src/baselines/sklearn_baseline.py` | Drive-only source (verified Phase 3.2) | **Locally adapted copy** — one line changed | Required for local execution |
| `src/metrics.py` | **RECONSTRUCTED — ORIGINAL SOURCE NOT RECOVERED** (Phase 4B) | Unchanged this phase, status unchanged | Not touched |
| `configs/base.yaml` | New file; `paths:` block differs from the Drive original by design, `data:`/`training:`/`eval:` blocks copied verbatim | New local file, not a modification of Drive provenance | Required for local execution |
| `configs/models/*.yaml` (12) | **Verbatim copy** of the Drive snapshot's `configs/models/*.yaml`, confirmed identical via `diff -rq` | Verbatim local copy, distinct from the adapted/reconstructed categories above | Enables local model-config loading |

After this phase, `src/config_utils.py`, `src/loader.py`,
`src/finetune/trainer.py`, and `src/baselines/sklearn_baseline.py` are
**no longer byte-identical** to their Drive-snapshot originals under
`external/original-drive/` — this is expected and intentional, and the
Drive originals themselves remain completely unmodified as the permanent
historical reference (re-confirmed below). This report is the permanent
record of exactly what changed and why, so the distinction is never lost.

## Dependency Changes

**FACT** — via `uv add pandas pyyaml` (not `pip`):
```
+ pandas==3.0.6
+ python-dateutil==2.9.0.post0
+ pyyaml==6.0.3
+ six==1.17.0
```
`pyproject.toml`'s `dependencies` list:
```diff
 dependencies = [
     "numpy>=2.4.6",
+    "pandas>=3.0.6",
+    "pyyaml>=6.0.3",
     "scikit-learn>=1.9.1",
 ]
```
`uv.lock` was regenerated accordingly (193 lines changed). **Not added**,
per the phase's explicit scope: `torch`, `transformers`, `gensim`,
`streamlit` — the completed smoke test (see below) confirms none of these
were needed to satisfy Tasks 1–8's minimal target, since no baseline or
transformer code path was executed.

## .gitignore Changes

Covered in "Changes Made" item 7 above. **Verification performed** (Task 6
requirement):
```
git check-ignore -q configs/models/tfidf_lr.yaml                                            → exit 1 (NOT ignored — fixed)
git check-ignore -q external/original-drive/MODELING/COGNITIVE DISTORTION/models/tfidf_lr    → exit 0 (still ignored — checkpoint protection intact)
```
Also spot-checked unrelated rules remain unaffected: `.venv/lib/foo.py`,
`results/predictions/foo.csv`, and
`external/original-drive/DATASETS/PREPROCESSING/COGNITIVE DISTORTION/data/interim/foo.csv`
all still correctly match their original rules (`.venv/`, `results/predictions/`,
`data/` respectively) — the negation is scoped exactly to `configs/models/`
paths and nothing else.

**FACT**: no raw dataset, model checkpoint, or `.venv`/cache path was
unignored by this change.

## Dataset Integrity Verification

**FACT**, recomputed this phase, compared against the existing stored
sidecar checksums (neither dataset file was modified in the process):

| File | Expected (stored `.sha256`) | Actual (recomputed) | Result |
|---|---|---|---|
| `dib_labeled.csv` | `ed154162554b68ba6980af8f3c7c01fa80a7962c4a0595fd61acafd142bdaea0` | `ed154162554b68ba6980af8f3c7c01fa80a7962c4a0595fd61acafd142bdaea0` | **MATCH** |
| `folds.json` | `730cd82a912828e902cdab26eb498c640195eca4901fde606f849372c5693c54` | `730cd82a912828e902cdab26eb498c640195eca4901fde606f849372c5693c54` | **MATCH** |

The `dib_labeled.csv` hash additionally matches the hash recorded in the
original notebook's saved execution output (first cross-checked in Phase 1,
and again in Phase 6) — three independent confirmations now exist that this
is the correct, experiment-matching dataset version.

## Configuration Resolution

**FACT**, from the smoke test: `cu.load_config()` successfully loaded
`configs/base.yaml` (`num_labels=11`, `seed=42` confirmed read correctly);
`cu.get_paths()` resolved `dataset_csv` to
`external/original-drive/DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION/dib_labeled.csv`,
confirmed to exist on disk via `os.path.exists`.

## Import Resolution

**FACT**: `import config_utils`, `import loader`, `from metrics import
compute_metrics, average_metrics` all resolved without needing any manual
`sys.path` manipulation beyond inserting `src/` once at the top of the
smoke-test script (the same one-time insertion any Python entry point would
need) — the three internal `sys.path.insert` fixes made each module
self-sufficient for resolving its sibling modules, satisfying Task 5's goal.
`src/baselines/sklearn_baseline.py` and `src/finetune/trainer.py` were not
imported in this smoke test (the former needs `gensim`, not yet added; the
latter needs `torch`/`transformers`, not yet added) — both out of this
phase's minimal scope, and their path-resolution line was still fixed
proactively since it required no additional dependency to fix correctly.

## Smoke Test

**FACT** — full transcript, `uv run python` (not `pip`), no training, no
cross-validation, no predictions generated, no existing results touched:

```
A. Python environment: OK (running under uv)
B. import config_utils: OK
   MODELING_ROOT = /Users/MAC/Projects/Tesis-Mbak-Ai
   CONFIG_DIR    = /Users/MAC/Projects/Tesis-Mbak-Ai/configs
C. load configuration: OK
   num_labels = 11 | seed = 42
D. resolve local dataset path: OK
   dataset_csv = external/original-drive/DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION/dib_labeled.csv
   both exist on disk: confirmed
E. load dataset: OK — 4628 rows, 16 columns
F. load folds: OK — 5 folds
G. validate basic dataset/fold structure: OK — label_ok=True, ids_unique=True, total_nan=0
   fold_0 split sizes: train=3345 val=364 test=919 (sum matches total rows)
H. tiny synthetic metrics.py call: OK — 1.0 / 2

SMOKE TEST: ALL STEPS PASSED
```
All 8 required steps (A–H) passed on the first attempt after the changes
above; no blocker was hit, so Task 8's "stop at first meaningful blocker"
branch was not triggered. Row counts (4,628), fold count (5), and fold_0
split sizes (3345/364/919) match exactly what was recorded in the original
notebook's saved output (cross-checked against Phase 1's reconnaissance).

## Dataset Safety Audit

**Verified, not assumed:**

```
Dataset files READ:      YES — dib_labeled.csv, folds.json (smoke test);
                          dib_labeled.sha256, folds.sha256 (checksum verification)
Dataset files CREATED:   NO
Dataset files MODIFIED:  NO  (confirmed: MD5 of external/original-drive base.yaml
                          and sha256 of both dataset files match pre-existing
                          values; no write operation was ever called against
                          them — loader.load_dataset only calls pd.read_csv)
Dataset files MOVED:     NO
Dataset files DELETED:   NO
Dataset files STAGED:    NO  (confirmed via `git status --short`: only
                          .gitignore, pyproject.toml, uv.lock, the 4 source
                          files, and new configs/ and docs/phases/ paths
                          appear — no path under external/original-drive/DATASETS
                          or external/original-drive/.../results/predictions
                          is staged)
Dataset files COMMITTED: NO  (no commit was made this phase)
```
All match the expected result stated in the phase instructions.

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
```

**Important clarification on the last line:** the 12 model-config YAMLs
under `external/original-drive/MODELING/COGNITIVE DISTORTION/configs/models/`
newly appear as `??` (untracked) — this is **not** a modification to any
file under `external/original-drive/`. Prior to this phase these files were
silently `.gitignore`-excluded (Phase 7's finding); the `.gitignore` fix in
this phase only changed their Git *visibility* status from "ignored" to
"untracked but trackable." No byte of any file under `external/original-drive/`
was written, confirmed by the MD5/SHA-256 checks above matching pre-existing
values.

**`git diff --stat`:**
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
Full per-file diffs for the four source files are reproduced in full in
"Changes Made" above — each is a single-line change, nothing else.

No commit was made. `git log --oneline -5` is unchanged from the baseline
recorded at the top of this report.

## Remaining Blockers

None discovered for the scope of Tasks 1–8 — the smoke test passed
end-to-end on the first attempt. Known, already-scoped-out blockers for
*later* phases (not blockers for this phase's objective):

- `gensim` not yet installed — needed before `src/baselines/sklearn_baseline.py`
  can actually run `run_svm_word2vec`.
- `torch`/`transformers` not yet installed — needed before
  `src/finetune/trainer.py` can actually run any transformer fold.
- `models/`, `results/` (top-level) don't exist on disk yet — not needed
  until something actually writes to them (no step in this phase did).
- The two DAPT placeholder configs remain unresolved placeholders,
  unchanged — out of scope, as previously established.

## Decisions

1. Adopted the `MODELING_ROOT`-as-repo-root design (§ Proposed Local Path
   Strategy) rather than introducing a new differently-named constant —
   preserves the existing identifier so the diff is a value change, not a
   rename, honoring Task 2's "do not rename" instruction even though it
   technically only mentioned functions.
2. Chose plain relative paths in `configs/base.yaml` over env-var
   templating inside the YAML — avoids any change to `get_paths()`'s code,
   keeping the fix contained entirely to path *values*, not logic.
3. Copied the 12 model YAMLs verbatim into a new local `configs/models/`
   rather than having `config_utils.py` read them directly from
   `external/original-drive/` — consistent with the standing principle
   (Phase 6.2/6.3) that the repository's runtime should not depend on the
   provenance snapshot directly.
4. Fixed the `.gitignore` collision with a scoped negation rule rather than
   restructuring the broader `models/`/`data/` rules — smallest possible
   change per Task 6's explicit instruction.

## Risks

- The `COGNITIVE_DISTORTION_PROJECT_ROOT` environment-variable override
  exists but is untested this phase (only the `__file__`-derived default
  path was exercised) — if a future Colab-compatibility phase relies on it,
  it should be explicitly tested there, not assumed to work from this
  phase's evidence alone.
- `configs/base.yaml`'s relative paths depend on the process's working
  directory being the repository root at execution time — true for `uv run`
  by default, but would silently break (wrong relative resolution, not a
  clean error) if something is ever invoked from a different working
  directory. Worth a defensive check in a future phase, not added here
  since it wasn't necessary to pass the smoke test.
- `configs/` now exists in two places with different content (Drive
  original at `external/original-drive/.../configs/base.yaml`, unmodified;
  new local `configs/base.yaml`, path-adapted) — a future reader must not
  confuse the two. This report and the provenance table above are the
  explicit record preventing that confusion.

## Recommended Phase 9

**Phase 9: Baseline experiment execution readiness** — add `gensim` (the
one remaining dependency needed to run all four baseline functions, not
just the three that avoid it), then execute the actual baseline pipeline
(`run_majority_class`, `run_tfidf_lr`, `run_tfidf_svm`, `run_svm_word2vec`)
against the local dataset, writing results to the new local `results/`
directory (not overwriting the existing Drive-snapshot results), and
compare the freshly-generated local results against the already-verified
stored results from `external/original-drive/.../results/` as the
reproduction check envisioned since the original migration plan. This is
explicitly **not** authorized to begin automatically — it requires its own
phase instruction, per the standing rule.

## Conclusion

The repository can now execute the full chain from environment through
basic metrics validation entirely locally, using `uv`, with zero Colab/
Drive-path dependencies remaining in the executed code path. Four
previously byte-identical files were modified with exactly one line each,
fully diffed and justified above; their provenance status is now
explicitly "locally adapted copy," not "byte-identical," and this report is
the permanent record of that transition. `src/metrics.py` remains
untouched and correctly labeled RECONSTRUCTED — ORIGINAL SOURCE NOT
RECOVERED. No dataset file was created, modified, moved, deleted, staged,
or committed. The `external/original-drive/` snapshot remains byte-for-byte
unmodified, confirmed via checksum. No commit was made this phase.
