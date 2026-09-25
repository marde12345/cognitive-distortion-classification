# Phase 10A — Repository Hygiene & Baseline Artifact Decision

## 1. Objective

Audit the repository state after Phase 9.1's successful baseline execution
and classify every modified/untracked path into TRACK / IGNORE /
INTENTIONALLY UNTRACKED / DO NOT TOUCH, without executing anything,
modifying source, or touching `external/original-drive/`. This phase
produces a recommendation and exact commands — it does not stage or commit
anything itself.

## 2. Initial Git State

**FACT — `git status --short`:**
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
**FACT — `git diff --stat`:**
```
 .gitignore                        |   8 +
 pyproject.toml                    |   3 +
 src/baselines/sklearn_baseline.py |   2 +-
 src/config_utils.py               |   5 +-
 src/finetune/trainer.py           |   2 +-
 src/loader.py                     |   2 +-
 uv.lock                           | 323 +++++++++++++++++++++++++++++++++++++-
 7 files changed, 335 insertions(+), 10 deletions(-)
```
**FACT — `git log --oneline -8`** (only 5 commits exist):
```
efc1f8d  feat: reconstruct metrics module
2fcfe00  chore: reconcile recovered source modules
b71999d  chore: add original Drive snapshot for provenance
a5dcdff  Repository cleanup: establish uv project foundation
feccab5  Initial thesis snapshot: original Colab notebook, unmodified
```
This is exactly the accumulated, uncommitted state left by Phases 8 and
9.1 combined — no drift, no unexpected changes.

## 3. File Classification (overview)

| Path | Git state | Category (see §11 for full table) |
|---|---|---|
| `.gitignore` | Modified | Source-adjacent config — TRACK |
| `pyproject.toml`, `uv.lock` | Modified | Dependency manifest — TRACK |
| `src/{config_utils,loader,finetune/trainer,baselines/sklearn_baseline}.py` | Modified | Locally-adapted source — TRACK |
| `configs/` | Untracked | Local runtime config — TRACK |
| `docs/phases/` | Untracked | Phase reports — TRACK |
| `scripts/` | Untracked | Repository tooling — TRACK |
| `results/` | Untracked | Mixed — see §4 |
| `external/original-drive/DATASETS/` | Untracked | Dataset + unrelated raw data — DO NOT TRACK (mostly), DO NOT TOUCH |
| `external/original-drive/.../configs/models/` | Untracked | Provenance config, visibility restored by Phase 8's `.gitignore` fix — TRACK |

## 4. Results Artifact Analysis

**FACT**, sizes measured directly:
```
results/metrics/      96 KB  (24 files: 4 models × [5 fold_N.json + 1 summary.json])
results/predictions/ 240 KB  (20 files: 4 models × 5 fold_N.csv)
```
**FACT** — `git check-ignore` confirms the existing rule set already
handles this split correctly with no change needed:
- `results/metrics/**/*.json` → **not ignored**, trackable.
- `results/predictions/**/*.csv` → **ignored**, matched by the existing
  `results/predictions/` rule.

**FACT** — `git add -n results/` (dry-run, nothing staged) confirms this in
practice: exactly the 24 metrics JSON files would be staged; none of the 20
prediction CSVs would be.

**Redundancy check against `external/original-drive/`** (FACT, `diff -q`):
```
majority_class/summary.json:  IDENTICAL to Drive original
tfidf_lr/summary.json:         IDENTICAL to Drive original
tfidf_svm/summary.json:        IDENTICAL to Drive original
svm_word2vec/summary.json:     DIFFERS from Drive original
```
(Consistent with Phase 9.1's full byte-for-byte comparison across all
fold-level files, not just summaries.)

**INFERENCE**: three of the four local metric sets are byte-identical
duplicates of data already committed under `external/original-drive/...`
(in `b71999d`). Committing them again under `results/` would not expose
any new information — their value is in *demonstrating* reproducibility
(the fact that local execution produced the same bytes), not in the bytes
themselves being new. The fourth (`svm_word2vec`) is **not** a duplicate —
it's a distinct, real result (different numbers) that only exists locally
and would be lost if not tracked somewhere.

**RECOMMENDATION**: track all 24 `results/metrics/**` files anyway,
despite three being duplicates — the duplication itself is the evidence of
reproduction (a reviewer diffing `results/metrics/tfidf_lr/summary.json`
against `external/original-drive/.../tfidf_lr/summary.json` and finding
them identical *is* the reproducibility proof), and omitting the three
"redundant" ones while keeping only `svm_word2vec` would break the set's
legibility as "the four baselines, locally reproduced." This is a
recommendation, not a predetermined conclusion — the instructions
explicitly flagged not to auto-duplicate everything, so flagging the
tradeoff explicitly: an alternative is to track only `svm_word2vec` (the
non-duplicate) and note the other three's exact-match status in this
report instead of in tracked files. Both are defensible; the first is
recommended for legibility, not asserted as the only correct choice.

**Prediction CSVs**: **RECOMMENDATION — remain ignored.** 240 KB across 4
models is not large in absolute terms, but per-row prediction dumps are
exactly the category the original migration plan (and Phase 6's dataset-
safety audit) already decided should stay local-only, and three of the
four are — again — byte-identical duplicates of data already in
`external/original-drive/`. No new argument for tracking them emerged this
phase.

## 5. Dataset / Original Drive Safety Verification

**FACT**, verified via `git diff -- external/original-drive/`: **empty
output** — zero modification to any currently-tracked file under
`external/original-drive/` (the `MODELING/COGNITIVE DISTORTION` subtree
committed in `b71999d`). `git status --short -- external/original-drive/`
shows no `M` (modified) entries at all — only pre-existing `??`
(untracked) entries for `DATASETS/` and `configs/models/`, both dating
from before this phase (Phase 6 and Phase 8 respectively, not new).

**FACT** — dataset directory sizes (measured, not estimated):
```
external/original-drive/DATASETS/                  929M
  RAW DATASETS/                                     913M  (includes ~805MB of
                                                            unrelated third-party
                                                            datasets, per Phase 6)
  PROCESSED DATASETS/                                3.6M (dib_labeled.csv,
                                                            folds.json, etc.)
  PREPROCESSING/                                      13M
```
**RECOMMENDATION**, unchanged from Phase 6/6.1's prior findings, reaffirmed
here rather than re-decided: none of this should be tracked. This phase
found no new evidence changing that conclusion.

## 6. Source Diff Analysis

**FACT**, all four diffs re-verified this phase (not assumed from memory):

```diff
src/baselines/sklearn_baseline.py:
- sys.path.insert(0, "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src")
+ sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

src/config_utils.py:
- MODELING_ROOT = "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION"
+ MODELING_ROOT = os.environ.get(
+     "COGNITIVE_DISTORTION_PROJECT_ROOT",
+     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
+ )

src/finetune/trainer.py:
- sys.path.insert(0, "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src")
+ sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

src/loader.py:
- sys.path.insert(0, "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src")
+ sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

**FACT**: these are exactly, and only, the four single-line changes
documented in Phase 8's report — confirmed by direct `git diff`
re-inspection this phase, not assumed to still be accurate. No additional
drift, no re-modification, no unexpected edits occurred between Phase 8
and now.

**FACT** — `src/metrics.py`: `git diff -- src/metrics.py` returns empty,
and a direct `diff` against `git show efc1f8d:src/metrics.py` confirms
byte-for-byte identity with the committed reconstruction. It remains
exactly what Phase 4B produced and Phase 4B's commit message states:
**RECONSTRUCTED — ORIGINAL SOURCE NOT RECOVERED.** Not rewritten, not
touched.

## 7. Config Analysis

**FACT**, full `configs/` listing: `configs/base.yaml` (1 file, locally
path-adapted per Phase 8, not a copy of the Drive original's `paths:`
block) + `configs/models/*.yaml` (12 files, verified verbatim copies of
`external/original-drive/MODELING/COGNITIVE DISTORTION/configs/models/` in
Phase 8).

**Classification**: source configuration required for local execution —
none of it is generated, none of it is dataset-related, none of it is
large or sensitive. **RECOMMENDATION: TRACK all 13 files.**

## 8. Script Analysis

**FACT**: `scripts/` contains exactly one file, `scripts/run_baseline.py`,
introduced in Phase 9 as the minimal execution entry point. It is
hand-written repository tooling (not a generated/temporary artifact) —
confirmed by inspection: it's a normal Python CLI script with no
execution-run-specific state baked into it, reusable for any future
baseline re-run. **RECOMMENDATION: TRACK.**

## 9. Phase Report Analysis

**FACT**, `docs/phases/` currently contains 5 files:
```
PHASE_6_3_final_architecture.md          (existing, untracked since creation)
PHASE_7_execution_readiness_audit.md     (existing, untracked since creation)
PHASE_8_local_execution_adaptation.md    (existing, untracked since creation)
PHASE_9_baseline_execution.md            (existing, untracked since creation)
PHASE_9.1_BASELINE_EXECUTION_RETRY.md    (existing, untracked since creation)
```
None have ever been committed — `docs/phases/` as a whole has been `??`
since it was first created. All five are permanent, append-only
documentation artifacts per the standing convention; none should be
deleted or modified. **RECOMMENDATION: TRACK all 5** (this Phase 10A
report will become a 6th, created after this audit).

## 10. `.gitignore` Analysis

**FACT**, current diff (already applied in Phase 8, re-verified unchanged
this phase):
```diff
+!**/configs/models/
+!**/configs/models/**
```
placed directly after the `models/` rule.

**No new `.gitignore` change is being made this phase** — the audit found
the existing rule set (as of Phase 8's fix) already correctly handles
every path encountered:
1. **Exact path protected**: any `configs/models/` directory at any depth
   (currently matches two: the Drive snapshot's and the new local one).
2. **Why not tracked otherwise**: without this negation, the broad
   `models/` rule (intended for checkpoint directories) would also hide
   these small config YAMLs, as Phase 7 discovered.
3. **Why it doesn't hide legitimate artifacts**: verified via
   `git check-ignore` on unrelated paths (`.venv/`, `results/predictions/`,
   `external/.../PREPROCESSING/.../data/interim/`) — all still correctly
   match their original rules, confirming the negation is scoped precisely
   to `configs/models/` and nothing else.
4. **Effect on nested `external/original-drive/` content**: yes, by
   design — it applies at any depth (`**/configs/models/`), which is
   exactly why it also un-hides the Drive snapshot's own
   `configs/models/*.yaml`, not only the new local copy.

**RECOMMENDATION**: no `.gitignore` change needed this phase. The rule set
inherited from Phase 8 is sufficient and correctly scoped for the current
`results/` split (§4) with zero modification.

## 11. Proposed Tracking Plan

| Path/category | Current state | Recommendation | Reason |
|---|---|---|---|
| `.gitignore` | Modified | **TRACK** | Already-verified, correctly-scoped fix from Phase 8 |
| `pyproject.toml` | Modified | **TRACK** | Dependency manifest, small, no secrets |
| `uv.lock` | Modified | **TRACK** | Lockfile, canonical dependency pin per project convention |
| `src/config_utils.py` | Modified | **TRACK** | Locally-adapted source, single-line diff, fully documented (Phase 8) |
| `src/loader.py` | Modified | **TRACK** | Same |
| `src/finetune/trainer.py` | Modified | **TRACK** | Same |
| `src/baselines/sklearn_baseline.py` | Modified | **TRACK** | Same |
| `src/metrics.py` | Unmodified (already committed) | **DO NOT TOUCH** | Reconstruction, already committed in `efc1f8d`, verified unchanged |
| `configs/base.yaml` | Untracked | **TRACK** | Local runtime config, small, path-adapted (documented) |
| `configs/models/*.yaml` (12) | Untracked | **TRACK** | Verbatim provenance copies, small |
| `scripts/run_baseline.py` | Untracked | **TRACK** | Repository tooling, hand-written, reusable |
| `docs/phases/*.md` (5 existing + this one) | Untracked | **TRACK** | Permanent, append-only documentation/provenance |
| `results/metrics/**/*.json` (24 files, 96KB) | Untracked | **TRACK** (recommended, see §4 for the alternative) | Small, non-sensitive aggregate metrics; 3/4 are exact-match duplicates serving as reproducibility evidence, 1/4 is genuinely new data |
| `results/predictions/**/*.csv` (20 files, 240KB) | Untracked, ignored | **IGNORE** (no change) | Per-row prediction dumps, mostly duplicates of already-committed Drive data, consistent with standing dataset-safety policy |
| `external/original-drive/DATASETS/` (929MB) | Untracked | **DO NOT TRACK** | Dataset + ~805MB of unrelated third-party raw data (Phase 6) |
| `external/original-drive/.../configs/models/` (12 files) | Untracked | **TRACK** | Provenance copies, visibility restored by Phase 8's `.gitignore` fix, small |
| `.venv/` | Not shown (ignored) | **IGNORE** (no change) | Standard, already correctly ignored |
| `notebooks/Modeling_Cognitive_Distortion (2).ipynb` | Already tracked, unmodified | **DO NOT TOUCH** | Immutable archival artifact |
| `external/original-drive/MODELING/COGNITIVE DISTORTION/**` (already tracked subset) | Already tracked, unmodified | **DO NOT TOUCH** | Committed provenance snapshot |

Every category was checked against the actual repository state this
phase, not assumed from the task instructions' suggested list.

## 12. Proposed Cleanup Commands (NOT executed)

```bash
# Source, config, tooling, dependency, and gitignore changes
git add .gitignore pyproject.toml uv.lock
git add src/config_utils.py src/loader.py src/finetune/trainer.py src/baselines/sklearn_baseline.py
git add configs/
git add scripts/

# Provenance: restored visibility of Drive-snapshot model configs
git add "external/original-drive/MODELING/COGNITIVE DISTORTION/configs/models/"

# Documentation
git add docs/phases/

# Baseline reproduction evidence (aggregate metrics only — NOT predictions)
git add results/metrics/

# Review before committing
git status --short
git diff --cached --stat

# Suggested commit message (for review, not applied)
git commit -m "feat: local execution adaptation, baseline reproduction, and repository hygiene"
```

**Explicitly NOT included in any proposed command:**
```bash
# git add external/original-drive/DATASETS/       ← dataset, DO NOT TRACK
# git add results/predictions/                     ← prediction dumps, IGNORE
```

None of the above were executed. `git status --short` remains exactly as
shown in §2 as of the end of this phase.

## 13. Risks / Unresolved Questions

1. **Whether to track all 24 `results/metrics/**` files or only the
   non-duplicate `svm_word2vec` ones** (§4) — presented as a
   recommendation with an explicit alternative, not decided unilaterally.
2. **Whether bundling source+config+tooling+docs+results into a single
   commit (as sketched in §12) is the right granularity**, versus splitting
   into several smaller commits (e.g. "local execution adaptation" separate
   from "baseline reproduction results") mirroring the fine-grained commit
   discipline used in Phases 3–4 — this phase provides one proposed
   sequence but does not treat it as the only correct one.
3. **The already-known `svm_word2vec` non-determinism** (Phase 9.1) means
   any *future* re-run of `scripts/run_baseline.py svm_word2vec` would
   likely overwrite `results/metrics/svm_word2vec/*` with slightly
   different numbers — worth deciding, before committing, whether this
   run's specific numbers should be treated as "the" reference local
   result or whether that designation needs more thought (e.g. averaging
   multiple runs) — not resolved here, flagged for the committer's
   awareness.

## 14. Recommended Next Phase

**Phase 10B (or a user-driven review/commit step)**: review this report's
tracking plan and proposed commands, decide on the `results/metrics/`
scope question (§4/§13.1) and commit granularity (§13.2), and — only once
decided — execute the actual `git add`/`git commit` sequence. This phase
does not recommend proceeding to transformer work; that remains a later,
separate decision per the standing execution order (Phase 6.3 §17).

## Final State

**FACT**, `git status --short` at the end of this phase — identical to the
start (§2), confirming nothing was staged or committed:
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
(`docs/phases/PHASE_10A_REPOSITORY_HYGIENE.md`, this report, is a new file
within the already-`??` `docs/phases/` directory — it does not change the
directory's status line.)

No files staged. No commit made. No experiment re-run. No transformer
training. No dataset modification. No modification under
`external/original-drive/`.
