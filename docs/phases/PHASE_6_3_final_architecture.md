# Phase 6.3 — Final Repository + Research Dashboard Architecture

## Objective

Produce the final architecture plan for the repository before execution begins,
covering preprocessing architecture, modeling architecture, experiment/result
artifact architecture, dashboard architecture, live inference architecture,
data boundaries, provenance, local execution through `uv`, dependency
boundaries, and future commit sequence — including, as a new requirement
introduced in this phase, a minimal research dashboard design (light-theme,
academic/research-lab visual direction, live demo first).

## Scope

**Planning only. No files modified. No execution performed. No experiments
performed. No dependencies installed. No commits made.** No Google Drive
access. No preprocessing or notebook execution.

## Actions Performed

1. Re-grounded the plan in actual current repository state (`git log --oneline -6`,
   `git status --short`, directory listing) rather than relying purely on
   prior-phase summaries.
2. Synthesized findings from Phases 1 through 6.2 (reconnaissance, source
   recovery, Drive reconciliation, bytecode forensics, metrics.py
   reconstruction, dataset safety, preprocessing reconnaissance, preprocessing
   reconciliation planning) into a single forward-looking architecture.
3. Resolved the preprocessing package-structure decision left open at the end
   of Phase 6.2 (chose Option A: `src/preprocessing/{common,cognitive_distortion}/`).
4. Designed a new research dashboard architecture per the user's new
   requirement: light-theme-only, live-demo-first information hierarchy,
   artifact-reading-only data layer (no second implementation of metric
   calculations), and a hard structural separation between the live-inference
   demo and the evidence-backed research report sections.
5. Produced a full 19-section architecture report (reproduced in full below).

## Files Changed

None.

## Files Created

- `docs/phases/PHASE_6_3_final_architecture.md` (this file) — created to
  satisfy the new standing reporting requirement introduced by the user in
  the message that triggered this file's creation. This is the first file
  under the new `docs/phases/` convention; it documents Phase 6.3, which had
  already been completed and reported in-chat prior to the reporting
  requirement being introduced.

## Files Deleted

None.

## Files Moved

None.

## Dataset Safety

No dataset files were read, created, modified, copied, moved, deleted,
staged, or committed during this phase. This phase touched no data at all —
it was pure architectural planning based on findings already established in
prior phases.

## Git Status

At the start of this phase:
```
?? external/original-drive/DATASETS/
```
(unchanged since Phase 6/6.1/6.2 — the downloaded dataset tree remains
untracked and unprotected by `.gitignore`, a known, already-reported state
pending the `.gitignore` decision from Phase 6/6.1).

After creating this report file, the only additional change is:
```
?? docs/phases/
```

## Commit Information

**No commit was made.** Phase 6.3 was planning-only and, per its own
conclusion, was not intended to produce a commit. Per the new reporting
requirement's own rule ("include the report file in the phase's Git commit
ONLY when that phase is supposed to be committed"), this report file is
therefore left uncommitted alongside it, pending a future phase where
documentation is deliberately committed (see the commit sequence in the
report body, item 13: "docs: final documentation pass", or an earlier
dedicated docs commit if the user prefers to commit `docs/phases/` sooner).

## Evidence / Verification Performed

- `git log --oneline -6` — confirmed the 5 existing commits (`feccab5`
  through `efc1f8d`) remain the full history, nothing rewritten.
- `git status --short` — confirmed the only pending change is the untracked
  `external/original-drive/DATASETS/` tree, consistent with Phase 6's
  findings.
- Directory listing of the repository root (excluding `.git`, `.venv`,
  `external`) — confirmed `src/`, `notebooks/`, `docs/`, `pyproject.toml`,
  `uv.lock` match the state left by Phase 4B's commit, with no drift.

## Decisions Made

1. **Preprocessing package structure: Option A** —
   `src/preprocessing/{common,cognitive_distortion}/`. Chosen because
   `COMMON` currently has exactly one consumer (no second pipeline exists to
   justify Option B's separate top-level package), and `preprocessing` as a
   name directly reuses the Drive folder's own naming (`DATASETS/PREPROCESSING/`)
   rather than inventing new vocabulary (as Option C's `data_pipeline` would).
2. **Dashboard framework: Streamlit**, recommended (not yet installed) —
   chosen because it lets the artifact-reading architecture stay pure Python
   with no API/serialization layer between the data-access module and the
   display code, avoiding the infrastructure a JS-SPA option would require
   (rejected outright as contradicting "minimal JavaScript" / "no
   unnecessary infrastructure").
3. **Dashboard structural separation**: live demo (`inference.py`) and
   research report (`data_access.py`) share no import path — `data_access.py`
   never imports `inference.py` and vice versa — enforced by the module
   import graph itself, not merely by convention, so the research-report
   layer cannot be contaminated by live-demo state and the live demo cannot
   accidentally trigger retraining or artifact regeneration.
4. **Single `DATA_ROOT` mechanism** reused for both preprocessing and
   modeling config path normalization, rather than inventing a second,
   preprocessing-specific mechanism.
5. **Notebook role confirmed as archival/reference-only** — no execution
   entry point going forward; that role shifts to `scripts/` once built.

## Unresolved Decisions

1. `DATA_ROOT` target ambiguity: pointing directly at
   `external/original-drive/DATASETS` versus a separate `data/` directory
   (copy or symlink) — not resolved, needs an explicit choice before the
   path-normalization phase.
2. Whether the `sys.path.insert` → proper package-import change (implied by
   choosing Option A's installable package structure) happens in the same
   commit as path normalization, or as a separate commit.
3. Streamlit's default visual identity versus the strict "no generic SaaS
   dashboard" requirement — flagged as a real implementation risk requiring
   deliberate CSS override work, not yet solved.
4. The DATASET section's data-summary JSON (row 3 of the dashboard
   information hierarchy) does not exist in any form yet and has no defined
   generation step.
5. `decontaminate.py`/DAPT-corpus reconciliation timing — still unresolved
   from Phase 6.2, not newly resolved here.
6. Development/testing dependency category is currently empty by omission —
   no prior phase has decided whether this project wants automated tests.

## Risks

- Architecture spans many not-yet-built components (preprocessing package,
  `scripts/`, `configs/` promotion, dashboard); sequencing errors (building
  the dashboard before the artifact schema it reads is stable, for example)
  could require rework if not followed in the order given in the report body.
- The dashboard's "must not become a second implementation of metric
  calculations" constraint is only as strong as future implementers'
  discipline in keeping `data_access.py` free of any import from
  `src/metrics.py`'s computational functions — worth a lint/review check
  when that phase arrives.
- No test infrastructure exists; as more source is reconciled and refactored
  (path normalization, package restructuring), regressions could go
  undetected without at least minimal validation beyond the manual
  hash-comparison approach used through Phase 4B.

## Next Recommended Phase

**Phase 7: Reconcile preprocessing source** (`src/preprocessing/`, path-normalized),
per the execution order derived in this phase's report body (section 17),
which itself must produce its own `docs/phases/PHASE_7_preprocessing_reconciliation.md`
report per the new standing requirement.

## Final Conclusion

Phase 6.3 completed a full architectural synthesis — preprocessing structure,
data boundaries, experiment artifact schema, dashboard design (data layer,
live-demo isolation, information hierarchy, visual design system),
dependency mapping, provenance model, and a concrete future phase sequence —
without modifying, executing, or committing anything. This file constitutes
that phase's permanent record under the newly-introduced `docs/phases/`
convention. Six concrete decisions remain open (listed above) and should be
resolved before Phase 7 begins.

---

## Full Phase 6.3 Report (as originally delivered)

### 1. Final Repository Architecture

```
cognitive-distortion-classification/
├── README.md                                    [TRACKED]
├── pyproject.toml, uv.lock                       [TRACKED]
├── .gitignore                                    [TRACKED]
├── .python-version                                [TRACKED]
│
├── docs/                                          [TRACKED]
│   ├── reconnaissance.md                          [TRACKED] (existing)
│   ├── phases/                                    [TRACKED] (new, this phase)
│   └── preprocessing_provenance.md                [TRACKED] (planned, Phase 6.2 §6)
│
├── notebooks/
│   └── Modeling_Cognitive_Distortion (2).ipynb    [TRACKED, PROVENANCE — frozen archival artifact]
│
├── external/original-drive/                       [TRACKED source subset; PROVENANCE]
│   ├── MODELING/COGNITIVE DISTORTION/             [TRACKED — already committed, b71999d]
│   └── DATASETS/                                  [LOCAL-ONLY — see Phase 6/6.1 boundary]
│
├── src/                                            [TRACKED]
│   ├── config_utils.py, loader.py, metrics.py     [TRACKED] (existing)
│   ├── finetune/, baselines/                       [TRACKED] (existing)
│   │
│   ├── preprocessing/                              [TRACKED] (planned — Phase 6.2/7)
│   │   ├── common/{io_utils,hash_utils,text_sim}.py
│   │   └── cognitive_distortion/
│   │       ├── {11 stage scripts}, config_utils.py
│   │       └── configs/preprocessing.yaml
│   │
│   └── dashboard/                                  [TRACKED] (planned — later phase)
│       ├── data_access.py    (reads artifacts only, never recomputes)
│       ├── inference.py      (live-demo model loading/prediction)
│       └── app.py            (entry point)
│
├── configs/                                        [TRACKED] (planned — promoted from external/)
│   ├── base.yaml
│   └── models/*.yaml
│
├── scripts/                                        [TRACKED] (planned)
│   ├── run_preprocessing.py
│   ├── run_baselines.py
│   ├── run_transformers.py
│   └── summarize_results.py
│
├── data/                                            [LOCAL-ONLY] (planned DATA_ROOT target)
│   └── (dib_labeled.csv, folds.json, dib_groups.json — never committed)
│
├── models/                                          [LOCAL-ONLY / GENERATED] (checkpoints)
│
└── results/                                         [MIXED]
    ├── metrics/                                     [TRACKED, GENERATED — small JSON]
    ├── predictions/                                 [LOCAL-ONLY, GENERATED]
    ├── figures/                                      [TRACKED, GENERATED — small]
    └── tables/                                        [TRACKED, GENERATED — small]
```

`configs/` and `scripts/` don't exist yet anywhere (currently only under
`external/original-drive/`) — their promotion to top-level tracked
directories is implied by the original migration plan but not yet executed.

### 2. Preprocessing Structure Decision

**Chosen: Option A — `src/preprocessing/{common,cognitive_distortion}/`.**

`COMMON` currently has exactly one consumer (confirmed in Phase 6.1 §5, no
second pipeline exists anywhere in the Drive snapshot). Option B's premise —
that `preprocessing_common` deserves top-level-under-`src` billing separate
from `preprocessing/` — is justified only if a second pipeline is
anticipated, and nothing in the evidence supports that; it would be
anticipatory abstraction, contrary to the standing project convention
("don't introduce unnecessary sophistication"). Option C's `data_pipeline`
naming is equally workable but introduces a new term with no evidentiary
grounding over `preprocessing`, which is literally what the Drive folder was
already called. Option A also keeps `src/` flat at one level of new nesting,
consistent with how `finetune/` and `baselines/` already sit as siblings.

### 3. Path Normalization Strategy

Single mechanism, reused for both modeling and preprocessing configs:

```
DATA_ROOT environment variable
  Colab default:  /content/drive/MyDrive/THESIS
  Local default:   <repo>/external/original-drive   (or <repo>/data — open decision)
```

Every hardcoded `/content/drive/MyDrive/THESIS/...` literal — in
`base.yaml`, `preprocessing.yaml`, and the 12 preprocessing files'
`sys.path.insert`/`DEFAULT_CONFIG_PATH` lines — gets replaced with a
`DATA_ROOT`-relative construction, read once at each config loader's entry
point. Design only; not implemented. Two sub-decisions remain open: (a)
whether `DATA_ROOT` points at `external/original-drive/DATASETS` directly or
a separate `data/` directory, and (b) whether `sys.path.insert` gets
replaced by proper package-relative imports now that everything would live
in one installable package.

### 4. Data Boundary

**Tracked source:** all Python (`src/**/*.py`), all YAML config, documentation
(`docs/`), dashboard source (`src/dashboard/`), scripts (`scripts/`).

**Local-only (never committed):** raw dataset, processed dataset
(`dib_labeled.csv`, `folds.json`, `dib_groups.json`, `*.sha256`),
intermediate CSVs, preprocessing/training logs, model checkpoints, full
per-row prediction CSVs.

**Generated but potentially trackable:** small aggregate JSON
(`results/metrics/<model>/summary.json`, `baseline_sanity_summary.json`,
`transformer_summary.json`), small figures/tables. Reasoning: these are
exactly the artifacts already committed in `b71999d` — small, non-sensitive
aggregate statistics, valuable as citable thesis evidence.

### 5. Experiment Artifact Architecture

Reusing the schema already validated end-to-end against real data in Phase
4B (recomputation matched stored results exactly across all 4 baselines):

```
results/
├── predictions/<model>/fold_N.csv        {sentence_id, y_true, y_pred, prob_0..prob_10}
├── metrics/<model>/fold_N.json            compute_metrics() output
├── metrics/<model>/summary.json           average_metrics() output
├── metrics/baseline_sanity_summary.json   cross-model comparison (already exists)
├── metrics/transformer_summary.json       cross-model comparison (not yet generated)
├── figures/                                confusion matrices, per-class bar charts
└── tables/                                  thesis-ready tables
```

**Critical constraint:** the dashboard's data-loading layer
(`src/dashboard/data_access.py`) must be a pure reader of these JSON/CSV
files — it calls no `compute_metrics`/`average_metrics` logic of its own. If
a metric doesn't exist in a `summary.json` file, the dashboard shows "not
yet available" rather than computing it inline.

### 6. Dashboard Architecture

**Framework evaluation:**
- **Streamlit**: single-process Python app, zero JS required, well-suited to
  a "research report + one interactive form" shape. Default styling leans
  generic-SaaS, but CSS override is achievable.
- **Flask/FastAPI + Jinja**: more markup/CSS control, but more code to build
  and maintain for the same outcome.
- **Full JS SPA**: rejected — contradicts "minimal JavaScript" and
  introduces a second toolchain (npm/node) alongside `uv`.

**Recommendation: Streamlit** — lets the artifact-reading architecture stay
pure Python with no serialization/API layer between data access and display.

**Layout:**
```
src/dashboard/
├── app.py            entry point — Streamlit page composition
├── data_access.py     reads results/metrics/**/*.json, results/predictions/**/*.csv
├── inference.py        loads a trained checkpoint, runs single-text prediction
└── styles.py / .css    design-system overrides
```

### 7. Dashboard Information Hierarchy

| # | Section | Content | Source artifact | Static/Dynamic | Needs model? | Needs dataset? |
|---|---|---|---|---|---|---|
| 1 | LIVE DEMO | text box, prediction + confidence | saved checkpoint under `models/<model>/` (not yet existing) | Dynamic | Yes | No |
| 2 | RESEARCH SUMMARY | pipeline diagram, one line per stage | `docs/reconnaissance.md` + artifact links | Static | No | No |
| 3 | DATASET | row count, class distribution, imbalance ratio | new dataset-summary JSON (not yet generated) | Static | No | Only at summary-generation time |
| 4 | PREPROCESSING | 9-stage pipeline, per-stage status | log-file presence checks | Static | No | Only file-existence checks |
| 5 | EXPERIMENTS | baseline table (4 completed) + transformer table (0 completed) | `results/metrics/*/summary.json` | Dynamic | No | No |
| 6 | EVALUATION | per-class F1, fold variance — baselines only; transformer explicitly pending | `summary.json`/`fold_N.json` | Dynamic | No | No |
| 7 | PROVENANCE / REPRODUCIBILITY | commit hash, data hash, seed | git metadata + `folds.json` metadata | Static/Dynamic | No | No |

No metric or result not already evidenced in a committed artifact is
invented — transformer rows explicitly render as "not yet executed."

### 8. Live Demo Architecture

```
user text input
    ↓
selected model (dropdown, populated from models/<name>/ with an actual checkpoint)
    ↓
tokenizer.from_pretrained(hf_model_id)
    ↓
model.eval() forward pass, single input, no gradient
    ↓
softmax → predicted label + per-class confidence
    ↓
rendered in the LIVE DEMO section
```

**Hard boundaries:** `src/dashboard/inference.py` never imports
`src/finetune/trainer.py`'s `train_one_fold`, never writes to `results/`,
`models/`, or dataset files.

**Graceful behavior when no checkpoint exists** (the current actual state —
all `models/*` directories are empty): the LIVE DEMO section renders the
form, but displays *"No trained model checkpoint is currently available. Run
the transformer training pipeline to enable live inference."* rather than
crashing, silently falling back to a baseline, or training on the spot.

### 9. Research Summary

```
Raw Dataset (4,628 rows)
    ↓  assign_ids_and_normalize, extract_spans, filter_raw
Cleaning / Normalization
    ↓  dedup_exact.py
Deduplication (exact)
    ↓  dedup_near.py — Jaccard-shingle clustering
Group Construction (group_id, dib_groups.json)
    ↓  build_folds.py — StratifiedGroupKFold, seed=42, 5 folds
Group-aware Split (folds.json, hash-verified)
    ↓  Baseline Models — ALL 4 COMPLETED, real Macro-F1 0.06/0.50/0.51/0.36
    ↓  Transformer Models — 7 configured, NOT YET EXECUTED
Evaluation — only 4 baselines have real results; transformer pending
    ↓
Final Model / Findings — NOT YET DETERMINED
```

### 10. Visual Design System

Light-only. Navy/blue single accent, white/near-white background, gray
borders, restrained shadows, editorial serif/sans headings, monospace for
technical values only. Prohibited: gradients, glassmorphism, dark mode,
decorative pills, excessive cards, oversized hero, AI sparkle iconography,
glow effects, animated counters, generic SaaS chrome. Not implemented as CSS
this phase.

### 11. Live Demo vs. Research Report

**LIVE DEMO**: interactive, stateless per-request, reads only a checkpoint,
never writes back to any results file. **RESEARCH REPORT**: purely a reader
of already-authoritative artifact files, no code path touches a checkpoint
or runs inference. The two share only `app.py` (page composition) and never
share state — a file-import-graph guarantee, not just a documented
convention.

### 12. Dependency Architecture

| Category | Packages | Already present? |
|---|---|---|
| A. Preprocessing | pandas, PyYAML, scipy (direct), scikit-learn (shared) | scikit-learn yes; rest no |
| B. Modeling | numpy, scikit-learn, pandas, PyYAML, torch, transformers, gensim | numpy/scikit-learn yes; rest no |
| C. Dashboard | streamlit | none yet |
| D. Dev/testing | undetermined | none |

Nothing added to `pyproject.toml` this phase.

### 13. Local Execution Workflow

```bash
uv sync                                              # works today
uv run python -m src.preprocessing.cognitive_distortion.assign_ids_and_normalize
                                                        # CANNOT run yet — path normalization required first
uv run python scripts/run_baselines.py                # CANNOT run yet — scripts/ doesn't exist,
                                                        # sklearn_baseline.py still hardcodes Colab paths
uv run python scripts/run_transformers.py --model X --fold Y
                                                        # same blocker + needs torch/transformers + GPU
uv run streamlit run src/dashboard/app.py             # CANNOT run yet — src/dashboard/ doesn't exist
```
No command in this list can actually run today.

### 14. Provenance Model

| Category | Example | How it's marked |
|---|---|---|
| 1. Original Drive snapshot | `external/original-drive/**` | Directory-level: unmodified copy |
| 2. Notebook-extracted source | `src/config_utils.py`, `src/loader.py`, `src/finetune/trainer.py` | SHA-256-verified-identical (commit `2fcfe00`) |
| 3. Reconciled source | `src/baselines/sklearn_baseline.py` (byte-identical); future `src/preprocessing/**` (NOT byte-identical — path-normalized) | Provenance doc states which reconciliation type each file underwent |
| 4. Reconstructed source | `src/metrics.py` | Header comment: "forensic reconstruction... should NOT be treated as the original source file" — **RECONSTRUCTED — ORIGINAL SOURCE NOT RECOVERED** |
| 5. Generated experiment artifacts | `results/metrics/*/summary.json` | Distinguished by directory + being data, not code |
| 6. Dashboard presentation artifacts | dataset-summary JSON | Marked as derived-for-display, regenerable from category 5 |

### 15. Commit Sequence (proposed, not executed)

```
1.  chore: reconcile preprocessing source modules
2.  chore: reconcile preprocessing configuration
3.  chore: define preprocessing dataset boundaries
4.  chore: add preprocessing dependencies
5.  docs: document preprocessing provenance
6.  chore: normalize modeling config paths (DATA_ROOT)
7.  chore: promote configs/ and scripts/ to top-level
8.  feat: add local preprocessing validation
9.  feat: reconcile baseline experiment run against local data
10. feat: add dashboard foundation
11. feat: add dashboard research report sections
12. feat: add dashboard live demo
13. docs: final documentation pass
```

### 16. Notebook Role

Untouched archival artifact, per every prior phase's explicit rule. Its
long-term role is reference only — the historical record of the original
Colab-based work and the byte-for-byte baseline every extraction was
verified against. It is not expected to become the execution entry point
going forward; that role shifts to `scripts/`.

### 17. Future Execution Phases

```
Phase 7:  Reconcile preprocessing source (src/preprocessing/, path-normalized)
Phase 8:  Define preprocessing/modeling dataset boundaries + dependency additions
Phase 9:  Normalize modeling config paths (DATA_ROOT)
Phase 10: Promote configs/ and scripts/ to top-level; write scripts/run_preprocessing.py
Phase 11: Validate preprocessing locally end-to-end (re-derive same hashes?)
Phase 12: Reconcile + re-validate baseline experiments against local data
Phase 13: Run transformer smoke test (1 model × 1 fold) — first GPU-dependent execution
Phase 14: Run full transformer experiments (7 models × 5 folds)
Phase 15: Build experiment artifact/report layer consumed by the dashboard
Phase 16: Build dashboard live demo (requires Phase 14's checkpoints)
Phase 17: Final documentation pass
```

### 18. Risks / Unresolved Decisions

See "Unresolved Decisions" and "Risks" sections above (deduplicated from the
original report body to avoid repetition within this file).

### 19. Phase 6.3 Conclusion

See "Final Conclusion" above.
