# Phase 19 — Transformer Model Comparison & Selection Evidence

## 19A. Reconnaissance

- Starting branch: `main`.
- Starting HEAD: `5f63ac2` (merge commit "merge: integrate Phase 18 live
  inference demo").
- Working-tree status at start: identical to the documented post-Phase-18
  baseline — only `src/finetune/trainer.py` modified (the pre-existing,
  uncommitted Phase 12 MPS device-selection fix), plus the same set of
  untracked files carried since earlier phases (`.claude/`, Phase 11–15
  reports, `external/original-drive/DATASETS/`, `results/logs/`,
  `results/metrics/<model>/` directories, `scripts/run_transformer.py`,
  `scripts/run_transformer_sweep.py`). Nothing unexpected was present.
- Commits since Phase 17/18 integration: none beyond the Phase 18 merge
  itself (`5f63ac2`) — `git log --oneline -8` shows no local commits
  between `5f63ac2` and the start of Phase 19.
- Phase 18 merge status: already merged into `main` (`bed7bba` is
  contained in `main` per `git branch --contains bed7bba`).
- Existing Phase 16 analysis artifacts: present and unmodified —
  `results/analysis/transformer/aggregate_metrics.json`,
  `statistical_tests.json`, `anomaly_sensitivity.json`,
  `fold_metrics.csv`, `per_class_metrics.csv`, `runtime_analysis.json`,
  plus a `plots/` directory.
- Existing Phase 18 checkpoint/demo state: the demo checkpoint
  (`models/demo/indobert_base_p1/fold_0/model.pt` + `checkpoint_meta.json`)
  exists only inside the `.claude/worktrees/phase18-manual/` filesystem,
  as documented in Phase 18 — `models/` is git-ignored and was never
  tracked, so the main checkout has no `models/` directory at all. No
  change was made to this.
- Dashboard state: no dashboard process was running at the start of
  Phase 19 (confirmed via `ps aux`).

**Changes introduced by Phase 19** (all new files; nothing pre-existing
was modified):
- `scripts/build_model_comparison.py`
- `results/analysis/transformer/comparison/model_comparison.csv`
- `results/analysis/transformer/comparison/model_comparison.json`
- `results/analysis/transformer/comparison/researcher_summary.md`
- `docs/phases/PHASE_19_TRANSFORMER_MODEL_COMPARISON.md` (this file)

**Changes that existed before Phase 19** (left untouched, not created or
modified by this phase): the uncommitted `src/finetune/trainer.py` MPS
fix and the full list of untracked files above.

## 19B. Source-of-Truth Rule

All numbers in this phase are read directly from the existing Phase 16
artifacts:
- `results/analysis/transformer/aggregate_metrics.json`
- `results/analysis/transformer/statistical_tests.json`
- `results/analysis/transformer/anomaly_sensitivity.json`

No training result was regenerated. No Phase 16 artifact was recomputed
or overwritten. Every number reproduced in this report and in
`results/analysis/transformer/comparison/` was cross-checked against
these source files by direct read (not re-derived independently), and
no discrepancy was found between this report's figures and the Phase 16
source values.

## 19C. Model Comparison Matrix

Produced by `scripts/build_model_comparison.py`, which reads
`aggregate_metrics.json` only (read-only — no recomputation of any
per-fold metric) and writes
`results/analysis/transformer/comparison/model_comparison.csv` /
`.json`. The official 5-fold aggregate is reproduced exactly as-is;
nothing here replaces it.

| model | macro_f1_mean | macro_f1_std | macro_f1_min | macro_f1_max | macro_f1_median | macro_f1_range | macro_f1_cv | accuracy_mean | weighted_f1_mean | n_folds | n_anomalous_folds | anomalous_fold_names |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| indobert_15g | 0.5639 | 0.0093 | 0.5527 | 0.5739 | 0.5665 | 0.0211 | 0.0165 | 0.6468 | 0.6520 | 5 | 0 | — |
| indobert_base_p1 | 0.5724 | 0.0203 | 0.5493 | 0.5923 | 0.5821 | 0.0430 | 0.0355 | 0.6643 | 0.6687 | 5 | 0 | — |
| indobertweet | 0.4775 | 0.2343 | 0.0592 | 0.5938 | 0.5860 | 0.5346 | 0.4907 | 0.6101 | 0.5785 | 5 | 1 | fold_1 |
| indoroberta_15g | 0.5681 | 0.0112 | 0.5579 | 0.5864 | 0.5667 | 0.0285 | 0.0198 | 0.6494 | 0.6531 | 5 | 0 | — |
| mbert | 0.4156 | 0.2000 | 0.0587 | 0.5271 | 0.4989 | 0.4684 | 0.4812 | 0.5500 | 0.5176 | 5 | 1 | fold_3 |
| nusabert | 0.5419 | 0.0280 | 0.5067 | 0.5848 | 0.5377 | 0.0781 | 0.0517 | 0.5845 | 0.5765 | 5 | 0 | — |
| xlmr | 0.5090 | 0.0175 | 0.4923 | 0.5344 | 0.5068 | 0.0421 | 0.0343 | 0.5400 | 0.5246 | 5 | 0 | — |

Full precision values are in `model_comparison.csv` / `.json`.

## 19D. Pairwise Comparison

All 21 pairwise mean Macro-F1 differences, reproduced unchanged from
`statistical_tests.json`'s `pairwise_mean_macro_f1_differences`
(positive = the first model's mean is higher):

| pair | mean diff |
|---|---|
| indobert_15g vs indobert_base_p1 | -0.0085 |
| indobert_15g vs indobertweet | +0.0865 |
| indobert_15g vs indoroberta_15g | -0.0041 |
| indobert_15g vs mbert | +0.1483 |
| indobert_15g vs nusabert | +0.0221 |
| indobert_15g vs xlmr | +0.0550 |
| indobert_base_p1 vs indobertweet | +0.0949 |
| indobert_base_p1 vs indoroberta_15g | +0.0043 |
| indobert_base_p1 vs mbert | +0.1568 |
| indobert_base_p1 vs nusabert | +0.0305 |
| indobert_base_p1 vs xlmr | +0.0634 |
| indobertweet vs indoroberta_15g | -0.0906 |
| indobertweet vs mbert | +0.0619 |
| indobertweet vs nusabert | -0.0644 |
| indobertweet vs xlmr | -0.0315 |
| indoroberta_15g vs mbert | +0.1525 |
| indoroberta_15g vs nusabert | +0.0262 |
| indoroberta_15g vs xlmr | +0.0591 |
| mbert vs nusabert | -0.1262 |
| mbert vs xlmr | -0.0934 |
| nusabert vs xlmr | +0.0329 |

These are descriptive differences only. No ranking is derived from
them, and no significance testing beyond the existing Phase 16 Friedman
test was performed.

## 19E. Stability Analysis

| model | mean Macro-F1 | std (ddof=1) | range | CV | anomalous folds |
|---|---|---|---|---|---|
| indobert_15g | 0.5639 | 0.0093 | 0.0211 | 0.0165 | 0 |
| indobert_base_p1 | 0.5724 | 0.0203 | 0.0430 | 0.0355 | 0 |
| indoroberta_15g | 0.5681 | 0.0112 | 0.0285 | 0.0198 | 0 |
| nusabert | 0.5419 | 0.0280 | 0.0781 | 0.0517 | 0 |
| xlmr | 0.5090 | 0.0175 | 0.0421 | 0.0343 | 0 |
| indobertweet | 0.4775 | 0.2343 | 0.5346 | 0.4907 | 1 (fold_1) |
| mbert | 0.4156 | 0.2000 | 0.4684 | 0.4812 | 1 (fold_3) |

indobertweet/fold_1 and mbert/fold_3 are the two identified anomalous
runs (near-zero Macro-F1, consistent with a collapsed training run on
that specific fold). Their numeric effect, per the existing Phase 16
sensitivity analysis (`anomaly_sensitivity.json`):

- indobertweet: official 5-fold mean = 0.4775; 4-fold sensitivity mean
  (excluding fold_1) = 0.5820.
- mbert: official 5-fold mean = 0.4156; 4-fold sensitivity mean
  (excluding fold_3) = 0.5048.

These sensitivity means are sensitivity analysis only. They do not
replace the official 5-fold aggregates shown in Section 19C, which
retain both anomalous folds.

## 19F. "Which Model Is Best?" — Evidence, Not a Verdict

- By official 5-fold mean Macro-F1, the highest observed mean is
  indobert_base_p1 (0.5724).
- The next observed means are indoroberta_15g (0.5681) and
  indobert_15g (0.5639).
- This is a descriptive ordering of one metric (Macro-F1 mean), not an
  overall model selection.
- The two anomalous runs materially affect the official aggregates of
  indobertweet and mbert (Section 19E).
- The Friedman test indicates evidence that model rankings are not all
  identical across folds (statistic = 18.60, df = 6, p = 0.0049), but
  it does not identify which pairs differ.
- Only 5 matched folds exist.
- No repeated seeds exist.
- Therefore the experiment provides comparative evidence, not a
  definitive universal model ranking.

Stating that indobert_base_p1 has the highest observed mean is a factual
observation. It is **not** a claim that indobert_base_p1 "wins,"
"is superior," or "should be selected" — that decision is left to the
human researcher.

## 19G. Explicit Demo Checkpoint Context

The Phase 18 live-inference demo uses one explicitly selected demo
checkpoint (model = indobert_base_p1, fold = fold_0), which recorded a
test Macro-F1 of 0.5923 at training time.

- This checkpoint was explicitly selected for the live demo, for
  demonstration purposes.
- It is **not** evidence that indobert_base_p1 is universally the best
  model.
- The demo checkpoint is one concrete trained fold, not an aggregate.
- The Phase 16 aggregate mean for indobert_base_p1 across all five
  folds is 0.5724.
- The fold-0 score of 0.5923 is a single fold-level result and must not
  be confused with the five-fold aggregate of 0.5724.

No change was made to the checkpoint, the dashboard, or the demo
selection in this phase.

## 19H. Researcher-Facing Summary

See `results/analysis/transformer/comparison/researcher_summary.md` for
the full standalone summary. In brief:

1. 7 transformer architectures were evaluated: indobert_15g,
   indobert_base_p1, indobertweet, indoroberta_15g, mbert, nusabert,
   xlmr.
2. Each was evaluated on the same 5 cross-validation folds.
3. Official statistics include all five folds for every model.
4. Two anomalous runs (indobertweet/fold_1, mbert/fold_3) were retained,
   not silently excluded.
5. The anomalies substantially lower the official aggregate statistics
   of exactly those two models.
6. Several other models (indobert_15g, indoroberta_15g, xlmr) show
   comparatively tight cross-fold behavior (CV < 0.035), with no
   anomalies.
7. A Friedman test found evidence of an overall difference across the 7
   models' fold-rank distributions (p = 0.0049), but this does not
   identify specific pairwise differences and rests on only 5 matched
   folds with no repeated seeds — weak evidence, not proof.
8. The Phase 18 demo checkpoint selection (indobert_base_p1, fold_0) is
   a separate, unrelated decision from statistical model comparison; it
   reflects one fold-level run chosen for demonstration, not a
   model-selection conclusion.

## 19I. No Automatic Selection

**No automatic model selection was performed.**

No ranking score, winner badge, "recommended model" label, tier, medal,
traffic-light indicator, or composite score was created anywhere in
this phase's artifacts or in the dashboard. The dashboard UI was not
modified by this phase.

## 19J. Reproducibility

`scripts/build_model_comparison.py`:
- Runs via `uv run python scripts/build_model_comparison.py`.
- Is deterministic (pure read-then-format, no randomness).
- Reads only existing artifacts
  (`results/analysis/transformer/aggregate_metrics.json`,
  `anomaly_sensitivity.json`) — it does not touch the dataset, does not
  create or modify any model checkpoint, and does not require network
  access.
- Uses only the Python standard library (`json`, `os`) — no new
  dependency was added.
- Reuses the existing Phase 16 output files as its sole input rather
  than duplicating or re-implementing `scripts/analyze_transformer_sweep.py`'s
  computation logic.

## 19K. Validation

Full test suite, run from the main checkout after Phase 19's file
additions:

```
uv run python -m unittest discover -s tests -p "test_*.py"
```

Result: 57/57 tests passing, 0 failures/errors (unchanged from the
Phase 18 baseline — Phase 19 added no new test files, since it
introduces no new executable logic beyond the read-only comparison
script).

Additional checks performed:
- No training process was started or is running.
- No dashboard process was started or is running.
- Dataset checksums (`dib_labeled.csv`, `folds.json`) unchanged from the
  established baseline.
- `results/metrics/`, `results/logs/`, `results/predictions/` unchanged
  (no diff against the pre-Phase-19 state).
- `results/analysis/transformer/aggregate_metrics.json`,
  `statistical_tests.json`, `anomaly_sensitivity.json`,
  `fold_metrics.csv`, `per_class_metrics.csv`, `runtime_analysis.json`,
  and `plots/` unchanged.
- Phase 18 checkpoint files (`models/demo/indobert_base_p1/fold_0/model.pt`,
  `checkpoint_meta.json`, inside the Phase 18 worktree) unchanged.
- No new model checkpoint was created anywhere.

## 19L. Deliverables

- `docs/phases/PHASE_19_TRANSFORMER_MODEL_COMPARISON.md` (this file).
- `results/analysis/transformer/comparison/model_comparison.csv`
- `results/analysis/transformer/comparison/model_comparison.json`
- `results/analysis/transformer/comparison/researcher_summary.md`
- `scripts/build_model_comparison.py`

No existing dataset, prediction file, or Phase 16 artifact was
duplicated or modified.

## 19M. Git Discipline

Reported in the final commit step below, immediately before committing:
`git status`, `git diff --stat`, and the exact file list are shown, and
it is confirmed that only Phase 19 files are staged — none of the
pre-existing uncommitted changes (`src/finetune/trainer.py` MPS fix,
untracked directories) are included in the Phase 19 commit.

## 19N. Final Report

See the end-of-turn summary in the session for the itemized final
report (starting HEAD, final HEAD, files created/modified, test result,
checksum result, artifact/checkpoint integrity result, training-occurred
answer, automatic-selection answer, the factual highest-mean-model
answer with its caveat, and final `git status`).
