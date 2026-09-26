# Phase 16 — Transformer Cross-Fold Statistical Analysis

## Objective

Analyze the completed 7-model × 5-fold transformer sweep (Phase 15) using
only existing artifacts under `results/`, producing a reproducible,
factual statistical comparison — without retraining, modifying training
code, altering the dataset, or silently excluding the two known anomalous
runs. No "best model" decision is made; model selection is left explicit
and human.

## Dataset and Experiment Scope

**FACT**: 7 models (`indobert_15g`, `indobert_base_p1`, `indobertweet`,
`indoroberta_15g`, `mbert`, `nusabert`, `xlmr`) × 5 folds
(`fold_0`..`fold_4`) = 35 combinations, all sourced from Phase 13
(`indobert_base_p1`/`fold_0`) and Phase 15 (the remaining 34). Real fold
test-set sizes, computed live from the dataset (not assumed): `{fold_0:
919, fold_1: 924, fold_2: 923, fold_3: 939, fold_4: 923}`.

## Artifact Validation

**FACT**, `scripts/analyze_transformer_sweep.py`'s validation pass (run
before any statistic was computed): all 35 combinations have
`results/metrics/<model>/<fold>.json`, `results/logs/<model>/<fold>.json`,
and `results/predictions/<model>/<fold>.csv` present; every execution log
reports `status == "completed"`; every metrics file contains `macro_f1`,
`weighted_f1`, `accuracy`, `per_class`, `history`; every prediction CSV's
row count exactly matches the corresponding fold's real test-set size.
**Zero problems found** — output: `Validation OK: all 7x5 combinations
structurally valid.`

## Aggregate Metrics

**CALCULATION**, sample standard deviation (`ddof=1`) used throughout, as
instructed, since the 5 folds are being treated as a sample of the
model's cross-fold behavior, not the entire population of possible folds.
This choice is stated explicitly and applied uniformly — no metric in
this report uses population std (`ddof=0`).

**Official aggregate = all 5 folds, including the two known anomalous
runs** (per instruction — anomalies are analyzed, not silently excluded
from the official numbers):

| Model | Macro-F1 mean | std | min | max | median | range | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| indobert_15g | 0.5639 | 0.0093 | 0.5527 | 0.5739 | 0.5665 | 0.0211 | 0.0165 |
| indobert_base_p1 | 0.5724 | 0.0203 | 0.5493 | 0.5923 | 0.5821 | 0.0430 | 0.0355 |
| indobertweet | 0.4775 | 0.2343 | 0.0592 | 0.5938 | 0.5860 | 0.5346 | 0.4907 |
| indoroberta_15g | 0.5681 | 0.0112 | 0.5579 | 0.5864 | 0.5667 | 0.0285 | 0.0198 |
| mbert | 0.4156 | 0.2000 | 0.0587 | 0.5271 | 0.4989 | 0.4684 | 0.4812 |
| nusabert | 0.5419 | 0.0280 | 0.5067 | 0.5848 | 0.5377 | 0.0781 | 0.0517 |
| xlmr | 0.5090 | 0.0175 | 0.4923 | 0.5344 | 0.5068 | 0.0421 | 0.0343 |

**OBSERVATION**: `indobertweet` and `mbert` show by far the largest std
and coefficient of variation (CV ≈ 0.49 and 0.48, roughly 10–30× the other
five models' CV) — this is driven entirely by their single anomalous fold
each (see Anomalous Run Analysis). Accuracy and Weighted-F1 tables show
the identical pattern and are provided in full in
`results/analysis/transformer/aggregate_metrics.json` and
`fold_metrics.csv` (not reproduced twice here for brevity — the Macro-F1
table above is representative of the same relative pattern).

## Fold-Level Results

**FACT**, Macro-F1 per model per fold (★ = flagged anomalous run):

| Model | fold_0 | fold_1 | fold_2 | fold_3 | fold_4 | Mean | Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| indobert_15g | 0.5665 | 0.5557 | 0.5527 | 0.5709 | 0.5739 | 0.5639 | 0.0093 |
| indobert_base_p1 | 0.5923 | 0.5821 | 0.5493 | 0.5517 | 0.5865 | 0.5724 | 0.0203 |
| indobertweet | 0.5574 | **0.0592★** | 0.5910 | 0.5860 | 0.5938 | 0.4775 | 0.2343 |
| indoroberta_15g | 0.5579 | 0.5691 | 0.5667 | 0.5602 | 0.5864 | 0.5681 | 0.0112 |
| mbert | 0.4897 | 0.4989 | 0.5037 | **0.0587★** | 0.5271 | 0.4156 | 0.2000 |
| nusabert | 0.5437 | 0.5377 | 0.5363 | 0.5067 | 0.5848 | 0.5419 | 0.0280 |
| xlmr | 0.4940 | 0.4923 | 0.5174 | 0.5344 | 0.5068 | 0.5090 | 0.0175 |

Full Accuracy and Weighted-F1 fold-level tables, plus fold-wise
(column-wise) descriptive statistics across all 7 models per fold, are in
`results/analysis/transformer/fold_metrics.csv` — this is raw per-run
data suitable for any downstream recomputation; not restated table-by-table
here to avoid duplicating what the CSV already holds exactly.

## Anomalous Run Analysis

Per instruction, both anomalies are analyzed in full, not removed from
the official aggregate.

### `indobertweet` / `fold_1`

**FACT**, metrics: `macro_f1=0.0592, accuracy=0.4827, weighted_f1=0.3143`.
**FACT**, epoch history:
```
epoch 1: train_loss=2.3700, val_macro_f1=0.0592
epoch 2: train_loss=2.4163, val_macro_f1=0.0034
epoch 3: train_loss=2.4127, val_macro_f1=0.0592   (early-stopped)
```
**FACT**, comparison against the same model's other 4 folds:
`{fold_0: 0.5574, fold_2: 0.5910, fold_3: 0.5860, fold_4: 0.5938}` — all
four cluster tightly around 0.55–0.59; `fold_1` is a clear outlier
relative to its own model's other folds, not merely relative to other
models.

**CALCULATION — official (5-fold, includes anomaly)**: mean=0.4775,
std=0.2343. **Sensitivity analysis — NOT the official result (4-fold,
excludes `fold_1`)**: mean=0.5820, std=0.0168. **Delta from exclusion**:
+0.1046 (mean increases by ~0.105 if the anomaly is excluded).

### `mbert` / `fold_3`

**FACT**, metrics: `macro_f1=0.0587, accuracy=0.4771, weighted_f1=0.3082`.
**FACT**, epoch history:
```
epoch 1: train_loss=2.4048, val_macro_f1=0.0162
epoch 2: train_loss=2.4016, val_macro_f1=0.0162
epoch 3: train_loss=2.3959, val_macro_f1=0.0587
epoch 4: train_loss=2.3927, val_macro_f1=0.0587
epoch 5: train_loss=2.3912, val_macro_f1=0.0587   (all 5 epochs ran)
```
**FACT**, comparison against the same model's other 4 folds:
`{fold_0: 0.4897, fold_1: 0.4989, fold_2: 0.5037, fold_4: 0.5271}` — all
four cluster around 0.49–0.53; `fold_3` is a clear outlier for this model
specifically.

**CALCULATION — official (5-fold, includes anomaly)**: mean=0.4156,
std=0.2000. **Sensitivity analysis — NOT the official result (4-fold,
excludes `fold_3`)**: mean=0.5048, std=0.0159. **Delta from exclusion**:
+0.0892.

**INTERPRETATION**: in both cases, `train_loss` remained near `ln(11) ≈
2.398` (the loss of a uniform 11-class predictor) throughout training and
never meaningfully decreased, indicating the classifier head did not learn
a useful signal for that specific fold — in contrast to every other fold
of the same model, where loss decreased substantially (e.g.
`indobertweet/fold_0`'s successful pattern is available in
`results/metrics/indobertweet/fold_0.json`'s history for direct
comparison). **The root cause is not established by the existing
artifacts** — the execution logs and metrics contain no information about
weight initialization, batch ordering, or any other factor that would
explain *why* these two specific (model, fold) combinations failed to
learn while the other 33 succeeded. **This remains unresolved.** No
further diagnosis was attempted this phase, per instruction not to
diagnose beyond what existing artifacts support.

**The official aggregate, as instructed, remains the full 5-fold result
for both models** (`indobertweet` mean=0.4775, `mbert` mean=0.4156) unless
a later, explicit decision changes this.

## Per-Class Analysis

**FACT**, computed for all 7 models × 11 classes (full table in
`results/analysis/transformer/per_class_metrics.csv`). Support (row count)
per class, summed across the 5 real test folds, is constant across models
since all models share the same folds: `{0: 2219, 1: 445, 2: 388, 3: 371,
4: 239, 5: 188, 6: 283, 7: 214, 8: 158, 9: 85, 10: 38}` — matching the
dataset's class distribution first established in Phase 1's reconnaissance.

**OBSERVATION**, using `indobert_base_p1` as a representative example
(the model with the highest official mean macro-F1): mean F1 per class
ranges from 0.8438 (label 3, support 371) down to **0.0954 (label 10,
support 38, std 0.132)** and 0.2851 (label 9, support 85, std 0.122) — the
two lowest-support classes show both the lowest mean F1 and by far the
highest fold-to-fold std across every model inspected, consistent with
the general expectation that very small per-fold support (label 10 has
only ~6–8 test rows per fold) makes per-class F1 highly sensitive to a
handful of predictions. **This is explicitly noted, not hidden**: any
per-class comparison involving labels 9 or 10 should be read with this
low-support caveat in mind, for every model, not only `indobert_base_p1`.

## Statistical Tests

**CALCULATION**, Friedman test (non-parametric repeated-measures test,
folds as the 5 matched blocks, 7 models as treatments), computed via
`scipy.stats.friedmanchisquare`:
```
chi2 statistic = 18.60
df = 6
p-value = 0.0049
```
**What this test does and does NOT establish**: it tests whether the
7 models' macro-F1 *rankings* differ systematically across the same 5
folds (a matched-samples design, correctly avoiding the mistake of
treating all 35 runs as independent observations, which they are not —
each fold's 7 models share the same train/val/test split). A p-value of
0.0049 is below the conventional 0.05 threshold, suggesting the models'
performance rankings are not identical across folds purely by chance.
**It does NOT identify which specific pairs of models differ**, does not
account for the two anomalous runs in any special way (they are included
in the ranks exactly as any other value), and — critically — **is being
computed on only 5 matched blocks**, a very small sample for this test.
**INTERPRETATION**: this result should be treated as weak-to-moderate
evidence that the models are not all performing identically, not as proof
of a specific ordering or of practical significance. No claim of a
"significant difference between model X and model Y" is made from this
test alone.

**Pairwise mean Macro-F1 differences** (21 pairs, full list in
`results/analysis/transformer/statistical_tests.json`) — reported as raw
mean differences only, with no significance test per pair (none was
requested or computed, avoiding an inflated multiple-comparisons claim).

## Confidence Intervals

**CALCULATION**, 95% t-intervals on each model's 5-fold macro-F1 mean
(`t.ppf(0.975, df=4)` × sample std / √5):

| Model | Mean | 95% CI |
|---|---:|---|
| indobert_15g | 0.5639 | [0.5524, 0.5755] |
| indobert_base_p1 | 0.5724 | [0.5472, 0.5976] |
| indobertweet | 0.4775 | [0.1866, 0.7684] |
| indoroberta_15g | 0.5681 | [0.5541, 0.5820] |
| mbert | 0.4156 | [0.1673, 0.6639] |
| nusabert | 0.5419 | [0.5071, 0.5766] |
| xlmr | 0.5090 | [0.4873, 0.5307] |

**Explicit limitation, per instruction**: these are **not** rigorous
confidence intervals over each model's true population performance. With
only `n=5` matched folds, the t-interval is a rough, illustrative
Wald-type interval, highly sensitive to individual fold values — visible
directly in `indobertweet` and `mbert`'s enormously wide intervals
(driven entirely by their one anomalous fold each), which should not be
read as "these two models are simply more variable in general," but as a
direct artifact of the single anomalous data point inflating the
sample variance with only 4 other points to offset it.

## Runtime Analysis

**FACT**, from `results/logs/transformer_sweep.json` (34 completed sweep
runs) plus Phase 13's separately-logged `indobert_base_p1`/`fold_0` run
(521.30s, not part of the sweep manifest since it predates the sweep and
was correctly skipped rather than re-run):

| Model | Mean (s) | Std | Min | Max | n |
|---|---:|---:|---:|---:|---:|
| indobert_15g | 537.30 | 55.05 | 510.12 | 635.66 | 5 |
| indobert_base_p1 | 523.29 | 1.88 | 520.61 | 524.96 | 4* |
| indobertweet | 485.45 | 98.15 | 312.85 | 558.76 | 5 |
| indoroberta_15g | 549.54 | 26.10 | 535.97 | 598.73 | 5 |
| mbert | 579.37 | 23.51 | 566.62 | 622.99 | 5 |
| nusabert | 534.94 | 16.60 | 524.56 | 563.63 | 5 |
| xlmr | 640.01 | 30.82 | 622.37 | 692.62 | 5 |

*`indobert_base_p1` shows `n=4` here because `fold_0`'s timing (521.30s)
lives in Phase 13's separate log, not the sweep manifest — including it
would give a 5th data point nearly identical to the other four (all
~520–525s), not changing the interpretation.

**FACT**: overall (34 completed sweep runs) mean=550.77s, median=536.07s,
min=312.85s, max=692.62s. **Fastest observed run**: `indobertweet/fold_1`
at 312.85s — this is the same anomalous, early-stopped run discussed
above (3 epochs instead of 5), explaining its short duration; this is a
runtime observation, not a quality signal. **Slowest observed run**:
`xlmr/fold_0` at 692.62s. **Total sweep wall-clock**: 5h12m8s (from Phase
15's report; not recomputed here since it is a wall-clock measurement, not
a sum of per-run JSON fields, and Phase 15 already established it
correctly).

**No model-quality inference is drawn from any runtime number.**

## Visualizations

**FACT**, 6 PNG files generated under `results/analysis/transformer/plots/`:
```
01_macro_f1_mean_std.png      - bar chart, mean ± std per model
02_accuracy_mean_std.png       - bar chart, mean ± std per model
03_weighted_f1_mean_std.png    - bar chart, mean ± std per model
04_macro_f1_across_folds.png   - line plot, one line per model across the
                                  5 folds, red X markers on the 2 flagged
                                  anomalous points
05_runtime_distribution.png     - box plot of per-run elapsed time, by model
06_per_class_f1_heatmap.png     - 7×11 heatmap, mean F1 per model per class
```
Model ordering in every plot follows `configs/models/*.yaml`'s filename
order (`indobert_15g, indobert_base_p1, indobertweet, indoroberta_15g,
mbert, nusabert, xlmr`) — an arbitrary but consistent ordering, explicitly
not a best-to-worst ranking (no plot sorts by performance).

## Interpretation

**INTERPRETATION** (explicitly qualified, not presented as fact):

- Five of the seven models (`indobert_15g`, `indobert_base_p1`,
  `indoroberta_15g`, `nusabert`, `xlmr`) show tight, consistent cross-fold
  behavior (CV between ~0.017 and ~0.052) — their 5-fold macro-F1 spreads
  are small relative to their means.
- `indobertweet` and `mbert`'s official aggregate statistics are dominated
  by one anomalous fold each; their *sensitivity-analysis* (4-fold,
  anomaly excluded) means — 0.5820 and 0.5048 respectively — would place
  them within the same general range as the other five models, **but this
  is explicitly the sensitivity result, not the official one**, and no
  decision has been made about which figure should represent these models
  going forward.
- The Friedman test's p=0.0049 provides some evidence the 7 models are not
  performing identically across folds, but with only 5 blocks this is
  weak evidence of a real difference, not a strong or definitive claim,
  and does not by itself identify a "winner."
- The confidence intervals, per their own stated limitation, should not be
  read as precise population estimates — they are most useful here for
  illustrating just how much the two anomalies widen uncertainty for their
  respective models.

**No model is called "best," "worst," "superior," or "inferior" anywhere
in this report.**

## Limitations

- Single sweep, single seed (`seed=42` throughout) — no repeated-trial
  variance estimate exists independent of the 5-fold spread itself.
- Root cause of the two anomalous runs is unresolved; this analysis cannot
  determine whether they reflect a rare-but-real training instability
  (relevant to any future re-run) or something specific to that exact
  run's conditions.
- The Friedman test and the CIs both rely on only 5 matched observations
  — appropriate given the actual data, but inherently limited in
  statistical power regardless of method.
- Per-class metrics for labels 9 and 10 (support 85 and 38 across all 5
  folds combined) carry high variance and limited interpretability for
  every model, not only the ones discussed by example.
- This analysis used only existing artifacts; it did not and could not
  verify whether the anomalous runs' predictions differ qualitatively
  (e.g. always predicting one class) beyond what the existing per-class
  metrics already show — no new inspection of the raw prediction CSVs
  beyond their row counts (already validated in Phase 15) was performed.

## Implications for Live Demo

**FACT**: `find models -type f` returns empty — **no persistent model
checkpoint exists for any of the 35 completed runs**, confirming the
expected state (unchanged since Phase 11; `train_one_fold` was not
modified this phase and still does not persist weights to disk). Any
future live-demo feature requiring a loaded model would need a specific
model/fold retrained with checkpoint persistence explicitly added — a
future, separate, explicitly-authorized change, not performed here.

Based on the descriptive evidence in this report (tight cross-fold
consistency, no anomaly, competitive official mean macro-F1 of 0.5724):

> **`indobert_base_p1` is a candidate for explicit human review** for any
> future checkpoint-training/live-demo work.

This is a candidate flag, not an automatic selection — `indoroberta_15g`
(mean 0.5681, tightest CV at 0.0198) and `indobert_15g` (mean 0.5639,
second-tightest CV at 0.0165) are comparably strong candidates by the same
descriptive criteria, and the sensitivity-analysis figures for
`indobertweet` (0.5820 excluding its anomaly) are not disqualifying either
once that caveat is understood. The final choice remains an explicit human
decision informed by, but not made by, this report.

## Reproducibility

Every number in this report is derived mechanically from
`scripts/analyze_transformer_sweep.py` (statistics) and
`scripts/plot_transformer_sweep.py` (plots), both read-only with respect
to `results/metrics/`, `results/logs/`, `results/predictions/`, and the
dataset. Re-running both scripts against the same `results/` tree will
reproduce identical output files, since no random sampling is involved
anywhere in the analysis itself (the only randomness in this phase's scope
was already fixed at training time via `seed=42`, outside this phase).

## Files Created

```
scripts/analyze_transformer_sweep.py
scripts/plot_transformer_sweep.py
results/analysis/transformer/aggregate_metrics.json
results/analysis/transformer/fold_metrics.csv
results/analysis/transformer/per_class_metrics.csv
results/analysis/transformer/anomaly_sensitivity.json
results/analysis/transformer/runtime_analysis.json
results/analysis/transformer/statistical_tests.json
results/analysis/transformer/plots/01_macro_f1_mean_std.png
results/analysis/transformer/plots/02_accuracy_mean_std.png
results/analysis/transformer/plots/03_weighted_f1_mean_std.png
results/analysis/transformer/plots/04_macro_f1_across_folds.png
results/analysis/transformer/plots/05_runtime_distribution.png
results/analysis/transformer/plots/06_per_class_f1_heatmap.png
docs/phases/PHASE_16_TRANSFORMER_STATISTICAL_ANALYSIS.md
```

## Files Modified

`pyproject.toml`, `uv.lock` — `matplotlib` added via `uv add` (genuinely
required for the requested visualizations; no other dependency was added).

## Files Untouched (verified, not assumed)

Every pre-existing file under `results/metrics/`, `results/logs/`,
`results/predictions/` (all 35 combinations' original artifacts plus the
sweep manifest); `external/original-drive/` in its entirety; the notebook;
`src/finetune/trainer.py`, `src/config_utils.py`, `src/loader.py`,
`src/metrics.py`; all `configs/*.yaml`; the existing dashboard
(`src/dashboard/`, `scripts/run_dashboard.py`).

## Dashboard Integration Preparation

**FACT**, verified by inspecting `src/dashboard/data_loader.py`: the
existing dashboard reads `results/logs/transformer_sweep.json` and
per-run `results/metrics/<model>/<fold>.json` /
`results/logs/<model>/<fold>.json` directly. **It contains no reference to
`results/analysis/` anywhere.** The new analysis artifacts from this phase
are therefore **not** automatically picked up by the existing dashboard —
a future dashboard phase would need to explicitly add code to read
`results/analysis/transformer/aggregate_metrics.json` (and the other new
files) and to serve/display the plot PNGs. This phase did not modify the
dashboard, per instruction.

## Safety Checks

**FACT**, before and after this phase's work:
```
Dataset checksums:      unchanged (ed154162... / 730cd82a..., verified twice)
git diff -- external/original-drive/:   empty (before and after)
git status --short notebooks/:           empty (before and after)
Training process started:                 NO — no `run_transformer*` process
                                          was ever invoked this phase
Existing metric/log/prediction files:      unchanged (git status shows every
                                          pre-existing results/ path as
                                          untouched; only new files/dirs
                                          under results/analysis/ appear)
Checkpoint created:                        NO (`find models -type f` empty)
```

## Git Status

**FACT**, final `git status --short` (relevant new lines only; full output
identical to Phase 15's end state plus the additions below):
```
 M pyproject.toml          (matplotlib dependency)
 M uv.lock                  (matplotlib + transitive deps)
?? results/analysis/         (new — all Phase 16 artifacts)
?? scripts/analyze_transformer_sweep.py
?? scripts/plot_transformer_sweep.py
?? docs/phases/PHASE_16_TRANSFORMER_STATISTICAL_ANALYSIS.md
```
All other paths (source, dataset, notebook, existing results, the
concurrently-merged dashboard from Phase 15's note) are unchanged from the
prior phase's end state. **Nothing was staged. Nothing was committed.**

## Conclusion

All 35 sweep combinations were validated structurally intact before any
statistic was computed. Aggregate, fold-level, per-class, runtime, and
statistical-test analyses were produced entirely from existing artifacts,
with sample standard deviation (`ddof=1`) used consistently and disclosed.
The two known anomalous runs (`indobertweet`/`fold_1`, `mbert`/`fold_3`)
were analyzed in detail — their metrics, epoch histories, and impact on
their models' aggregates (via an explicitly-labeled sensitivity analysis,
not a silent exclusion) are fully documented, with root cause left
unresolved rather than speculated. A Friedman test (p=0.0049, 5 blocks)
provides weak-to-moderate evidence the 7 models do not perform identically
across folds, without identifying a winner. No model was automatically
selected; `indobert_base_p1` is flagged as one reasonable **candidate for
explicit human review** for future checkpoint/live-demo work, alongside
two comparably-strong alternatives, with the final decision left to the
user. No training process was started, no checkpoint was created, the
dataset and `external/original-drive/` remain unchanged, and nothing was
committed.
