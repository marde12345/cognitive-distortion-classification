# Transformer Model Comparison — Researcher Summary

Source: Phase 16 official aggregates (`results/analysis/transformer/aggregate_metrics.json`,
`statistical_tests.json`, `anomaly_sensitivity.json`). No values here are
recomputed or overwritten; this is a reorganization of existing numbers.

## What was evaluated

- 7 transformer architectures: indobert_15g, indobert_base_p1, indobertweet,
  indoroberta_15g, mbert, nusabert, xlmr.
- Each evaluated on the same 5 cross-validation folds.
- Official statistics include all five folds for every model — no fold was
  dropped from the official aggregate.

## Official 5-fold mean Macro-F1 (descriptive, not a ranking)

| model | mean | std (ddof=1) | min | max | anomalous folds |
|---|---|---|---|---|---|
| indobert_base_p1 | 0.5724 | 0.0203 | 0.5493 | 0.5923 | none |
| indoroberta_15g | 0.5681 | 0.0112 | 0.5579 | 0.5864 | none |
| indobert_15g | 0.5639 | 0.0093 | 0.5527 | 0.5739 | none |
| nusabert | 0.5419 | 0.0280 | 0.5067 | 0.5848 | none |
| xlmr | 0.5090 | 0.0175 | 0.4923 | 0.5344 | none |
| indobertweet | 0.4775 | 0.2343 | 0.0592 | 0.5938 | fold_1 |
| mbert | 0.4156 | 0.2000 | 0.0587 | 0.5271 | fold_3 |

The highest observed official 5-fold mean Macro-F1 belongs to
indobert_base_p1 (0.5724). This is a factual ordering of one metric on
this dataset and fold split — it is not a model-selection decision.

## Effect of the two anomalous runs

Two runs (indobertweet/fold_1, mbert/fold_3) show near-zero Macro-F1,
consistent with a collapsed/degenerate training run for that specific
fold. Both are retained in the official aggregate (not silently
dropped). A separate sensitivity view, excluding only the anomalous
fold, is provided for context and is clearly labeled as sensitivity
analysis, never as a replacement official number:

| model | official 5-fold mean | 4-fold sensitivity mean (excl. anomaly) |
|---|---|---|
| indobertweet | 0.4775 | 0.5820 |
| mbert | 0.4156 | 0.5048 |

The anomalies substantially lower the official aggregate statistics for
these two models specifically. Several other models (indobert_15g,
indoroberta_15g, xlmr) show comparatively tight cross-fold behavior
(CV under 0.035), with no anomalous folds observed.

## Friedman test and its limitations

A Friedman test (non-parametric, repeated-measures, folds as blocks)
gives statistic = 18.60, df = 6, p = 0.0049 — indicating evidence that
the 7 models' fold-rank distributions are not all identical. This does
**not** identify which specific pairs of models differ, does not
specially account for the two anomalous runs, and rests on only 5
matched folds with no repeated seeds. It is weak evidence of an overall
difference, not proof of a specific ranking.

## Phase 18 demo checkpoint — a separate, unrelated decision

The Phase 18 live-inference demo uses one explicitly selected checkpoint
(model = indobert_base_p1, fold = fold_0, test Macro-F1 = 0.5923). This
was chosen for demonstration purposes only. It is a single fold-level
result, not the five-fold aggregate (0.5724), and its selection is not
evidence that indobert_base_p1 is universally the best model — it is
one concrete trained run used to show live inference working.

## Bottom line

The experiment provides comparative evidence across 7 architectures on
one dataset and one 5-fold split. It does not provide a definitive,
universal model ranking. **No automatic model selection was performed.**
The final choice of which model (if any) to carry forward remains an
explicit human decision, to be made with awareness of the anomalous
runs, the small number of folds, and the absence of repeated-seed
variance estimates.
