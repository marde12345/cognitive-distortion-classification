# Repository Reconnaissance

This document records the findings of the initial reconnaissance of
`notebooks/Modeling_Cognitive_Distortion (2).ipynb`, performed before any
extraction or refactoring took place. It reflects the repository state as
observed at the time of the initial thesis snapshot (commit `feccab5`).

Findings are marked as either **FACT** (directly observed in the notebook,
its saved outputs, or the GitHub repository), **INFERENCE** (a conclusion
drawn from those facts but not directly verifiable from this repository
alone), or **RECOMMENDATION** (a proposed course of action, not a decision
that has been made).

## 1. Current Repository State (FACT)

- Before this migration began, the GitHub repository
  (`marde12345/cognitive-distortion-classification`) was empty — no files,
  no README, no prior commits.
- The only artifact available was a single Colab notebook,
  `Modeling_Cognitive_Distortion (2).ipynb` (31 cells), now preserved at
  `notebooks/Modeling_Cognitive_Distortion (2).ipynb`.
- No `.gitignore`, `requirements.txt`, `src/`, `configs/`, `scripts/`, or
  `results/` existed locally prior to this migration.
- The notebook programmatically writes several Python source files
  (`config_utils.py`, `loader.py`, `trainer.py`) as string literals into a
  Google Drive path at runtime, rather than storing them as standalone
  files in this repository.
- No secrets, API keys, tokens, or credentials were found anywhere in the
  notebook's source cells.

## 2. Current Research Pipeline (FACT, traced from code)

```
raw/labeled data (dib_labeled.csv, origin outside notebook scope)
    -> validate_dataset() [config_utils.py] - schema, NaN, id-uniqueness,
       label-set, dataset<->folds.json hash check
    -> folds.json (external, pre-built 5-fold split; construction code not
       present in this notebook)
    -> loader.load_folds() / get_fold_data() - filters by train/val/test
       sentence_id sets per fold
    -> baselines.sklearn_baseline (run_majority_class, run_tfidf_lr,
       run_tfidf_svm, run_svm_word2vec)
    -> finetune.trainer.train_one_fold() - 7 HF transformer models x 5 folds
    -> metrics.compute_metrics() - per-fold test metrics
    -> save_predictions_and_metrics() - predictions/<model>/fold_N.csv,
       metrics/<model>/fold_N.json
    -> finalize() / metrics.average_metrics() - metrics/<model>/summary.json
```

Seed handling: `config_utils.set_seed(42)` is called before each
independent experiment cell and sets Python, NumPy, `torch`, and CUDA
seeds, with `torch.backends.cudnn.deterministic = True`.

Leakage guards observed in code: `load_folds()` explicitly checks that
`train`, `val`, and `test` id sets are pairwise disjoint per fold and
raises if not; the dataset's SHA-256 hash is checked against
`folds.json`'s recorded `data_hash` before use. Word2Vec vectors are
retrained per fold on train-only text (documented in the notebook as a
deliberate leakage-avoidance choice).

**INFERENCE / unverifiable:** the code that originally constructed
`folds.json` (including whether `group_id` was used for group-aware
splitting) is not present in this notebook, so the split's construction
cannot be independently verified from this repository alone.

## 3. Experiment Inventory (FACT, from saved notebook outputs)

| Experiment | Status | Evidence |
|---|---|---|
| Majority-class baseline | COMPLETED | Saved output: Macro-F1 0.0589 ± 0.0002 |
| TF-IDF + Logistic Regression | COMPLETED | Saved output: Macro-F1 0.5022 ± 0.0094 |
| TF-IDF + Linear SVM | COMPLETED | Saved output: Macro-F1 0.5106 ± 0.0164 |
| SVM (RBF) + Word2Vec | COMPLETED | Saved output: Macro-F1 0.3628 ± 0.0117 |
| Transformer smoke test (1 model x 1 fold) | FAILED | Saved cell output is a `RuntimeError: GPU tidak aktif` traceback — training never started |
| Full transformer fine-tuning (7 models x 5 folds) | NOT EXECUTED | Corresponding cell has `execution_count: None` and zero saved outputs |
| DAPT backbone config | PLANNED / PLACEHOLDER | YAML config contains an explicit `PLACEHOLDER` comment; no checkpoint referenced exists |
| DAPT+TAPT config | PLANNED / PLACEHOLDER | Same; references a non-existent checkpoint directory template |

**No transformer results currently exist in this repository.** Any
performance figures for transformer models appearing in notebook markdown
are pre-registered *expectations* to check for after a future run, not
recorded results.

Dataset statistics (FACT, from saved output): 4,628 rows, 11 labels,
class imbalance ratio (max/min) of 58.39, dataset/fold hash confirmed
matching (`ed154162554b68ba...`).

## 4. Missing Source Files (FACT — blocking)

Two modules are imported by the notebook but their source is never
written anywhere within it, unlike `config_utils.py`, `loader.py`, and
`trainer.py`, which the notebook generates and writes itself:

```
src/baselines/sklearn_baseline.py   (imported in cells running the baselines)
src/metrics.py                       (imported inside the generated trainer.py)
```

**INFERENCE:** based on the path convention used by every other generated
file, these most likely live at:
```
/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src/metrics.py
/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src/baselines/sklearn_baseline.py
```
This has not yet been confirmed against the actual Google Drive contents.

Without these two files, none of the notebook's experiment-running cells
can currently be executed from a clean environment.

## 5. Reproducibility Issues (FACT, prioritized)

1. **HIGH** — `sklearn_baseline.py` and `metrics.py` are missing from any
   version-controlled or notebook-embedded source (see section 4).
2. **HIGH** — All paths are hard-coded to
   `/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/...`,
   including inside the string literals that get written out as
   `trainer.py`. Nothing is currently parameterized.
3. **HIGH** — The dataset (`dib_labeled.csv`) and `folds.json` are not
   present in this repository, and the code that built `folds.json` is
   not visible anywhere in the notebook — only its *validation* is shown.
4. **MEDIUM** — The notebook's saved `execution_count` values are not
   monotonic with cell order, indicating cells were run out of sequence
   across sessions. A fresh top-to-bottom run is not guaranteed to
   reproduce the saved outputs without inspection.
5. **MEDIUM** — No dependency versions are pinned anywhere in the
   original notebook (e.g. `pip install -q gensim` with no version).
6. **LOW** — `DataLoader(shuffle=True)` relies on the global seed rather
   than an explicit `generator=`; likely fine given `num_workers=0` is
   implied, but not made explicit in the code.

## 6. Methodology Risks (FACT/INFERENCE, prioritized)

- **CRITICAL (forward-looking risk)** — Because no transformer results
  currently exist, there is a risk that a future run's numbers could be
  reported without being traceable back to an actually-saved,
  hash-verified `metrics/<model>/summary.json`. This has not happened;
  it is a risk given the current gap, not an observed incident.
- **HIGH (unverifiable)** — Whether `group_id` was used for group-aware
  fold splitting cannot be confirmed from this repository; only its
  presence and non-null validation is observed.
- **MEDIUM (observation, not a defect)** — Class imbalance is severe
  (58.39:1); `class_weight="balanced"` (sklearn) and inverse-frequency
  class weighting (transformer trainer) are used consistently across
  both baseline and transformer code paths.
- **MEDIUM (observation)** — The SVM+Word2Vec baseline, described in the
  notebook as the official comparison baseline from prior published work,
  scores lower (0.3628) than the TF-IDF baselines (0.50–0.51). This is a
  legitimate result to be addressed in the thesis narrative, not a bug.

## 7. Google Drive Dependencies (FACT)

Everything the notebook reads or writes lives under
`/content/drive/MyDrive/THESIS/`, including:
- `DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION/dib_labeled.csv`
- `DATASETS/PROCESSED DATASETS/COGNITIVE DISTORTION/folds.json`
- `MODELING/COGNITIVE DISTORTION/src/` (partially — see section 4)
- `MODELING/COGNITIVE DISTORTION/models/` (checkpoints, once training runs)
- `MODELING/COGNITIVE DISTORTION/results/` (predictions, metrics)

None of these have been downloaded or migrated into this repository as of
this phase.

## 8. Proposed Migration Architecture (RECOMMENDATION — not yet implemented)

The following are proposed directions, not decisions already made:

- Use `uv` (`pyproject.toml` + `uv.lock`) as the canonical dependency
  manager, rather than a manually maintained `requirements.txt`.
- Target Python 3.11 for the project. **This is a recommendation, not a
  constraint derived from the code** — the original notebook does not
  specify a Python version anywhere. 3.11 was chosen as a modern version
  broadly compatible with the observed dependencies (`torch`,
  `transformers`, `scikit-learn`, `gensim`, `pandas`, `pyyaml`), and
  should ideally be confirmed against the actual Colab runtime's Python
  version when next available.
- Keep the dataset, fold definitions, and model checkpoints outside Git
  entirely, accessed via a configurable external data root (mechanism
  not yet implemented).
- Recover `src/metrics.py` and `src/baselines/sklearn_baseline.py`
  verbatim from Google Drive before any code extraction or refactoring
  proceeds, per the "do not invent replacement implementations" rule.
- Extract `config_utils.py`, `loader.py`, and `trainer.py` out of the
  notebook's embedded string literals into real, version-controlled
  files with no logic changes, once the two missing files above are
  recovered and the extracted modules can actually be imported together.

No source recovery, extraction, dependency installation, or experiment
execution has occurred as of this document's creation. This document
will be revisited and updated as later migration phases complete.
