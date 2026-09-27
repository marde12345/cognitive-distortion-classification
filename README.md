# Cognitive Distortion Classification

**A research project that trains and compares language models to recognize
patterns of unhelpful thinking ("cognitive distortions") in Indonesian
text.**

This README has two equivalent halves: **English** and **Bahasa
Indonesia**. Pick whichever you're more comfortable with — both explain
the same project in the same order.

- [English](#english)
- [Bahasa Indonesia](#bahasa-indonesia)

---

# English

## 1. What is this project, in one minute?

Cognitive distortions are unhelpful thinking patterns studied in
psychology — for example, assuming the worst will happen ("Jumping to
Conclusions") or labeling yourself harshly after one mistake
("Labeling"). This project tries to automatically detect **11** such
categories (including "No Distortion") in a piece of Indonesian text,
using machine learning language models.

```
Input:                          Model:                              Output:
An Indonesian sentence   →      A language model fine-tuned to  →   The predicted
or short text                   recognize 11 categories of          category, plus
                                 cognitive distortion                evaluation metrics
```

**Why does this experiment exist?** To find out how well different
pretrained Indonesian/multilingual language models can be *fine-tuned*
(see glossary below) to do this classification task, and to compare them
fairly, honestly, and reproducibly — without exaggerating what the
results do or don't prove.

This is a **research/thesis project**. The live text-classification demo
included in this repository is explicitly **not** a clinical, diagnostic,
or medically validated tool. A prediction is a model's best guess, not a
mental-health assessment of anyone.

## 2. Plain-language glossary

You'll see these words throughout the project. Here's what each one
means, in one or two sentences:

| Term | Plain-language meaning |
|---|---|
| **Transformer** | A type of neural network architecture that is currently the standard way to build language-understanding models (e.g. BERT, RoBERTa). You don't need to know how it works internally to use this repository. |
| **Fine-tuning** | Taking a language model that already "understands" general language, and further training it on our specific task (classifying cognitive distortions) so it specializes in it. |
| **Epoch** | One full pass through the training data. Training usually repeats several epochs so the model can gradually improve. |
| **Fold** | One way of splitting the dataset into a training part and a held-out test part, so we can check how the model performs on data it has never seen. This project uses 5 different folds (fold_0 … fold_4) to get a more reliable picture than just one split would give. |
| **Seed** | A starting number for the random number generator. Two runs with the same data and the same seed produce the same result; two runs with different seeds may differ slightly purely due to randomness (e.g. random shuffling, random initialization). |
| **Macro-F1** | A single accuracy-like score that treats every one of the 11 categories as equally important, regardless of how common that category is in the data. It's explained further in Section 11. |
| **Accuracy** | The percentage of predictions the model got exactly right. |
| **Checkpoint** | A saved copy of a model's trained weights, so it can be reloaded later and used to make predictions without retraining. |
| **Inference** | Using an already-trained model to make a prediction on new text (as opposed to training it). |
| **MPS** | "Metal Performance Shaders" — Apple's technology that lets PyTorch use the GPU on Apple Silicon Macs (M1/M2/M3/etc.) to train models faster than on the CPU alone. |
| **Dashboard** | A local web page (in your browser) that shows the experiment's results, progress, and comparisons, without you needing to read raw JSON files. |

## 3. How the project works, step by step

```
Dataset (Indonesian sentences, each labeled with 1 of 11 categories)
   ↓
Split into 5 evaluation "folds" (train/validation/test)
   ↓
Fine-tune each language model on each fold
   ↓
Measure performance (Accuracy, Macro-F1, Weighted-F1) on held-out test data
   ↓
Aggregate and statistically compare the models across folds
   ↓
View everything in a local dashboard
   ↓
Try live predictions with one explicitly chosen trained model (demo)
```

Each step is a distinct, inspectable stage — nothing is hidden inside one
giant script. The sections below map each step to actual files and
commands in this repository.

## 4. Repository structure

```
.
├── configs/          Settings: which models to use, dataset columns, training hyperparameters
├── docs/             Written documentation, including a history of project "phases"
├── external/         Where the raw dataset lives on your machine (not stored in Git)
├── notebooks/        The original Google Colab notebook this project was migrated from
├── results/          Everything the experiments produce: metrics, logs, analysis, comparisons
├── scripts/          Commands you actually run (train, analyze, view dashboard, predict)
├── src/              The Python code that the scripts rely on
├── tests/            Automatic checks that verify the code still behaves correctly
├── pyproject.toml    The list of Python packages this project depends on
└── README.md         This file
```

| Folder | What it means for you |
|---|---|
| `configs/` | Small text files that describe *which* models to train and *how* (learning rate, batch size, number of epochs, etc.), without touching any code. |
| `docs/phases/` | A written history of how this project was built, phase by phase — useful if you want the full backstory, but not required to run anything. |
| `external/original-drive/` | Where the dataset and fold definitions are expected to live locally. This folder is **not** included in the Git repository (see Section 12). |
| `notebooks/` | The original research notebook (Google Colab), kept as a historical reference. |
| `results/` | Generated output: per-model metrics, statistical analysis, and model-comparison reports. Large raw outputs are not stored in Git; the smaller analysis summaries are. |
| `scripts/` | Small, runnable programs — this is where you'll spend most of your time as a user. |
| `src/` | The reusable code (dataset loading, training logic, dashboard logic, inference logic) that the scripts import. |
| `tests/` | Automated tests. Running them tells you whether your setup is healthy. |

## 5. Important files, grouped by purpose

### Training

- **`src/finetune/trainer.py`** — the core function that actually
  fine-tunes one model on one fold: loads the data, trains for a few
  epochs, evaluates it, and returns the results. This file is used by
  every training script below; it never runs on its own.
- **`scripts/run_baseline.py`** — trains a simple, fast, non-transformer
  baseline model (e.g. TF-IDF + Logistic Regression) so there's a simple
  reference point to compare transformer results against.

> **Note on `main`:** the scripts that automatically run a full
> multi-model transformer sweep (`run_transformer.py`,
> `run_transformer_sweep.py`) exist in this project's history but are
> **not currently part of the `main` branch**. On `main`, the transformer
> results already present under `results/` were produced this way, but
> re-running that full sweep requires those orchestration scripts, which
> live on a separate development branch. If you only have `main`, you can
> still run baselines (above) and everything described in Sections 8–11
> below using the results already committed to `results/`.

### Data

- **`src/loader.py`** — loads the dataset CSV and the fold definitions,
  and splits the data into train/validation/test for a given fold.
- **`src/config_utils.py`** — reads the `configs/*.yaml` files, resolves
  file paths, sets the random seed, and verifies the dataset/fold files
  match their expected checksums (see Section 17).

### Analysis

- **`scripts/analyze_transformer_sweep.py`** — reads the raw per-model,
  per-fold metrics and produces the aggregated statistics (mean, standard
  deviation, etc.) and a Friedman statistical test comparing the 7
  models. Writes its output under `results/analysis/transformer/`.
- **`scripts/plot_transformer_sweep.py`** — generates the plot images
  shown in the dashboard's analysis sections.
- **`scripts/build_model_comparison.py`** — reads the analysis output
  above (without recomputing anything) and produces a compact
  side-by-side model-comparison table and a plain-language researcher
  summary under `results/analysis/transformer/comparison/`.

### Dashboard

- **`scripts/run_dashboard.py`** — starts a local web server that serves
  the dashboard (see Section 8).
- **`src/dashboard/`** — the code behind the dashboard: reading result
  files safely (`data_loader.py`, `analysis_loader.py`), turning them
  into a display-ready shape (`state.py`), and the actual web page
  (`static/index.html`).

### Inference (the live demo)

- **`scripts/train_demo_checkpoint.py`** — trains and saves **one**
  specific, explicitly chosen model+fold combination as a reusable
  checkpoint, so it can answer predictions without retraining every time.
- **`scripts/run_inference.py`** — loads that saved checkpoint from the
  command line and prints a prediction for a piece of text you provide.
- **`src/dashboard/inference.py`** — the same prediction logic, wired
  into the dashboard's "Live Demo" section and its `/api/infer` endpoint.

### Configuration

- **`configs/base.yaml`** — shared settings: dataset column names, number
  of categories (11), max text length, default training seed, etc.
- **`configs/models/*.yaml`** — one file per model, naming which
  pretrained model to fine-tune and its training hyperparameters (see
  Section 13 for the full list).

## 6. Installation for beginners

You do **not** need to already know Python, Git, or machine learning to
follow these steps — just copy and run each command in a terminal.

### 6.1 Required software

| Requirement | Why | Notes |
|---|---|---|
| **Git** | To download ("clone") this repository | Required |
| **Python 3.11+** | The language this project is written in | Required (see `.python-version`) |
| **`uv`** | A fast Python package/environment manager this project uses instead of plain `pip` | Required — see below |
| **macOS with Apple Silicon (M-series chip)** | Only needed if you want training to use your Mac's GPU via MPS | Optional / environment-specific — everything also works on CPU, just slower |

`uv` handles creating an isolated Python environment (a private,
project-specific copy of Python and its packages, so this project's
dependencies never conflict with anything else on your computer) and
installing every dependency listed in `pyproject.toml`. If you don't have
`uv` yet, see <https://docs.astral.sh/uv/getting-started/installation/>.

### 6.2 Get the repository

```bash
git clone <this-repository-url>
cd Tesis-Mbak-Ai
```

("Cloning" just means downloading a copy of the project, together with
its full history, onto your computer.)

### 6.3 Install dependencies

```bash
uv sync
```

This one command reads `pyproject.toml` and `uv.lock`, creates a local
`.venv/` folder (the isolated environment mentioned above), and installs
every required package (PyTorch, Hugging Face Transformers, scikit-learn,
etc.) at the exact versions this project was built and tested with.

### 6.4 Verify the installation works

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

`uv run` means "run this command inside the project's isolated
environment." This command runs the project's automated tests. If you
see `OK` at the end, your setup is healthy.

## 7. Your First Run (5 minutes)

This is the shortest safe path to confirm everything works, without
starting any long training.

1. **Install dependencies** (if you haven't already):
   ```bash
   uv sync
   ```
2. **Run the automated tests** (takes a few seconds):
   ```bash
   uv run python -m unittest discover -s tests -p "test_*.py"
   ```
   Expect something like `Ran 57 tests ... OK`.
3. **Start the dashboard**:
   ```bash
   uv run python scripts/run_dashboard.py
   ```
4. **Open your browser** and go to: <http://127.0.0.1:8765>
5. **Confirm the page loads** and shows sections like "Sweep Overview"
   and "Model Comparison" with real numbers from the experiments already
   recorded in this repository.
6. Press `Ctrl+C` in the terminal to stop the dashboard when you're done.

You have not trained anything yet — you've only viewed results that were
already produced. Training is covered in Section 9.

## 8. Running the dashboard

**What it is:** a local web page that reads the JSON/CSV files under
`results/` and displays them as tables, charts, and plain-language
explanations, so you don't have to open raw data files yourself.

**Why it exists:** to make the experiment's progress and results
understandable without needing to read code or JSON.

**How to start it:**
```bash
uv run python scripts/run_dashboard.py
```
Then open <http://127.0.0.1:8765> in your browser.

**What "localhost" (127.0.0.1) means:** the dashboard is only reachable
from your own computer — no one else on your network or the internet can
open it, unless you deliberately start it with `--host 0.0.0.0` (which
allows other devices on the *same local network* to connect; it still
does not expose it to the public internet).

**Major sections you'll see:**

| Section | What it shows |
|---|---|
| Live Demo | Type in text and get a live prediction, once a demo checkpoint has been trained (Section 16). |
| Sweep Overview | How many training runs are done, failed, or still pending. |
| Model × Fold Matrix | A grid showing the status of every model/fold combination. |
| Model Comparison | Macro-F1/Accuracy/Weighted-F1 for each of the 7 models, side by side. |
| Cross-Fold Stability | How consistent each model's score is across the 5 folds. |
| Anomalies | Any run that behaved very differently from the rest — shown honestly, never hidden. |
| Statistical Analysis | The result of a formal statistical test comparing the models (Section 11). |
| Glossary | A short in-dashboard explanation of the same kind of terms as Section 2 above. |

**How to stop it:** press `Ctrl+C` in the terminal where it's running.

## 9. Running experiments

There are four distinct "levels" of experiment in this project, from
smallest/fastest to largest/slowest. It's important to know which one a
command belongs to before running it.

| Level | Name | Roughly how long | Available on `main`? |
|---|---|---|---|
| A | Single baseline | Seconds to a couple of minutes | ✅ Yes |
| B | Single transformer fine-tuning | Minutes | Logic exists, exposed via the demo-checkpoint script (9.2) |
| C | Full transformer sweep (7 models × 5 folds) | Hours | ⚠️ Orchestration script not on `main` (see below) — results already exist |
| D | Multi-seed experiment (7 models × 5 folds × 3 seeds) | Much longer than C | 🚧 Not on `main` — in development on a separate branch |

### 9.A A single baseline

The simplest possible experiment — a quick, non-transformer model, useful
as a sanity check or a simple reference point:

```bash
uv run python scripts/run_baseline.py tfidf_lr
```
Other options: `majority_class`, `tfidf_svm`, `svm_word2vec`.

### 9.B Fine-tuning one transformer model

`src/finetune/trainer.py` contains the reusable fine-tuning logic used
throughout this project. On `main`, this is used via the demo-checkpoint
script (Section 16) rather than a general-purpose single-run script.

### 9.C The full transformer sweep

The 7 transformer models listed in Section 10, each evaluated on the same
5 folds, is **35 training runs** in total (7 × 5). This is how the
results already shown in the dashboard's "Model Comparison" section were
produced. ⚠️ **Long-running**: on the original hardware used for this
project (Apple Silicon Mac, MPS), the full sweep took several hours.

> As noted in Section 5, the orchestration script that runs this sweep
> automatically currently lives on a separate development branch, not on
> `main`. The results of that sweep are already available on `main`
> under `results/`, and can be explored via the dashboard and the
> analysis scripts without re-running any training. If you only have
> `main`, there is no single command here that will "just run the whole
> sweep" — and that's intentional, so nobody accidentally starts an
> hours-long job by mistake.

### 9.D Multi-seed experiment

Running the same model/fold combination with different random seeds
(Section 4) helps distinguish "this model is genuinely more consistent"
from "this particular run got lucky."

The planned protocol for this experiment is:

- **7 models** (Section 10)
- **5 folds** (fold_0 … fold_4)
- **3 seeds**: `42`, `43`, `44`
- 7 × 5 × 3 = **105 planned runs**

**Important:** 105 is the number of planned *training runs* (combinations
of model + fold + seed) — it is **not** 105 different models. The same 7
models are simply each trained 15 times (5 folds × 3 seeds) instead of 5
times, to check how stable their results are.

**On `main`, this protocol and its orchestration scripts are not
present** — this experiment is being developed and, as of this writing,
executed on a separate branch, and is not something you can start from a
plain `main` checkout. This README will not tell you to run it, because
there is nothing on `main` to run yet. Once it is merged, this section
will be updated with the real command and real results.

> ⚠️ **Do not attempt to run a full sweep or multi-seed experiment
> "just to see what happens."** These are multi-hour (Level C) to
> multi-day (Level D) training jobs. Only start Level C/D deliberately,
> when you specifically intend to (re)produce the full experiment, and
> preferably when you can leave your computer running undisturbed.

## 10. Long-running experiments: what to expect

If you do run a long training command (Sections 9.1–9.3):

- **Where logs go:** structured JSON execution logs are written under
  `results/logs/<model>/<fold>.json`, recording status (`running`,
  `completed`, or `failed`), timing, and per-epoch training history.
- **How to check progress:** while a run is active, its log file's
  `status` field will say `"running"`; once finished, it becomes
  `"completed"` or `"failed"`.
- **What indicates completion:** a `"status": "completed"` log entry
  together with a matching file under `results/metrics/<model>/<fold>.json`.
- **If a run fails:** the log records `"status": "failed"` and an error
  message — it is never silently discarded, so you can see exactly what
  went wrong.
- **Resuming:** re-running the same baseline command is safe; it does not
  corrupt or need to repeat work that already finished successfully.

## 11. Understanding the results

Three metrics appear throughout this project:

- **Accuracy** — the percentage of predictions that were exactly correct.
  Simple, but can be misleading if one category is much more common than
  the others.
- **Weighted-F1** — like Macro-F1 below, but categories with more
  examples count for more in the final average.
- **Macro-F1** — calculates an F1 score *separately for each of the 11
  categories*, then averages those 11 scores **equally**, regardless of
  how many examples each category had. This matters here because some
  distortion categories are much rarer than others in the dataset; Macro-
  F1 prevents the score from being dominated by just the most common
  category ("No Distortion").

**How to read a comparison table, correctly:**

- *Observed result*: "Model A had a higher observed mean Macro-F1 than
  Model B in this experiment" — this is simply reading a number off a
  table.
- *Statistical evidence*: a formal test (this project uses a Friedman
  test — see `results/analysis/transformer/statistical_tests.json`) can
  tell you whether the differences across all 7 models are unlikely to
  be pure chance, across the 5 folds used. It does **not** tell you which
  *specific pair* of models differs, and 5 folds is a small sample.
- *Researcher decision*: choosing to actually use one model over another
  for some downstream purpose is a judgment call a person makes, informed
  by the evidence above — this project does not automatically declare a
  "winner," and neither should you when reading its results.

See `results/analysis/transformer/comparison/researcher_summary.md` for
a full plain-language write-up of what the current results do and don't
show.

## 12. Dataset and folds

- The raw dataset (Indonesian sentences, each labeled with one of the 11
  categories) and its fold definitions are expected at
  `external/original-drive/DATASETS/...` on your machine. **This data is
  not included in the Git repository** — you need to obtain and place it
  there yourself before running any training or baseline script.
- A **fold** is one way of splitting the dataset into a training group, a
  validation group, and a held-out test group. This project defines 5
  folds so every model can be evaluated on 5 different train/test splits
  instead of just one, giving a more reliable picture of performance.
- **Every model uses the exact same 5 folds.** This is essential for a
  fair comparison — if models were evaluated on different splits of the
  data, differences in their scores could come from the data split
  rather than the model itself.
- **Do not casually modify the dataset or the fold file.** Doing so
  breaks reproducibility and invalidates comparisons with every result
  already recorded in `results/`. See Section 17 for how this project
  guards against that.

## 13. Models used in this project

| Model config name | Underlying pretrained model | Plain-language description |
|---|---|---|
| `indobert_15g` | `cahya/bert-base-indonesian-1.5G` | An Indonesian-language BERT model. |
| `indobert_base_p1` | `indobenchmark/indobert-base-p1` | Another Indonesian BERT model, from a different research group. |
| `indobertweet` | `indolem/indobertweet-base-uncased` | A BERT model pretrained specifically on Indonesian social-media/Twitter-style text. |
| `indoroberta_15g` | `cahya/roberta-base-indonesian-1.5G` | An Indonesian RoBERTa model (a variant of the BERT-style architecture). |
| `mbert` | `bert-base-multilingual-cased` | A multilingual BERT model trained on 100+ languages, including Indonesian. |
| `nusabert` | `LazarusNLP/NusaBERT-base` | A BERT model trained with a focus on Indonesian and regional Indonesian languages. |
| `xlmr` | `xlm-roberta-base` | A multilingual RoBERTa-style model, also trained on many languages. |

These 7 are all fine-tuned and evaluated the same way, on the same data
and folds, so their results can be fairly compared. This project does not
claim any one of them is universally "the best" — see Section 11.

## 14. Results and generated files

Running experiments produces files under `results/` and `models/`. Here's
what lives where, and which of them are actually stored in Git versus
generated locally on your own machine:

| Location | What's in it | Committed to Git? |
|---|---|---|
| `results/metrics/<model>/foldN.json` | The evaluation numbers (Accuracy, Macro-F1, Weighted-F1) for one model on one fold. | Baseline models' metrics are committed; not every model's metrics are. |
| `results/logs/<model>/foldN.json` | A record of one training run: status (`running`/`completed`/`failed`), timing, per-epoch history. | Only created when you actually run training locally. |
| `results/predictions/` | The model's raw predictions on the test data. | **No** — listed in `.gitignore`, since these can be regenerated and are not needed for comparison. |
| `results/analysis/transformer/` | The aggregated statistics, plots, and comparison summaries covered in Sections 11 and 16. | **Yes** — these are the small, important summaries this project keeps in Git. |
| `models/` | Saved model weights ("checkpoints"), including the live-demo checkpoint (Section 16). | **No** — listed in `.gitignore`. Model weight files are large binary files (`*.pt`, `*.bin`, etc.) and are deliberately never committed. |

This is why, if you look at this repository on GitHub, you will **not**
find a `model.pt` file anywhere — checkpoints only exist locally, on
whichever machine actually ran the training, under the git-ignored
`models/` folder.

## 15. Phase history (why so many documents?)

This project was built incrementally, in numbered "phases," each with its
own short report under `docs/phases/`. You do not need to read them to
use the repository, but they explain *why* things are structured the way
they are:

| Phase | What it did |
|---|---|
| 6.3 | Planned the final repository/dashboard architecture before any code was written. |
| 7 | Audited whether the repository was actually ready to run locally. |
| 8 | Adapted the original Colab code to run outside Colab, locally. |
| 9 / 9.1 | Ran and re-verified the simple baseline models. |
| 10A / 10B | General repository cleanup and a documented commit of that cleanup. |
| 15 (dashboard) | Built the first version of the local results dashboard. |
| 16 | Added the statistical analysis (aggregate metrics, Friedman test, anomaly handling). |
| 17 | Enhanced the dashboard with more sections and LAN-access support. |
| 18 | Added the live inference demo (one explicitly chosen trained checkpoint). |
| 19 | Built the descriptive model-comparison summary discussed in Section 11. |

(A multi-seed experiment — repeating the sweep with several random seeds
for more robust comparisons — was prepared on a separate development
branch after Phase 19, but is not part of `main` as of this README.)

## 16. How the live inference demo works

1. `scripts/train_demo_checkpoint.py` trains **one specific, explicitly
   named** model+fold combination (by default `indobert_base_p1` /
   `fold_0`) and saves its weights to `models/demo/<model>/<fold>/`. It
   refuses to silently overwrite an existing checkpoint.
2. Once that checkpoint exists, you can get a prediction two ways:
   - **From the command line:**
     ```bash
     uv run python scripts/run_inference.py \
       --checkpoint models/demo/indobert_base_p1/fold_0 \
       --text "some Indonesian sentence here"
     ```
   - **From the dashboard's "Live Demo" section**, by typing text into
     the box in your browser.
3. This checkpoint was **explicitly chosen for demonstration purposes**
   — it is one specific trained run, not automatically "the best model"
   selected by any algorithm (see Section 11). A demo checkpoint is a
   demonstration artifact, not automatically "the final research model."
4. As stated in Section 1: this demo is a research artifact, not a
   clinical or diagnostic tool.

## 17. Reproducibility

To reproduce this project's results, you need the same:

- **Dataset** — the exact same CSV file (see Section 12).
- **Fold definitions** — the exact same train/validation/test split file.
- **Model configuration** — the `configs/models/*.yaml` files, unchanged.
- **Random seed** — fixed at `42` for the sweep described in Section 9.C.
- **Scripts** — the same training/evaluation code.
- **Environment** — the same Python and package versions, which `uv
  sync` gives you automatically from `uv.lock`.

**Why this matters:** if any of the above changes silently, a "better" or
"worse" result might just reflect a different dataset or setup, not a
genuinely better model.

**Checksums:** this project computes a SHA-256 checksum (a short digital
fingerprint) of the dataset and fold files, and compares it to a known
expected value before running any experiment. If the files ever change
unexpectedly, the checksum will not match, and this is used as an early
warning sign that something about the data has changed.

## 18. Troubleshooting

| Problem | Likely cause | Simple solution |
|---|---|---|
| `uv: command not found` | `uv` isn't installed yet | Install it from <https://docs.astral.sh/uv/getting-started/installation/>, then retry. |
| `ModuleNotFoundError` for a package like `torch` or `transformers` | Dependencies weren't installed, or you ran plain `python` instead of `uv run python` | Run `uv sync` first, and always prefix commands with `uv run`. |
| Dataset/fold file not found error | The dataset hasn't been placed under `external/original-drive/DATASETS/...` on your machine | Obtain the dataset and place it at the expected path (Section 12) before running training/baseline scripts. |
| A Hugging Face model seems to hang or fail on first use | The pretrained model weights need to download from the internet the first time | Check your internet connection; the download only happens once and is then cached locally. |
| Training is very slow | You're running on CPU only, or MPS isn't available | This is expected on non-Apple-Silicon machines; training will still work, just more slowly. |
| `Address already in use` when starting the dashboard | Another process (maybe a previous dashboard) is already using port 8765 | Stop the other process, or start this one on a different port: `uv run python scripts/run_dashboard.py --port 8800`. |
| A long-running script seems to have stopped | The process may have crashed or been interrupted | Check `results/logs/<model>/<fold>.json` for a `"failed"` status and error message; safe to re-run. |
| Dashboard loads but shows no results | You're running it from a different results directory than expected | Make sure you run it from the repository root, or pass `--results-dir <path>` explicitly. |
| I don't see any `model.pt` file in the repository | Model weight files are deliberately not stored in Git (see Section 14) | This is expected — checkpoints are only created locally when you run training on your own machine. |
| Some files under `results/` seem to be "missing" from Git | Some result types (predictions, model weights) are intentionally listed in `.gitignore` | See Section 14 for exactly which result folders are committed and which are generated locally only. |

## 19. FAQ

**Do I need to understand machine learning to run this project?**
No — Sections 6–8 only require copying commands into a terminal.
Understanding the *results* more deeply benefits from Section 11, but
isn't required to get the dashboard running.

**Can I just run everything at once?**
There is deliberately no single "run everything" command. Baselines
(Level A) are quick and safe to run anytime. The full sweep (Level C) and
multi-seed experiment (Level D) described in Section 9 are multi-hour (or
longer) jobs — you should only start those on purpose, not by accident.

**Why don't I see `model.pt` anywhere in the repository?**
Trained model weight files are intentionally excluded from Git (see
Section 14) because they are large binary files. They only exist locally
on whichever machine actually ran the training.

**Why are some results ignored by Git?**
Files under `results/predictions/` and everything under `models/` are
listed in `.gitignore` on purpose — they can be regenerated from a
training run and don't need to be stored permanently in the repository's
history. See Section 14 for the full breakdown of what is and isn't
committed.

**What is a transformer?**
See the glossary in Section 2 — briefly, it's the type of neural network
architecture used for all 7 language models in this project.

**What is a fold?**
A way of splitting the dataset so the model is tested on data it never
trained on. See Sections 2 and 12.

**Why are there multiple seeds (in the broader project)?**
To check whether a model's performance is consistent or just due to
random luck in one particular run. See Section 9.4 for what currently
exists on `main`.

**Where are the results?**
Under `results/` — see Section 4, and view them more easily through the
dashboard (Section 8).

**Where are the trained models?**
Trained model weights ("checkpoints") are not stored in Git (they're
large binary files); only the demo checkpoint described in Section 16 is
created locally when you run `train_demo_checkpoint.py`.

**How do I know whether an experiment finished?**
Check the relevant file under `results/logs/` for `"status":
"completed"` (Section 10), or look at the dashboard's overview section.

**Can I run this on a normal laptop?**
Yes. Baselines (Section 9.1) run quickly on any laptop. Transformer
fine-tuning is heavier and benefits from an Apple Silicon Mac (MPS) or a
machine with a GPU, but will still run on CPU alone — just more slowly.

**Why does training take a long time?**
Fine-tuning a language model involves many repeated passes over the
dataset (epochs), and doing this for 7 models × 5 folds adds up. See
Section 9.3.

**What should I do before changing the dataset?**
Don't, unless you specifically intend to start a new, separate
experiment — changing it invalidates the checksum-verified comparability
of every existing result in `results/`. See Section 17.

## Researcher quick reference (English)

```bash
# Install dependencies
uv sync

# Run the automated test suite
uv run python -m unittest discover -s tests -p "test_*.py"

# Start the dashboard (http://127.0.0.1:8765)
uv run python scripts/run_dashboard.py

# Run a single baseline experiment (fast)
uv run python scripts/run_baseline.py tfidf_lr

# ⚠️ Long-running: train and save the one explicitly chosen demo checkpoint
uv run python scripts/train_demo_checkpoint.py --model indobert_base_p1 --fold fold_0

# Run inference against a saved checkpoint
uv run python scripts/run_inference.py --checkpoint models/demo/indobert_base_p1/fold_0 --text "your text here"

# Regenerate the descriptive model-comparison summary from existing results
uv run python scripts/build_model_comparison.py
```

---

# Klasifikasi Distorsi Kognitif

# Bahasa Indonesia

## 1. Apa proyek ini, dalam satu menit?

Distorsi kognitif adalah pola pikir yang tidak membantu, yang dipelajari
dalam psikologi — misalnya selalu mengira hal terburuk akan terjadi
("Jumping to Conclusions"), atau memberi label buruk pada diri sendiri
hanya karena satu kesalahan ("Labeling"). Proyek ini mencoba mendeteksi
secara otomatis **11** kategori semacam itu (termasuk kategori "Tidak Ada
Distorsi") dari sebuah teks berbahasa Indonesia, menggunakan model bahasa
berbasis machine learning.

```
Input:                              Model:                                  Output:
Sebuah kalimat/teks     →           Model bahasa yang telah dilatih    →    Kategori yang
berbahasa Indonesia                 ulang (fine-tuned) untuk mengenali      diprediksi, beserta
                                     11 kategori distorsi kognitif           metrik evaluasinya
```

**Kenapa eksperimen ini dibuat?** Untuk mengetahui seberapa baik
berbagai model bahasa (pretrained) Indonesia/multibahasa dapat
di-*fine-tune* (lihat istilah di bawah) untuk tugas klasifikasi ini, dan
untuk membandingkannya secara adil, jujur, dan dapat direproduksi — tanpa
melebih-lebihkan apa yang sebenarnya bisa atau tidak bisa dibuktikan oleh
hasilnya.

Ini adalah **proyek riset/skripsi**. Demo klasifikasi teks langsung
("live demo") dalam repositori ini secara eksplisit **bukan** alat
klinis, diagnostik, atau tervalidasi secara medis. Prediksi model hanyalah
tebakan terbaik model, bukan penilaian kesehatan mental terhadap siapa
pun.

## 2. Daftar istilah dalam bahasa sederhana

Istilah-istilah berikut akan sering muncul. Berikut penjelasannya dalam
satu-dua kalimat:

| Istilah | Penjelasan sederhana |
|---|---|
| **Transformer** | Sebuah jenis arsitektur neural network yang saat ini menjadi standar untuk membangun model pemahaman bahasa (contoh: BERT, RoBERTa). Anda tidak perlu memahami cara kerjanya secara internal untuk memakai repositori ini. |
| **Fine-tuning** | Mengambil model bahasa yang sudah "memahami" bahasa secara umum, lalu melatihnya lebih lanjut khusus untuk tugas kita (mengklasifikasikan distorsi kognitif) agar model tersebut menjadi ahli di tugas itu. |
| **Epoch** | Satu putaran penuh melewati seluruh data latih. Pelatihan biasanya mengulang beberapa epoch agar model bisa membaik secara bertahap. |
| **Fold** | Salah satu cara membagi dataset menjadi bagian latih dan bagian uji yang terpisah, agar kita bisa mengecek performa model pada data yang belum pernah dilihatnya. Proyek ini memakai 5 fold berbeda (fold_0 … fold_4) agar gambarannya lebih dapat dipercaya dibanding hanya satu pembagian saja. |
| **Seed** | Angka awal untuk generator bilangan acak. Dua kali menjalankan proses dengan data yang sama dan seed yang sama akan menghasilkan hasil yang sama persis; seed yang berbeda bisa menghasilkan sedikit perbedaan murni karena faktor acak (misalnya urutan data yang diacak, atau inisialisasi awal model). |
| **Macro-F1** | Satu skor mirip akurasi yang memperlakukan setiap dari 11 kategori sebagai sama pentingnya, terlepas dari seberapa sering kategori itu muncul di data. Dijelaskan lebih lanjut di Bagian 11. |
| **Accuracy (akurasi)** | Persentase prediksi yang benar-benar tepat. |
| **Checkpoint** | Salinan tersimpan dari bobot (weights) model yang sudah dilatih, sehingga bisa dimuat ulang kapan saja untuk membuat prediksi tanpa perlu melatih ulang dari awal. |
| **Inference** | Menggunakan model yang sudah dilatih untuk membuat prediksi pada teks baru (berbeda dengan melatihnya). |
| **MPS** | "Metal Performance Shaders" — teknologi dari Apple yang memungkinkan PyTorch memakai GPU pada Mac ber-chip Apple Silicon (M1/M2/M3/dst.) agar pelatihan model lebih cepat dibanding hanya memakai CPU. |
| **Dashboard** | Sebuah halaman web lokal (dibuka di browser) yang menampilkan hasil, progres, dan perbandingan eksperimen, tanpa Anda perlu membaca file JSON mentah. |

## 3. Cara kerja proyek, langkah demi langkah

```
Dataset (kalimat berbahasa Indonesia, masing-masing diberi label salah satu dari 11 kategori)
   ↓
Dibagi menjadi 5 "fold" evaluasi (latih/validasi/uji)
   ↓
Setiap model bahasa di-fine-tune pada setiap fold
   ↓
Performa diukur (Accuracy, Macro-F1, Weighted-F1) pada data uji yang belum pernah dilihat
   ↓
Hasil digabungkan dan dibandingkan secara statistik antar model
   ↓
Semuanya bisa dilihat lewat dashboard lokal
   ↓
Coba prediksi langsung dengan satu model terlatih yang dipilih secara eksplisit (demo)
```

Setiap langkah adalah tahap yang terpisah dan bisa diperiksa satu per
satu — tidak ada yang disembunyikan dalam satu skrip raksasa. Bagian-
bagian di bawah ini menghubungkan setiap langkah dengan file dan
perintah nyata di repositori ini.

## 4. Struktur repositori

```
.
├── configs/          Pengaturan: model apa saja yang dipakai, kolom dataset, hyperparameter pelatihan
├── docs/             Dokumentasi tertulis, termasuk riwayat "fase" proyek
├── external/         Tempat dataset mentah berada di komputer Anda (tidak disimpan di Git)
├── notebooks/        Notebook Google Colab asli, cikal bakal proyek ini
├── results/          Segala hasil eksperimen: metrik, log, analisis, perbandingan
├── scripts/          Perintah yang benar-benar Anda jalankan (latih, analisis, buka dashboard, prediksi)
├── src/              Kode Python yang dipakai oleh skrip-skrip tersebut
├── tests/            Pemeriksaan otomatis yang memastikan kode masih berjalan dengan benar
├── pyproject.toml    Daftar paket Python yang dibutuhkan proyek ini
└── README.md         File ini
```

| Folder | Artinya untuk Anda |
|---|---|
| `configs/` | File teks kecil yang menjelaskan model *apa* yang dilatih dan *bagaimana* (learning rate, batch size, jumlah epoch, dll.), tanpa perlu menyentuh kode. |
| `docs/phases/` | Riwayat tertulis bagaimana proyek ini dibangun, fase demi fase — berguna jika Anda ingin tahu latar belakang lengkapnya, tapi tidak wajib dibaca untuk menjalankan proyek. |
| `external/original-drive/` | Tempat dataset dan definisi fold seharusnya berada di komputer Anda. Folder ini **tidak** disertakan dalam repositori Git (lihat Bagian 12). |
| `notebooks/` | Notebook riset asli (Google Colab), disimpan sebagai referensi historis. |
| `results/` | Keluaran yang dihasilkan: metrik per model, analisis statistik, dan laporan perbandingan model. Keluaran mentah yang besar tidak disimpan di Git; ringkasan analisis yang lebih kecil disimpan. |
| `scripts/` | Program-program kecil yang bisa dijalankan — di sinilah Anda akan paling sering berinteraksi sebagai pengguna. |
| `src/` | Kode yang dipakai berulang (memuat dataset, logika pelatihan, logika dashboard, logika inference) yang di-import oleh skrip-skrip di atas. |
| `tests/` | Tes otomatis. Menjalankannya memberi tahu Anda apakah instalasi Anda sudah sehat. |

## 5. File-file penting, dikelompokkan berdasarkan fungsinya

### Pelatihan (Training)

- **`src/finetune/trainer.py`** — fungsi inti yang benar-benar
  melakukan fine-tuning satu model pada satu fold: memuat data, melatih
  selama beberapa epoch, mengevaluasinya, lalu mengembalikan hasilnya.
  File ini dipakai oleh semua skrip pelatihan di bawah; tidak pernah
  dijalankan sendiri.
- **`scripts/run_baseline.py`** — melatih model dasar (baseline) yang
  sederhana dan cepat, bukan transformer (misalnya TF-IDF + Logistic
  Regression), agar ada titik pembanding sederhana terhadap hasil
  transformer.

> **Catatan tentang `main`:** skrip yang menjalankan seluruh sweep
> (rangkaian eksperimen) multi-model transformer secara otomatis
> (`run_transformer.py`, `run_transformer_sweep.py`) memang ada dalam
> riwayat proyek ini, tetapi **belum menjadi bagian dari branch `main`**.
> Di `main`, hasil transformer yang sudah ada di `results/` diproduksi
> dengan cara ini, tetapi untuk menjalankan ulang seluruh sweep tersebut
> dibutuhkan skrip orkestrasi itu, yang saat ini berada di branch
> pengembangan terpisah. Jika Anda hanya memiliki `main`, Anda tetap bisa
> menjalankan baseline (di atas) dan semua hal di Bagian 8–11 di bawah
> menggunakan hasil yang sudah tersimpan di `results/`.

### Data

- **`src/loader.py`** — memuat CSV dataset dan definisi fold, lalu
  membagi data menjadi latih/validasi/uji untuk satu fold tertentu.
- **`src/config_utils.py`** — membaca file `configs/*.yaml`, menentukan
  lokasi file, mengatur random seed, dan memverifikasi bahwa file
  dataset/fold cocok dengan checksum yang diharapkan (lihat Bagian 16).

### Analisis

- **`scripts/analyze_transformer_sweep.py`** — membaca metrik mentah
  per model per fold, lalu menghasilkan statistik gabungan (rata-rata,
  standar deviasi, dll.) dan uji statistik Friedman yang membandingkan
  ke-7 model. Hasilnya ditulis ke `results/analysis/transformer/`.
- **`scripts/plot_transformer_sweep.py`** — menghasilkan gambar-gambar
  grafik yang ditampilkan di bagian analisis dashboard.
- **`scripts/build_model_comparison.py`** — membaca hasil analisis di
  atas (tanpa menghitung ulang apa pun) dan menghasilkan tabel
  perbandingan model yang ringkas serta ringkasan berbahasa sederhana
  untuk peneliti, di `results/analysis/transformer/comparison/`.

### Dashboard

- **`scripts/run_dashboard.py`** — menjalankan server web lokal yang
  menyajikan dashboard (lihat Bagian 8).
- **`src/dashboard/`** — kode di balik dashboard: membaca file hasil
  secara aman (`data_loader.py`, `analysis_loader.py`), mengubahnya
  menjadi bentuk siap tampil (`state.py`), dan halaman webnya sendiri
  (`static/index.html`).

### Inference (demo langsung)

- **`scripts/train_demo_checkpoint.py`** — melatih dan menyimpan **satu**
  kombinasi model+fold yang dipilih secara eksplisit sebagai checkpoint
  yang bisa dipakai ulang, agar bisa menjawab prediksi tanpa perlu
  dilatih ulang setiap kali.
- **`scripts/run_inference.py`** — memuat checkpoint tersimpan itu dari
  command line dan mencetak prediksi untuk teks yang Anda berikan.
- **`src/dashboard/inference.py`** — logika prediksi yang sama,
  terhubung ke bagian "Live Demo" dashboard dan endpoint `/api/infer`.

### Konfigurasi

- **`configs/base.yaml`** — pengaturan bersama: nama kolom dataset,
  jumlah kategori (11), panjang teks maksimum, seed pelatihan default,
  dst.
- **`configs/models/*.yaml`** — satu file per model, menentukan model
  pretrained mana yang di-fine-tune beserta hyperparameter pelatihannya
  (lihat daftar lengkap di Bagian 13).

## 6. Instalasi untuk pemula

Anda **tidak perlu** sudah menguasai Python, Git, atau machine learning
untuk mengikuti langkah-langkah ini — cukup salin dan jalankan setiap
perintah di terminal.

### 6.1 Perangkat lunak yang dibutuhkan

| Kebutuhan | Kenapa | Catatan |
|---|---|---|
| **Git** | Untuk mengunduh ("clone") repositori ini | Wajib |
| **Python 3.11+** | Bahasa pemrograman yang dipakai proyek ini | Wajib (lihat `.python-version`) |
| **`uv`** | Pengelola paket/environment Python yang cepat, dipakai proyek ini sebagai pengganti `pip` biasa | Wajib — lihat di bawah |
| **macOS dengan chip Apple Silicon (seri M)** | Hanya dibutuhkan jika ingin pelatihan memakai GPU Mac Anda lewat MPS | Opsional / tergantung lingkungan — semuanya tetap berjalan di CPU, hanya lebih lambat |

`uv` menangani pembuatan environment Python yang terisolasi (salinan
Python dan paket-paketnya yang khusus untuk proyek ini, sehingga
dependensi proyek ini tidak bentrok dengan apa pun di komputer Anda) dan
menginstal semua dependensi yang tercantum di `pyproject.toml`. Jika
belum punya `uv`, lihat
<https://docs.astral.sh/uv/getting-started/installation/>.

### 6.2 Mengunduh repositori

```bash
git clone <this-repository-url>
cd Tesis-Mbak-Ai
```

("Clone" artinya mengunduh salinan proyek beserta seluruh riwayatnya ke
komputer Anda.)

### 6.3 Instal dependensi

```bash
uv sync
```

Satu perintah ini membaca `pyproject.toml` dan `uv.lock`, membuat folder
lokal `.venv/` (environment terisolasi yang disebut di atas), dan
menginstal semua paket yang dibutuhkan (PyTorch, Hugging Face
Transformers, scikit-learn, dll.) pada versi persis yang dipakai saat
proyek ini dibangun dan diuji.

### 6.4 Memastikan instalasi berhasil

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

`uv run` artinya "jalankan perintah ini di dalam environment terisolasi
proyek." Perintah ini menjalankan tes otomatis proyek. Jika muncul `OK`
di akhir, instalasi Anda sudah sehat.

## 7. Percobaan Pertama Anda (5 menit)

Ini adalah jalur teraman dan tercepat untuk memastikan semuanya berjalan,
tanpa memulai pelatihan yang memakan waktu lama.

1. **Instal dependensi** (jika belum):
   ```bash
   uv sync
   ```
2. **Jalankan tes otomatis** (hanya beberapa detik):
   ```bash
   uv run python -m unittest discover -s tests -p "test_*.py"
   ```
   Harapannya seperti `Ran 57 tests ... OK`.
3. **Jalankan dashboard**:
   ```bash
   uv run python scripts/run_dashboard.py
   ```
4. **Buka browser** dan kunjungi: <http://127.0.0.1:8765>
5. **Pastikan halamannya terbuka** dan menampilkan bagian seperti "Sweep
   Overview" dan "Model Comparison" dengan angka-angka nyata dari
   eksperimen yang sudah tercatat di repositori ini.
6. Tekan `Ctrl+C` di terminal untuk menghentikan dashboard bila sudah
   selesai.

Anda belum melatih apa pun — Anda baru saja melihat hasil yang sudah ada
sebelumnya. Cara menjalankan pelatihan dibahas di Bagian 9.

## 8. Menjalankan dashboard

**Apa itu:** sebuah halaman web lokal yang membaca file JSON/CSV di
`results/` dan menampilkannya sebagai tabel, grafik, dan penjelasan
berbahasa sederhana, sehingga Anda tidak perlu membuka file data mentah
sendiri.

**Kenapa ada:** agar progres dan hasil eksperimen mudah dipahami tanpa
perlu membaca kode atau JSON.

**Cara menjalankannya:**
```bash
uv run python scripts/run_dashboard.py
```
Lalu buka <http://127.0.0.1:8765> di browser Anda.

**Arti "localhost" (127.0.0.1):** dashboard hanya bisa diakses dari
komputer Anda sendiri — tidak ada orang lain di jaringan Anda atau di
internet yang bisa membukanya, kecuali Anda sengaja menjalankannya dengan
`--host 0.0.0.0` (yang memungkinkan perangkat lain di *jaringan lokal
yang sama* untuk terhubung; ini tetap tidak mengekspos dashboard ke
internet publik).

**Bagian-bagian utama yang akan Anda lihat:**

| Bagian | Isinya |
|---|---|
| Live Demo | Ketik teks dan dapatkan prediksi langsung, setelah checkpoint demo dilatih (Bagian 16). |
| Sweep Overview | Berapa banyak proses pelatihan yang selesai, gagal, atau masih tertunda. |
| Model × Fold Matrix | Tabel yang menampilkan status setiap kombinasi model/fold. |
| Model Comparison | Macro-F1/Accuracy/Weighted-F1 untuk ketujuh model, berdampingan. |
| Cross-Fold Stability | Seberapa konsisten skor tiap model di kelima fold. |
| Anomalies | Proses pelatihan mana pun yang berperilaku sangat berbeda dari yang lain — ditampilkan apa adanya, tidak disembunyikan. |
| Statistical Analysis | Hasil uji statistik formal yang membandingkan model-model (Bagian 11). |
| Glossary | Penjelasan singkat di dalam dashboard untuk istilah-istilah seperti di Bagian 2. |

**Cara menghentikannya:** tekan `Ctrl+C` di terminal tempat dashboard
berjalan.

## 9. Menjalankan eksperimen

Ada empat "level" eksperimen yang berbeda dalam proyek ini, dari yang
paling kecil/cepat sampai yang paling besar/lama. Penting untuk tahu
level mana yang sebuah perintah masuki sebelum menjalankannya.

| Level | Nama | Perkiraan durasi | Tersedia di `main`? |
|---|---|---|---|
| A | Baseline tunggal | Beberapa detik sampai beberapa menit | ✅ Ya |
| B | Fine-tuning satu model transformer | Beberapa menit | Logikanya ada, diakses lewat skrip checkpoint demo (9.B) |
| C | Sweep transformer penuh (7 model × 5 fold) | Beberapa jam | ⚠️ Skrip orkestrasinya tidak ada di `main` (lihat di bawah) — hasilnya sudah ada |
| D | Eksperimen multi-seed (7 model × 5 fold × 3 seed) | Jauh lebih lama dari C | 🚧 Tidak ada di `main` — sedang dikembangkan di branch terpisah |

### 9.A Satu baseline

Eksperimen paling sederhana — model cepat non-transformer, berguna
sebagai pemeriksaan awal atau titik pembanding sederhana:

```bash
uv run python scripts/run_baseline.py tfidf_lr
```
Pilihan lain: `majority_class`, `tfidf_svm`, `svm_word2vec`.

### 9.B Fine-tuning satu model transformer

`src/finetune/trainer.py` berisi logika fine-tuning yang dipakai
berulang di seluruh proyek ini. Di `main`, ini dipakai lewat skrip
checkpoint demo (Bagian 16), bukan lewat skrip serba-guna untuk satu
kali jalan.

### 9.C Sweep transformer penuh

Ketujuh model transformer yang tercantum di Bagian 13, masing-masing
dievaluasi pada 5 fold yang sama, berarti **35 proses pelatihan** total
(7 × 5). Begitulah hasil yang sudah tampil di bagian "Model Comparison"
dashboard diproduksi. ⚠️ **Memakan waktu lama**: pada perangkat keras asli
yang dipakai proyek ini (Mac Apple Silicon, MPS), sweep penuh ini
memakan waktu beberapa jam.

> Seperti disebutkan di Bagian 5, skrip orkestrasi yang menjalankan sweep
> ini secara otomatis saat ini berada di branch pengembangan terpisah,
> bukan di `main`. Hasil sweep tersebut sudah tersedia di `main` dalam
> folder `results/`, dan bisa dijelajahi lewat dashboard maupun skrip
> analisis tanpa perlu menjalankan ulang pelatihan apa pun. Jika Anda
> hanya memiliki `main`, tidak ada satu perintah pun di sini yang
> "langsung menjalankan seluruh sweep" — dan itu memang disengaja, agar
> tidak ada yang tanpa sadar memulai proses yang memakan waktu berjam-jam.

### 9.D Eksperimen multi-seed

Menjalankan kombinasi model/fold yang sama dengan seed acak yang berbeda
(Bagian 4) membantu membedakan "model ini memang konsisten baik" dari
"proses ini kebetulan beruntung."

Protokol yang direncanakan untuk eksperimen ini adalah:

- **7 model** (Bagian 13)
- **5 fold** (fold_0 … fold_4)
- **3 seed**: `42`, `43`, `44`
- 7 × 5 × 3 = **105 proses pelatihan yang direncanakan**

**Penting:** 105 adalah jumlah *proses pelatihan* yang direncanakan
(kombinasi model + fold + seed) — **bukan** 105 model yang berbeda. Tujuh
model yang sama masing-masing dilatih 15 kali (5 fold × 3 seed), bukan 5
kali seperti sebelumnya, untuk memeriksa seberapa stabil hasilnya.

**Di `main`, protokol ini beserta skrip orkestrasinya belum ada** —
eksperimen ini sedang dikembangkan dan, pada saat README ini ditulis,
dijalankan di branch terpisah, dan bukan sesuatu yang bisa Anda mulai
dari checkout `main` biasa. README ini tidak akan menyuruh Anda
menjalankannya, karena memang belum ada apa pun di `main` untuk
dijalankan. Setelah digabungkan (merge), bagian ini akan diperbarui
dengan perintah dan hasil yang sebenarnya.

> ⚠️ **Jangan mencoba menjalankan sweep penuh atau eksperimen multi-seed
> "hanya untuk coba-coba."** Ini adalah proses pelatihan yang memakan
> waktu berjam-jam (Level C) hingga berhari-hari (Level D). Jalankan
> Level C/D hanya ketika Anda memang secara sengaja ingin
> mereproduksi eksperimen penuh, dan sebaiknya saat komputer Anda bisa
> dibiarkan menyala tanpa gangguan.

## 10. Eksperimen yang berjalan lama: apa yang perlu diketahui

Jika Anda menjalankan perintah pelatihan yang lama (Bagian 9.1–9.3):

- **Ke mana log disimpan:** log eksekusi terstruktur (JSON) ditulis di
  `results/logs/<model>/<fold>.json`, mencatat status (`running`,
  `completed`, atau `failed`), waktu, dan riwayat pelatihan per epoch.
- **Cara memeriksa progres:** selama proses masih berjalan, field
  `status` pada file log akan bertuliskan `"running"`; setelah selesai,
  berubah menjadi `"completed"` atau `"failed"`.
- **Tanda proses selesai:** entri log `"status": "completed"` disertai
  file yang cocok di `results/metrics/<model>/<fold>.json`.
- **Jika proses gagal:** log mencatat `"status": "failed"` beserta pesan
  errornya — tidak pernah dibuang diam-diam, sehingga Anda bisa melihat
  persis apa yang salah.
- **Melanjutkan (resume):** menjalankan ulang perintah baseline yang sama
  aman dilakukan; tidak merusak atau perlu mengulang pekerjaan yang sudah
  selesai dengan sukses.

## 11. Memahami hasil

Tiga metrik yang muncul di seluruh proyek ini:

- **Accuracy (akurasi)** — persentase prediksi yang benar-benar tepat.
  Sederhana, tapi bisa menyesatkan jika satu kategori jauh lebih sering
  muncul dibanding yang lain.
- **Weighted-F1** — mirip Macro-F1 di bawah, tapi kategori dengan lebih
  banyak contoh data punya bobot lebih besar dalam rata-rata akhirnya.
- **Macro-F1** — menghitung skor F1 *secara terpisah untuk masing-masing
  dari 11 kategori*, lalu merata-ratakan ke-11 skor tersebut **secara
  setara**, tidak peduli berapa banyak contoh yang dimiliki tiap
  kategori. Ini penting di sini karena beberapa kategori distorsi jauh
  lebih jarang muncul dibanding yang lain dalam dataset; Macro-F1
  mencegah skor didominasi hanya oleh kategori paling umum ("Tidak Ada
  Distorsi").

**Cara membaca tabel perbandingan dengan benar:**

- *Observasi (hasil yang teramati)*: "Model A memiliki rata-rata Macro-F1
  yang teramati lebih tinggi dibanding Model B pada eksperimen ini" —
  ini sekadar membaca angka dari tabel.
- *Bukti statistik*: sebuah uji formal (proyek ini memakai uji Friedman —
  lihat `results/analysis/transformer/statistical_tests.json`) bisa
  memberi tahu apakah perbedaan di antara ketujuh model kemungkinan besar
  bukan sekadar kebetulan, berdasarkan 5 fold yang dipakai. Uji ini
  **tidak** memberi tahu pasangan model *spesifik* mana yang berbeda, dan
  5 fold adalah sampel yang kecil.
- *Keputusan peneliti*: memilih untuk benar-benar memakai satu model
  dibanding model lain untuk suatu keperluan adalah keputusan yang
  diambil oleh manusia, dengan mempertimbangkan bukti-bukti di atas —
  proyek ini tidak secara otomatis mengumumkan "pemenang," dan Anda pun
  sebaiknya tidak melakukannya saat membaca hasilnya.

Lihat `results/analysis/transformer/comparison/researcher_summary.md`
untuk penjelasan lengkap berbahasa sederhana tentang apa yang bisa dan
tidak bisa disimpulkan dari hasil saat ini.

## 12. Dataset dan fold

- Dataset mentah (kalimat berbahasa Indonesia, masing-masing diberi
  label salah satu dari 11 kategori) dan definisi fold-nya diharapkan
  berada di `external/original-drive/DATASETS/...` di komputer Anda.
  **Data ini tidak disertakan dalam repositori Git** — Anda perlu
  memperolehnya sendiri dan menempatkannya di sana sebelum menjalankan
  skrip pelatihan atau baseline apa pun.
- **Fold** adalah salah satu cara membagi dataset menjadi kelompok
  latih, kelompok validasi, dan kelompok uji yang terpisah. Proyek ini
  mendefinisikan 5 fold agar setiap model bisa dievaluasi pada 5
  pembagian latih/uji yang berbeda, bukan hanya satu, sehingga gambaran
  performanya lebih dapat dipercaya.
- **Semua model memakai persis 5 fold yang sama.** Ini penting agar
  perbandingannya adil — jika model-model dievaluasi pada pembagian data
  yang berbeda-beda, perbedaan skornya bisa jadi berasal dari pembagian
  datanya, bukan dari modelnya sendiri.
- **Jangan sembarangan mengubah dataset atau file fold.** Melakukannya
  akan merusak reproduktifitas dan membuat perbandingan dengan semua
  hasil yang sudah tercatat di `results/` menjadi tidak valid. Lihat
  Bagian 17 untuk cara proyek ini melindungi diri dari hal tersebut.

## 13. Daftar model yang dipakai dalam proyek ini

| Nama konfigurasi model | Model pretrained yang mendasarinya | Penjelasan sederhana |
|---|---|---|
| `indobert_15g` | `cahya/bert-base-indonesian-1.5G` | Model BERT berbahasa Indonesia. |
| `indobert_base_p1` | `indobenchmark/indobert-base-p1` | Model BERT berbahasa Indonesia lain, dari kelompok riset yang berbeda. |
| `indobertweet` | `indolem/indobertweet-base-uncased` | Model BERT yang dilatih khusus pada teks gaya media sosial/Twitter berbahasa Indonesia. |
| `indoroberta_15g` | `cahya/roberta-base-indonesian-1.5G` | Model RoBERTa berbahasa Indonesia (varian dari arsitektur bergaya BERT). |
| `mbert` | `bert-base-multilingual-cased` | Model BERT multibahasa yang dilatih pada 100+ bahasa, termasuk Indonesia. |
| `nusabert` | `LazarusNLP/NusaBERT-base` | Model BERT yang dilatih dengan fokus pada bahasa Indonesia dan bahasa daerah Indonesia. |
| `xlmr` | `xlm-roberta-base` | Model bergaya RoBERTa multibahasa, juga dilatih pada banyak bahasa. |

Ketujuh model ini semuanya di-fine-tune dan dievaluasi dengan cara yang
sama, pada data dan fold yang sama, sehingga hasilnya bisa dibandingkan
secara adil. Proyek ini tidak mengklaim salah satu di antaranya secara
universal "yang terbaik" — lihat Bagian 11.

## 14. Hasil dan file yang dihasilkan

Menjalankan eksperimen menghasilkan file-file di bawah `results/` dan
`models/`. Berikut penjelasan isinya masing-masing, dan mana yang
benar-benar tersimpan di Git dibanding yang hanya dihasilkan secara lokal
di komputer Anda sendiri:

| Lokasi | Isinya | Tersimpan di Git? |
|---|---|---|
| `results/metrics/<model>/foldN.json` | Angka evaluasi (Accuracy, Macro-F1, Weighted-F1) untuk satu model pada satu fold. | Metrik model baseline tersimpan; tidak semua model tersimpan. |
| `results/logs/<model>/foldN.json` | Catatan satu proses pelatihan: status (`running`/`completed`/`failed`), waktu, riwayat per epoch. | Hanya dibuat jika Anda benar-benar menjalankan pelatihan secara lokal. |
| `results/predictions/` | Prediksi mentah model pada data uji. | **Tidak** — tercantum di `.gitignore`, karena bisa dihasilkan ulang dan tidak dibutuhkan untuk perbandingan. |
| `results/analysis/transformer/` | Statistik gabungan, grafik, dan ringkasan perbandingan yang dibahas di Bagian 11 dan 17. | **Ya** — ini adalah ringkasan kecil dan penting yang disimpan proyek ini di Git. |
| `models/` | Bobot model tersimpan ("checkpoint"), termasuk checkpoint demo langsung (Bagian 16). | **Tidak** — tercantum di `.gitignore`. File bobot model adalah file biner besar (`*.pt`, `*.bin`, dll.) dan sengaja tidak pernah di-commit. |

Inilah kenapa, jika Anda melihat repositori ini di GitHub, Anda **tidak**
akan menemukan file `model.pt` di mana pun — checkpoint hanya ada secara
lokal, di komputer mana pun yang benar-benar menjalankan pelatihan, di
dalam folder `models/` yang diabaikan Git.

## 15. Riwayat fase (kenapa dokumennya begitu banyak?)

Proyek ini dibangun secara bertahap, dalam "fase" bernomor, masing-masing
dengan laporan singkatnya sendiri di `docs/phases/`. Anda tidak perlu
membacanya untuk memakai repositori ini, tapi dokumen tersebut
menjelaskan *kenapa* strukturnya seperti sekarang:

| Fase | Yang dilakukan |
|---|---|
| 6.3 | Merencanakan arsitektur akhir repositori/dashboard sebelum ada kode yang ditulis. |
| 7 | Mengaudit apakah repositori sudah benar-benar siap dijalankan secara lokal. |
| 8 | Mengadaptasi kode Colab asli agar bisa berjalan di luar Colab, secara lokal. |
| 9 / 9.1 | Menjalankan dan memverifikasi ulang model-model baseline sederhana. |
| 10A / 10B | Pembersihan repositori secara umum, beserta commit yang mendokumentasikannya. |
| 15 (dashboard) | Membangun versi pertama dashboard hasil eksperimen lokal. |
| 16 | Menambahkan analisis statistik (metrik gabungan, uji Friedman, penanganan anomali). |
| 17 | Meningkatkan dashboard dengan lebih banyak bagian dan dukungan akses jaringan lokal (LAN). |
| 18 | Menambahkan demo inference langsung (satu checkpoint terlatih yang dipilih secara eksplisit). |
| 19 | Membangun ringkasan perbandingan model deskriptif yang dibahas di Bagian 11. |

(Eksperimen multi-seed — mengulang sweep dengan beberapa seed acak untuk
perbandingan yang lebih kuat — telah disiapkan di branch pengembangan
terpisah setelah Fase 19, tetapi belum menjadi bagian dari `main` saat
README ini ditulis.)

## 16. Cara kerja demo inference langsung

1. `scripts/train_demo_checkpoint.py` melatih **satu kombinasi
   model+fold yang dipilih secara eksplisit** (defaultnya
   `indobert_base_p1` / `fold_0`) dan menyimpan bobotnya ke
   `models/demo/<model>/<fold>/`. Skrip ini menolak untuk menimpa
   checkpoint yang sudah ada secara diam-diam.
2. Setelah checkpoint itu ada, Anda bisa mendapatkan prediksi dengan dua
   cara:
   - **Lewat command line:**
     ```bash
     uv run python scripts/run_inference.py \
       --checkpoint models/demo/indobert_base_p1/fold_0 \
       --text "kalimat berbahasa Indonesia di sini"
     ```
   - **Lewat bagian "Live Demo" di dashboard**, dengan mengetik teks pada
     kotak yang tersedia di browser.
3. Checkpoint ini **dipilih secara eksplisit untuk keperluan
   demonstrasi** — ia adalah satu proses pelatihan tertentu, bukan
   otomatis "model terbaik" yang dipilih oleh algoritma apa pun (lihat
   Bagian 11). Checkpoint demo adalah artefak demonstrasi, bukan otomatis
   "model riset final."
4. Seperti disebutkan di Bagian 1: demo ini adalah artefak riset, bukan
   alat klinis atau diagnostik.

## 17. Reproduktifitas

Untuk mereproduksi hasil proyek ini, Anda memerlukan hal yang sama
persis:

- **Dataset** — file CSV yang persis sama (lihat Bagian 12).
- **Definisi fold** — file pembagian latih/validasi/uji yang persis
  sama.
- **Konfigurasi model** — file `configs/models/*.yaml`, tanpa perubahan.
- **Random seed** — tetap `42` untuk sweep yang dijelaskan di Bagian 9.3.
- **Skrip** — kode pelatihan/evaluasi yang sama.
- **Environment** — versi Python dan paket yang sama, yang otomatis
  didapat dari `uv sync` berdasarkan `uv.lock`.

**Kenapa ini penting:** jika salah satu hal di atas berubah secara diam-
diam, hasil yang tampak "lebih baik" atau "lebih buruk" mungkin hanya
mencerminkan dataset atau pengaturan yang berbeda, bukan model yang
benar-benar lebih baik.

**Checksum:** proyek ini menghitung checksum SHA-256 (semacam sidik jari
digital singkat) dari file dataset dan fold, lalu membandingkannya
dengan nilai yang sudah diketahui sebelum menjalankan eksperimen apa pun.
Jika file tersebut berubah tanpa disengaja, checksum-nya tidak akan
cocok, dan ini dipakai sebagai tanda peringatan dini bahwa ada sesuatu
pada data yang telah berubah.

## 18. Pemecahan masalah (Troubleshooting)

| Masalah | Kemungkinan penyebab | Solusi sederhana |
|---|---|---|
| `uv: command not found` | `uv` belum terinstal | Instal dari <https://docs.astral.sh/uv/getting-started/installation/>, lalu coba lagi. |
| `ModuleNotFoundError` untuk paket seperti `torch` atau `transformers` | Dependensi belum terinstal, atau Anda menjalankan `python` biasa alih-alih `uv run python` | Jalankan `uv sync` terlebih dahulu, dan selalu awali perintah dengan `uv run`. |
| Error file dataset/fold tidak ditemukan | Dataset belum ditempatkan di `external/original-drive/DATASETS/...` di komputer Anda | Peroleh datasetnya dan tempatkan di path yang diharapkan (Bagian 12) sebelum menjalankan skrip pelatihan/baseline. |
| Model Hugging Face terlihat macet atau gagal saat pertama kali dipakai | Bobot model pretrained perlu diunduh dari internet untuk pertama kalinya | Periksa koneksi internet Anda; unduhan hanya terjadi sekali dan setelah itu disimpan (cache) secara lokal. |
| Pelatihan sangat lambat | Anda hanya menjalankan di CPU, atau MPS tidak tersedia | Ini wajar di komputer non-Apple-Silicon; pelatihan tetap akan berjalan, hanya lebih lambat. |
| `Address already in use` saat menjalankan dashboard | Ada proses lain (mungkin dashboard sebelumnya) yang sudah memakai port 8765 | Hentikan proses lain tersebut, atau jalankan di port berbeda: `uv run python scripts/run_dashboard.py --port 8800`. |
| Skrip yang berjalan lama tampak berhenti | Proses mungkin gagal (crash) atau terinterupsi | Periksa `results/logs/<model>/<fold>.json` untuk status `"failed"` dan pesan errornya; aman untuk dijalankan ulang. |
| Dashboard terbuka tapi tidak menampilkan hasil apa pun | Anda menjalankannya dari direktori hasil yang berbeda dari yang diharapkan | Pastikan Anda menjalankannya dari root repositori, atau berikan `--results-dir <path>` secara eksplisit. |
| Tidak ada file `model.pt` di repositori | File bobot model sengaja tidak disimpan di Git (lihat Bagian 14) | Ini wajar — checkpoint hanya dibuat secara lokal saat Anda menjalankan pelatihan di komputer Anda sendiri. |
| Beberapa file di `results/` sepertinya "hilang" dari Git | Beberapa jenis hasil (prediksi, bobot model) memang sengaja dicantumkan di `.gitignore` | Lihat Bagian 14 untuk rincian lengkap folder hasil mana yang tersimpan di Git dan mana yang hanya dihasilkan secara lokal. |

## 19. FAQ

**Apakah saya perlu memahami machine learning untuk menjalankan proyek
ini?**
Tidak — Bagian 6–8 hanya membutuhkan Anda menyalin perintah ke terminal.
Memahami *hasilnya* lebih dalam akan terbantu oleh Bagian 11, tapi tidak
wajib untuk sekadar menjalankan dashboard.

**Bisakah saya langsung menjalankan semuanya sekaligus?**
Sengaja tidak ada satu perintah "jalankan semuanya." Baseline (Level A)
cepat dan aman dijalankan kapan saja. Sweep penuh (Level C) dan
eksperimen multi-seed (Level D) yang dijelaskan di Bagian 9 adalah proses
yang memakan waktu berjam-jam (atau lebih) — Anda sebaiknya hanya
memulainya dengan sengaja, bukan tanpa sadar.

**Kenapa saya tidak melihat `model.pt` di mana pun dalam repositori?**
File bobot model terlatih sengaja tidak disertakan di Git (lihat Bagian
14) karena ukurannya besar (file biner). File itu hanya ada secara lokal
di komputer mana pun yang benar-benar menjalankan pelatihan.

**Kenapa beberapa hasil diabaikan oleh Git?**
File di `results/predictions/` dan semua isi `models/` memang sengaja
dicantumkan di `.gitignore` — file-file itu bisa dihasilkan ulang dari
sebuah proses pelatihan dan tidak perlu disimpan permanen dalam riwayat
repositori. Lihat Bagian 14 untuk rincian lengkapnya.

**Apa itu transformer?**
Lihat daftar istilah di Bagian 2 — singkatnya, ini adalah jenis
arsitektur neural network yang dipakai untuk ketujuh model bahasa dalam
proyek ini.

**Apa itu fold?**
Cara membagi dataset agar model diuji pada data yang belum pernah
dilatihkan padanya. Lihat Bagian 2 dan 12.

**Kenapa ada banyak seed (dalam proyek yang lebih luas)?**
Untuk memeriksa apakah performa model konsisten atau hanya kebetulan
beruntung pada satu proses tertentu. Lihat Bagian 9.4 untuk apa yang
sudah ada di `main` saat ini.

**Di mana hasilnya disimpan?**
Di folder `results/` — lihat Bagian 4, dan lebih mudah dilihat lewat
dashboard (Bagian 8).

**Di mana model yang sudah dilatih disimpan?**
Bobot model terlatih ("checkpoint") tidak disimpan di Git (karena
ukurannya besar); hanya checkpoint demo yang dijelaskan di Bagian 16 yang
dibuat secara lokal saat Anda menjalankan `train_demo_checkpoint.py`.

**Bagaimana saya tahu apakah sebuah eksperimen sudah selesai?**
Periksa file terkait di `results/logs/` untuk status `"completed"`
(Bagian 10), atau lihat bagian overview di dashboard.

**Bisakah ini dijalankan di laptop biasa?**
Bisa. Baseline (Bagian 9.1) berjalan cepat di laptop mana pun. Fine-
tuning transformer lebih berat dan lebih diuntungkan dengan Mac Apple
Silicon (MPS) atau komputer dengan GPU, tapi tetap bisa berjalan hanya
dengan CPU — hanya lebih lambat.

**Kenapa pelatihan memakan waktu lama?**
Fine-tuning model bahasa melibatkan banyak putaran berulang pada dataset
(epoch), dan melakukannya untuk 7 model × 5 fold menjadi cukup banyak.
Lihat Bagian 9.3.

**Apa yang harus saya lakukan sebelum mengubah dataset?**
Jangan, kecuali Anda memang berniat memulai eksperimen baru yang
terpisah — mengubahnya akan membuat perbandingan yang sudah
terverifikasi checksum-nya dengan semua hasil yang ada di `results/`
menjadi tidak valid. Lihat Bagian 17.

## Referensi cepat peneliti (Bahasa Indonesia)

```bash
# Instal dependensi
uv sync

# Jalankan seluruh tes otomatis
uv run python -m unittest discover -s tests -p "test_*.py"

# Jalankan dashboard (http://127.0.0.1:8765)
uv run python scripts/run_dashboard.py

# Jalankan satu eksperimen baseline (cepat)
uv run python scripts/run_baseline.py tfidf_lr

# ⚠️ Memakan waktu lama: melatih dan menyimpan satu checkpoint demo yang dipilih secara eksplisit
uv run python scripts/train_demo_checkpoint.py --model indobert_base_p1 --fold fold_0

# Jalankan inference terhadap checkpoint yang tersimpan
uv run python scripts/run_inference.py --checkpoint models/demo/indobert_base_p1/fold_0 --text "teks Anda di sini"

# Buat ulang ringkasan perbandingan model deskriptif dari hasil yang sudah ada
uv run python scripts/build_model_comparison.py
```
