# Phase 17 — Transformer Dashboard Enhancement, Public Access & Live Demo Readiness

## 1. Objective

Enhance the existing Phase 15 dashboard to integrate the Phase 16
statistical analysis, add plain-language explanations, present the two
known anomalous runs and their sensitivity analysis without hiding them,
provide a truthful (currently disabled) Live Demo section, and add
explicit, safe network-binding support — all while remaining read-only
with respect to experiment artifacts, preserving the existing minimal
stdlib/vanilla-JS architecture, and not modifying `main` directly.

## 2. Baseline

**FACT**, recorded before any change:
```
main HEAD: c2705d4  feat: add transformer statistical analysis
           (descends from f4012f9 merge: integrate Phase 15 transformer sweep dashboard)
```
Pre-existing uncommitted files on `main` (e.g. `src/finetune/trainer.py`,
`.claude/`, `docs/phases/PHASE_11-15*.md`,
`external/original-drive/DATASETS/`, `results/logs/`, `results/metrics/`,
`scripts/run_transformer*.py`) were **not touched, reset, stashed, or
cleaned** — this phase's work happened in a separate git worktree, and
`main`'s working tree was never written to.

**OBSERVATION on worktree setup**: the first attempt to create an
isolated worktree used the default "fresh" base-ref policy, which branches
from `origin/main` — since local `main` is 9+ commits ahead of
`origin/main` (nothing has been pushed in any prior phase), that worktree
was missing all of Phases 11–16's work entirely. This was caught before
any edits were made, the empty worktree was removed, and a worktree was
created manually via `git worktree add -b phase17-dashboard-enhancement
<path> HEAD`, correctly branching from local `main`'s actual tip
(`c2705d4`). All Phase 17 work happened in this corrected worktree.

## 3. Dashboard Changes (summary)

Five files changed, all within the existing `src/dashboard/` +
`scripts/run_dashboard.py` + `tests/` architecture — no new frontend
framework, no new third-party Python dependency:

- **New**: `src/dashboard/analysis_loader.py` — read-only loaders for the
  6 Phase 16 artifact files plus a plot-filename allowlist.
- **Modified**: `src/dashboard/state.py` — added `build_analysis_state()`
  and `build_live_demo_state()`, wired into `build_dashboard_state()`.
- **Modified**: `scripts/run_dashboard.py` — added `--host`, LAN-IP
  detection/printing, tunnel-friendly startup guidance, and an
  allowlisted `/api/analysis/plots/<name>` route.
- **Modified**: `src/dashboard/static/index.html` — 8 new sections
  (Overview, Live Demo, Model Comparison, Cross-Fold Stability,
  Anomalies, Per-Class Analysis, Statistical Analysis, Runtime, Glossary)
  plus supporting CSS and JS render functions.
- **New**: `tests/test_dashboard_analysis.py` — 28 new tests.

## 4. Model Comparison

**FACT**: the new "Model Comparison" section reads
`analysis.aggregate_metrics` (from `aggregate_metrics.json`, unmodified)
and renders, per model: Macro-F1 mean/std/min/max/median/range/CV, plus
Accuracy and Weighted-F1 mean±std — all numbers come from the JSON file,
none are computed in JavaScript. The three Phase 16 bar-chart PNGs
(`01_macro_f1_mean_std.png`, `02_accuracy_mean_std.png`,
`03_weighted_f1_mean_std.png`) are displayed via `<img>` tags fetched from
the new plot route, reusing the already-generated, already-reviewed Phase
16 plots rather than re-implementing charting in JavaScript.

**FACT**: model order is fixed to the experiment/configuration order
(`indobert_15g, indobert_base_p1, indobertweet, indoroberta_15g, mbert,
nusabert, xlmr`) via `state.MODEL_ORDER`, with an explicit on-page note:
"Models are listed in experiment/configuration order, not ranked by
performance." Verified by a dedicated test
(`test_model_order_is_fixed_experiment_order_not_a_ranking`).

## 5. Cross-Fold Stability

**FACT**: displays the Phase 16 line-plot PNG
(`04_macro_f1_across_folds.png`, which already marks the two anomalous
points with red X markers) alongside the two required explanatory
paragraphs, reproduced verbatim from the phase instructions, about what a
fold is and the limitation of having only five.

## 6. Anomaly Presentation

**FACT**: reads `anomaly_sensitivity.json` (unmodified) and, for each of
the two flagged runs, renders: Macro-F1/Accuracy/Weighted-F1 for that run,
the **official** 5-fold mean±std, and the **sensitivity-analysis** 4-fold
mean±std explicitly labeled "Sensitivity analysis — NOT the official
result", plus the fixed sentence "The available artifacts do not
establish the root cause." No root cause is speculated anywhere in the
new code or copy. Verified against real data during validation (§16):
the rendered `indobertweet/fold_1` and `mbert/fold_3` macro-F1 values
(0.0592, 0.0587) match Phase 16's report exactly.

## 7. Per-Class Analysis

**FACT**: displays the Phase 16 heatmap PNG
(`06_per_class_f1_heatmap.png`) plus the required plain-language
explanation of Macro-F1 and the low-support caveat for labels 9 and 10,
worded generally (not tied to one example model) per the instruction not
to overstate conclusions from low-support classes.

## 8. Statistical Analysis

**FACT**: reads `statistical_tests.json`'s `friedman_test` object
(unmodified) and displays the exact statistic, df, and p-value it
contains (matching Phase 16: chi-square=18.60, df=6, p=0.0049 — verified
identical during validation, not retyped from the phase brief), followed
by the three required caveats verbatim: it does not identify which pair
of models differs, it does not automatically select a model, and it is
based on only five matched folds.

## 9. Runtime Analysis

**FACT**: reads `runtime_analysis.json`'s `per_model` object and renders
mean/min/max per model in the fixed experiment order, plus the Phase 16
box-plot PNG (`05_runtime_distribution.png`), with the required
plain-language runtime-vs-quality disclaimer.

## 10. Plain-Language Explanations

**FACT**: all 8 requested glossary terms (Model, Fold, Accuracy,
Macro-F1, Weighted-F1, Standard deviation, Anomaly, Confidence interval)
are present as collapsible `<details>/<summary>` elements, using the
exact wording given in the phase instructions.

## 11. Live-Demo Status

**FACT**, checked first per instruction: `find models -type f` (against
the real project `models/` directory) returns empty — no checkpoint
exists anywhere in the repository, consistent with every prior phase's
finding (Phase 11 onward: `train_one_fold` never persists weights).

**Implementation**: `build_live_demo_state()` checks for any file under
`<results_dir>/../models/`; when none exists (the current, verified
reality), the dashboard shows a fixed, honest disabled state: **"Live
inference is not available yet."** with the explanation "The experiments
produced evaluation results and predictions, but the trained model
weights were not persisted for later inference. This dashboard is
prepared to support live inference once a checkpoint is added." No
inference code, no mocked prediction, and no confidence-score fabrication
exists anywhere in the new code — the checkpoint-found branch of
`build_live_demo_state()` only reports *that* a checkpoint file exists and
lists its path; it does not attempt to load or run any model, since doing
so was explicitly out of scope for this phase (no checkpoint exists to
test that path against in any case).

## 12. Network Binding

**FACT**: `scripts/run_dashboard.py` gained a `--host` argument, default
`127.0.0.1` (verified by test `test_default_host_is_localhost` — parsing
an empty argument list yields `127.0.0.1`, never `0.0.0.0`, so the safe
default cannot silently regress). Verified live: the server bound to and
accepted connections on both `127.0.0.1:8767` and `0.0.0.0:8766` in
separate manual runs (§16). `local_lan_ip()` uses a side-effect-free UDP
socket trick to report the machine's actual LAN address (verified live:
returned `192.168.61.162`, a genuine private-range address), and returns
`None` on failure rather than raising (verified by a dedicated test that
monkeypatches `socket.socket` to always raise).

## 13. Public-Access Documentation

**FACT**: when `--host 0.0.0.0` is used, the script's startup output
(reviewed by direct source inspection, since redirected/backgrounded
stdout was buffered during interactive testing and not captured verbatim
— the code path itself was read and confirmed correct) prints: the local
URL, the LAN URL (explicitly labeled as LAN, not public), an explicit
statement that a private address like `192.168.x.x`/`10.x.x.x` is **not**
a public internet address and is unreachable from mobile data/GSM, and a
tunnel-command example. When `--host` is left at the default, it instead
prints a note that `--host 0.0.0.0` is needed for LAN access. This
satisfies the instruction to document both the "same LAN" and "internet/
external network" scenarios distinctly, and to never call a private
address public.

## 14. Security Considerations

**FACT**: the HTTP handler's routing is a fixed `if path == "..."` /
`if path.startswith("/api/analysis/plots/")` chain — there is no generic
static-file-serving code path anywhere in `run_dashboard.py`, so no route
exists that could be tricked into serving an arbitrary repository file by
construction, not merely by input sanitization. The one route that takes
a filename from the URL (`/api/analysis/plots/<name>`) resolves it through
`analysis_loader.resolve_plot_path`, which rejects any name not
byte-for-byte equal to one of the 6 known plot filenames before any
filesystem access is attempted.

**FACT, verified live** (§16): a traversal attempt
(`/api/analysis/plots/..%2f..%2f..%2fetc%2fpasswd`), an unknown filename
(`/api/analysis/plots/evil.png`), and a direct request for `/.git/config`
all returned `404`, with no file outside the allowlisted plot set ever
read. The public-access warning text ("Public access makes this dashboard
reachable by people outside your local network. Do not expose sensitive
files or credentials.") is printed at `0.0.0.0` startup, per instruction.

## 15. Tests

**FACT**: `tests/test_dashboard_analysis.py` (new, 28 tests) covers:
aggregate-metrics loading (valid/missing/malformed), fold-metrics and
per-class-metrics CSV loading (valid/missing/empty), anomaly/runtime/
statistical-test loading, plot discovery and the allowlist (valid file,
missing directory, known-but-missing file, unknown name, explicit
traversal string), `build_analysis_state` (all-missing case, fixed model
order, one-malformed-file-does-not-blank-others isolation),
`build_live_demo_state` (no checkpoint, missing `models/` dir, checkpoint
present), `build_dashboard_state` including the new keys under both a
missing and a complete manifest, and dashboard-server host/LAN-IP
handling (default host, explicit override, LAN-IP failure handling). All
fixtures use `tempfile.TemporaryDirectory()`; none write to the real
`results/` tree.

**FACT**, full suite result:
```
uv run python -m unittest discover -s tests -p "test_*.py"
----------------------------------------------------------------------
Ran 44 tests in 0.038s

OK
```
(16 pre-existing Phase 15 tests + 28 new Phase 17 tests, all passing.)

## 16. Real-Data Validation

**FACT**, dashboard started against the real repository's `results/`
directory (the main checkout's, an absolute path, read-only —
`--results-dir /Users/MAC/Projects/Tesis-Mbak-Ai/results`), not synthetic
fixtures, to confirm genuine end-to-end behavior:

```
GET /                                             -> 200
GET /api/state                                     -> 200
GET /api/model?model=indobert_base_p1              -> 200
GET /api/analysis/plots/04_macro_f1_across_folds.png -> 200, 77059 bytes,
                                                         confirmed valid
                                                         840x540 PNG
GET /api/analysis/plots/..%2f..%2f..%2fetc%2fpasswd  -> 404
GET /api/analysis/plots/evil.png                     -> 404
GET /.git/config                                      -> 404
```
`/api/state`'s `overview` matched Phase 15's real numbers exactly
(`completed_runs: 34, failed_runs: 0, skipped_runs: 1, avg_runtime_seconds:
550.77`), and its `analysis.anomaly_sensitivity` matched Phase 16's
documented anomaly macro-F1 values exactly (`indobertweet/fold_1:
0.05919...`, `mbert/fold_3: 0.05873...`). All six `*_available` flags
were `True` and `plots_complete` was `True`. No server-side error or
traceback appeared in the server's log output during this session.

A second live run confirmed `--host 0.0.0.0 --port 8766` binds
successfully and accepts a connection via `127.0.0.1` (the loopback
address is always reachable on a `0.0.0.0`-bound socket) — this
demonstrates the bind succeeded; it does not by itself demonstrate LAN
reachability from a second physical device, which was not available to
test in this environment.

Both validation servers were stopped after testing; neither was left
running.

## 17. Safety Checks

**FACT**, verified directly (not via `git diff`, since this worktree-
isolated session cannot run git operations against the main checkout —
verified instead by direct file/process inspection):
```
dib_labeled.csv SHA-256: ed154162554b68ba6980af8f3c7c01fa80a7962c4a0595fd61acafd142bdaea0  (unchanged)
folds.json SHA-256:      730cd82a912828e902cdab26eb498c640195eca4901fde606f849372c5693c54  (unchanged)
results/logs/transformer_sweep.json: completed=34, failed=0, skipped=1  (unchanged from Phase 15)
find <project>/models -type f:        empty (no checkpoint created)
ps aux | grep run_transformer:        empty (no training process running, before or after)
```
No training process was started at any point in this phase. No file
under `results/metrics/`, `results/logs/`, `results/predictions/`, or
`external/original-drive/` was modified — the dashboard code only opens
these files for reading.

## 18. Files Created

```
src/dashboard/analysis_loader.py
tests/test_dashboard_analysis.py
docs/phases/PHASE_17_TRANSFORMER_DASHBOARD_ENHANCEMENT.md
```

## 19. Files Modified

```
scripts/run_dashboard.py       (+97 lines: --host, LAN-IP helper, plot route, startup guidance)
src/dashboard/state.py          (+115 lines: build_analysis_state, build_live_demo_state, wiring)
src/dashboard/static/index.html (+329 lines: new sections, CSS, render functions)
```

## 20. Files Intentionally Untouched

Per instruction, verified via `git status`/direct inspection, not merely
assumed: `src/finetune/trainer.py`, `.claude/` (worktree remnants),
`docs/phases/PHASE_11-15*.md`, `external/original-drive/DATASETS/`,
`results/logs/`, `results/metrics/`, `results/predictions/`,
`scripts/run_transformer.py`, `scripts/run_transformer_sweep.py`,
`configs/`, the notebook — none of these were staged, modified, reset,
stashed, or cleaned by this phase's work, which took place entirely in a
separate worktree/branch.

## 21. Limitations

- LAN reachability from a genuinely separate physical device was not
  tested (only loopback-via-0.0.0.0 was verified) — this environment has
  no second device available to confirm real cross-device LAN access.
- The `0.0.0.0` startup guidance text was verified correct by source
  inspection, not by capturing live stdout, since output buffering under
  the background/redirected test harness prevented direct capture in this
  session; the underlying `print()` calls were read and are not
  conditional on anything that would behave differently at runtime.
- No tunnel tool (cloudflared/ngrok/tailscale) was installed, launched,
  or tested — per instruction, the dashboard only needed to be
  tunnel-*compatible* (listen on a plain host/port), which was verified;
  actually exposing it externally was correctly left to the user's own
  tunnel setup.
- The live-demo "checkpoint present" code path (listing checkpoint files)
  was tested only with a synthetic fixture file, since no real checkpoint
  exists anywhere in this repository to test against natively.
- This phase enhances the dashboard's data presentation; it does not
  change, recompute, or re-validate any Phase 15/16 number — every value
  shown is read verbatim from already-committed artifacts.

## 22. How to Run Locally

```bash
uv run python scripts/run_dashboard.py --results-dir results --host 127.0.0.1 --port 8765
```
Then open `http://127.0.0.1:8765/` in a browser. Add `--host 0.0.0.0` to
also allow other devices on the same LAN to connect at
`http://<this-machine's-LAN-IP>:8765/` (printed at startup).

## 23. How to Expose Through a Secure Tunnel

Terminal 1 (dashboard, bound to loopback only — the tunnel provider
connects locally, so `0.0.0.0` is not required for this method):
```bash
uv run python scripts/run_dashboard.py --results-dir results --host 127.0.0.1 --port 8765
```
Terminal 2 (using whichever tunnel tool is already installed; none is
installed or required by this repository):
```bash
cloudflared tunnel --url http://127.0.0.1:8765
# or: ngrok http 8765
# or: tailscale funnel 8765
```
The tunnel tool prints an externally-reachable URL; share that URL, not
the LAN address, with someone outside the local network.

## Conclusion

The dashboard now integrates every Phase 16 statistical artifact through
new read-only loaders, presents model comparisons, cross-fold stability,
the two anomalous runs (with an explicitly-labeled, non-official
sensitivity analysis), per-class results, the Friedman test, and runtime
data — all read from disk, never hardcoded or recomputed — alongside a
plain-language glossary and a truthful, currently-disabled Live Demo
section reflecting the verified absence of any model checkpoint. Network
binding now supports explicit LAN exposure via `--host 0.0.0.0` with
accurate LAN-vs-public-internet guidance and tunnel-compatible operation,
while defaulting safely to localhost. The plot-serving route is
allowlist-only with no general file-serving capability, verified live
against traversal and `.git` access attempts. All 44 tests pass (28 new).
Real-data validation against the actual Phase 15/16 artifacts (not
fixtures) reproduced the exact known numbers. No training was run, no
checkpoint was fabricated, no experiment artifact was modified, and all
pre-existing uncommitted changes on `main` were left untouched — this
phase's work is isolated to the `phase17-dashboard-enhancement` branch,
not merged into `main`.
