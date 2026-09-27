"""Write the static Phase 20 multi-seed experiment protocol manifest.

This script performs NO training. It only reads the existing dataset/fold
checksums (via the same validated load path config_utils.load_folds already
uses) and the existing model configs, and writes a small, machine-readable
description of the planned 7 x 5 x 3 = 105-run protocol to
results/multiseed/manifest.json.

The manifest exists so that, once the 105 runs are eventually executed
(in a later, explicitly-authorized phase), every run's (model, fold, seed)
identity can be checked against a single frozen definition of "the same
experimental protocol" -- not re-derived ad hoc each time.

Re-running this script is safe and deterministic: given an unchanged
dataset/folds/model-config set, it produces byte-identical content (aside
from the creation timestamp, which is refreshed on each write).
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config_utils as cu

PROTOCOL_VERSION = "1.0"
EXPERIMENT_ID = "phase20_multiseed_transformer_sweep"
SEEDS = [42, 43, 44]

MULTISEED_ROOT = os.path.join("results", "multiseed")
MANIFEST_PATH = os.path.join(MULTISEED_ROOT, "manifest.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_transformer_models():
    """Same discovery rule as scripts/run_transformer_sweep.py: a model is
    included only if model_type == "transformer" and its raw YAML does not
    contain the literal string "PLACEHOLDER"."""
    import yaml

    models = []
    for fname in sorted(os.listdir(cu.MODELS_CONFIG_DIR)):
        if not fname.endswith(".yaml"):
            continue
        path = os.path.join(cu.MODELS_CONFIG_DIR, fname)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        cfg = yaml.safe_load(raw) or {}
        if cfg.get("model_type") != "transformer":
            continue
        if "PLACEHOLDER" in raw:
            continue
        models.append(cfg["model_name"])
    return models


def discover_folds():
    import loader

    paths = cu.get_paths()
    folds_data = loader.load_folds(paths, verbose=False)
    return sorted(folds_data["folds"].keys())


def main():
    paths = cu.get_paths()
    models = discover_transformer_models()
    folds = discover_folds()

    dataset_sha256 = sha256_file(paths["dataset_csv"])
    folds_sha256 = sha256_file(paths["folds_json"])

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": models,
        "folds": folds,
        "seeds": SEEDS,
        "total_expected_runs": len(models) * len(folds) * len(SEEDS),
        "dataset_csv_sha256": dataset_sha256,
        "folds_json_sha256": folds_sha256,
        "notes": (
            "This manifest describes the PLANNED Phase 20 multi-seed "
            "protocol only. It does not imply any run has been executed. "
            "The same fold definitions (folds.json, unmodified) are used "
            "for every model and every seed; the seed is the only intended "
            "experimental variation. Output for each run is written under "
            "results/multiseed/, never overwriting the Phase 13/15 "
            "single-seed sweep under results/metrics /results/logs "
            "/results/predictions."
        ),
    }

    os.makedirs(MULTISEED_ROOT, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {MANIFEST_PATH}")
    print(f"Models ({len(models)}): {models}")
    print(f"Folds ({len(folds)}): {folds}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Total expected runs: {manifest['total_expected_runs']}")


if __name__ == "__main__":
    main()
