"""Utility untuk load config, validasi data, dan set seed."""
import os
import sys
import random
import json
import copy
import yaml
import numpy as np


MODELING_ROOT = "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION"
CONFIG_DIR = os.path.join(MODELING_ROOT, "configs")
MODELS_CONFIG_DIR = os.path.join(CONFIG_DIR, "models")


def _require(cfg, keys, context=""):
    cur = cfg
    path_so_far = []
    for k in keys:
        path_so_far.append(k)
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(
                f"Key '{'.'.join(path_so_far)}' tidak ditemukan"
                + (f" ({context})" if context else "")
            )
        cur = cur[k]
    return cur


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(model_name=None, config_path=None):
    if config_path is not None:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    base_path = os.path.join(CONFIG_DIR, "base.yaml")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"base.yaml tidak ditemukan: {base_path}")
    with open(base_path, encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    if model_name is None:
        return base

    model_path = os.path.join(MODELS_CONFIG_DIR, f"{model_name}.yaml")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Config model tidak ditemukan: {model_path}. "
            f"Pilihan: {list_models()}"
        )
    with open(model_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(base, model_cfg)
    merged.setdefault("model_name", model_name)
    return merged


def list_models():
    if not os.path.isdir(MODELS_CONFIG_DIR):
        return []
    return sorted(
        f.replace(".yaml", "")
        for f in os.listdir(MODELS_CONFIG_DIR)
        if f.endswith(".yaml")
    )


def get_paths(model_name=None, cfg=None):
    if cfg is None:
        cfg = load_config()

    processed_dir = _require(cfg, ["paths", "processed_dataset_dir"], "base.yaml")
    models_root = _require(cfg, ["paths", "models_root"], "base.yaml")
    results_root = _require(cfg, ["paths", "results_root"], "base.yaml")

    paths = {
        "processed_dataset_dir": processed_dir,
        "dataset_csv": os.path.join(processed_dir, "dib_labeled.csv"),
        "folds_json": os.path.join(processed_dir, "folds.json"),
        "groups_json": os.path.join(processed_dir, "dib_groups.json"),
        "models_root": models_root,
        "results_root": results_root,
        "predictions_dir": os.path.join(results_root, "predictions"),
        "metrics_dir": os.path.join(results_root, "metrics"),
        "figures_dir": os.path.join(results_root, "figures"),
        "tables_dir": os.path.join(results_root, "tables"),
    }

    if model_name is not None:
        paths["model_dir"] = os.path.join(models_root, model_name)
        paths["model_predictions_dir"] = os.path.join(
            results_root, "predictions", model_name
        )
        paths["model_metrics_dir"] = os.path.join(
            results_root, "metrics", model_name
        )

    return paths


def ensure_dirs(*paths):
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(cfg=None, paths=None, verbose=True):
    import pandas as pd

    if paths is None:
        paths = get_paths(cfg=cfg)
    if cfg is None:
        cfg = load_config()

    dataset_csv = paths["dataset_csv"]
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_csv}")

    df = pd.read_csv(dataset_csv)

    id_col = cfg["data"]["id_column"]
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]
    group_col = cfg["data"]["group_column"]
    required_cols = [id_col, text_col, label_col, group_col]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ada: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    num_labels_cfg = cfg["data"]["num_labels"]
    labels_unique = sorted(df[label_col].dropna().unique().tolist())
    expected_labels = list(range(num_labels_cfg))
    label_ok = (labels_unique == expected_labels)

    nan_counts = {
        "id": int(df[id_col].isna().sum()),
        "text": int(df[text_col].isna().sum()),
        "label": int(df[label_col].isna().sum()),
        "group": int(df[group_col].isna().sum()),
    }
    total_nan = sum(nan_counts.values())

    n_dup_ids = int(df[id_col].duplicated().sum())
    ids_unique = (n_dup_ids == 0)

    label_dist = df[label_col].value_counts().sort_index().to_dict()
    n_rows = len(df)
    imbalance = (
        max(label_dist.values()) / min(label_dist.values())
        if label_dist else None
    )

    hash_status = "not_checked"
    hash_expected = None
    hash_actual = None
    folds_json = paths.get("folds_json")
    if folds_json and os.path.exists(folds_json):
        with open(folds_json, encoding="utf-8") as f:
            folds_meta = json.load(f).get("metadata", {})
        hash_expected = folds_meta.get("data_hash")
        hash_actual = _sha256_file(dataset_csv)
        if hash_expected and hash_expected == hash_actual:
            hash_status = "match"
        elif hash_expected:
            hash_status = "mismatch"
        else:
            hash_status = "no_hash_in_folds"

    summary = {
        "n_rows": n_rows,
        "n_labels_data": len(labels_unique),
        "n_labels_config": num_labels_cfg,
        "labels_unique": labels_unique,
        "labels_expected": expected_labels,
        "label_ok": label_ok,
        "nan_counts": nan_counts,
        "total_nan": total_nan,
        "ids_unique": ids_unique,
        "n_dup_ids": n_dup_ids,
        "label_distribution": label_dist,
        "class_imbalance_ratio": imbalance,
        "hash_status": hash_status,
        "hash_expected": hash_expected,
        "hash_actual": hash_actual,
    }

    if verbose:
        print("=" * 60)
        print("VALIDASI DATASET")
        print("=" * 60)
        print(f"File       : {dataset_csv}")
        print(f"Total baris: {n_rows}")
        print()
        print(f"Kolom wajib        : {required_cols}")
        print(f"Kolom tersedia     : {list(df.columns)}")
        print()
        print(f"Label unik di data       : {labels_unique}")
        print(f"Label diharapkan (config): {expected_labels}")
        print(f"Label cocok              : {'YA' if label_ok else 'TIDAK'}")
        print()
        print(f"sentence_id unik : {'YA' if ids_unique else 'TIDAK'} "
              f"(duplikat: {n_dup_ids})")
        print()
        print("NaN per kolom:")
        for k, v in nan_counts.items():
            print(f"  {k}: {v}")
        print()
        print(f"Distribusi label ({len(label_dist)} kelas):")
        for lbl, cnt in label_dist.items():
            pct = cnt / n_rows * 100
            print(f"  Label {lbl:>2}: {cnt:>5} ({pct:5.2f}%)")
        if imbalance:
            print()
            print(f"Rasio imbalance (max/min): {imbalance:.2f}")
        print()
        if hash_status == "match":
            print(f"Hash dataset vs folds.json: COCOK ({hash_actual[:16]}...)")
        elif hash_status == "mismatch":
            print(f"Hash dataset vs folds.json: BEDA")
            print(f"  diharapkan: {hash_expected[:16]}...")
            print(f"  aktual    : {hash_actual[:16]}...")
        elif hash_status == "no_hash_in_folds":
            print("Hash dataset: tidak ada di metadata folds.json")
        else:
            print("Hash dataset: folds.json tidak ada, dilewati")
        print("=" * 60)

    if not label_ok:
        raise ValueError(
            f"Label TIDAK cocok dengan config. "
            f"Data: {labels_unique}, Config: {expected_labels}."
        )
    if total_nan > 0:
        raise ValueError(f"Ada NaN di kolom wajib: {nan_counts}")
    if not ids_unique:
        raise ValueError(
            f"Ada {n_dup_ids} sentence_id duplikat. Periksa pipeline preprocessing."
        )

    return summary


def load_folds(paths=None, cfg=None, verbose=True):
    if paths is None:
        paths = get_paths(cfg=cfg)

    folds_json = paths["folds_json"]
    if not os.path.exists(folds_json):
        raise FileNotFoundError(f"folds.json tidak ditemukan: {folds_json}")

    with open(folds_json, encoding="utf-8") as f:
        data = json.load(f)

    if "folds" not in data or "metadata" not in data:
        raise ValueError("Skema folds.json tidak sesuai (folds/metadata).")

    required_meta = ["data_hash", "n_splits", "seed", "total_rows"]
    missing_meta = [k for k in required_meta if k not in data["metadata"]]
    if missing_meta:
        raise ValueError(f"Metadata folds.json kurang: {missing_meta}")

    for fold_name, fold in data["folds"].items():
        for split in ("train_ids", "val_ids", "test_ids"):
            if split not in fold:
                raise ValueError(f"{fold_name} tidak punya key '{split}'")

    dataset_csv = paths["dataset_csv"]
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_csv}")

    hash_actual = _sha256_file(dataset_csv)
    hash_expected = data["metadata"]["data_hash"]

    if hash_actual != hash_expected:
        raise ValueError(
            f"Hash dataset TIDAK cocok dengan folds.json. "
            f"Dataset sudah berubah sejak fold dibuat. "
            f"diharapkan: {hash_expected[:16]}..., aktual: {hash_actual[:16]}... "
            f"Jalankan ulang preprocessing build_folds."
        )

    for fold_name, fold in data["folds"].items():
        train = set(fold["train_ids"])
        val = set(fold["val_ids"])
        test = set(fold["test_ids"])
        if train & val:
            raise ValueError(f"{fold_name}: train dan val tidak kosong")
        if train & test:
            raise ValueError(f"{fold_name}: train dan test tidak kosong")
        if val & test:
            raise ValueError(f"{fold_name}: val dan test tidak kosong")

    if verbose:
        print("=" * 60)
        print("LOAD FOLDS")
        print("=" * 60)
        print(f"File      : {folds_json}")
        print(f"Data hash : {hash_expected[:16]}... (cocok dengan dataset)")
        print(f"N splits  : {data['metadata']['n_splits']}")
        print(f"Seed      : {data['metadata']['seed']}")
        print(f"Total rows: {data['metadata']['total_rows']}")
        print()
        for name, fold in data["folds"].items():
            print(f"{name}: train={len(fold['train_ids'])}, "
                  f"val={len(fold['val_ids'])}, test={len(fold['test_ids'])}")
        print("=" * 60)

    return data


def get_fold_data(df, folds_data, fold_name, cfg):
    if fold_name not in folds_data["folds"]:
        raise KeyError(
            f"Fold '{fold_name}' tidak ada. "
            f"Tersedia: {list(folds_data['folds'].keys())}"
        )

    fold = folds_data["folds"][fold_name]
    id_col = cfg["data"]["id_column"]

    train_df = df[df[id_col].isin(fold["train_ids"])].reset_index(drop=True)
    val_df = df[df[id_col].isin(fold["val_ids"])].reset_index(drop=True)
    test_df = df[df[id_col].isin(fold["test_ids"])].reset_index(drop=True)

    return train_df, val_df, test_df
