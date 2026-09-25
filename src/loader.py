"""Load dataset dan folds untuk modeling."""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, "/content/drive/MyDrive/THESIS/MODELING/COGNITIVE DISTORTION/src")
import config_utils as cu


def load_dataset(cfg, paths):
    return pd.read_csv(paths["dataset_csv"])


def load_folds(paths, verbose=True):
    """Reuse cu.load_folds: validasi hash + skema + irisan split."""
    return cu.load_folds(paths=paths, verbose=verbose)


def get_fold_data(df, folds_data, fold_name, cfg):
    """Filter DataFrame per fold. Pakai id_column dari cfg."""
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


def get_texts_labels(df, cfg):
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]
    return df[text_col].tolist(), df[label_col].tolist()
