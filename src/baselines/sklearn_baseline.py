"""Baseline sanity: majority-class, TF-IDF + LR, TF-IDF + SVM."""
import os
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from gensim.models import Word2Vec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import get_fold_data, get_texts_labels
from metrics import compute_metrics, average_metrics
import config_utils as cu


def _save_fold(fold_name, model_name, test_ids, y_true, y_pred,
               paths, num_labels):
    pred_dir = os.path.join(paths["results_root"], "predictions", model_name)
    met_dir = os.path.join(paths["results_root"], "metrics", model_name)
    cu.ensure_dirs(pred_dir, met_dir)

    df_pred = pd.DataFrame({
        "sentence_id": test_ids,
        "y_true": list(y_true),
        "y_pred": list(y_pred),
    })
    df_pred.to_csv(os.path.join(pred_dir, f"{fold_name}.csv"), index=False)

    metrics = compute_metrics(y_true, y_pred, num_labels)
    metrics["fold"] = fold_name
    with open(os.path.join(met_dir, f"{fold_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _finalize(model_name, fold_metrics, paths, num_labels):
    met_dir = os.path.join(paths["results_root"], "metrics", model_name)
    summary = average_metrics(fold_metrics, num_labels)
    with open(os.path.join(met_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_majority_class(df, folds_data, cfg, paths, model_name="majority_class"):
    num_labels = cfg["data"]["num_labels"]
    id_col = cfg["data"]["id_column"]
    fold_names = sorted(folds_data["folds"].keys())
    fold_metrics = []

    for fold_name in fold_names:
        train_df, _, test_df = get_fold_data(df, folds_data, fold_name, cfg)
        _, y_train = get_texts_labels(train_df, cfg)
        _, y_test = get_texts_labels(test_df, cfg)

        majority = Counter(y_train).most_common(1)[0][0]
        y_pred = [majority] * len(y_test)

        m = _save_fold(fold_name, model_name,
                       test_df[id_col].tolist(), y_test, y_pred,
                       paths, num_labels)
        fold_metrics.append(m)

    return _finalize(model_name, fold_metrics, paths, num_labels)


def _run_tfidf(df, folds_data, cfg, paths, model_name,
               clf_class, clf_params, vec_params):
    num_labels = cfg["data"]["num_labels"]
    id_col = cfg["data"]["id_column"]
    fold_names = sorted(folds_data["folds"].keys())
    fold_metrics = []

    for fold_name in fold_names:
        train_df, _, test_df = get_fold_data(df, folds_data, fold_name, cfg)
        X_train, y_train = get_texts_labels(train_df, cfg)
        X_test, y_test = get_texts_labels(test_df, cfg)

        vec = TfidfVectorizer(**vec_params)
        X_train_v = vec.fit_transform(X_train)
        X_test_v = vec.transform(X_test)

        clf = clf_class(**clf_params)
        clf.fit(X_train_v, y_train)
        y_pred = clf.predict(X_test_v)

        m = _save_fold(fold_name, model_name,
                       test_df[id_col].tolist(), y_test, y_pred,
                       paths, num_labels)
        fold_metrics.append(m)

    return _finalize(model_name, fold_metrics, paths, num_labels)


def run_tfidf_lr(df, folds_data, cfg, paths):
    mc = cu.load_config("tfidf_lr")
    vec_params = {
        "max_features": mc["vectorizer"]["max_features"],
        "ngram_range": tuple(mc["vectorizer"]["ngram_range"]),
        "min_df": mc["vectorizer"]["min_df"],
    }
    clf_params = {
        "C": mc["hyperparameters"]["C"],
        "max_iter": mc["hyperparameters"]["max_iter"],
        "class_weight": mc["hyperparameters"]["class_weight"],
        "random_state": mc["random_state"],
    }
    return _run_tfidf(df, folds_data, cfg, paths, "tfidf_lr",
                      LogisticRegression, clf_params, vec_params)


def run_tfidf_svm(df, folds_data, cfg, paths):
    mc = cu.load_config("tfidf_svm")
    vec_params = {
        "max_features": mc["vectorizer"]["max_features"],
        "ngram_range": tuple(mc["vectorizer"]["ngram_range"]),
        "min_df": mc["vectorizer"]["min_df"],
    }
    clf_params = {
        "C": mc["hyperparameters"]["C"],
        "max_iter": mc["hyperparameters"]["max_iter"],
        "class_weight": mc["hyperparameters"]["class_weight"],
        "random_state": mc["random_state"],
    }
    return _run_tfidf(df, folds_data, cfg, paths, "tfidf_svm",
                      LinearSVC, clf_params, vec_params)


def run_svm_word2vec(df, folds_data, cfg, paths):
    """Baseline DIB: SVM (RBF) + Word2Vec (dilatih pada train fold)."""
    num_labels = cfg["data"]["num_labels"]
    id_col = cfg["data"]["id_column"]
    mc = cu.load_config("svm_word2vec")
    w2v_params = mc.get("word2vec", {})
    clf_params = mc["hyperparameters"]

    fold_names = sorted(folds_data["folds"].keys())
    fold_metrics = []

    for fold_name in fold_names:
        train_df, _, test_df = get_fold_data(df, folds_data, fold_name, cfg)
        X_train_texts, y_train = get_texts_labels(train_df, cfg)
        X_test_texts, y_test = get_texts_labels(test_df, cfg)

        # Tokenisasi sederhana: split per spasi, lowercase
        train_tokens = [str(t).lower().split() for t in X_train_texts]
        test_tokens = [str(t).lower().split() for t in X_test_texts]

        # Latih Word2Vec pada train fold
        w2v = Word2Vec(
            sentences=train_tokens,
            vector_size=w2v_params.get("vector_size", 100),
            window=w2v_params.get("window", 5),
            min_count=w2v_params.get("min_count", 2),
            epochs=w2v_params.get("epochs", 20),
            workers=4,
            seed=42,
        )

        def text_to_vec(tokens):
            vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
            if not vecs:
                return np.zeros(w2v.vector_size)
            return np.mean(vecs, axis=0)

        X_train_vec = np.array([text_to_vec(t) for t in train_tokens])
        X_test_vec = np.array([text_to_vec(t) for t in test_tokens])

        clf = SVC(**clf_params)
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        m = _save_fold(fold_name, "svm_word2vec",
                       test_df[id_col].tolist(), y_test, y_pred,
                       paths, num_labels)
        fold_metrics.append(m)

    return _finalize("svm_word2vec", fold_metrics, paths, num_labels)
