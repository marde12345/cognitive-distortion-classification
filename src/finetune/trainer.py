"""Fine-tuning transformer untuk klasifikasi 11 kelas."""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_utils as cu


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _compute_class_weights(labels, num_labels):
    counts = np.bincount(labels, minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)


def train_one_fold(
    model_name, hf_model_id,
    train_df, val_df, test_df,
    cfg, paths,
    num_labels=11, max_len=128, batch_size=16, learning_rate=2e-5,
    weight_decay=0.01, warmup_ratio=0.1, num_epochs=5, patience=2,
    fp16=True, device=None, fold_name="fold_0", checkpoint_path=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    id_col = cfg["data"]["id_column"]
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    train_texts = train_df[text_col].tolist()
    train_labels = train_df[label_col].tolist()
    val_texts = val_df[text_col].tolist()
    val_labels = val_df[label_col].tolist()
    test_texts = test_df[text_col].tolist()
    test_labels = test_df[label_col].tolist()

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id, num_labels=num_labels
    ).to(device)

    train_ds = TextDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, max_len)
    test_ds = TextDataset(test_texts, test_labels, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Class-weighted loss: baca flag dari cfg
    use_class_weight = cfg["training"].get("class_weighted_loss", True)
    if use_class_weight:
        class_weights = _compute_class_weights(train_labels, num_labels).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if fp16 and device == "cuda" else None

    def _eval(loader):
        model.eval()
        preds, probs_all, labels_all = [], [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
                probs_all.extend(probs.cpu().numpy().tolist())
                labels_all.extend(labels.cpu().numpy().tolist())
        macro_f1 = f1_score(labels_all, preds, average="macro", zero_division=0)
        return preds, probs_all, labels_all, macro_f1

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attn)
                    loss = loss_fn(outputs.logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attn)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        _, _, _, val_f1 = _eval(val_loader)
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_macro_f1": val_f1})
        print(f"  Epoch {epoch+1}/{num_epochs} | loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop di epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    _, _, _, val_f1 = _eval(val_loader)
    test_preds, test_probs, test_labels_eval, _ = _eval(test_loader)

    from metrics import compute_metrics
    val_metrics = {"macro_f1": val_f1}
    test_metrics = compute_metrics(test_labels_eval, test_preds, num_labels)
    test_metrics["history"] = history

    if checkpoint_path is not None:
        # Opt-in only: every existing caller (the sweep runner, the
        # single-run script) omits this argument, so this branch never
        # executes for them and their behavior is unchanged. Saves just
        # enough to reconstruct inference later: weights, the exact
        # Hugging Face model id (so the matching architecture/tokenizer
        # can be re-downloaded), and the training/evaluation metadata
        # needed to interpret the checkpoint. No dataset content is
        # written here.
        import os as _os
        _os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(model.state_dict(), _os.path.join(checkpoint_path, "model.pt"))
        checkpoint_meta = {
            "model_name": model_name,
            "hf_model_id": hf_model_id,
            "num_labels": num_labels,
            "max_len": max_len,
            "fold_name": fold_name,
            "seed": cfg["training"]["seed"],
            "device_used": device,
            "training_config": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "configured_epochs": num_epochs,
                "completed_epochs": len(history),
                "patience": patience,
                "class_weighted_loss": use_class_weight,
            },
            "val_macro_f1": val_metrics["macro_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_accuracy": test_metrics["accuracy"],
        }
        with open(_os.path.join(checkpoint_path, "checkpoint_meta.json"), "w") as _f:
            json.dump(checkpoint_meta, _f, indent=2)

    return val_metrics, test_metrics, test_preds, test_probs, test_labels_eval


def save_predictions_and_metrics(model_name, fold_name, test_df, test_preds,
                                 test_probs, test_metrics, cfg, paths):
    id_col = cfg["data"]["id_column"]
    pred_dir = os.path.join(paths["results_root"], "predictions", model_name)
    met_dir = os.path.join(paths["results_root"], "metrics", model_name)
    cu.ensure_dirs(pred_dir, met_dir)

    df_pred = pd.DataFrame({
        "sentence_id": test_df[id_col].tolist(),
        "y_true": test_df[cfg["data"]["label_column"]].tolist(),
        "y_pred": test_preds,
    })
    for i in range(cfg["data"]["num_labels"]):
        df_pred[f"prob_{i}"] = [p[i] for p in test_probs]
    df_pred.to_csv(os.path.join(pred_dir, f"{fold_name}.csv"), index=False)

    with open(os.path.join(met_dir, f"{fold_name}.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


def finalize(model_name, fold_metrics, paths, num_labels):
    from metrics import average_metrics
    met_dir = os.path.join(paths["results_root"], "metrics", model_name)
    cu.ensure_dirs(met_dir)
    summary = average_metrics(fold_metrics, num_labels)
    with open(os.path.join(met_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
