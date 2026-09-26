"""Standalone, read-only inference over a persisted Phase 18 demo checkpoint.

Loads a checkpoint written by scripts/train_demo_checkpoint.py (a
state_dict plus a small JSON metadata file) and runs single-text
inference in eval/no-grad mode. Never modifies the checkpoint, the
dataset, or any experiment artifact. Never trains or fine-tunes anything.

The 11 class labels this project trains on come from the dataset's
`label_11` column (0..10). Human-readable names for these numbers are
NOT invented here -- they are the exact category names already recorded
in this project's own preprocessing provenance
(external/original-drive/DATASETS/PREPROCESSING/COGNITIVE DISTORTION/
configs/preprocessing.yaml, `label_map`, discovered and documented in the
Phase 6.1 reconnaissance). Label 1 merges three original categories
("Jumping to Conclusions", "Mind Reading", "Fortune-telling") into one
class; the project's own documentation names the merged class "Jumping to
Conclusions", which is the name used here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# Exact reverse mapping of the label_map already documented in
# external/original-drive/DATASETS/PREPROCESSING/COGNITIVE DISTORTION/
# configs/preprocessing.yaml (read-only project provenance; not modified,
# not copied verbatim here, just its already-established int->name pairing).
LABEL_NAMES: dict[int, str] = {
    0: "No Distortion",
    1: "Jumping to Conclusions",
    2: "Labeling",
    3: "Should statement",
    4: "Discounting the positives",
    5: "Mental filter",
    6: "Personalization and Blame",
    7: "Overgeneralization",
    8: "All-or-nothing",
    9: "Magnification or Minimization",
    10: "Emotional Reasoning",
}


class CheckpointLoadError(Exception):
    """Raised when a checkpoint directory is missing or malformed."""


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint_meta(checkpoint_dir: Path) -> dict[str, Any]:
    meta_path = checkpoint_dir / "checkpoint_meta.json"
    if not meta_path.is_file():
        raise CheckpointLoadError(f"missing checkpoint_meta.json under {checkpoint_dir}")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointLoadError(f"malformed checkpoint_meta.json: {exc}") from exc


class DemoModel:
    """Loads a checkpoint once and serves repeated inference calls.

    Import of `transformers`/`torch` model classes is deferred to
    ``__init__`` so that modules which only need ``LABEL_NAMES`` or the
    metadata loader (e.g. dashboard state-building, which must work even
    with no checkpoint present) never pay the cost of importing torch.
    """

    def __init__(self, checkpoint_dir: Path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.checkpoint_dir = checkpoint_dir
        self.meta = load_checkpoint_meta(checkpoint_dir)

        model_path = checkpoint_dir / "model.pt"
        if not model_path.is_file():
            raise CheckpointLoadError(f"missing model.pt under {checkpoint_dir}")

        self.device = resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.meta["hf_model_id"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.meta["hf_model_id"], num_labels=self.meta["num_labels"]
        )
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        """Run inference on one piece of text. Never modifies model weights."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        max_len = self.meta.get("max_len", 128)
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().tolist()

        ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        predicted_label = ranked[0]

        return {
            "predicted_label": predicted_label,
            "predicted_label_name": LABEL_NAMES.get(predicted_label, str(predicted_label)),
            "confidence": probs[predicted_label],
            "top_k": [
                {
                    "label": i,
                    "label_name": LABEL_NAMES.get(i, str(i)),
                    "probability": probs[i],
                }
                for i in ranked[:top_k]
            ],
            "model_name": self.meta["model_name"],
            "fold": self.meta["fold_name"],
        }
