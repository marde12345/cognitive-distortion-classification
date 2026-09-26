"""Standalone, deterministic inference over a persisted Phase 18 demo checkpoint.

Loads the checkpoint, tokenizes the given text the same way training did,
runs one forward pass in eval/no-grad mode, and prints the predicted
class (numeric label + the project's own documented category name),
confidence, and top-k alternatives. Never trains, never modifies the
checkpoint or the dataset.

Usage:
    uv run python scripts/run_inference.py \
        --checkpoint models/demo/indobert_base_p1/fold_0 \
        --text "some Indonesian text here"

This is a research demonstration of an experimental classifier. It is not
a clinical, diagnostic, or medically validated tool, and a prediction is
not a mental-health assessment of the person who wrote the text.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dashboard.inference import CheckpointLoadError, DemoModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        model = DemoModel(args.checkpoint)
    except CheckpointLoadError as exc:
        print(f"Could not load checkpoint at {args.checkpoint}: {exc}")
        sys.exit(1)

    try:
        result = model.predict(args.text, top_k=args.top_k)
    except ValueError as exc:
        print(f"Invalid input: {exc}")
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
