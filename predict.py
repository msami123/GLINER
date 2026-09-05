from __future__ import annotations

import argparse
import json

import torch
from gliner import GLiNER

from project_utils import best_model, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract classification signals from one issue string.")
    parser.add_argument("issue", help="Raw transaction issue text.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", help="Checkpoint path; defaults to the latest checkpoint.")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = resolve_path(args.model) if args.model else best_model(config["output_dir"])
    threshold = args.threshold if args.threshold is not None else float(config["evaluation"]["threshold"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(str(model_path), map_location=device)
    entities = model.predict_entities(args.issue, [config["label"]], threshold=threshold)
    print(json.dumps(entities, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
