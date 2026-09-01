from __future__ import annotations

import argparse
from collections import Counter

import torch
from gliner import GLiNER
from tqdm import tqdm

from project_utils import latest_checkpoint, load_config, load_jsonl, resolve_path, tokens_to_text, write_json


def prediction_span(entity: dict, text: str) -> tuple[int, int, str]:
    start = int(entity["start"])
    end = int(entity["end"])
    entity_text = str(entity.get("text", ""))
    if text[start:end] != entity_text and text[start : end + 1] == entity_text:
        end += 1
    return start, end, str(entity["label"])


def score_dataset(model: GLiNER, rows: list[dict], label: str, threshold: float) -> dict:
    counts: Counter[str] = Counter()
    mistakes: list[dict] = []
    for row in tqdm(rows, desc="Evaluating"):
        tokens = row["tokenized_text"]
        text, offsets = tokens_to_text(tokens)
        gold = {
            (offsets[start][0], offsets[end][1], span_label)
            for start, end, span_label in row["ner"]
        }
        predicted_entities = model.predict_entities(text, [label], threshold=threshold)
        predicted = {prediction_span(entity, text) for entity in predicted_entities}
        true_positive = gold & predicted
        counts["tp"] += len(true_positive)
        counts["fp"] += len(predicted - gold)
        counts["fn"] += len(gold - predicted)
        if gold != predicted and len(mistakes) < 100:
            mistakes.append(
                {
                    "text": text,
                    "gold": sorted(gold),
                    "predicted": sorted(predicted),
                    "raw_predictions": predicted_entities,
                }
            )

    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "examples": len(rows),
        "true_positives": counts["tp"],
        "false_positives": counts["fp"],
        "false_negatives": counts["fn"],
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mistake_samples": mistakes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained GLiNER checkpoint.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--model", help="Checkpoint path; defaults to the latest checkpoint.")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = resolve_path(args.model) if args.model else latest_checkpoint(config["output_dir"])
    threshold = args.threshold if args.threshold is not None else float(config["evaluation"]["threshold"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_path}")
    model = GLiNER.from_pretrained(str(model_path), map_location=device)
    rows = load_jsonl(config["data"][args.split])
    report = score_dataset(model, rows, config["label"], threshold)
    report["model"] = str(model_path)
    report["split"] = args.split
    report_path = f"reports/{args.split}_threshold_{threshold:.2f}.json"
    write_json(report_path, report)
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall:    {report['recall']:.4f}")
    print(f"F1:        {report['f1']:.4f}")
    print(f"Report:    {resolve_path(report_path)}")


if __name__ == "__main__":
    main()

