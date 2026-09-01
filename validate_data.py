from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from project_utils import PROJECT_ROOT, load_config, load_jsonl, resolve_path, write_json


def fingerprint(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_split(path: Path, expected_label: str) -> tuple[dict, set[str]]:
    rows = load_jsonl(path)
    errors: list[str] = []
    labels: Counter[str] = Counter()
    token_lengths: list[int] = []
    fingerprints: list[str] = []
    span_count = 0

    for row_number, row in enumerate(rows, start=1):
        tokens = row.get("tokenized_text")
        spans = row.get("ner")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            errors.append(f"row {row_number}: tokenized_text must be a list of strings")
            continue
        if not tokens:
            errors.append(f"row {row_number}: tokenized_text is empty")
        if not isinstance(spans, list) or not spans:
            errors.append(f"row {row_number}: ner must contain at least one span")
            continue
        token_lengths.append(len(tokens))
        fingerprints.append(fingerprint(row))
        for span in spans:
            span_count += 1
            if not isinstance(span, list) or len(span) != 3:
                errors.append(f"row {row_number}: invalid span {span!r}")
                continue
            start, end, label = span
            if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start <= end < len(tokens)):
                errors.append(f"row {row_number}: span out of bounds {span!r}")
            if label != expected_label:
                errors.append(f"row {row_number}: unexpected label {label!r}")
            labels[str(label)] += 1

    duplicate_count = len(fingerprints) - len(set(fingerprints))
    if duplicate_count:
        errors.append(f"{duplicate_count} duplicate rows")
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Validation failed for {path}:\n{preview}")

    try:
        display_path = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(path)
    report = {
        "file": display_path,
        "examples": len(rows),
        "spans": span_count,
        "labels": dict(labels),
        "min_tokens": min(token_lengths),
        "max_tokens": max(token_lengths),
        "average_tokens": round(sum(token_lengths) / len(token_lengths), 2),
        "duplicates": duplicate_count,
    }
    return report, set(fingerprints)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all GLiNER training splits.")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    reports: dict[str, dict] = {}
    seen: dict[str, set[str]] = {}
    for split, value in config["data"].items():
        path = resolve_path(value)
        reports[split], seen[split] = validate_split(path, config["label"])
        print(f"OK {split}: {reports[split]['examples']} examples, {reports[split]['spans']} spans")

    split_names = list(seen)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = seen[left] & seen[right]
            if overlap:
                raise ValueError(f"Data leakage: {len(overlap)} identical rows in {left} and {right}")

    reports["split_overlap"] = 0
    write_json("reports/data_validation.json", reports)
    print("OK: no duplicate rows and no overlap between train/validation/test")


if __name__ == "__main__":
    main()
