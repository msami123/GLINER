from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = "config.json") -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = resolve_path(path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {source}, line {line_number}: {exc}") from exc
    return rows


def write_json(path: str | Path, value: Any) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def latest_checkpoint(output_dir: str | Path) -> Path:
    root = resolve_path(output_dir)
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(
            f"No trained checkpoint found under {root}. Run train.py first, "
            "or pass --model with a checkpoint path."
        )
    return max(checkpoints, key=lambda item: item[0])[1]


def best_model(output_dir: str | Path) -> Path:
    root=resolve_path(output_dir)
    for path in (root/'best',root/'final',root):
        if (path/'gliner_config.json').exists(): return path
    return latest_checkpoint(root)


def file_hash(path: str | Path) -> str:
    digest=hashlib.sha256()
    with resolve_path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def tokens_to_text(tokens: Iterable[str]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if pieces:
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        offsets.append((start, cursor))
    return " ".join(pieces), offsets
