from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone

import torch
from gliner import GLiNER

from project_utils import load_config, load_jsonl, resolve_path, write_json
from validate_data import validate_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GLiNER to extract classification signals.")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    train_path = resolve_path(config["data"]["train"])
    validation_path = resolve_path(config["data"]["validation"])
    validate_split(train_path, config["label"])
    validate_split(validation_path, config["label"])
    train_data = load_jsonl(train_path)
    validation_data = load_jsonl(validation_path)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. Run check_environment.py before training.")
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    use_bf16 = bool(training.get("bf16", True) and bf16_supported)
    if training.get("bf16", True) and not use_bf16:
        print("WARNING: BF16 is not supported; training will use FP32.")

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_name": config["model_name"],
        "label": config["label"],
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "bf16": use_bf16,
        "config": config,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    print(f"Loading base model: {config['model_name']}")
    model = GLiNER.from_pretrained(config["model_name"]).to(dtype=torch.float32)
    print(f"Training on {len(train_data)} examples; validation: {len(validation_data)}")
    print(f"Output: {output_dir}")
    model.train_model(
        train_dataset=train_data,
        eval_dataset=validation_data,
        output_dir=str(output_dir),
        max_steps=int(training["max_steps"]),
        lr_scheduler_type=training["scheduler_type"],
        warmup_steps=int(training["warmup_steps"]),
        per_device_train_batch_size=int(training["train_batch_size"]),
        per_device_eval_batch_size=int(training["eval_batch_size"]),
        learning_rate=float(training["learning_rate"]),
        others_lr=float(training["others_lr"]),
        weight_decay=float(training["weight_decay"]),
        others_weight_decay=float(training["others_weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        focal_loss_alpha=float(training["focal_loss_alpha"]),
        focal_loss_gamma=float(training["focal_loss_gamma"]),
        focal_loss_prob_margin=float(training["focal_loss_prob_margin"]),
        loss_reduction=training["loss_reduction"],
        negatives=float(training["negatives"]),
        masking=training["masking"],
        save_steps=int(training["save_steps"]),
        logging_steps=int(training["logging_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        bf16=use_bf16,
    )
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "training_complete.json", manifest)
    print("Training complete. Run evaluate.py on the saved checkpoint.")


if __name__ == "__main__":
    main()

