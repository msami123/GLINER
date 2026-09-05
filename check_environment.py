from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"System: {platform.platform()}")
    try:
        import torch
    except ImportError:
        raise SystemExit("Install CUDA-enabled PyTorch using the official selector, then requirements.txt. See PC_HANDOFF.md.")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU was not detected. Check the NVIDIA driver and your PyTorch installation.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    print(f"BF16 supported: {bf16}")
    if not bf16:
        print("Set training.bf16 to false in config.json before training.")


if __name__ == "__main__":
    main()
