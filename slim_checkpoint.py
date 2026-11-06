#!/usr/bin/env python3
"""
Slim an F5-TTS training checkpoint for inference.

A training checkpoint (model_last.pt, model_6000.pt, ...) holds the model,
the EMA copy of the model, and the optimizer state. It is about 5 GB.
Inference needs only the EMA weights, about 1.3 GB.

This script memory-maps the .pt file, so it does not load 5 GB into RAM.
It writes the EMA weights as a .safetensors file that run_inference.py
and the F5-TTS loader accept as a checkpoint.

Usage:
    python slim_checkpoint.py checkpoints/jfk-f5tts-30epochs/model_last.pt
    # writes checkpoints/jfk-f5tts-30epochs/model_last_ema.safetensors
"""

import argparse
import os

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Extract EMA weights from an F5-TTS training checkpoint.")
    parser.add_argument("checkpoint", help="Path to a training checkpoint (.pt)")
    parser.add_argument("--output", help="Output .safetensors path (default: <name>_ema.safetensors next to the input)")
    args = parser.parse_args()

    src = args.checkpoint
    dst = args.output or os.path.splitext(src)[0] + "_ema.safetensors"

    print(f"Reading  {src}  ({os.path.getsize(src) / 2**30:.2f} GiB, memory-mapped)")
    ckpt = torch.load(src, map_location="cpu", weights_only=True, mmap=True)

    if "ema_model_state_dict" not in ckpt:
        raise SystemExit("This checkpoint has no ema_model_state_dict. Is it a training checkpoint?")

    ema = ckpt["ema_model_state_dict"]
    step = ema.get("step")
    if step is not None:
        print(f"EMA step : {int(step)}")

    # Keep tensors only. The loader drops 'initted' and 'step' itself,
    # but safetensors cannot store non-tensor values, so drop them here.
    tensors = {k: v.contiguous() for k, v in ema.items() if k not in ("initted", "step") and torch.is_tensor(v)}

    save_file(tensors, dst)
    print(f"Wrote    {dst}  ({os.path.getsize(dst) / 2**30:.2f} GiB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
