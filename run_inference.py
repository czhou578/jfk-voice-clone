#!/usr/bin/env python3
"""
F5-TTS Inference — Fine-tuned JFK Voice
========================================
Run this after fine-tuning to synthesize speech in JFK's voice.

Usage:
    python run_inference.py --text "Ask not what your country can do for you."
    python run_inference.py --text "We choose to go to the Moon." --checkpoint 5000
"""

import argparse
import os
import soundfile as sf
import torch

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these paths
# ─────────────────────────────────────────────────────────────
F5_REPO        = os.path.expanduser("~/F5-TTS")
CKPT_DIR       = os.path.join(F5_REPO, "ckpts/F5TTS_Base")
DATASET_DIR    = os.path.expanduser("~/jfk_dataset")

# A short clean JFK clip to use as voice reference during inference
# Pick a 3-10 second clip from your dataset that sounds clear and representative
REFERENCE_CLIP = os.path.join(DATASET_DIR, "wavs/jfk_00001.wav")
REFERENCE_TEXT = "We choose to go to the Moon in this decade and do the other things."

OUTPUT_DIR     = "./jfk_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",       required=True, help="Text to synthesize")
    parser.add_argument("--checkpoint", default=None,  help="Checkpoint step number (e.g. 5000). Defaults to latest.")
    parser.add_argument("--ref_clip",   default=REFERENCE_CLIP, help="Path to reference WAV clip")
    parser.add_argument("--ref_text",   default=REFERENCE_TEXT, help="Transcript of reference clip")
    parser.add_argument("--output",     default="output.wav", help="Output filename")
    parser.add_argument("--nfe",        type=int, default=32, help="NFE steps (higher = better quality, slower)")
    return parser.parse_args()


def find_checkpoint(ckpt_dir: str, step: str = None) -> str:
    """Find the best checkpoint to use."""
    if step:
        path = os.path.join(ckpt_dir, f"model_{step}.pt")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Find latest checkpoint automatically
    pts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("model_") and f.endswith(".pt")],
        key=lambda x: int(x.replace("model_", "").replace(".pt", ""))
    )
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    latest = os.path.join(ckpt_dir, pts[-1])
    print(f"    Using checkpoint: {pts[-1]}")
    return latest


def main():
    args = parse_args()

    print("=" * 50)
    print("  F5-TTS Inference — JFK Voice")
    print("=" * 50)
    print(f"  Text      : {args.text}")
    print(f"  Reference : {args.ref_clip}")
    print(f"  NFE steps : {args.nfe}")

    import sys
    sys.path.insert(0, os.path.join(F5_REPO, "src"))

    from f5_tts.api import F5TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device    : {device}")
    print("=" * 50)

    # Find checkpoint
    ckpt_path = find_checkpoint(CKPT_DIR, args.checkpoint)

    # Load fine-tuned model
    print("\nLoading fine-tuned model...")
    tts = F5TTS(
        model_type="F5-TTS",
        ckpt_file=ckpt_path,
        device=device,
    )

    # Run inference
    print("Synthesizing...")
    wav, sr, _ = tts.infer(
        ref_file=args.ref_clip,
        ref_text=args.ref_text,
        gen_text=args.text,
        nfe_step=args.nfe,
        cross_fade_duration=0.15,
        speed=1.0,
    )

    # Save output
    output_path = os.path.join(OUTPUT_DIR, args.output)
    sf.write(output_path, wav, sr)

    print(f"\n✅ Saved to: {output_path}")
    print(f"   Sample rate: {sr}Hz")
    print(f"   Duration: {len(wav)/sr:.2f}s")


if __name__ == "__main__":
    main()