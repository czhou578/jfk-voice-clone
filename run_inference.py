#!/usr/bin/env python3
"""
F5-TTS Inference — Fine-tuned JFK Voice
========================================
Run this after fine-tuning to synthesize speech in JFK's voice.

Usage:
    # Single sentence
    python run_inference.py --text "Ask not what your country can do for you."

    # From a text file (one sentence per line, or paragraphs separated by blank lines)
    python run_inference.py --file speech.txt

    # With custom settings
    python run_inference.py --file speech.txt --checkpoint 2000 --nfe 64
"""

import argparse
import os
import re

import numpy as np
import soundfile as sf
import torch

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these paths
# ─────────────────────────────────────────────────────────────
F5_REPO        = os.path.abspath(os.path.join(os.path.dirname(__file__), "F5-TTS"))
CKPT_DIR       = "/root/F5-TTS-ckpts"
DATASET_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "jfk_dataset"))

# A short clean JFK clip to use as voice reference during inference
# Pick a 3-10 second clip from your dataset that sounds clear and representative
REFERENCE_CLIP = os.path.join(DATASET_DIR, "wavs/jfk_00002.wav")
REFERENCE_TEXT = "We choose to go to the Moon in this decade and do the other things."

OUTPUT_DIR     = "./jfk_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="F5-TTS Inference — Synthesize speech in JFK's voice"
    )
    # Text input: either --text or --file (one required)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize (single sentence)")
    text_group.add_argument("--file", help="Path to a text file with sentences/paragraphs")

    parser.add_argument("--checkpoint", default=None,  help="Checkpoint step number (e.g. 2000). Defaults to latest.")
    parser.add_argument("--ref_clip",   default=REFERENCE_CLIP, help="Path to reference WAV clip")
    parser.add_argument("--ref_text",   default=REFERENCE_TEXT, help="Transcript of reference clip")
    parser.add_argument("--output",     default="output.wav", help="Output filename (or prefix for multi-sentence)")
    parser.add_argument("--nfe",        type=int, default=32, help="NFE steps (higher = better quality, slower)")
    parser.add_argument("--pause",      type=float, default=0.5, help="Pause in seconds between sentences (default: 0.5)")
    parser.add_argument("--split",      action="store_true", help="Save each sentence as a separate WAV file")
    return parser.parse_args()


def read_text_file(filepath):
    """Read a text file and split into sentences.

    Supports two formats:
      1. One sentence per line
      2. Paragraphs separated by blank lines (auto-split into sentences)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Check if the file uses one-sentence-per-line format
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    # If most lines end with sentence-ending punctuation, treat as one-per-line
    ending_punct_count = sum(1 for l in lines if l and l[-1] in ".!?\"'")
    if ending_punct_count >= len(lines) * 0.5:
        return lines

    # Otherwise, join everything and split into sentences
    full_text = " ".join(lines)
    # Split on sentence-ending punctuation followed by a space
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    return [s.strip() for s in sentences if s.strip()]


def find_checkpoint(ckpt_dir: str, step: str = None) -> str:
    """Find the best checkpoint to use.

    Searches ckpt_dir recursively for model_last.pt files, picking the most
    recently modified one (skipping backup dirs). Handles nested structures like
    jfk_e30_b6000/jfk/ that arise from mv-into-existing-directory behaviour.
    """
    # Collect all .pt file paths under ckpt_dir, excluding backups
    all_pt_files = []
    for root, dirs, files in os.walk(ckpt_dir):
        # Skip backup directories
        dirs[:] = [d for d in dirs if "backup" not in d]
        for f in files:
            if f.endswith(".pt") and not f.startswith("pretrained_"):
                all_pt_files.append(os.path.join(root, f))

    if not all_pt_files:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")

    if step:
        matches = [p for p in all_pt_files if os.path.basename(p) == f"model_{step}.pt"]
        if matches:
            return matches[0]
        available = sorted(set(os.path.basename(p) for p in all_pt_files))
        raise FileNotFoundError(
            f"Checkpoint model_{step}.pt not found under {ckpt_dir}\n"
            f"    Available: {', '.join(available) or 'none'}"
        )

    # Prefer model_last.pt — pick the most recently modified one
    last_pts = [p for p in all_pt_files if os.path.basename(p) == "model_last.pt"]
    if last_pts:
        best = max(last_pts, key=os.path.getmtime)
        print(f"    Using checkpoint: {best}")
        return best

    # Fall back to highest-numbered checkpoint
    numbered = []
    for p in all_pt_files:
        name = os.path.basename(p)
        if name.startswith("model_") and name != "model_last.pt":
            try:
                numbered.append((int(name.replace("model_", "").replace(".pt", "")), p))
            except ValueError:
                pass
    if numbered:
        _, latest = max(numbered, key=lambda x: x[0])
        print(f"    Using checkpoint: {latest}")
        return latest

    raise FileNotFoundError(f"No usable checkpoints found under {ckpt_dir}")


def main():
    args = parse_args()

    # Determine input sentences
    if args.file:
        sentences = read_text_file(args.file)
        source_label = f"File: {args.file} ({len(sentences)} sentences)"
    else:
        sentences = [args.text]
        source_label = f"Text: {args.text}"

    print("=" * 50)
    print("  F5-TTS Inference — JFK Voice")
    print("=" * 50)
    print(f"  Input     : {source_label}")
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
        model="F5TTS_Base",
        ckpt_file=ckpt_path,
        device=device,
    )

    # Generate audio for each sentence
    all_wavs = []
    sr = None
    output_base = args.output.replace(".wav", "")

    for i, sentence in enumerate(sentences):
        label = f"[{i+1}/{len(sentences)}]" if len(sentences) > 1 else ""
        print(f"\n{label} Synthesizing: {sentence}")

        wav, sr, _ = tts.infer(
            ref_file=args.ref_clip,
            ref_text=args.ref_text,
            gen_text=sentence,
            nfe_step=args.nfe,
            cross_fade_duration=0.15,
            speed=1.0,
        )
        all_wavs.append(wav)

        # Save individual files if --split
        if args.split and len(sentences) > 1:
            split_path = os.path.join(OUTPUT_DIR, f"{output_base}_{i+1:03d}.wav")
            sf.write(split_path, wav, sr)
            print(f"    → Saved: {split_path}")

    # Concatenate all audio with pauses between sentences
    if len(all_wavs) > 1:
        pause_samples = int(args.pause * sr)
        silence = np.zeros(pause_samples, dtype=all_wavs[0].dtype)

        combined = []
        for i, wav in enumerate(all_wavs):
            combined.append(wav)
            if i < len(all_wavs) - 1:
                combined.append(silence)
        final_wav = np.concatenate(combined)
    else:
        final_wav = all_wavs[0]

    # Save combined output
    output_path = os.path.join(OUTPUT_DIR, args.output)
    sf.write(output_path, final_wav, sr)

    print(f"\n{'=' * 50}")
    print(f"  ✅ Saved to: {output_path}")
    print(f"     Sentences : {len(sentences)}")
    print(f"     Sample rate: {sr}Hz")
    print(f"     Duration: {len(final_wav)/sr:.2f}s")
    if args.split and len(sentences) > 1:
        print(f"     Individual files: {output_base}_001.wav — {output_base}_{len(sentences):03d}.wav")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()