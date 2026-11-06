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
    python run_inference.py --file speech.txt --checkpoint 6000 --nfe 64

    # Point at one checkpoint file, run on the CPU
    python run_inference.py --text "Hello." --ckpt_file checkpoints/jfk-f5tts-30epochs/model_last_ema.safetensors --device cpu
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
PROJECT_DIR    = os.path.abspath(os.path.dirname(__file__))
F5_REPO        = os.path.join(PROJECT_DIR, "F5-TTS")
# Checkpoints live under ./checkpoints by default, where finetune_ft_tts.sh
# also writes them. Set JFK_CKPT_DIR to search another folder.
CKPT_DIR       = os.environ.get("JFK_CKPT_DIR", os.path.join(PROJECT_DIR, "checkpoints"))
DATASET_DIR    = os.path.join(PROJECT_DIR, "jfk_dataset")

# A short clean JFK clip to use as voice reference during inference
# Pick a 3-10 second clip from your dataset that sounds clear and representative
REFERENCE_CLIP = os.path.join(DATASET_DIR, "wavs/jfk_00206.wav")
REFERENCE_TEXT = "And I therefore want to commend the members of the Congress and the House and the Senate, the Chairman of the subcommittees,"

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
    parser.add_argument("--ckpt_dir",   default=CKPT_DIR, help="Directory to search for checkpoints (default: ./checkpoints, or $JFK_CKPT_DIR).")
    parser.add_argument("--ckpt_file",  default=None, help="Exact checkpoint file (.safetensors or .pt). Skips the search.")
    parser.add_argument("--device",     default="auto", choices=["auto", "cuda", "cpu"], help="Compute device (default: cuda if available, else cpu)")
    parser.add_argument("--ref_clip",   default=REFERENCE_CLIP, help="Path to reference WAV clip")
    parser.add_argument("--ref_text",   default=REFERENCE_TEXT, help="Transcript of reference clip")
    parser.add_argument("--output",     default="output.wav", help="Output filename (or prefix for multi-sentence)")
    parser.add_argument("--nfe",        type=int, default=32, help="NFE steps (higher = better quality, slower)")
    parser.add_argument("--pause",      type=float, default=0.5, help="Pause in seconds between sentences (default: 0.5)")
    parser.add_argument("--split",      action="store_true", help="Save each sentence as a separate WAV file")
    parser.add_argument("--mode",       default="lines", choices=["lines", "sentences"],
                        help="'lines': one TTS call per line (good for lyrics). 'sentences': split on punctuation (good for prose/speeches).")
    return parser.parse_args()


def read_text_file(filepath, mode="lines"):
    """Read a text file and return a list of TTS units.

    mode='lines'     : one entry per non-empty line (good for lyrics).
    mode='sentences' : split on sentence-ending punctuation (good for prose/speeches).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if mode == "lines":
        return [line.strip() for line in content.splitlines() if line.strip()]

    # sentences mode: join non-empty lines within each paragraph, then split on .!?
    paragraphs = []
    current = []
    for line in content.splitlines():
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    sentences = []
    for para in paragraphs:
        parts = re.split(r'(?<=[.!?])\s+', para)
        sentences.extend(s.strip() for s in parts if s.strip())
    return sentences


def find_checkpoint(ckpt_dir: str, step: str = None) -> str:
    """Find the best checkpoint to use.

    Searches ckpt_dir recursively for checkpoint files (skipping backup dirs
    and pretrained_* files). Accepts .safetensors (slim EMA weights, see
    slim_checkpoint.py) and .pt (full training checkpoints).

    Order of preference without --checkpoint:
      1. model_last*.safetensors, most recently modified
      2. model_last.pt, most recently modified
      3. the highest numbered model_<step> file
    """
    found = []
    for root, dirs, files in os.walk(ckpt_dir):
        dirs[:] = [d for d in dirs if "backup" not in d]
        for f in files:
            if f.startswith("pretrained_"):
                continue
            if f.endswith(".pt") or f.endswith(".safetensors"):
                found.append(os.path.join(root, f))

    if not found:
        raise FileNotFoundError(
            f"No checkpoints found under {ckpt_dir}\n"
            f"    Download one first, see README.md 'Phase 3'."
        )

    def stem(path):
        name = os.path.basename(path)
        name = re.sub(r"\.(pt|safetensors)$", "", name)
        return re.sub(r"_ema$", "", name)

    if step:
        matches = [p for p in found if stem(p) == f"model_{step}"]
        if matches:
            matches.sort(key=lambda p: (not p.endswith(".safetensors"), -os.path.getmtime(p)))
            print(f"    Using checkpoint: {matches[0]}")
            return matches[0]
        available = sorted(set(os.path.basename(p) for p in found))
        raise FileNotFoundError(
            f"Checkpoint model_{step} not found under {ckpt_dir}\n"
            f"    Available: {', '.join(available) or 'none'}"
        )

    for ext in (".safetensors", ".pt"):
        last = [p for p in found if stem(p) == "model_last" and p.endswith(ext)]
        if last:
            best = max(last, key=os.path.getmtime)
            print(f"    Using checkpoint: {best}")
            return best

    numbered = []
    for p in found:
        m = re.fullmatch(r"model_(\d+)", stem(p))
        if m:
            numbered.append((int(m.group(1)), p))
    if numbered:
        _, latest = max(numbered, key=lambda x: x[0])
        print(f"    Using checkpoint: {latest}")
        return latest

    raise FileNotFoundError(f"No usable checkpoints found under {ckpt_dir}")


def main():
    args = parse_args()

    # Determine input sentences
    if args.file:
        sentences = read_text_file(args.file, mode=args.mode)
        source_label = f"File: {args.file} ({len(sentences)} units, mode={args.mode})"
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

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"  Device    : {device}")
    print("=" * 50)

    # Find checkpoint
    if args.ckpt_file:
        ckpt_path = args.ckpt_file
        print(f"    Using checkpoint: {ckpt_path}")
    else:
        ckpt_path = find_checkpoint(args.ckpt_dir, args.checkpoint)

    # Load fine-tuned model.
    # Build it inside a device context. Without this, F5-TTS initializes the
    # 337M weights with random values on the CPU and then moves them to the
    # GPU. On the DGX Spark that costs 4 s per run with free host memory,
    # and about 14 s when the host is short of memory.
    print("\nLoading fine-tuned model...")
    with torch.device(device):
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