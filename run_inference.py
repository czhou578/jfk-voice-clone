#!/usr/bin/env python3
"""
F5-TTS Inference — Fine-tuned JFK Voice
========================================
Run this after fine-tuning to synthesize speech in JFK's voice.

Usage:
    # One sentence, or several in one process (the model loads once)
    python run_inference.py --text "Ask not what your country can do for you."
    python run_inference.py --text "First sentence." --text "Second sentence."

    # From a text file. Prose: one model call per paragraph. Lyrics: one per line.
    python run_inference.py --file speech.txt --mode paragraphs
    python run_inference.py --file lyrics.txt --mode lines

    # With custom settings
    python run_inference.py --file speech.txt --checkpoint 6000 --nfe 64

    # Point at one checkpoint file, run on the CPU
    python run_inference.py --text "Hello." --ckpt_file checkpoints/jfk-f5tts-30epochs/model_last_ema.safetensors --device cpu

    # Another reference clip from the dataset; the transcript comes from metadata.csv
    python run_inference.py --text "Hello." --ref_clip jfk_dataset/wavs/jfk_00001.wav

    # Reproducible output
    python run_inference.py --text "Hello." --seed 7
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
# Checkpoints live under ./checkpoints by default, where finetune_ft_tts.sh
# also writes them. Set JFK_CKPT_DIR to search another folder.
CKPT_DIR       = os.environ.get("JFK_CKPT_DIR", os.path.join(PROJECT_DIR, "checkpoints"))
DATASET_DIR    = os.path.join(PROJECT_DIR, "jfk_dataset")

# A short clean JFK clip to use as voice reference during inference.
# Pick a 3-10 second clip from the dataset that sounds clear and representative.
# The script reads its transcript from metadata.csv unless --ref_text is given.
REFERENCE_CLIP = os.path.join(DATASET_DIR, "wavs/jfk_00206.wav")
METADATA_CSV   = os.path.join(DATASET_DIR, "metadata.csv")

OUTPUT_DIR     = os.path.join(PROJECT_DIR, "jfk_outputs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="F5-TTS Inference — Synthesize speech in JFK's voice"
    )
    # Text input: either --text or --file (one required)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", action="append", help="Text to synthesize. Repeat the flag for several sentences in one run.")
    text_group.add_argument("--file", help="Path to a text file with sentences/paragraphs")

    parser.add_argument("--checkpoint", default=None,  help="Checkpoint step number (e.g. 2000). Defaults to latest.")
    parser.add_argument("--ckpt_dir",   default=CKPT_DIR, help="Directory to search for checkpoints (default: ./checkpoints, or $JFK_CKPT_DIR).")
    parser.add_argument("--ckpt_file",  default=None, help="Exact checkpoint file (.safetensors or .pt). Skips the search.")
    parser.add_argument("--device",     default="auto", choices=["auto", "cuda", "cpu"], help="Compute device (default: cuda if available, else cpu)")
    parser.add_argument("--ref_clip",   default=REFERENCE_CLIP, help="Path to reference WAV clip")
    parser.add_argument("--ref_text",   default=None, help="Transcript of the reference clip (default: looked up in jfk_dataset/metadata.csv)")
    parser.add_argument("--output",     default="output.wav", help="Output filename (or prefix for multi-sentence)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Folder for the output files (default: jfk_outputs/ in the repo)")
    parser.add_argument("--seed",       type=int, default=None, help="Random seed for reproducible output (default: a new random seed per sentence)")
    parser.add_argument("--nfe",        type=int, default=32, help="Flow-matching steps: 16 for drafts, 32 default, 64 for final renders (default: 32)")
    parser.add_argument("--pause",      type=float, default=0.5, help="Pause in seconds between sentences (default: 0.5)")
    parser.add_argument("--split",      action="store_true", help="Save each sentence as a separate WAV file")
    parser.add_argument("--mode",       default="lines", choices=["lines", "sentences", "paragraphs"],
                        help="How --file is split into model calls. 'lines': one call per line (lyrics). "
                             "'paragraphs': one call per blank-line-separated paragraph; F5-TTS chunks and cross-fades inside it (prose, natural joins). "
                             "'sentences': one call per sentence, with --pause between them (prose with fixed pauses).")
    return parser.parse_args()


def read_text_file(filepath, mode="lines"):
    """Read a text file and return a list of TTS units, one model call each.

    mode='lines'      : one entry per non-empty line (lyrics).
    mode='paragraphs' : one entry per paragraph; paragraphs are separated by
                        blank lines, and their lines are joined (prose).
                        F5-TTS splits a long paragraph into chunks of about
                        200 characters and cross-fades them, so this mode
                        gives natural joins instead of fixed pauses.
    mode='sentences'  : one entry per sentence, split on .!? (prose, when a
                        fixed --pause between sentences is wanted).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if mode == "lines":
        return [line.strip() for line in content.splitlines() if line.strip()]

    # paragraphs and sentences: join non-empty lines within each paragraph
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

    if mode == "paragraphs":
        return paragraphs

    sentences = []
    for para in paragraphs:
        parts = re.split(r'(?<=[.!?])\s+', para)
        sentences.extend(s.strip() for s in parts if s.strip())
    return sentences


def lookup_ref_text(ref_clip: str, metadata_csv: str = METADATA_CSV) -> str:
    """Return the transcript of a dataset clip from the pipe-delimited metadata.csv.

    Matches on the file name. Raises SystemExit with a hint when the clip is
    not in the dataset, because a wrong transcript degrades the output.
    """
    clip_name = os.path.basename(ref_clip)
    if os.path.isfile(metadata_csv):
        with open(metadata_csv, "r", encoding="utf-8") as f:
            for line in f:
                name, sep, text = line.rstrip("\n").partition("|")
                if sep and os.path.basename(name) == clip_name and text.strip():
                    return text.strip()
    raise SystemExit(
        f"No transcript for {clip_name} in {metadata_csv}.\n"
        f"    Pass --ref_text with the exact transcript of the reference clip."
    )


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
        sentences = [t.strip() for t in args.text if t.strip()]
        source_label = f"Text: {' | '.join(sentences)}"
    if not sentences:
        raise SystemExit(f"No text to synthesize in {args.file or '--text'}.")

    ref_text = args.ref_text if args.ref_text else lookup_ref_text(args.ref_clip)

    # Output names: <base>.wav, and <base>_001.wav ... with --split
    output_base = os.path.splitext(os.path.basename(args.output))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("  F5-TTS Inference — JFK Voice")
    print("=" * 50)
    print(f"  Input     : {source_label}")
    print(f"  Reference : {args.ref_clip}")
    print(f"  Ref text  : {ref_text}")
    print(f"  NFE steps : {args.nfe}")
    print(f"  Seed      : {args.seed if args.seed is not None else 'random'}")

    # F5-TTS comes from the editable install of the submodule (README step 3).
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

    for i, sentence in enumerate(sentences):
        label = f"[{i+1}/{len(sentences)}]" if len(sentences) > 1 else ""
        print(f"\n{label} Synthesizing: {sentence}")

        wav, sr, _ = tts.infer(
            ref_file=args.ref_clip,
            ref_text=ref_text,
            gen_text=sentence,
            nfe_step=args.nfe,
            cross_fade_duration=0.15,
            speed=1.0,
            seed=args.seed,
        )
        print(f"    seed: {tts.seed}")
        all_wavs.append(wav)

        # Save individual files if --split
        if args.split and len(sentences) > 1:
            split_path = os.path.join(args.output_dir, f"{output_base}_{i+1:03d}.wav")
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
    output_path = os.path.join(args.output_dir, f"{output_base}.wav")
    sf.write(output_path, final_wav, sr)

    print(f"\n{'=' * 50}")
    print(f"  ✅ Saved to: {output_path}")
    print(f"     Units     : {len(sentences)}")
    print(f"     Sample rate: {sr}Hz")
    print(f"     Duration: {len(final_wav)/sr:.2f}s")
    if args.split and len(sentences) > 1:
        print(f"     Individual files: {output_base}_001.wav — {output_base}_{len(sentences):03d}.wav")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()