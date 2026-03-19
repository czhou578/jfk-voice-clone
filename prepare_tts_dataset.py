#!/usr/bin/env python3
"""
JFK Voice TTS Dataset Preparation Pipeline
==========================================
Tools used:
  - yt-dlp        : Download audio from YouTube
  - resemble-enhance : Denoise / remove crowd noise & applause
  - WhisperX      : VAD, segmentation, and transcription

Usage:
  python prepare_tts_dataset.py --url "https://www.youtube.com/watch?v=hN7Nu6Ym40E" --output ./jfk_dataset --speaker jfk --model large-v2

Requirements (install before running):
  pip install yt-dlp resemble-enhance whisperx torch torchaudio soundfile
  git lfs install
"""

import os
import sys
import argparse
import subprocess
import csv
import shutil
import soundfile as sf

# ─────────────────────────────────────────────────────────────
# STEP 0: Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TTS Dataset Preparation Pipeline")
    parser.add_argument("--url",        required=True,  help="YouTube video URL")
    parser.add_argument("--output",     default="./jfk_dataset", help="Output directory for final dataset")
    parser.add_argument("--speaker",    default="jfk",  help="Speaker name (used in filenames)")
    parser.add_argument("--model",      default="large-v2", help="Whisper model size (tiny/base/small/medium/large-v2)")
    parser.add_argument("--language",   default="en",   help="Language code (default: en)")
    parser.add_argument("--min_sec",    type=float, default=3.0,  help="Minimum clip duration in seconds")
    parser.add_argument("--max_sec",    type=float, default=15.0, help="Maximum clip duration in seconds")
    parser.add_argument("--hf_token",   default=None,
        help="HuggingFace token (required for WhisperX diarization). Get one at https://huggingface.co/settings/tokens")
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────
# STEP 1: Download audio from YouTube
# ─────────────────────────────────────────────────────────────

def download_audio(url: str, out_path: str) -> str:
    """Download best audio from YouTube, convert to 16kHz mono WAV."""
    print("\n[1/4] Downloading audio from YouTube...")
    wav_path = os.path.join(out_path, "raw_download.wav")
    if os.path.exists(wav_path):
        print(f"    ✓ Found existing raw audio at: {wav_path}, skipping download.")
        return wav_path

    os.makedirs(out_path, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "-ar 16000 -ac 1",  # 16kHz mono
        "-o", wav_path,
        url
    ]
    result = subprocess.run(cmd, check=True)
    print(f"    ✓ Downloaded to: {wav_path}")
    return wav_path

# ─────────────────────────────────────────────────────────────
# STEP 2: Denoise with resemble-enhance
# ─────────────────────────────────────────────────────────────

def denoise_audio(input_wav: str, out_path: str) -> str:
    """Remove crowd noise, applause, and background noise."""
    print("\n[2/4] Denoising audio with resemble-enhance...")

    from resemble_enhance.enhancer.inference import denoise
    import torch
    import torchaudio

    denoised_dir = os.path.join(out_path, "denoised")
    os.makedirs(denoised_dir, exist_ok=True)
    denoised_path = os.path.join(denoised_dir, "denoised.wav")

    if os.path.exists(denoised_path):
        print(f"    ✓ Found existing denoised audio at: {denoised_path}, skipping denoise.")
        return denoised_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Using device: {device}")

    dwav, sr = torchaudio.load(input_wav)
    dwav = dwav.mean(dim=0)  # ensure mono

    enhanced, out_sr = denoise(dwav, sr, device)
    torchaudio.save(denoised_path, enhanced.unsqueeze(0).cpu(), out_sr)

    print(f"    ✓ Denoised audio saved to: {denoised_path}")
    return denoised_path

# ─────────────────────────────────────────────────────────────
# STEP 3: VAD + Transcription with WhisperX
# ─────────────────────────────────────────────────────────────

def transcribe_and_segment(input_wav: str, out_path: str, args) -> list:
    """
    Run WhisperX to:
      - Apply VAD (automatically strips silences and non-speech)
      - Transcribe with word-level timestamps
      - Return list of segments: {start, end, text}
    """
    print("\n[3/4] Running WhisperX (VAD + transcription)...")

    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    # Load model
    model = whisperx.load_model(
        args.model,
        device,
        compute_type=compute_type,
        language=args.language
    )

    # Load audio (WhisperX handles its own resampling)
    audio = whisperx.load_audio(input_wav)

    # Transcribe (VAD filtering is ON by default — silences are skipped)
    result = model.transcribe(audio, batch_size=16, language=args.language)
    print(f"    ✓ Transcription complete. Detected {len(result['segments'])} raw segments.")

    # Align to get word-level timestamps
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False
    )

    return result["segments"]

# ─────────────────────────────────────────────────────────────
# STEP 4: Slice audio + build metadata.csv
# ─────────────────────────────────────────────────────────────

def slice_and_export(denoised_wav: str, segments: list, out_path: str, args) -> int:
    """
    Slice denoised audio into clips using WhisperX segment timestamps.
    Filter clips by min/max duration. Export wavs + metadata.csv.
    """
    print("\n[4/4] Slicing audio and building dataset...")

    wavs_dir = os.path.join(out_path, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    audio_data, sample_rate = sf.read(denoised_wav)

    metadata_path = os.path.join(out_path, "metadata.csv")
    kept = 0
    skipped = 0

    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        writer.writerow(["file", "text"])  # header

        for i, seg in enumerate(segments):
            start = seg.get("start", None)
            end   = seg.get("end",   None)
            text  = seg.get("text",  "").strip()

            # Skip segments missing timestamps or text
            if start is None or end is None or not text:
                skipped += 1
                continue

            duration = end - start

            # Filter by duration
            if duration < args.min_sec or duration > args.max_sec:
                skipped += 1
                continue

            # Slice audio
            start_sample = int(start * sample_rate)
            end_sample   = int(end   * sample_rate)
            clip = audio_data[start_sample:end_sample]

            # Save clip
            clip_name = f"{args.speaker}_{i:05d}.wav"
            clip_path = os.path.join(wavs_dir, clip_name)
            sf.write(clip_path, clip, sample_rate)

            writer.writerow([clip_name, text])
            kept += 1

    print(f"    ✓ Kept {kept} clips | Skipped {skipped} clips")
    print(f"    ✓ Dataset saved to: {out_path}")
    print(f"    ✓ Metadata CSV: {metadata_path}")
    return kept

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    workdir    = os.path.join(args.output, "_work")
    final_dir  = args.output
    os.makedirs(workdir,   exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    print("=" * 55)
    print("  TTS Dataset Preparation Pipeline")
    print("=" * 55)
    print(f"  URL     : {args.url}")
    print(f"  Output  : {final_dir}")
    print(f"  Speaker : {args.speaker}")
    print(f"  Model   : {args.model}")
    print(f"  Clips   : {args.min_sec}s – {args.max_sec}s")
    print("=" * 55)

    # Run pipeline
    raw_wav      = download_audio(args.url, workdir)
    denoised_wav = denoise_audio(raw_wav, workdir)
    segments     = transcribe_and_segment(denoised_wav, workdir, args)
    total_clips  = slice_and_export(denoised_wav, segments, final_dir, args)

    print("\n" + "=" * 55)
    print(f"  ✅ Done! {total_clips} clips ready for TTS fine-tuning.")
    print(f"  Dataset: {final_dir}/wavs/")
    print(f"  Labels:  {final_dir}/metadata.csv")
    print("=" * 55)
    print("\n  Next step: point your TTS trainer (Coqui/XTTS/Piper)")
    print(f"  at: {final_dir}")

if __name__ == "__main__":
    main()
