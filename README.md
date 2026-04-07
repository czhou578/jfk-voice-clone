# JFK Voice Clone — F5-TTS Fine-Tuning Pipeline

Clone JFK's voice using [F5-TTS](https://github.com/SWivid/F5-TTS). This repo contains everything needed to go from a raw YouTube speech to a fully fine-tuned TTS model that speaks in JFK's voice.

## Overview

The pipeline has **three phases**:

| Phase | Script | What it does |
|-------|--------|--------------|
| **1. Dataset Preparation** | `prepare_tts_dataset.py` | Downloads audio → denoises → transcribes → slices into labeled clips |
| **2. Fine-Tuning** | `finetune_ft_tts.sh` | Converts dataset for F5-TTS, configures training, and launches fine-tuning |
| **3. Inference** | `run_inference.py` | Synthesizes new speech using your fine-tuned model |

---

## Prerequisites

- **GPU**: NVIDIA GPU with ≥24 GB VRAM (tested on RTX 4090). Training will also work on cloud GPUs (RunPod, etc.)
- **OS**: Linux (Ubuntu recommended). The training scripts use `apt-get` and bash.
- **Python**: 3.10 or 3.11
- **CUDA**: 12.1+
- **Storage**: ≥40 GB free (RunPod 40 GB volume recommended — smaller volumes will run out of space)

> **Lessons learned:** RunPod with a 40 GB volume is the minimum viable setup. Smaller storage will cause failures during training.

---

## Phase 1 — Dataset Preparation

This phase takes a YouTube video of JFK speaking and produces a clean, segmented dataset of labeled audio clips ready for TTS fine-tuning.

### What the pipeline does internally

1. **Downloads** audio from YouTube via `yt-dlp` (converted to 16 kHz mono WAV)
2. **Denoises** the audio with `resemble-enhance` (removes crowd noise, applause, background)
3. **Transcribes + segments** with `WhisperX` (VAD strips silences, produces word-aligned timestamps)
4. **Slices** the denoised audio into 3–15 second clips and writes a `metadata.csv`

### Step 1.1 — Clone this repository

```bash
git clone --recurse-submodules https://github.com/czhou578/jfk-voice-clone.git
cd jfk-voice-clone
```

> The `--recurse-submodules` flag is important — it pulls in the [F5-TTS fork](https://github.com/czhou578/F5-TTS) as a submodule.

### Step 1.2 — Install system dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    pkg-config ffmpeg git-lfs \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

```bash
git lfs install
git lfs pull     # Downloads any LFS-tracked WAV/model files
```

### Step 1.3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

The key packages are:
- `yt-dlp` — YouTube audio download
- `resemble-enhance` — Audio denoising
- `whisperx` — VAD + transcription + alignment
- `soundfile` — Audio I/O

### Step 1.4 — Run the dataset preparation script

```bash
python prepare_tts_dataset.py \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output ./jfk_dataset \
    --speaker jfk \
    --model large-v2
```

**All available arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | *(required)* | YouTube video URL |
| `--output` | `./jfk_dataset` | Output directory |
| `--speaker` | `jfk` | Speaker name (used in filenames) |
| `--model` | `large-v2` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`) |
| `--language` | `en` | Language code |
| `--min_sec` | `3.0` | Minimum clip duration (seconds) |
| `--max_sec` | `15.0` | Maximum clip duration (seconds) |
| `--hf_token` | `None` | HuggingFace token (for WhisperX diarization) |

### Step 1.5 — Verify the dataset

After the script completes, your `jfk_dataset/` folder should look like:

```
jfk_dataset/
├── metadata.csv          # Pipe-delimited: filename|transcript
├── wavs/
│   ├── jfk_00000.wav
│   ├── jfk_00001.wav
│   ├── jfk_00002.wav
│   └── ... (hundreds of clips)
└── _work/                # Intermediate files (raw download, denoised audio)
    ├── raw_download.wav
    └── denoised/
        └── denoised.wav
```

Spot-check a few clips and their transcripts in `metadata.csv` to make sure quality looks good:

```bash
head -5 jfk_dataset/metadata.csv
```

Expected format:
```
audio_file|text
jfk_00000.wav|Remarks to the Fair Employment Practices Committee...
jfk_00001.wav|The President's Committee on Equal Employment Opportunity...
```

> **Tip:** To build a larger dataset, run the script multiple times with different `--url` values pointing to different JFK speeches. The clips will accumulate in the same `jfk_dataset/wavs/` directory (just make sure not to overwrite `metadata.csv` — merge them manually).

### Alternative: Google Colab

If you don't have a local GPU for dataset preparation, use the included Colab notebook:

1. Upload `prepare_tts_colab.ipynb` to Google Colab
2. Set runtime to **GPU (T4)**
3. Run all cells in order — the notebook installs dependencies, writes the script, and runs the pipeline
4. Download the resulting `jfk_dataset/` folder

---

## Phase 2 — Fine-Tuning F5-TTS

This phase takes the prepared dataset and fine-tunes the F5-TTS base model on JFK's voice.

> **Important:** This should be run on a machine with a powerful GPU. An RTX 4090 (24 GB VRAM) or equivalent cloud GPU is recommended.

### Step 2.1 — Configure paths

Open `finetune_ft_tts.sh` and edit the config section at the top to match your environment:

```bash
DATASET_PATH="/workspace/jfk-voice-clone/jfk_dataset"   # Where your dataset lives
F5_REPO="/workspace/jfk-voice-clone/F5-TTS"              # F5-TTS submodule path
CONDA_ENV="f5-tts"                                        # Virtual environment name
EPOCHS=10                                                  # 10 to start, 20-30 for better accent
LEARNING_RATE=1e-5
BATCH_SIZE=4000                                            # Safe for 24 GB VRAM
GRAD_ACCUM=1
```

### Step 2.2 — Run the fine-tuning script

```bash
chmod +x finetune_ft_tts.sh
./finetune_ft_tts.sh
```

### What the script does (6 steps)

Here's exactly what happens when you run it:

#### [1/6] Validates F5-TTS repo
Checks that the F5-TTS submodule exists at the configured path.

#### [2/6] Sets up Python environment
- Installs system packages (`python3-venv`, `ffmpeg`)
- Creates a Python virtual environment
- Installs PyTorch with CUDA 12.1 support
- Installs F5-TTS and its dependencies (`pip install -e .`)
- Installs `accelerate` and `tensorboard`

#### [3/6] Prepares the dataset
- Copies `wavs/` and `metadata.csv` into F5-TTS's expected `data/jfk/` directory
- Converts the metadata to use **absolute paths** (F5-TTS requirement):
  ```
  # Before (your format)
  jfk_00000.wav|Some transcript text

  # After (F5-TTS format)
  /workspace/jfk-voice-clone/F5-TTS/data/jfk/wavs/jfk_00000.wav|Some transcript text
  ```

#### [4/6] Runs F5-TTS dataset preparation
- Executes `prepare_csv_wavs.py` to convert the CSV + WAVs into:
  - `raw.arrow` — Efficient binary dataset format
  - `duration.json` — Audio duration metadata
  - `vocab.txt` — Character vocabulary

#### [5/6] Configures accelerate
Writes a single-GPU config file for HuggingFace Accelerate (fp16 mixed precision, 1 GPU).

#### [6/6] Launches fine-tuning
Runs the actual training with these settings:

| Setting | Value |
|---------|-------|
| Base model | `F5TTS_Base` |
| Learning rate | `1e-5` |
| Batch size | `4000` (frame-based) |
| Epochs | `10` |
| Warmup steps | `200` |
| Save every | `1000` updates |
| Mixed precision | `fp16` |
| Logger | TensorBoard |

**Expected training time:** 2–4 hours on an RTX 4090 for ~1150 clips at 10 epochs.

### Step 2.3 — Monitor training (optional)

In a second terminal, run:

```bash
tensorboard --logdir /workspace/jfk-voice-clone/F5-TTS/ckpts/F5TTS_Base/logs
```

### Step 2.4 — Find your checkpoints

After training, checkpoints are saved to:

```
F5-TTS/ckpts/F5TTS_Base/
├── model_1000.pt
├── model_2000.pt
├── model_3000.pt
└── ...
```

---

## Phase 3 — Inference

Generate speech in JFK's voice using your fine-tuned model.

### Step 3.1 — Pick a reference clip

The inference script needs a short (3–10 second), clean audio clip of JFK as a voice reference. By default, it uses `jfk_dataset/wavs/jfk_00001.wav`.

Open `run_inference.py` and verify / update these constants if needed:

```python
REFERENCE_CLIP = os.path.join(DATASET_DIR, "wavs/jfk_00001.wav")
REFERENCE_TEXT = "We choose to go to the Moon in this decade and do the other things."
```

> **Important:** The `REFERENCE_TEXT` must be an accurate transcript of the reference clip. Pick a clip from your dataset and copy its transcript from `metadata.csv`.

### Step 3.2 — Run inference

Basic usage (uses the latest checkpoint automatically):
```bash
python run_inference.py --text "Ask not what your country can do for you."
```

Use a specific checkpoint:
```bash
python run_inference.py \
    --text "We choose to go to the Moon." \
    --checkpoint 5000
```

**All available arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--text` | *(required)* | Text to synthesize |
| `--checkpoint` | latest | Checkpoint step number (e.g. `5000`) |
| `--ref_clip` | `jfk_dataset/wavs/jfk_00001.wav` | Path to reference WAV clip |
| `--ref_text` | *(see script)* | Transcript of reference clip |
| `--output` | `output.wav` | Output filename |
| `--nfe` | `32` | NFE steps (higher = better quality, slower) |

### Step 3.3 — Find your output

Generated audio is saved to:
```
jfk_outputs/output.wav
```

---

## Repository Structure

```
jfk-voice-clone/
├── README.md                    # This file
├── plan.md                      # Setup notes (system deps, Git LFS)
├── requirements.txt             # Python dependencies for dataset prep
├── .gitattributes               # Git LFS tracking (*.wav, *.pt, *.safetensors)
├── .gitmodules                  # F5-TTS submodule reference
│
├── prepare_tts_dataset.py       # Phase 1: Dataset preparation pipeline
├── prepare_tts_colab.ipynb      # Phase 1: Colab notebook version
├── update_nb.py                 # Utility: updates Colab notebook deps
│
├── finetune_ft_tts.sh           # Phase 2: Fine-tuning script
├── run_inference.py             # Phase 3: Inference script
│
├── F5-TTS/                      # Git submodule — F5-TTS fork
│
└── jfk_dataset/                 # Generated dataset (tracked via Git LFS)
    ├── metadata.csv             # ~1150 clips, pipe-delimited
    └── wavs/                    # Audio clips (3–15 sec each)
```

---

## Quick Start (TL;DR)

```bash
# 1. Clone
git clone --recurse-submodules https://github.com/czhou578/jfk-voice-clone.git
cd jfk-voice-clone

# 2. Install deps
sudo apt-get update && sudo apt-get install -y pkg-config ffmpeg git-lfs \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev
git lfs install && git lfs pull
pip install -r requirements.txt

# 3. Build dataset (or use the included one)
python prepare_tts_dataset.py \
    --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
    --output ./jfk_dataset --speaker jfk --model large-v2

# 4. Fine-tune (edit paths in script first!)
chmod +x finetune_ft_tts.sh
./finetune_ft_tts.sh

# 5. Generate speech
python run_inference.py --text "Ask not what your country can do for you."
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'soundfile'` | Run `pip install -r requirements.txt` |
| Out of disk space during training | Use ≥40 GB storage (RunPod 40 GB volume) |
| `raw.arrow not created` during training | Check that `metadata.csv` paths point to existing WAV files |
| CUDA out of memory | Reduce `BATCH_SIZE` in `finetune_ft_tts.sh` (try `2000`) |
| Detached HEAD in F5-TTS | The script skips `git pull` intentionally to avoid this |
| Poor voice quality | Increase `EPOCHS` to 20–30 for better accent fidelity |
| Inference sounds robotic | Try a different/longer reference clip and increase `--nfe` to 64 |
