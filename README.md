# JFK Voice Clone — F5-TTS Fine-Tuning Pipeline

Clone JFK's voice with [F5-TTS](https://github.com/SWivid/F5-TTS). This repo holds the full path from a raw YouTube speech to a fine-tuned TTS model, plus the script that reads new text in JFK's voice.

The fine-tuned checkpoints are on HuggingFace, so you do not need to train again. To generate speech, follow [Generate speech now](#generate-speech-now) and skip the training phases.

Last verified on 2026-11-05 on a DGX Spark (GB10, aarch64, CUDA 13) with Python 3.11 and torch 2.14.1+cu130.

## Overview

The pipeline has three phases:

| Phase | Script | What it does |
|-------|--------|--------------|
| 1. Dataset preparation | `prepare_tts_dataset.py` | Downloads audio, denoises, transcribes, and slices into labeled clips |
| 2. Fine-tuning | `finetune_ft_tts.sh` | Converts the dataset for F5-TTS, configures training, and launches fine-tuning |
| 3. Inference | `run_inference.py` | Reads new text with the fine-tuned model |

Two helper scripts support phase 3:

| Script | What it does |
|--------|--------------|
| `slim_checkpoint.py` | Extracts the 1.3 GB inference weights from a 5 GB training checkpoint |
| `update_nb.py` | Updates the dependency cell of the Colab notebook |

---

## Generate speech now

This is the short path. It takes about 15 minutes, most of it downloads.

### Step 1 — Clone with the submodule

```bash
git clone --recurse-submodules https://github.com/czhou578/jfk-voice-clone.git
cd jfk-voice-clone
```

The `--recurse-submodules` flag pulls the [F5-TTS fork](https://github.com/czhou578/F5-TTS) into `F5-TTS/`. The fork is upstream 1.1.17 plus three small commits for training.

### Step 2 — Pull the Git LFS files

The dataset, the reference clips, and the sample outputs are Git LFS files. Without this step they are 130-byte pointer files, and inference fails on the reference clip.

With sudo:

```bash
sudo apt-get install -y git-lfs
```

Without sudo, install the release binary for your user:

```bash
gh release download -R git-lfs/git-lfs --pattern 'git-lfs-linux-arm64-*.tar.gz'   # use amd64 on x86
tar xzf git-lfs-linux-arm64-*.tar.gz && cp git-lfs-*/git-lfs ~/.local/bin/
```

Then, inside the repo:

```bash
git lfs install --local
git lfs pull          # about 712 MB
```

### Step 3 — Create the Python environment

Use Python 3.11 and the torch wheels that match your CUDA driver. The three torch packages must come from the same index, or audio loading fails.

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu130 torch torchaudio torchcodec
uv pip install --python .venv/bin/python -e ./F5-TTS
```

Pick the index for your driver: `cu130` for CUDA 13 (DGX Spark), `cu128` or `cu126` for CUDA 12 on x86. PyPI torch wheels for aarch64 do not support the GB10.

Check the result:

```bash
.venv/bin/python -c "import torch, f5_tts; print(torch.__version__, torch.cuda.is_available())"
```

### Step 4 — Download a checkpoint

The fine-tuned checkpoints are on HuggingFace under [czhou578](https://huggingface.co/czhou578). Each repo holds a 5 GB training checkpoint and the 1.3 GB base model it started from.

| HuggingFace repo | Epochs | Batch (frames) | Final loss | Note |
|------------------|--------|----------------|------------|------|
| `czhou578/jfk-voice-clone-f5tts` | 10 | 4000 | ~0.70 | Baseline |
| `czhou578/jfk-f5tts-20epochs` | 20 | 4000 | 0.644 | |
| `czhou578/jfk-f5tts-30epochs` | 30 | 4000 | 0.602 | Best checkpoint on HuggingFace |

The 30-epoch, batch 6000 run reached loss 0.487 (see `EXPERIMENTATION_RESULTS.md`). It is the default in `finetune_ft_tts.sh`, but no upload of it exists on HuggingFace.

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('czhou578/jfk-f5tts-30epochs', 'model_last.pt', local_dir='checkpoints/jfk-f5tts-30epochs')
"
```

### Step 5 — Slim the checkpoint

A training checkpoint carries the optimizer state, and the loader reads the whole 5 GB file into memory. The slim script keeps only the EMA weights that inference uses.

```bash
.venv/bin/python slim_checkpoint.py checkpoints/jfk-f5tts-30epochs/model_last.pt
# writes checkpoints/jfk-f5tts-30epochs/model_last_ema.safetensors (1.26 GiB)
```

The script memory-maps the input, so it needs about 1.8 GB of RAM and a few seconds. You can delete the 5 GB `.pt` file afterwards.

### Step 6 — Generate speech

```bash
.venv/bin/python run_inference.py --text "Ask not what your country can do for you, ask what you can do for your country."
```

The output lands in `jfk_outputs/output.wav` at 24 kHz. The script finds the slim checkpoint under `checkpoints/` on its own.

Read a whole speech from a file. Use `--mode sentences` for prose and `--mode lines` for lyrics:

```bash
.venv/bin/python run_inference.py --file jfk_undelivered_speech.txt --mode sentences --output undelivered.wav
.venv/bin/python run_inference.py --file taylor_swift_lyrics.txt --mode lines --output blank_space.wav
```

Measured on the DGX Spark for one 5-second sentence at 32 NFE steps:

| Device | Wall time | Peak host memory |
|--------|-----------|------------------|
| GPU (GB10) | 26 s | 2.6 GB |
| CPU (20 cores) | 115 s | 3.8 GB |

Both outputs transcribe word for word to the input text.

---

## Prerequisites

### For inference

- Python 3.11 (3.10 also works with upstream F5-TTS)
- `uv` or `pip`, `ffmpeg`, `git-lfs`
- A CUDA GPU with 3 GB free, or a CPU with 4 GB of RAM free
- 8 GB of disk for the environment and one checkpoint

### For training (phases 1 and 2)

- **GPU**: NVIDIA GPU with at least 24 GB VRAM (tested on RTX 4090 and RTX 4000 Ada on RunPod)
- **OS**: Linux. The training script uses `apt-get` and bash and expects the repo at `/workspace/jfk-voice-clone`.
- **CUDA**: 12.1 or later. The script pins torch 2.4.0+cu121 for x86 RunPod images.
- **Storage**: at least 40 GB free. A RunPod 40 GB volume is the minimum that works.

---

## Running beside another GPU process

On the DGX Spark, the vLLM server holds about 111 GB of the 121 GB unified memory. Inference still fits, but run it inside a memory-capped scope so that the kernel's OOM killer picks the TTS process and not the server:

```bash
systemd-run --user --scope -p MemoryMax=8G -- choom -n 1000 -- \
    .venv/bin/python run_inference.py --text "Hello from the Oval Office."
```

Add `--device cpu` to keep the GPU untouched. CPU inference takes about four times longer.

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

## Phase 3 — Inference in detail

[Generate speech now](#generate-speech-now) covers the normal path. This section lists every option.

### The reference clip

F5-TTS clones a voice from one reference clip plus its exact transcript. The default is `jfk_dataset/wavs/jfk_00206.wav`, an 8.2-second clip, with its transcript from `metadata.csv`:

```python
REFERENCE_CLIP = os.path.join(DATASET_DIR, "wavs/jfk_00206.wav")
REFERENCE_TEXT = "And I therefore want to commend the members of the Congress and the House and the Senate, the Chairman of the subcommittees,"
```

To use another clip, pass `--ref_clip` and `--ref_text` together. Pick a clean clip of 3–10 seconds from `jfk_dataset/wavs/` and copy its transcript from `metadata.csv` exactly. A wrong transcript degrades the output.

### Where the script looks for checkpoints

The script searches `checkpoints/` recursively. It skips `backup` folders and `pretrained_*` files. It accepts `.safetensors` files from `slim_checkpoint.py` and full `.pt` training checkpoints. With no `--checkpoint`, it prefers `model_last_ema.safetensors`, then `model_last.pt`, then the highest numbered `model_<step>` file.

Set `JFK_CKPT_DIR` to search another folder, for example on a RunPod box:

```bash
JFK_CKPT_DIR=/root/F5-TTS-ckpts python run_inference.py --text "..."
```

### All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--text` | | Text to synthesize. Use this or `--file` |
| `--file` | | Text file to synthesize |
| `--mode` | `lines` | `lines`: one TTS call per line, for lyrics. `sentences`: split on `.!?`, for prose |
| `--checkpoint` | latest | Checkpoint step number, for example `6000` |
| `--ckpt_dir` | `./checkpoints` | Folder to search; `JFK_CKPT_DIR` also sets it |
| `--ckpt_file` | | One checkpoint file; skips the search |
| `--device` | `auto` | `cuda` if available, else `cpu`; or force one |
| `--ref_clip` | `jfk_dataset/wavs/jfk_00206.wav` | Reference WAV clip |
| `--ref_text` | *(see script)* | Exact transcript of the reference clip |
| `--output` | `output.wav` | Output filename inside `jfk_outputs/` |
| `--nfe` | `32` | Flow-matching steps. Higher is better and slower; 64 for final renders |
| `--pause` | `0.5` | Silence in seconds between sentences |
| `--split` | off | Also save each sentence as its own file, `name_001.wav`, `name_002.wav`, ... |

### Sample inputs and outputs

| File | Use |
|------|-----|
| `jfk_undelivered_speech.txt` | The Dallas Trade Mart speech; use `--mode sentences` |
| `covfefe_speech.txt` | The 2025 inaugural address, prose; use `--mode sentences` |
| `taylor_swift_lyrics.txt` | "Blank Space" lyrics; use `--mode lines` |
| `jfk_outputs/*.wav` | Outputs from the experiment runs, named by run settings |

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
| CSV error? | Make sure to git lfs pull so that csv is populated with real data, and not just the lfs pointer |
| No valid audio files were processed | Make sure that wav files are not git lfs pointers | 

[HuggingFace Repos](https://huggingface.co/czhou578)
