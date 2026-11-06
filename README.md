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
| `slim_checkpoint.py` | Extracts the inference weights from a 5 GB training checkpoint, 0.63 GB in float16 |
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
# writes checkpoints/jfk-f5tts-30epochs/model_last_ema.safetensors (0.63 GiB, float16)
```

The script memory-maps the input, so it needs about 1.8 GB of RAM and a few seconds. You can delete the 5 GB `.pt` file afterwards.

The output is float16 by default. The F5-TTS loader runs the model in float16 on a CUDA GPU, so the GPU result is the same as with float32 weights, and the file loads in a third of the time. For CPU inference with the full-precision weights, add `--dtype float32` (1.26 GiB).

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

### For training (phase 2)

- **Machine**: the DGX Spark, with the same `.venv` as inference. The training script adds `tensorboard` to it with `uv`.
- **GPU memory**: about 20 GB at batch size 6000. Stop the vLLM server first; it holds most of the unified memory.
- **Storage**: about 12 GB per run under `checkpoints/`.

Phase 1 has its own dependencies, listed in its section below.

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

This phase takes a YouTube video of JFK speaking and produces a clean, segmented dataset of labeled audio clips ready for TTS fine-tuning. The included `jfk_dataset/` already holds 1,151 clips (2.35 hours), so you only need this phase for a new dataset.

### What the pipeline does

1. **Downloads** audio from YouTube with `yt-dlp`, converted to 16 kHz mono WAV
2. **Denoises** the audio with `resemble-enhance` to remove crowd noise, applause, and background
3. **Transcribes and segments** with `WhisperX`. VAD strips silences and produces word-aligned timestamps
4. **Slices** the denoised audio into 3–15 second clips and writes `metadata.csv`

### Step 1.1 — Install system dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    pkg-config ffmpeg git-lfs \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

### Step 1.2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

The key packages are `yt-dlp`, `resemble-enhance`, `whisperx`, and `soundfile`. These are not needed for inference.

### Step 1.3 — Run the dataset preparation script

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
| `--speaker` | `jfk` | Speaker name, used in filenames |
| `--model` | `large-v2` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2` |
| `--language` | `en` | Language code |
| `--min_sec` | `3.0` | Minimum clip duration in seconds |
| `--max_sec` | `15.0` | Maximum clip duration in seconds |
| `--hf_token` | `None` | HuggingFace token for WhisperX diarization |

### Step 1.4 — Verify the dataset

After the script completes, `jfk_dataset/` looks like this:

```
jfk_dataset/
├── metadata.csv          # Pipe-delimited: filename|transcript
├── wavs/
│   ├── jfk_00000.wav
│   ├── jfk_00001.wav
│   └── ... (hundreds of clips)
└── _work/                # Intermediate files (raw download, denoised audio)
```

Spot-check a few clips and their transcripts:

```bash
head -5 jfk_dataset/metadata.csv
```

Expected format:

```
audio_file|text
jfk_00000.wav|Remarks to the Fair Employment Practices Committee from the Cabinet Room of the White House, April 11, 1961.
jfk_00001.wav|The President's Committee on Equal Employment Opportunity is meeting for the first time today.
```

To build a larger dataset, run the script several times with different `--url` values. The clips accumulate in `jfk_dataset/wavs/`. Merge the `metadata.csv` files by hand, because each run overwrites that file.

### Alternative: Google Colab

If you have no local GPU for dataset preparation, use the Colab notebook:

1. Upload `prepare_tts_colab.ipynb` to Google Colab
2. Set the runtime to GPU (T4)
3. Run all cells in order. The notebook installs dependencies, writes the script, and runs the pipeline
4. Download the resulting `jfk_dataset/` folder

---

## Phase 2 — Fine-Tuning F5-TTS

This phase fine-tunes the F5-TTS base model on the JFK dataset. Run it on the DGX Spark from the repo folder, with the `.venv` from [Generate speech now](#generate-speech-now) step 3 in place.

### Step 2.1 — Configure the run

Open `finetune_ft_tts.sh` and edit the config block at the top. The defaults are the best settings from `EXPERIMENTATION_RESULTS.md`:

```bash
EPOCHS=30                    # 10 for a quick baseline, 30 for the best accent
LEARNING_RATE=1e-5
BATCH_SIZE=6000              # frames per batch; 6000 was the best run, about 20 GB of GPU memory
GRAD_ACCUM=1
```

The paths derive from the script's own location: the dataset at `jfk_dataset/`, the submodule at `F5-TTS/`, the environment at `.venv/`, and the checkpoints at `checkpoints/`.

Each run gets a tag `e${EPOCHS}_b${BATCH_SIZE}_lr${LEARNING_RATE}`. The trainer writes to `checkpoints/jfk`, and the script moves that folder to `checkpoints/jfk_<tag>` when training ends. If that archive folder exists, the script appends a timestamp instead of overwriting it.

### Step 2.2 — Run the script

```bash
DRY_RUN=1 ./finetune_ft_tts.sh   # checks the environment and the dataset, stops before training
./finetune_ft_tts.sh
```

### What the script does

1. **Checks the F5-TTS submodule.** It stops with a hint when the submodule is missing.
2. **Checks the environment.** It activates `.venv`, verifies torch, CUDA, F5-TTS, and `accelerate`, and installs `tensorboard` with `uv` when it is missing.
3. **Prepares the dataset.** Pulls the LFS files when they are still pointers, and writes `F5-TTS/data/jfk/metadata.csv` with absolute paths to the WAV files in `jfk_dataset/wavs/`. It does not copy the WAV files.
4. **Runs `prepare_csv_wavs.py`.** This writes `raw.arrow`, `duration.json`, and `vocab.txt`. The vocab is a copy of the pretrained Emilia vocab, so the fine-tuned model keeps the same token set as the base model. The script skips this step when `raw.arrow` is newer than `metadata.csv`.
5. **Writes the accelerate config** to `checkpoints/accelerate_config.yaml`. Single GPU, fp16 mixed precision. Your own accelerate default config stays untouched.
6. **Launches training.** It points the fork at `checkpoints/` through `F5TTS_CKPT_ROOT`, then moves any old `checkpoints/jfk` to an archive folder named after that run's tag. A crashed run keeps its checkpoints this way.

Training settings:

| Setting | Value |
|---------|-------|
| Base model | `F5TTS_Base` |
| Learning rate | `1e-5` |
| Batch size | `6000` frames |
| Epochs | `30` |
| Warmup | `200` updates |
| Save every | `1000` updates, last checkpoint every `2000` |
| Kept on disk | the newest numbered checkpoint and `model_last.pt` |
| Mixed precision | `fp16` |
| Logger | TensorBoard |

Training time on the GB10 is not yet measured. An RTX 4090 took about 1 minute per epoch for 1,151 clips.

### The fork's changes

The F5-TTS submodule is the fork `czhou578/F5-TTS` at upstream 1.1.17 plus two changes:

- `finetune_cli.py` writes checkpoints to `$F5TTS_CKPT_ROOT/<dataset>` when that variable is set, else to upstream's `ckpts/<dataset>`. The training script sets it to `checkpoints/`.
- `pyproject.toml` drops the `pydantic` pin so that a current gradio installs.

Disk cleanup uses upstream's `--keep_last_n_checkpoints 1`, which the training script passes. It keeps the newest numbered checkpoint and `model_last.pt`.

Upstream is 28 commits ahead, at release 1.1.22. Inference with this fork works on current torch, so there is no need to update the submodule.

### Step 2.3 — Monitor training

The trainer writes TensorBoard logs under `runs/` inside the F5-TTS folder:

```bash
.venv/bin/tensorboard --logdir F5-TTS/runs
```

### Step 2.4 — Find and upload the checkpoints

While training runs, checkpoints are in `checkpoints/jfk/`:

```
checkpoints/jfk/
├── .exp_tag                      # the run's tag, used to name the archive folder
├── model_last.pt                 # latest, 5 GB, holds optimizer state
├── model_6000.pt                 # saved every 1000 updates, older ones deleted
└── pretrained_model_1200000.pt   # the base model
```

After the script moves the folder, upload the run to HuggingFace:

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='checkpoints/jfk_e30_b6000_lr1e-5', repo_id='czhou578/jfk-f5tts-30epochs-b6000', repo_type='model')
"
```

See `EXPERIMENTATION_PLAN.md` for the experiment matrix and `EXPERIMENTATION_RESULTS.md` for the losses.

---

## Phase 3 — Inference in detail

[Generate speech now](#generate-speech-now) covers the normal path. This section lists every option.

### The reference clip

F5-TTS clones a voice from one reference clip plus its exact transcript. The default is `jfk_dataset/wavs/jfk_00206.wav`, an 8.2-second clip. The script reads the transcript from `jfk_dataset/metadata.csv` by file name.

To use another clip from the dataset, pass `--ref_clip` alone. Pick a clean clip of 3–10 seconds from `jfk_dataset/wavs/`. For a clip outside the dataset, pass `--ref_text` with its exact transcript too. A wrong transcript degrades the output.

Pass `--seed` to make the output reproducible. Without it, each sentence gets a new random seed, and the script prints the seed it used. Use the same seed when you compare checkpoints by ear.

### Where the script looks for checkpoints

The script searches `checkpoints/` recursively. It skips `backup` folders and `pretrained_*` files. It accepts `.safetensors` files from `slim_checkpoint.py` and full `.pt` training checkpoints. With no `--checkpoint`, it prefers `model_last_ema.safetensors`, then `model_last.pt`, then the highest numbered `model_<step>` file.

Set `JFK_CKPT_DIR` to search another folder, for example an external drive:

```bash
JFK_CKPT_DIR=/mnt/models/jfk .venv/bin/python run_inference.py --text "..."
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
| `--ref_text` | from `metadata.csv` | Exact transcript of the reference clip; needed for clips outside the dataset |
| `--output` | `output.wav` | Output filename inside the output folder |
| `--output_dir` | `jfk_outputs/` | Output folder; not committed |
| `--seed` | random | Random seed for reproducible output |
| `--nfe` | `32` | Flow-matching steps. Higher is better and slower; 64 for final renders |
| `--pause` | `0.5` | Silence in seconds between sentences |
| `--split` | off | Also save each sentence as its own file, `name_001.wav`, `name_002.wav`, ... |

### Sample inputs and outputs

| File | Use |
|------|-----|
| `jfk_undelivered_speech.txt` | The Dallas Trade Mart speech; use `--mode sentences` |
| `covfefe_speech.txt` | The 2025 inaugural address, prose; use `--mode sentences` |
| `taylor_swift_lyrics.txt` | "Blank Space" lyrics; use `--mode lines` |
| `samples/*.wav` | Outputs from the experiment runs, named by run settings |

---

## Continuous integration

Every pull request runs the static checks in `.github/workflows/ci.yml`: a Python syntax and undefined-name check on the scripts in the repo root, and a bash syntax check on the training script. The checks do not fetch the LFS files or the submodule, so they finish in about a minute.

Dependabot opens a pull request each week when a GitHub Action or a pinned pip package has a newer version. When the checks pass on a Dependabot pull request, the workflow enables auto-merge, and GitHub squash-merges it. The repo setting "Allow auto-merge" must stay on for this. The F5-TTS submodule is pinned by hand and is not updated by Dependabot.

---

## Repository Structure

```
jfk-voice-clone/
├── README.md                    # This file
├── EXPERIMENTATION_PLAN.md      # Experiment matrix for epochs, batch size, learning rate
├── EXPERIMENTATION_RESULTS.md   # Final losses of each run
├── requirements.txt             # Python dependencies for dataset preparation only
├── .gitattributes               # Git LFS rules: *.wav, *.pt, *.safetensors, jfk_dataset/**
├── .gitignore                   # Ignores .venv/, checkpoints/, and jfk_outputs/
├── .gitmodules                  # F5-TTS submodule reference
├── .github/dependabot.yml       # Dependabot: weekly checks for GitHub Actions and pip updates
├── .github/workflows/ci.yml     # CI: static checks on every PR; auto-merges Dependabot PRs that pass
│
├── prepare_tts_dataset.py       # Phase 1: dataset preparation pipeline
├── prepare_tts_colab.ipynb      # Phase 1: Colab notebook version
├── update_nb.py                 # Utility: updates the Colab notebook dependencies
│
├── finetune_ft_tts.sh           # Phase 2: fine-tuning script (DGX Spark)
├── slim_checkpoint.py           # Phase 3: extract inference weights from a training checkpoint
├── run_inference.py             # Phase 3: inference script
│
├── jfk_undelivered_speech.txt   # Sample input, prose
├── covfefe_speech.txt           # Sample input, prose
├── taylor_swift_lyrics.txt      # Sample input, lyrics
│
├── F5-TTS/                      # Git submodule: F5-TTS fork at 1.1.17 + 3 commits
├── jfk_dataset/                 # 1,151 clips and metadata.csv (Git LFS)
├── samples/                     # Audio from the experiment runs (Git LFS)
├── jfk_outputs/                 # Your generated audio (not committed)
├── checkpoints/                 # Downloaded, slimmed, and training checkpoints (not committed)
└── .venv/                       # Local Python environment (not committed)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No checkpoints found under .../checkpoints` | Download and slim a checkpoint. See steps 4 and 5 above |
| Reference clip error, or `No valid audio files were processed` | The WAV files are Git LFS pointers. Run `git lfs install --local && git lfs pull` |
| CSV error during training | `metadata.csv` is a Git LFS pointer. Run `git lfs pull` |
| `torch.cuda.is_available()` is `False` on the DGX Spark | Install torch from the `cu130` index, not from PyPI |
| `torchaudio.load` fails or complains about torchcodec | Install `torch`, `torchaudio`, and `torchcodec` together from the same index |
| Inference killed, or the machine swaps | Run inside `systemd-run --user --scope -p MemoryMax=8G -- choom -n 1000 -- ...`, or add `--device cpu` |
| `ModuleNotFoundError: No module named 'soundfile'` | Run `uv pip install --python .venv/bin/python -e ./F5-TTS` |
| CUDA out of memory during training | Stop the vLLM server, or reduce `BATCH_SIZE` in `finetune_ft_tts.sh` to `4000` or `2000` |
| Out of disk space during training | Each run needs about 12 GB under `checkpoints/`. Delete archived runs that you have uploaded |
| `raw.arrow not created` during training | Check that the paths in `metadata.csv` point to existing WAV files |
| Detached HEAD in F5-TTS | Normal for a submodule. Run `git submodule update --init` to restore the pinned commit |
| Poor voice quality | Use the 30-epoch checkpoint, a longer clean reference clip, and `--nfe 64` |
| Output repeats or mumbles | A sign of overfitting. Use fewer epochs, or lower the learning rate; see `EXPERIMENTATION_RESULTS.md` |

[HuggingFace repos](https://huggingface.co/czhou578)
