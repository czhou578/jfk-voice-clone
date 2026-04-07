#!/usr/bin/env bash
# =============================================================================
# F5-TTS Fine-tuning Pipeline for JFK Voice Dataset
# =============================================================================
# RTX 4090 (24GB VRAM) optimized settings
# Expected training time: 2-4 hours for 1151 clips @ 10 epochs
#
# USAGE:
#   chmod +x finetune_f5tts.sh
#   ./finetune_f5tts.sh
#
# REQUIREMENTS:
#   - CUDA 12.1+ drivers installed
#   - conda or python 3.10/3.11 available
#   - Your dataset at ~/jfk_dataset/ (wavs/ + metadata.csv)
# =============================================================================

set -e  # Exit on any error

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these to match your paths
# ─────────────────────────────────────────────────────────────
DATASET_PATH="/workspace/jfk-voice-clone/jfk_dataset"
DATASET_NAME="jfk"                     # name used internally by F5-TTS
F5_REPO="/workspace/jfk-voice-clone/F5-TTS"    # where to clone the repo
CONDA_ENV="f5-tts"                     # virtual environment name
EPOCHS=30                              # start with 10, increase to 20-30 for better accent fidelity
LEARNING_RATE=3e-5
BATCH_SIZE=6000                        # safe for RTX 4090 24GB
GRAD_ACCUM=1
# Each experiment gets its own checkpoint dir; the trainer always reads/writes
# /root/F5-TTS-ckpts/{DATASET_NAME}, so we rename any prior run out of the way.
EXP_TAG="e${EPOCHS}_b${BATCH_SIZE}"    # unique tag for this experiment
CKPT_BASE="/root/F5-TTS-ckpts/$DATASET_NAME"  # path finetune_cli.py always uses
CKPT_DIR="/root/F5-TTS-ckpts/${DATASET_NAME}_${EXP_TAG}"  # where we'll archive this run

echo "=============================================="
echo " F5-TTS Fine-tuning Pipeline"
echo "=============================================="
echo " Dataset : $DATASET_PATH"
echo " Epochs  : $EPOCHS"
echo " GPU     : RTX 4000 Ada (20GB)"
echo "=============================================="

# ─────────────────────────────────────────────────────────────
# STEP 1: Clone F5-TTS repo
# ─────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Cloning F5-TTS (official repo)..."
if [ ! -d "$F5_REPO" ]; then
    echo "ERROR: F5-TTS repo missing"
    exit 1
else
    echo "    ✓ Repo existing, skipping pull to avoid detached HEAD detached issues..."
fi
cd "$F5_REPO"

# ─────────────────────────────────────────────────────────────
# STEP 2: Create conda environment + install dependencies
# ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Setting up python virtual environment: $CONDA_ENV"

apt-get update -y && apt-get install -y python3-venv ffmpeg || true

python3 -m venv "/workspace/$CONDA_ENV"
source "/workspace/$CONDA_ENV/bin/activate"

# Update pip
pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 (works for RTX 4090)
pip install --no-cache-dir torch==2.4.0+cu121 torchaudio==2.4.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install F5-TTS and its dependencies
pip install --no-cache-dir -e .

# Install accelerate for training
pip install --no-cache-dir accelerate tensorboard

echo "    ✓ Environment ready"

# ─────────────────────────────────────────────────────────────
# STEP 3: Validate and copy dataset into F5-TTS data directory
# ─────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Preparing dataset..."

DATA_DIR="$F5_REPO/data/${DATASET_NAME}"
mkdir -p "$DATA_DIR"

# Ensure Git LFS files are pulled (metadata.csv AND wavs may be tracked by LFS)
echo "    Ensuring Git LFS files are pulled..."
REPO_ROOT="$(git -C "$DATASET_PATH" rev-parse --show-toplevel 2>/dev/null || echo "$DATASET_PATH")"
(cd "$REPO_ROOT" && git lfs install 2>/dev/null && git lfs pull 2>/dev/null) || true

# Validate that metadata.csv is not a Git LFS pointer
if head -1 "$DATASET_PATH/metadata.csv" | grep -q "^version https://git-lfs"; then
    echo "ERROR: metadata.csv is a Git LFS pointer, not the actual file."
    echo "       Run 'git lfs pull' in the dataset repo first."
    exit 1
fi

# Validate that wav files are not Git LFS pointers
SAMPLE_WAV=$(ls "$DATASET_PATH/wavs/"*.wav 2>/dev/null | head -1)
if [ -n "$SAMPLE_WAV" ] && head -1 "$SAMPLE_WAV" | grep -q "^version https://git-lfs"; then
    echo "ERROR: WAV files are Git LFS pointers, not actual audio."
    echo "       Run 'git lfs pull' in the dataset repo first."
    exit 1
fi

# Copy your wavs and metadata.csv into F5-TTS expected location
cp -r "$DATASET_PATH/wavs" "$DATA_DIR/wavs"
cp "$DATASET_PATH/metadata.csv" "$DATA_DIR/metadata.csv"

# Fix metadata format — F5-TTS expects absolute paths: /absolute/path.wav|transcript
echo "    Converting metadata to use absolute paths..."
python3 -c "
import sys, os
csv_path = sys.argv[1]
wavs_dir = sys.argv[2]
with open(csv_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open(csv_path, 'w', encoding='utf-8') as f:
    f.write('audio_file|text\n')
    for line in lines[1:]:
        if '|' in line:
            filename, text = line.split('|', 1)
            if not filename.startswith('/'):
                filename = os.path.join(wavs_dir, os.path.basename(filename))
            f.write(f'{filename}|{text}')
" "$DATA_DIR/metadata.csv" "$DATA_DIR/wavs"

HEAD=$(head -2 "$DATA_DIR/metadata.csv")
echo "    Sample rows:"
echo "$HEAD"
echo "    ✓ Dataset copied to $DATA_DIR"

# ─────────────────────────────────────────────────────────────
# STEP 4: Run the official dataset preparation script
# This converts metadata.csv + wavs into raw.arrow + duration.json
# which F5-TTS needs internally for efficient training
# ─────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Running F5-TTS dataset preparation (prepare_csv_wavs.py)..."

cd "$F5_REPO/src/f5_tts/train/datasets"

python prepare_csv_wavs.py \
    "$DATA_DIR/metadata.csv" \
    "$DATA_DIR"

# Verify output files were created
if [ ! -f "$DATA_DIR/raw.arrow" ]; then
    echo "ERROR: raw.arrow not created. Check that metadata.csv paths are correct."
    exit 1
fi
if [ ! -f "$DATA_DIR/duration.json" ]; then
    echo "ERROR: duration.json not created. Check that WAV files exist."
    exit 1
fi

echo "    ✓ raw.arrow created"
echo "    ✓ duration.json created"
echo "    ✓ vocab.txt created"

# ─────────────────────────────────────────────────────────────
# STEP 5: Configure accelerate for single GPU (RTX 4090)
# ─────────────────────────────────────────────────────────────
echo ""
echo "[5/6] Configuring accelerate for single GPU..."

# Write accelerate config directly (avoids interactive prompt)
mkdir -p ~/.cache/huggingface/accelerate/
cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "    ✓ Accelerate configured for single RTX 4000 Ada"

# ─────────────────────────────────────────────────────────────
# STEP 6: Launch fine-tuning
# ─────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Starting fine-tuning..."
echo ""

# Fix F5-TTS looking for _custom suffix
ln -sfn "$DATA_DIR" "$F5_REPO/data/${DATASET_NAME}_custom"

echo "  Settings:"
echo "    Epochs            : $EPOCHS"
echo "    Learning rate     : $LEARNING_RATE"
echo "    Batch size        : $BATCH_SIZE (frame-based)"
echo "    Grad accum steps  : $GRAD_ACCUM"
echo ""

cd "$F5_REPO"

# finetune_cli.py ALWAYS uses /root/F5-TTS-ckpts/{dataset_name} (hardcoded).
# Move any prior run out of the way so we start completely fresh.
if [ -d "$CKPT_BASE" ]; then
    BACKUP="${CKPT_BASE}_backup_$(date +%s)"
    echo "    Moving old checkpoint dir → $BACKUP"
    mv "$CKPT_BASE" "$BACKUP"
fi
mkdir -p "$CKPT_BASE"
echo "    Checkpoints → $CKPT_BASE (fresh, will archive to $CKPT_DIR after training)"

accelerate launch src/f5_tts/train/finetune_cli.py \
    --exp_name        F5TTS_Base \
    --learning_rate   $LEARNING_RATE \
    --batch_size_per_gpu $BATCH_SIZE \
    --batch_size_type frame \
    --max_samples     64 \
    --grad_accumulation_steps $GRAD_ACCUM \
    --max_grad_norm   1 \
    --epochs          $EPOCHS \
    --num_warmup_updates 200 \
    --save_per_updates 1000 \
    --last_per_updates  2000 \
    --dataset_name    "$DATASET_NAME" \
    --tokenizer       custom \
    --tokenizer_path  "data/$DATASET_NAME/vocab.txt" \
    --finetune \
    --logger          tensorboard

echo ""
echo "=============================================="
echo " ✅ Fine-tuning complete!"
# Archive the finished checkpoints under the experiment-tagged name
mv "$CKPT_BASE" "$CKPT_DIR"
echo " Checkpoints archived to: $CKPT_DIR"
echo "=============================================="
echo ""
echo " To monitor training in real-time, open a second terminal and run:"
echo "   tensorboard --logdir $F5_REPO/ckpts/F5TTS_Base/logs"
echo ""
echo " To run inference with your fine-tuned model, see: run_inference.py"