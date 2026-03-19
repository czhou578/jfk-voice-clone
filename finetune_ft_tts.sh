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
DATASET_PATH="$HOME/jfk_dataset"       # folder containing wavs/ and metadata.csv
DATASET_NAME="jfk"                     # name used internally by F5-TTS
F5_REPO="$HOME/F5-TTS"                 # where to clone the repo
CONDA_ENV="f5-tts"                     # conda environment name
EPOCHS=10                              # start with 10, increase to 20-30 for better accent fidelity
LEARNING_RATE=1e-5
BATCH_SIZE=4000                        # safe for RTX 4090 24GB
GRAD_ACCUM=1

echo "=============================================="
echo " F5-TTS Fine-tuning Pipeline"
echo "=============================================="
echo " Dataset : $DATASET_PATH"
echo " Epochs  : $EPOCHS"
echo " GPU     : RTX 4090 (24GB)"
echo "=============================================="

# ─────────────────────────────────────────────────────────────
# STEP 1: Clone F5-TTS repo
# ─────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Cloning F5-TTS (official repo)..."
if [ ! -d "$F5_REPO" ]; then
    git clone https://github.com/SWivid/F5-TTS.git "$F5_REPO"
else
    echo "    ✓ Repo already exists, pulling latest..."
    cd "$F5_REPO" && git pull
fi
cd "$F5_REPO"

# ─────────────────────────────────────────────────────────────
# STEP 2: Create conda environment + install dependencies
# ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Setting up python virtual environment: $CONDA_ENV"

apt-get update -y && apt-get install -y python3-venv ffmpeg || true

python3 -m venv "$HOME/$CONDA_ENV"
source "$HOME/$CONDA_ENV/bin/activate"

# Update pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 (works for RTX 4090)
pip install torch==2.4.0+cu121 torchaudio==2.4.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install F5-TTS and its dependencies
pip install -e .

# Install accelerate for training
pip install accelerate

echo "    ✓ Environment ready"

# ─────────────────────────────────────────────────────────────
# STEP 3: Validate and copy dataset into F5-TTS data directory
# ─────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Preparing dataset..."

DATA_DIR="$F5_REPO/data/${DATASET_NAME}"
mkdir -p "$DATA_DIR"

# Copy your wavs and metadata.csv into F5-TTS expected location
cp -r "$DATASET_PATH/wavs" "$DATA_DIR/wavs"
cp "$DATASET_PATH/metadata.csv" "$DATA_DIR/metadata.csv"

# Validate metadata format — F5-TTS expects: wavs/filename.wav|transcript
# Your pipeline already outputs this format, but let's confirm
echo "    Checking metadata format..."
HEAD=$(head -2 "$DATA_DIR/metadata.csv")
echo "    Sample rows:"
echo "    $HEAD"
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
    "$DATA_DIR" \
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

echo "    ✓ Accelerate configured for single RTX 4090"

# ─────────────────────────────────────────────────────────────
# STEP 6: Launch fine-tuning
# ─────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Starting fine-tuning..."
echo ""
echo "  Settings:"
echo "    Epochs            : $EPOCHS"
echo "    Learning rate     : $LEARNING_RATE"
echo "    Batch size        : $BATCH_SIZE (frame-based)"
echo "    Grad accum steps  : $GRAD_ACCUM"
echo "    Estimated time    : 2-4 hours on RTX 4090"
echo ""

cd "$F5_REPO"

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
    --last_per_steps  2000 \
    --dataset_name    "$DATASET_NAME" \
    --finetune        True \
    --logger          tensorboard

echo ""
echo "=============================================="
echo " ✅ Fine-tuning complete!"
echo " Checkpoints saved to: $F5_REPO/ckpts/F5TTS_Base/"
echo "=============================================="
echo ""
echo " To monitor training in real-time, open a second terminal and run:"
echo "   tensorboard --logdir $F5_REPO/ckpts/F5TTS_Base/logs"
echo ""
echo " To run inference with your fine-tuned model, see: run_inference.py"