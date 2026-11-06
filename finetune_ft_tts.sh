#!/usr/bin/env bash
# =============================================================================
# F5-TTS Fine-tuning Pipeline for the JFK Voice Dataset
# =============================================================================
# Runs on the DGX Spark (GB10) from the repo folder. It reuses the .venv that
# inference uses (README.md, "Generate speech now", step 3).
#
# USAGE:
#   ./finetune_ft_tts.sh              # full run
#   DRY_RUN=1 ./finetune_ft_tts.sh    # check the setup, stop before training
#
# REQUIREMENTS:
#   - .venv with torch, torchaudio, torchcodec from the cu130 index and
#     F5-TTS installed editable (README.md step 3)
#   - uv on the PATH, to add accelerate and tensorboard to .venv
#   - The dataset at jfk_dataset/ (wavs/ + metadata.csv), pulled with git lfs
#   - About 12 GB of disk per run under checkpoints/
#   - About 20 GB of GPU memory at batch size 6000. Stop the vLLM server first.
# =============================================================================

set -e  # Exit on any error

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these to change a run
# ─────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="$PROJECT_DIR/jfk_dataset"
DATASET_NAME="jfk"                     # name used internally by F5-TTS
F5_REPO="$PROJECT_DIR/F5-TTS"          # the submodule
VENV_DIR="$PROJECT_DIR/.venv"          # the inference environment, reused for training
CKPT_ROOT="$PROJECT_DIR/checkpoints"   # where run_inference.py looks for checkpoints
EPOCHS=30                              # 10 for a quick baseline, 30 for the best accent
LEARNING_RATE=1e-5
BATCH_SIZE=6000                        # frames per batch; 6000 was the best run, about 20 GB of GPU memory
GRAD_ACCUM=1
# Each experiment gets its own checkpoint dir. The fork's finetune_cli.py
# writes to $F5TTS_CKPT_ROOT/{DATASET_NAME}; the script sets that variable to
# $CKPT_ROOT and archives each finished run under its experiment tag.
EXP_TAG="e${EPOCHS}_b${BATCH_SIZE}_lr${LEARNING_RATE}"  # unique tag for this experiment
CKPT_BASE="$CKPT_ROOT/$DATASET_NAME"   # the active run
CKPT_DIR="$CKPT_ROOT/${DATASET_NAME}_${EXP_TAG}"  # where the finished run is archived

echo "=============================================="
echo " F5-TTS Fine-tuning Pipeline"
echo "=============================================="
echo " Project : $PROJECT_DIR"
echo " Dataset : $DATASET_PATH"
echo " Epochs  : $EPOCHS"
echo "=============================================="

# ─────────────────────────────────────────────────────────────
# STEP 1: Check the F5-TTS submodule
# ─────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Checking the F5-TTS submodule..."
if [ ! -f "$F5_REPO/pyproject.toml" ]; then
    echo "ERROR: F5-TTS submodule is missing. Run: git submodule update --init"
    exit 1
fi
echo "    ✓ Submodule present at $F5_REPO"

# ─────────────────────────────────────────────────────────────
# STEP 2: Check the Python environment
# ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Checking the Python environment: $VENV_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "ERROR: $VENV_DIR not found."
    echo "       Create it first, see README.md 'Generate speech now', step 3."
    exit 1
fi
source "$VENV_DIR/bin/activate"

python - <<'EOF' || exit 1
import sys
try:
    import torch, f5_tts
except ImportError as e:
    sys.exit(f"ERROR: {e}. Set up .venv first, see README.md 'Generate speech now', step 3.")
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False. Install torch from the cu130 index, see README.md.")
print(f"    ✓ torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
EOF

# accelerate comes with F5-TTS. tensorboard does not; install it once. It has
# no torch dependency, so the install leaves the torch packages alone.
if ! python -c "import accelerate" >/dev/null 2>&1; then
    echo "ERROR: accelerate is missing. Reinstall F5-TTS, see README.md step 3."
    exit 1
fi
if ! python -c "import tensorboard" >/dev/null 2>&1; then
    echo "    Installing tensorboard into $VENV_DIR..."
    uv pip install --python "$VENV_DIR/bin/python" tensorboard
fi
echo "    ✓ Environment ready"

# ─────────────────────────────────────────────────────────────
# STEP 3: Validate the dataset and write the F5-TTS metadata
# ─────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Preparing dataset..."

DATA_DIR="$F5_REPO/data/${DATASET_NAME}"
mkdir -p "$DATA_DIR"

is_lfs_pointer() { head -c 40 "$1" 2>/dev/null | grep -q "^version https://git-lfs"; }

SAMPLE_WAV=$(ls "$DATASET_PATH/wavs/"*.wav 2>/dev/null | head -1)
if [ -z "$SAMPLE_WAV" ]; then
    echo "ERROR: no WAV files in $DATASET_PATH/wavs"
    exit 1
fi

# The dataset is in Git LFS. Pull it when the files are still pointers.
if is_lfs_pointer "$DATASET_PATH/metadata.csv" || is_lfs_pointer "$SAMPLE_WAV"; then
    echo "    Dataset files are Git LFS pointers, running git lfs pull..."
    (cd "$PROJECT_DIR" && git lfs install --local && git lfs pull)
fi
if is_lfs_pointer "$DATASET_PATH/metadata.csv" || is_lfs_pointer "$SAMPLE_WAV"; then
    echo "ERROR: the dataset files are still Git LFS pointers."
    echo "       Install git-lfs (README.md step 2) and run 'git lfs pull'."
    exit 1
fi

# Write metadata.csv into the F5-TTS data directory with absolute paths.
# F5-TTS expects: /absolute/path.wav|transcript. The paths point at the WAV
# files in $DATASET_PATH/wavs, so the script does not copy the audio.
echo "    Writing metadata with absolute paths..."
python3 -c "
import sys, os
src_path, dst_path, wavs_dir = sys.argv[1:4]
with open(src_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open(dst_path, 'w', encoding='utf-8') as f:
    f.write('audio_file|text\n')
    for line in lines[1:]:
        if '|' in line:
            filename, text = line.split('|', 1)
            if not filename.startswith('/'):
                filename = os.path.join(wavs_dir, os.path.basename(filename))
            f.write(f'{filename}|{text}')
" "$DATASET_PATH/metadata.csv" "$DATA_DIR/metadata.csv" "$DATASET_PATH/wavs"

HEAD=$(head -2 "$DATA_DIR/metadata.csv")
echo "    Sample rows:"
echo "$HEAD"
echo "    ✓ Metadata written to $DATA_DIR"

# ─────────────────────────────────────────────────────────────
# STEP 4: Run the official dataset preparation script
# This converts metadata.csv + wavs into raw.arrow + duration.json
# which F5-TTS needs internally for efficient training
# ─────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Running F5-TTS dataset preparation (prepare_csv_wavs.py)..."

cd "$F5_REPO/src/f5_tts/train/datasets"

# The conversion reads every WAV file. Skip it when the output is newer
# than the dataset's metadata.csv.
if [ "$DATA_DIR/raw.arrow" -nt "$DATASET_PATH/metadata.csv" ] \
   && [ -f "$DATA_DIR/duration.json" ] && [ -f "$DATA_DIR/vocab.txt" ]; then
    echo "    ✓ raw.arrow is newer than metadata.csv, skipping conversion"
else
    python prepare_csv_wavs.py \
        "$DATA_DIR/metadata.csv" \
        "$DATA_DIR"
fi

# Verify output files were created
if [ ! -f "$DATA_DIR/raw.arrow" ]; then
    echo "ERROR: raw.arrow not created. Check that metadata.csv paths are correct."
    exit 1
fi
if [ ! -f "$DATA_DIR/duration.json" ]; then
    echo "ERROR: duration.json not created. Check that WAV files exist."
    exit 1
fi
if [ ! -f "$DATA_DIR/vocab.txt" ]; then
    echo "ERROR: vocab.txt not created."
    exit 1
fi

echo "    ✓ raw.arrow created"
echo "    ✓ duration.json created"
echo "    ✓ vocab.txt created"

# ─────────────────────────────────────────────────────────────
# STEP 5: Write the accelerate config for a single GPU
# ─────────────────────────────────────────────────────────────
echo ""
echo "[5/6] Writing the accelerate config..."

# Single GPU, fp16 mixed precision. The file lives in the project, so your
# own accelerate default config stays untouched.
mkdir -p "$CKPT_ROOT"
ACCEL_CONFIG="$CKPT_ROOT/accelerate_config.yaml"
cat > "$ACCEL_CONFIG" << EOF
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

echo "    ✓ Accelerate config written to $ACCEL_CONFIG"

# ─────────────────────────────────────────────────────────────
# STEP 6: Launch fine-tuning
# ─────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Starting fine-tuning..."
echo ""

# F5-TTS looks for data/<name>_custom
ln -sfn "$DATA_DIR" "$F5_REPO/data/${DATASET_NAME}_custom"

# The fork's finetune_cli.py writes the checkpoints under this folder, where
# run_inference.py looks for them.
export F5TTS_CKPT_ROOT="$CKPT_ROOT"

echo "  Settings:"
echo "    Epochs            : $EPOCHS"
echo "    Learning rate     : $LEARNING_RATE"
echo "    Batch size        : $BATCH_SIZE (frame-based)"
echo "    Grad accum steps  : $GRAD_ACCUM"
echo ""

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "    DRY_RUN=1: setup complete, not starting training."
    exit 0
fi

cd "$F5_REPO"

# Move any prior run out of the way so we start completely fresh. The prior
# run is archived, never deleted, so a crashed run keeps its checkpoints.
# If the archive name is taken, a timestamp is appended.
archive_run() {
    local src="$1" dest="$2"
    if [ -e "$dest" ]; then
        dest="${dest}_$(date +%Y%m%d_%H%M%S)"
    fi
    mv "$src" "$dest"
    echo "    Archived checkpoints to: $dest"
}

if [ -d "$CKPT_BASE" ]; then
    # The previous run wrote its tag into .exp_tag. Fall back to a timestamp.
    PREV_TAG="$(cat "$CKPT_BASE/.exp_tag" 2>/dev/null || date +%Y%m%d_%H%M%S)"
    archive_run "$CKPT_BASE" "$CKPT_ROOT/${DATASET_NAME}_${PREV_TAG}"
fi
mkdir -p "$CKPT_BASE"
echo "$EXP_TAG" > "$CKPT_BASE/.exp_tag"
echo "    Checkpoints → $CKPT_BASE (fresh, will archive to $CKPT_DIR after training)"

accelerate launch --config_file "$ACCEL_CONFIG" src/f5_tts/train/finetune_cli.py \
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
    --keep_last_n_checkpoints 1 \
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
archive_run "$CKPT_BASE" "$CKPT_DIR"
echo "=============================================="
echo ""
echo " To monitor training in real-time, open a second terminal and run:"
echo "   $VENV_DIR/bin/tensorboard --logdir $F5_REPO/runs"
echo ""
echo " To run inference with your fine-tuned model:"
echo "   $VENV_DIR/bin/python slim_checkpoint.py $CKPT_DIR/model_last.pt"
echo "   $VENV_DIR/bin/python run_inference.py --text \"Ask not what your country can do for you.\""
