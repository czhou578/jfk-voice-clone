# F5-TTS Fine-Tuning Experimentation Plan

## Current Baseline

Your current training run serves as the baseline to beat:

| Setting | Value |
|---|---|
| **Epochs** | 10 |
| **Batch size** | 4000 (frame-based) |
| **Learning rate** | 1e-5 |
| **Grad accumulation** | 1 |
| **Training time** | ~10 min (RTX 4090) |
| **Final loss** | ~0.7 |
| **Dataset** | 1,151 clips, 2.35 hours |

---

## Experiment Matrix

### Round 1: Epoch Sweep (keep batch size = 4000)

| Experiment | Epochs | Expected Time | What to Watch |
|---|---|---|---|
| E1 (baseline) | 10 | ~10 min | Already done — loss ~0.7 |
| E2 | 20 | ~20 min | Does loss plateau or keep dropping? |
| E3 | 30 | ~30 min | Better accent fidelity? Check for overfitting |
| E4 | 50 | ~50 min | Likely overfitting territory — compare quality |

> [!TIP]
> Listen to outputs at each epoch count to find the sweet spot where JFK's voice character is strongest without artifacts or repetition (signs of overfitting).

### Round 2: Batch Size Sweep (use best epoch count from Round 1)

| Experiment | Batch Size (frames) | VRAM Usage | Effect |
|---|---|---|---|
| B1 | 2000 | ~12 GB | More updates per epoch, noisier gradients |
| B2 (baseline) | 4000 | ~16 GB | Current setting |
| B3 | 6000 | ~20 GB | Smoother gradients, fewer updates per epoch |
| B4 | 8000 | ~23 GB | Max for 24GB GPU — may OOM, try with caution |

> [!WARNING]
> Batch size 8000 may cause CUDA OOM on a 24GB card. If it fails, drop to 6000.

### Round 3: Learning Rate Tuning (use best epoch + batch size)

| Experiment | Learning Rate | Notes |
|---|---|---|
| L1 | 5e-6 | More conservative, may need more epochs |
| L2 (baseline) | 1e-5 | Current setting |
| L3 | 3e-5 | Faster convergence, risk of instability |
| L4 | 5e-5 | Aggressive — watch for loss spikes |

---

## How to Run Each Experiment

Edit the variables at the top of `finetune_ft_tts.sh`:

```bash
# Example: 30 epochs, batch size 6000, lr 3e-5
EPOCHS=30
BATCH_SIZE=6000
LEARNING_RATE=3e-5
```

Then run:

```bash
./finetune_ft_tts.sh
```

> [!IMPORTANT]
> Each run **overwrites** the previous checkpoint at `/root/F5-TTS-ckpts/jfk/`. Before starting a new experiment, save the previous checkpoint:
> ```bash
> # Save previous run before starting new one
> cp -r /root/F5-TTS-ckpts/jfk /root/F5-TTS-ckpts/jfk_e10_b4000_lr1e5
> ```

---

## Evaluation Protocol

For each experiment, generate the **same set of test sentences** and compare:

```bash
source /workspace/f5-tts/bin/activate

# Standard evaluation sentences
python run_inference.py --text "Ask not what your country can do for you." --output "eval_ask_not.wav"
python run_inference.py --text "We choose to go to the Moon in this decade." --output "eval_moon.wav"
python run_inference.py --text "The torch has been passed to a new generation of Americans." --output "eval_torch.wav"
python run_inference.py --text "Today we celebrate the birthday of artificial intelligence." --output "eval_unseen.wav"
```

### What to Listen For

| Quality | Good Sign | Bad Sign (Overfitting) |
|---|---|---|
| **Voice match** | Sounds like JFK's cadence and tone | Sounds generic or robotic |
| **Clarity** | Words are clear and natural | Mumbling, slurring, or repeating |
| **Prosody** | Natural pauses and emphasis | Monotone or unnatural rhythm |
| **Unseen text** | Handles novel phrases well | Only sounds good on training-like text |

---

## Suggested Experiment Order

1. **E2 (20 epochs)** — Quick check if more training helps
2. **E3 (30 epochs)** — The most commonly recommended range for voice fine-tuning
3. **B3 (batch 6000)** — Try larger batch with best epoch count
4. **L3 (lr 3e-5)** — If convergence seems slow, try a higher LR
5. **E4 (50 epochs)** — Only if 30 epochs wasn't enough

---

## Naming Convention for Saved Checkpoints

```
/root/F5-TTS-ckpts/
├── jfk/                          ← active (latest run)
├── jfk_e10_b4000_lr1e5/          ← baseline
├── jfk_e30_b4000_lr1e5/          ← round 1 winner
├── jfk_e30_b6000_lr1e5/          ← round 2 test
└── jfk_e30_b6000_lr3e5/          ← round 3 test
```

## Upload Best Result

Once you find the best combination, upload to HuggingFace:

```bash
source /workspace/f5-tts/bin/activate
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='/root/F5-TTS-ckpts/jfk',
    repo_id='czhou578/jfk-voice-clone-f5tts',
    repo_type='model'
)
print('✅ Upload complete!')
"
```
