# F5-TTS Fine-Tuning Experiment Results

Final training loss of each run from `EXPERIMENTATION_PLAN.md`. Dataset: 1,151 clips, 2.35 hours. Lower is better, but a very low loss with poor audio means overfitting.

### Round 1: Epoch Sweep (batch size 4000, learning rate 1e-5)

| Experiment | Epochs | Final loss | Note |
|---|---|---|---|
| E1 (baseline) | 10 | ~0.70 | |
| E2 | 20 | 0.644 | |
| E3 | 30 | 0.602 | Best of round 1 |
| E4 | 50 | 0.97 | Overfitting: loss rose, output repeats |

### Round 2: Batch Size Sweep (30 epochs, learning rate 1e-5)

| Experiment | Batch size (frames) | GPU memory | Final loss | Note |
|---|---|---|---|---|
| B1 | 2000 | ~12 GB | not run | |
| B2 (baseline) | 4000 | ~16 GB | 0.602 | Same run as E3 |
| B3 | 6000 | ~20 GB | 0.487 | Best run overall; the default in `finetune_ft_tts.sh` |

### Round 3: Learning Rate Sweep (30 epochs, batch size 6000)

| Experiment | Learning rate | Final loss | Note |
|---|---|---|---|
| L1 | 5e-6 | 0.571 | |
| L2 (baseline) | 1e-5 | 0.487 | Same run as B3 |
| L3 | 3e-5 | 0.528 | |
| L4 | 5e-5 | 0.875 | Unstable |
