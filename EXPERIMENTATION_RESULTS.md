### Round 1: Epoch Sweep (keep batch size = 4000)

| Experiment | Epochs | Expected Time | What to Watch |
|---|---|---|---|
| E1 (baseline) | 10 | ~10 min | Already done — loss ~0.7 |
| E2 | 20 | ~20 min | Does loss plateau or keep dropping? | Final loss = 0.644 |
| E3 | 30 | ~30 min | Better accent fidelity? Check for overfitting | Final loss = 0.602 |
| E4 | 50 | ~50 min | Likely overfitting territory — compare quality | Final loss = 0.97 ⚠️ overfitting |

### Round 2: Batch Size Sweep (use best epoch count from Round 1)

| Experiment | Batch Size (frames) | VRAM Usage | Effect |
|---|---|---|---|
| B1 | 2000 | ~12 GB | More updates per epoch, noisier gradients |
| B2 (baseline) | 4000 | ~16 GB | Current setting | Already done - loss ~0.7 |
| B3 | 6000 | ~20 GB | Smoother gradients, fewer updates per epoch | Final loss = 0.487 |
| B4 | 8000 | ~23 GB | Max for 24GB GPU — may OOM, try with caution |