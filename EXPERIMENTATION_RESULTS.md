### Round 1: Epoch Sweep (keep batch size = 4000)

| Experiment | Epochs | Expected Time | What to Watch |
|---|---|---|---|
| E1 (baseline) | 10 | ~10 min | Already done — loss ~0.7 |
| E2 | 20 | ~20 min | Does loss plateau or keep dropping? | Final loss = 0.644 |
| E3 | 30 | ~30 min | Better accent fidelity? Check for overfitting | Final loss = 0.602 |
| E4 | 50 | ~50 min | Likely overfitting territory — compare quality | Final loss = 0.97 ⚠️ overfitting |