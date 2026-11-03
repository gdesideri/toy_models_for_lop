# Result 2: training histories and subsequent feature acquisition

Dataset synthetic, mlp_2x256, history lr 0.03, batch 256, T = 40000; protocol `e616a01efb47`, analysis `c31a19c7ae2b`; seeds [0, 1, 2, 3, 4, 5].
Acquisition: targets ['f4', 'f5', 'f6'], 3 head replicas, full / fresh 2000 updates (continued to 8000) at lr 0.03, frozen decoders up to 8000 (lrs [0.01, 0.03, 0.1], weight decays [0.0, 0.0001, 0.001], nonlinear width 64). Plain SGD, momentum 0.

Measurement status: {'complete': 918}

Deficit = fresh - checkpoint test accuracy, update-weighted mean over the primary budget (positive: worse than fresh). Seed means [95% t interval over seeds], targets averaged within seed.

## Family A

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| A_persistent_rho0 | init | 0 | 49.8 [47.9, 51.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| A_persistent_rho0 | repeat_f1 | 20000 | 83.7 [82.5, 84.9] | +0.5 [+0.3, +0.6] | +0.3 [+0.1, +0.4] | +0.3 [+0.2, +0.4] | 87.3 [87.0, 87.6] | 91.4 [91.0, 91.8] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| A_persistent_rho0 | repeat_f1 | 40000 | 83.9 [82.7, 85.1] | +0.5 [+0.4, +0.7] | +0.3 [+0.2, +0.4] | +0.3 [+0.2, +0.4] | 87.3 [87.0, 87.6] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| A_renewed_rho0 | init | 0 | 49.8 [47.9, 51.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| A_renewed_rho0 | repeat_f1 | 20000 | 83.7 [82.5, 84.9] | +0.5 [+0.3, +0.6] | +0.3 [+0.1, +0.4] | +0.3 [+0.2, +0.4] | 87.3 [87.0, 87.6] | 91.4 [91.0, 91.8] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| A_renewed_rho0 | repeat_f1 | 40000 | 83.9 [82.7, 85.1] | +0.5 [+0.4, +0.7] | +0.3 [+0.2, +0.4] | +0.3 [+0.2, +0.4] | 87.3 [87.0, 87.6] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| A_persistent_rho0.25 | init | 0 | 49.8 [47.9, 51.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| A_persistent_rho0.25 | f1_persistent_rho0.25 | 20000 | 70.5 [68.6, 72.4] | +0.3 [+0.1, +0.4] | +0.2 [+0.1, +0.3] | +0.2 [+0.1, +0.3] | 87.6 [87.2, 88.0] | 91.4 [91.1, 91.8] | 91.7 [91.3, 92.0] | no difference beyond the thresholds: 18 |
| A_persistent_rho0.25 | f1_persistent_rho0.25 | 40000 | 70.4 [68.7, 72.2] | +0.3 [+0.2, +0.4] | +0.3 [+0.1, +0.4] | +0.2 [+0.1, +0.4] | 87.6 [87.4, 87.8] | 91.4 [91.0, 91.8] | 91.7 [91.3, 92.0] | no difference beyond the thresholds: 18 |
| A_renewed_rho0.25 | init | 0 | 49.8 [47.9, 51.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| A_renewed_rho0.25 | f1_renewed_rho0.25_L2000 | 20000 | 70.5 [68.0, 72.9] | +0.3 [+0.2, +0.4] | +0.2 [+0.1, +0.3] | +0.2 [+0.2, +0.3] | 87.7 [87.3, 88.1] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| A_renewed_rho0.25 | f1_renewed_rho0.25_L2000 | 40000 | 70.3 [68.5, 72.0] | +0.3 [+0.2, +0.4] | +0.3 [+0.2, +0.4] | +0.3 [+0.2, +0.4] | 87.7 [87.2, 88.1] | 91.4 [91.0, 91.8] | 91.7 [91.3, 92.0] | no difference beyond the thresholds: 18 |
| A_persistent_rho0.75 | init | 0 | 49.8 [47.9, 51.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| A_persistent_rho0.75 | f1_persistent_rho0.75 | 20000 | 57.7 [56.0, 59.3] | +0.2 [+0.1, +0.3] | +0.2 [+0.1, +0.3] | +0.2 [+0.1, +0.3] | 87.6 [87.2, 87.9] | 91.5 [91.1, 91.8] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| A_persistent_rho0.75 | f1_persistent_rho0.75 | 40000 | 57.7 [56.0, 59.3] | +0.2 [+0.1, +0.3] | +0.2 [+0.1, +0.3] | +0.2 [+0.1, +0.3] | 87.6 [87.2, 88.0] | 91.5 [91.1, 91.8] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| A_renewed_rho0.75 | init | 0 | 49.8 [47.9, 51.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| A_renewed_rho0.75 | f1_renewed_rho0.75_L2000 | 20000 | 57.3 [54.9, 59.7] | +0.1 [-0.2, +0.3] | -0.0 [-0.2, +0.1] | -0.0 [-0.1, +0.1] | 85.0 [84.5, 85.6] | 91.7 [91.5, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.75 | f1_renewed_rho0.75_L2000 | 40000 | 55.8 [53.4, 58.1] | +0.2 [-0.0, +0.4] | -0.1 [-0.3, +0.1] | -0.0 [-0.2, +0.1] | 84.0 [83.2, 84.7] | 91.8 [91.5, 92.1] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 18 |

## Family B

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| B_repeat | init | 0 | 49.8 [47.9, 51.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| B_repeat | repeat_f1 | 20000 | 83.7 [82.5, 84.9] | +0.5 [+0.3, +0.6] | +0.3 [+0.1, +0.4] | +0.3 [+0.2, +0.4] | 87.3 [87.0, 87.6] | 91.4 [91.0, 91.8] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| B_repeat | repeat_f1 | 40000 | 83.9 [82.7, 85.1] | +0.5 [+0.4, +0.7] | +0.3 [+0.2, +0.4] | +0.3 [+0.2, +0.4] | 87.3 [87.0, 87.6] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| B_complement | init | 0 | 49.4 [47.6, 51.2] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| B_complement | alt_f1_not_f1_L2000 | 20000 | 84.0 [82.7, 85.4] | +0.2 [+0.1, +0.3] | -0.1 [-0.2, -0.0] | -0.0 [-0.1, +0.0] | 85.5 [84.6, 86.4] | 91.8 [91.4, 92.1] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 16; no difference beyond the thresholds: 2 |
| B_complement | alt_f1_not_f1_L2000 | 40000 | 83.4 [82.1, 84.8] | +0.2 [+0.1, +0.3] | -0.0 [-0.2, +0.1] | -0.0 [-0.1, +0.1] | 85.0 [84.2, 85.7] | 91.7 [91.3, 92.1] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 17; no difference beyond the thresholds: 1 |
| B_f2 | init | 0 | 48.8 [47.2, 50.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| B_f2 | alt_f1_f2_L2000 | 20000 | 77.8 [75.9, 79.7] | +0.3 [+0.3, +0.4] | +0.3 [+0.2, +0.3] | +0.2 [+0.2, +0.3] | 87.1 [86.5, 87.7] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| B_f2 | alt_f1_f2_L2000 | 40000 | 77.1 [75.1, 79.1] | +0.3 [+0.2, +0.4] | +0.2 [+0.1, +0.3] | +0.2 [+0.1, +0.3] | 86.7 [86.1, 87.3] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 8; no difference beyond the thresholds: 10 |
| B_rot0pi | init | 0 | 49.8 [47.9, 51.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| B_rot0pi | repeat_f1 | 20000 | 83.7 [82.5, 84.9] | +0.5 [+0.3, +0.6] | +0.3 [+0.1, +0.4] | +0.3 [+0.2, +0.4] | 87.3 [87.0, 87.6] | 91.4 [91.0, 91.8] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| B_rot0pi | repeat_f1 | 40000 | 83.9 [82.7, 85.1] | +0.5 [+0.4, +0.7] | +0.3 [+0.2, +0.4] | +0.3 [+0.2, +0.4] | 87.3 [87.0, 87.6] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| B_rot0.125pi | init | 0 | 49.8 [47.4, 52.2] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| B_rot0.125pi | alt_f1_rot0.125pi_L2000 | 20000 | 85.4 [84.4, 86.5] | +0.6 [+0.4, +0.7] | +0.3 [+0.2, +0.5] | +0.3 [+0.2, +0.5] | 87.2 [86.8, 87.6] | 91.3 [91.0, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 2; no difference beyond the thresholds: 16 |
| B_rot0.125pi | alt_f1_rot0.125pi_L2000 | 40000 | 85.1 [83.9, 86.3] | +0.6 [+0.4, +0.7] | +0.3 [+0.1, +0.5] | +0.3 [+0.2, +0.5] | 87.2 [86.7, 87.6] | 91.3 [91.0, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 3; no difference beyond the thresholds: 15 |
| B_rot0.25pi | init | 0 | 49.4 [47.7, 51.1] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| B_rot0.25pi | alt_f1_rot0.25pi_L2000 | 20000 | 83.2 [81.9, 84.5] | +0.3 [+0.2, +0.5] | +0.2 [+0.0, +0.4] | +0.2 [+0.1, +0.4] | 87.0 [86.6, 87.3] | 91.5 [91.2, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 3; no difference beyond the thresholds: 15 |
| B_rot0.25pi | alt_f1_rot0.25pi_L2000 | 40000 | 81.9 [80.3, 83.4] | +0.3 [+0.1, +0.5] | +0.2 [+0.0, +0.4] | +0.2 [+0.1, +0.4] | 86.8 [86.5, 87.1] | 91.5 [91.2, 91.8] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 6; no difference beyond the thresholds: 12 |
| B_rot0.5pi | init | 0 | 49.4 [47.6, 51.2] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| B_rot0.5pi | alt_f1_not_f1_L2000 | 20000 | 84.0 [82.7, 85.4] | +0.2 [+0.1, +0.3] | -0.1 [-0.2, -0.0] | -0.0 [-0.1, +0.0] | 85.5 [84.6, 86.4] | 91.8 [91.4, 92.1] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 16; no difference beyond the thresholds: 2 |
| B_rot0.5pi | alt_f1_not_f1_L2000 | 40000 | 83.4 [82.1, 84.8] | +0.2 [+0.1, +0.3] | -0.0 [-0.2, +0.1] | -0.0 [-0.1, +0.1] | 85.0 [84.2, 85.7] | 91.7 [91.3, 92.1] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 17; no difference beyond the thresholds: 1 |

## Family C

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| C_L100 | init | 0 | 48.8 [47.2, 50.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| C_L100 | alt_f1_f2_L100 | 20000 | 73.2 [71.1, 75.4] | +0.4 [+0.2, +0.5] | +0.3 [+0.3, +0.4] | +0.3 [+0.2, +0.4] | 85.8 [85.3, 86.3] | 91.3 [91.0, 91.6] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 14; no difference beyond the thresholds: 4 |
| C_L100 | alt_f1_f2_L100 | 40000 | 73.6 [71.9, 75.2] | +0.5 [+0.3, +0.7] | +0.4 [+0.2, +0.6] | +0.4 [+0.2, +0.5] | 85.0 [84.6, 85.4] | 91.3 [91.1, 91.5] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 18 |
| C_L500 | init | 0 | 48.8 [47.2, 50.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| C_L500 | alt_f1_f2_L500 | 20000 | 76.0 [74.3, 77.7] | +0.3 [+0.1, +0.4] | +0.2 [+0.2, +0.3] | +0.2 [+0.1, +0.3] | 86.5 [86.0, 87.0] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 8; no difference beyond the thresholds: 10 |
| C_L500 | alt_f1_f2_L500 | 40000 | 75.0 [73.4, 76.5] | +0.3 [+0.2, +0.4] | +0.3 [+0.2, +0.3] | +0.3 [+0.2, +0.3] | 86.2 [85.6, 86.7] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 11; no difference beyond the thresholds: 7 |
| C_L2000 | init | 0 | 48.8 [47.2, 50.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| C_L2000 | alt_f1_f2_L2000 | 20000 | 77.8 [75.9, 79.7] | +0.3 [+0.3, +0.4] | +0.3 [+0.2, +0.3] | +0.2 [+0.2, +0.3] | 87.1 [86.5, 87.7] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| C_L2000 | alt_f1_f2_L2000 | 40000 | 77.1 [75.1, 79.1] | +0.3 [+0.2, +0.4] | +0.2 [+0.1, +0.3] | +0.2 [+0.1, +0.3] | 86.7 [86.1, 87.3] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 8; no difference beyond the thresholds: 10 |
| C_L10000 | init | 0 | 48.8 [47.2, 50.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 87.3 [86.8, 87.7] | 91.7 [91.3, 92.0] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |
| C_L10000 | alt_f1_f2_L10000 | 20000 | 81.8 [80.1, 83.4] | +0.5 [+0.3, +0.6] | +0.3 [+0.2, +0.4] | +0.3 [+0.2, +0.4] | 86.9 [86.5, 87.3] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 3; no difference beyond the thresholds: 15 |
| C_L10000 | alt_f1_f2_L10000 | 40000 | 79.9 [78.5, 81.3] | +0.4 [+0.3, +0.5] | +0.2 [+0.1, +0.3] | +0.2 [+0.1, +0.3] | 87.1 [86.6, 87.7] | 91.4 [91.1, 91.7] | 91.7 [91.3, 92.0] | full adaptation beats the tested frozen decoders: 4; no difference beyond the thresholds: 14 |

## Memorisation in comparison A (history bank, each checkpoint against the label table of the last block it was trained on)

| condition | age | overwrite fraction | labels actually changed | assigned-label acc | overwritten subset: assigned / clean acc | kept subset clean acc | status |
|---|---|---|---|---|---|---|---|
| A_persistent_rho0.25 | 20000 | 0.250 | 0.124 | 100.0 | 100.0 / 50.4 | 100.0 | overwritten labels fitted |
| A_persistent_rho0.25 | 40000 | 0.250 | 0.124 | 100.0 | 100.0 / 50.4 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.25 | 20000 | 0.250 | 0.126 | 100.0 | 100.0 / 49.7 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.25 | 40000 | 0.250 | 0.136 | 100.0 | 100.0 / 45.6 | 100.0 | overwritten labels fitted |
| A_persistent_rho0.75 | 20000 | 0.750 | 0.362 | 100.0 | 100.0 / 51.7 | 100.0 | overwritten labels fitted |
| A_persistent_rho0.75 | 40000 | 0.750 | 0.362 | 100.0 | 100.0 / 51.7 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.75 | 20000 | 0.750 | 0.371 | 100.0 | 100.0 / 50.5 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.75 | 40000 | 0.750 | 0.385 | 100.0 | 100.0 / 48.7 | 100.0 | overwritten labels fitted |

## Paired contrasts within families (full adaptation, target mean, primary budget)

| family | age | x | y | y - x deficit (points) |
|---|---|---|---|---|
| A | 0 | A_persistent_rho0 | A_persistent_rho0.25 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0 | A_persistent_rho0.75 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0 | A_renewed_rho0.25 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0 | A_renewed_rho0.75 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0.25 | A_persistent_rho0.75 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0.25 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0.25 | A_renewed_rho0.25 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0.25 | A_renewed_rho0.75 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0.75 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0.75 | A_renewed_rho0.25 | +0.0 [+0.0, +0.0] |
| A | 0 | A_persistent_rho0.75 | A_renewed_rho0.75 | +0.0 [+0.0, +0.0] |
| A | 0 | A_renewed_rho0 | A_renewed_rho0.25 | +0.0 [+0.0, +0.0] |
| A | 0 | A_renewed_rho0 | A_renewed_rho0.75 | +0.0 [+0.0, +0.0] |
| A | 0 | A_renewed_rho0.25 | A_renewed_rho0.75 | +0.0 [+0.0, +0.0] |
| A | 20000 | A_persistent_rho0 | A_persistent_rho0.25 | -0.2 [-0.4, -0.1] |
| A | 20000 | A_persistent_rho0 | A_persistent_rho0.75 | -0.3 [-0.5, -0.1] |
| A | 20000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 20000 | A_persistent_rho0 | A_renewed_rho0.25 | -0.2 [-0.4, +0.0] |
| A | 20000 | A_persistent_rho0 | A_renewed_rho0.75 | -0.4 [-0.6, -0.3] |
| A | 20000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.1 [-0.2, +0.0] |
| A | 20000 | A_persistent_rho0.25 | A_renewed_rho0 | +0.2 [+0.1, +0.4] |
| A | 20000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +0.1 [-0.1, +0.2] |
| A | 20000 | A_persistent_rho0.25 | A_renewed_rho0.75 | -0.2 [-0.4, -0.0] |
| A | 20000 | A_persistent_rho0.75 | A_renewed_rho0 | +0.3 [+0.1, +0.5] |
| A | 20000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +0.2 [-0.0, +0.3] |
| A | 20000 | A_persistent_rho0.75 | A_renewed_rho0.75 | -0.1 [-0.4, +0.1] |
| A | 20000 | A_renewed_rho0 | A_renewed_rho0.25 | -0.2 [-0.4, +0.0] |
| A | 20000 | A_renewed_rho0 | A_renewed_rho0.75 | -0.4 [-0.6, -0.3] |
| A | 20000 | A_renewed_rho0.25 | A_renewed_rho0.75 | -0.3 [-0.5, -0.0] |
| A | 40000 | A_persistent_rho0 | A_persistent_rho0.25 | -0.2 [-0.4, -0.1] |
| A | 40000 | A_persistent_rho0 | A_persistent_rho0.75 | -0.3 [-0.5, -0.1] |
| A | 40000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 40000 | A_persistent_rho0 | A_renewed_rho0.25 | -0.2 [-0.4, -0.0] |
| A | 40000 | A_persistent_rho0 | A_renewed_rho0.75 | -0.4 [-0.5, -0.2] |
| A | 40000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.1 [-0.2, +0.0] |
| A | 40000 | A_persistent_rho0.25 | A_renewed_rho0 | +0.2 [+0.1, +0.4] |
| A | 40000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +0.0 [-0.1, +0.1] |
| A | 40000 | A_persistent_rho0.25 | A_renewed_rho0.75 | -0.1 [-0.3, -0.0] |
| A | 40000 | A_persistent_rho0.75 | A_renewed_rho0 | +0.3 [+0.1, +0.5] |
| A | 40000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +0.1 [-0.0, +0.2] |
| A | 40000 | A_persistent_rho0.75 | A_renewed_rho0.75 | -0.0 [-0.2, +0.2] |
| A | 40000 | A_renewed_rho0 | A_renewed_rho0.25 | -0.2 [-0.4, -0.0] |
| A | 40000 | A_renewed_rho0 | A_renewed_rho0.75 | -0.4 [-0.5, -0.2] |
| A | 40000 | A_renewed_rho0.25 | A_renewed_rho0.75 | -0.1 [-0.4, +0.1] |
| B | 0 | B_complement | B_f2 | +0.0 [+0.0, +0.0] |
| B | 0 | B_complement | B_repeat | +0.0 [+0.0, +0.0] |
| B | 0 | B_complement | B_rot0.125pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_complement | B_rot0.25pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_complement | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_complement | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_f2 | B_repeat | +0.0 [+0.0, +0.0] |
| B | 0 | B_f2 | B_rot0.125pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_f2 | B_rot0.25pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_f2 | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_f2 | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_repeat | B_rot0.125pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_repeat | B_rot0.25pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_repeat | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_repeat | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_rot0.125pi | B_rot0.25pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_rot0.125pi | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_rot0.125pi | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_rot0.25pi | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_rot0.25pi | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 0 | B_rot0.5pi | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 20000 | B_complement | B_f2 | +0.1 [-0.0, +0.3] |
| B | 20000 | B_complement | B_repeat | +0.3 [+0.2, +0.4] |
| B | 20000 | B_complement | B_rot0.125pi | +0.4 [+0.2, +0.5] |
| B | 20000 | B_complement | B_rot0.25pi | +0.1 [-0.0, +0.3] |
| B | 20000 | B_complement | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 20000 | B_complement | B_rot0pi | +0.3 [+0.2, +0.4] |
| B | 20000 | B_f2 | B_repeat | +0.2 [-0.1, +0.4] |
| B | 20000 | B_f2 | B_rot0.125pi | +0.2 [+0.1, +0.4] |
| B | 20000 | B_f2 | B_rot0.25pi | +0.0 [-0.2, +0.2] |
| B | 20000 | B_f2 | B_rot0.5pi | -0.1 [-0.3, +0.0] |
| B | 20000 | B_f2 | B_rot0pi | +0.2 [-0.1, +0.4] |
| B | 20000 | B_repeat | B_rot0.125pi | +0.1 [-0.0, +0.2] |
| B | 20000 | B_repeat | B_rot0.25pi | -0.1 [-0.3, +0.0] |
| B | 20000 | B_repeat | B_rot0.5pi | -0.3 [-0.4, -0.2] |
| B | 20000 | B_repeat | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 20000 | B_rot0.125pi | B_rot0.25pi | -0.2 [-0.4, -0.0] |
| B | 20000 | B_rot0.125pi | B_rot0.5pi | -0.4 [-0.5, -0.2] |
| B | 20000 | B_rot0.125pi | B_rot0pi | -0.1 [-0.2, +0.0] |
| B | 20000 | B_rot0.25pi | B_rot0.5pi | -0.1 [-0.3, +0.0] |
| B | 20000 | B_rot0.25pi | B_rot0pi | +0.1 [-0.0, +0.3] |
| B | 20000 | B_rot0.5pi | B_rot0pi | +0.3 [+0.2, +0.4] |
| B | 40000 | B_complement | B_f2 | +0.1 [-0.1, +0.3] |
| B | 40000 | B_complement | B_repeat | +0.3 [+0.2, +0.4] |
| B | 40000 | B_complement | B_rot0.125pi | +0.4 [+0.2, +0.5] |
| B | 40000 | B_complement | B_rot0.25pi | +0.1 [-0.1, +0.3] |
| B | 40000 | B_complement | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 40000 | B_complement | B_rot0pi | +0.3 [+0.2, +0.4] |
| B | 40000 | B_f2 | B_repeat | +0.2 [-0.0, +0.4] |
| B | 40000 | B_f2 | B_rot0.125pi | +0.3 [+0.1, +0.5] |
| B | 40000 | B_f2 | B_rot0.25pi | +0.0 [-0.2, +0.3] |
| B | 40000 | B_f2 | B_rot0.5pi | -0.1 [-0.3, +0.1] |
| B | 40000 | B_f2 | B_rot0pi | +0.2 [-0.0, +0.4] |
| B | 40000 | B_repeat | B_rot0.125pi | +0.1 [-0.1, +0.2] |
| B | 40000 | B_repeat | B_rot0.25pi | -0.2 [-0.3, -0.1] |
| B | 40000 | B_repeat | B_rot0.5pi | -0.3 [-0.4, -0.2] |
| B | 40000 | B_repeat | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 40000 | B_rot0.125pi | B_rot0.25pi | -0.2 [-0.4, -0.0] |
| B | 40000 | B_rot0.125pi | B_rot0.5pi | -0.4 [-0.5, -0.2] |
| B | 40000 | B_rot0.125pi | B_rot0pi | -0.1 [-0.2, +0.1] |
| B | 40000 | B_rot0.25pi | B_rot0.5pi | -0.1 [-0.3, +0.1] |
| B | 40000 | B_rot0.25pi | B_rot0pi | +0.2 [+0.1, +0.3] |
| B | 40000 | B_rot0.5pi | B_rot0pi | +0.3 [+0.2, +0.4] |
| C | 0 | C_L100 | C_L10000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L100 | C_L2000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L100 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L2000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2000 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 20000 | C_L100 | C_L10000 | +0.1 [-0.1, +0.3] |
| C | 20000 | C_L100 | C_L2000 | -0.0 [-0.2, +0.1] |
| C | 20000 | C_L100 | C_L500 | -0.1 [-0.2, -0.0] |
| C | 20000 | C_L10000 | C_L2000 | -0.1 [-0.2, -0.1] |
| C | 20000 | C_L10000 | C_L500 | -0.2 [-0.3, -0.1] |
| C | 20000 | C_L2000 | C_L500 | -0.1 [-0.2, +0.0] |
| C | 40000 | C_L100 | C_L10000 | -0.1 [-0.3, +0.1] |
| C | 40000 | C_L100 | C_L2000 | -0.2 [-0.4, +0.0] |
| C | 40000 | C_L100 | C_L500 | -0.2 [-0.4, +0.0] |
| C | 40000 | C_L10000 | C_L2000 | -0.1 [-0.2, -0.0] |
| C | 40000 | C_L10000 | C_L500 | -0.1 [-0.3, +0.0] |
| C | 40000 | C_L2000 | C_L500 | +0.0 [-0.1, +0.1] |

## Result 1 artifacts

- (none): not provided

## Statements and limitations

- Positive deficit = the historical checkpoint learns the target worse than the fresh network (same seed, head weights and minibatches); negative = beneficial transfer. Differences between histories can exist while every history outperforms fresh networks.
- Worse current-task performance (e.g. under rapid switching) is not loss of plasticity by itself; the outcome is subsequent acquisition under a fixed diagnostic budget.
- A gap that disappears with the continued (longer) trajectory supports a finite-budget barrier, not a permanent inability.
- Frozen-decoder insufficiency does not prove that the information is absent from the representation; only the tested decoder classes, grids and budgets are limited.
- A fair-coin overwrite changes about half of the overwritten labels: rho is the overwrite fraction, not the mislabelling fraction (the actual disagreement is reported). If overwritten labels are not fitted, the history did not memorise them.
- The complement is an output-remapping control; not every rule change demands a new feature. Rotations keep the population difficulty of the rule; label disagreement need not order acquisition monotonically.
- History lrs, budgets, block lengths and decoder grids are pilot settings, not established optima.
- Qualification thresholds (descriptive, not tests): {"fresh_learnable": 0.9, "frozen_solves": 0.95, "gap_pp": 5.0, "threshold": 0.9}.
