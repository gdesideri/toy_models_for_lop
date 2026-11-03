# Result 2: training histories and subsequent feature acquisition

Dataset synthetic, mlp_2x64, history lr 0.03, batch 256, T = 40000; protocol `eb0f2fb410d6`, analysis `c31a19c7ae2b`; seeds [0, 1, 2, 3, 4, 5].
Acquisition: targets ['f4', 'f5', 'f6'], 3 head replicas, full / fresh 2000 updates (continued to 8000) at lr 0.03, frozen decoders up to 8000 (lrs [0.01, 0.03, 0.1], weight decays [0.0, 0.0001, 0.001], nonlinear width 64). Plain SGD, momentum 0.

Measurement status: {'complete': 918}

Deficit = fresh - checkpoint test accuracy, update-weighted mean over the primary budget (positive: worse than fresh). Seed means [95% t interval over seeds], targets averaged within seed.

## Family A

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| A_persistent_rho0 | init | 0 | 50.1 [48.4, 51.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0 | repeat_f1 | 20000 | 85.5 [84.1, 87.0] | +1.4 [+1.0, +1.8] | +0.7 [+0.4, +1.0] | +0.7 [+0.4, +1.0] | 74.1 [73.0, 75.2] | 92.2 [91.8, 92.6] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0 | repeat_f1 | 40000 | 85.7 [84.4, 86.9] | +1.5 [+1.1, +1.9] | +0.7 [+0.4, +1.0] | +0.8 [+0.5, +1.1] | 73.8 [72.3, 75.4] | 92.1 [91.7, 92.5] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0 | init | 0 | 50.1 [48.4, 51.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0 | repeat_f1 | 20000 | 85.5 [84.1, 87.0] | +1.4 [+1.0, +1.8] | +0.7 [+0.4, +1.0] | +0.7 [+0.4, +1.0] | 74.1 [73.0, 75.2] | 92.2 [91.8, 92.6] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0 | repeat_f1 | 40000 | 85.7 [84.4, 86.9] | +1.5 [+1.1, +1.9] | +0.7 [+0.4, +1.0] | +0.8 [+0.5, +1.1] | 73.8 [72.3, 75.4] | 92.1 [91.7, 92.5] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.25 | init | 0 | 50.1 [48.4, 51.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.25 | f1_persistent_rho0.25 | 20000 | 70.3 [68.0, 72.6] | +0.8 [+0.6, +1.1] | +1.1 [+0.8, +1.4] | +0.9 [+0.7, +1.2] | 75.4 [73.9, 77.0] | 91.7 [91.5, 91.9] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.25 | f1_persistent_rho0.25 | 40000 | 70.4 [68.0, 72.8] | +0.9 [+0.7, +1.1] | +1.2 [+0.9, +1.5] | +1.0 [+0.8, +1.3] | 75.3 [73.9, 76.7] | 91.6 [91.4, 91.8] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.25 | init | 0 | 50.1 [48.4, 51.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.25 | f1_renewed_rho0.25_L2000 | 20000 | 70.8 [68.8, 72.8] | +1.1 [+0.8, +1.4] | +1.5 [+1.2, +1.7] | +1.2 [+1.0, +1.4] | 75.1 [73.7, 76.5] | 91.4 [91.2, 91.6] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.25 | f1_renewed_rho0.25_L2000 | 40000 | 69.3 [67.0, 71.5] | +1.3 [+1.1, +1.5] | +1.7 [+1.4, +2.0] | +1.5 [+1.3, +1.7] | 75.3 [73.9, 76.7] | 91.1 [91.0, 91.3] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.75 | init | 0 | 50.1 [48.4, 51.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.75 | f1_persistent_rho0.75 | 20000 | 56.1 [54.6, 57.7] | +0.8 [+0.4, +1.2] | +1.3 [+1.1, +1.4] | +1.0 [+0.9, +1.1] | 75.9 [74.7, 77.0] | 91.6 [91.2, 91.9] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.75 | f1_persistent_rho0.75 | 40000 | 56.1 [54.5, 57.6] | +0.9 [+0.5, +1.3] | +1.4 [+1.3, +1.5] | +1.1 [+1.0, +1.3] | 75.8 [74.7, 77.0] | 91.4 [91.1, 91.8] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.75 | init | 0 | 50.1 [48.4, 51.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.75 | f1_renewed_rho0.75_L2000 | 20000 | 56.8 [53.6, 60.0] | +1.1 [+0.3, +1.9] | +1.8 [+1.4, +2.3] | +1.5 [+1.1, +1.9] | 74.4 [72.7, 76.1] | 91.0 [90.4, 91.7] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.75 | f1_renewed_rho0.75_L2000 | 40000 | 56.5 [53.0, 60.1] | +2.1 [+1.6, +2.6] | +2.4 [+2.2, +2.7] | +2.1 [+1.9, +2.3] | 72.4 [71.2, 73.5] | 90.4 [90.1, 90.8] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |

## Family B

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| B_repeat | init | 0 | 50.1 [48.4, 51.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_repeat | repeat_f1 | 20000 | 85.5 [84.1, 87.0] | +1.4 [+1.0, +1.8] | +0.7 [+0.4, +1.0] | +0.7 [+0.4, +1.0] | 74.1 [73.0, 75.2] | 92.2 [91.8, 92.6] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_repeat | repeat_f1 | 40000 | 85.7 [84.4, 86.9] | +1.5 [+1.1, +1.9] | +0.7 [+0.4, +1.0] | +0.8 [+0.5, +1.1] | 73.8 [72.3, 75.4] | 92.1 [91.7, 92.5] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_complement | init | 0 | 51.0 [49.7, 52.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_complement | alt_f1_not_f1_L2000 | 20000 | 86.5 [85.4, 87.5] | +2.1 [+1.8, +2.4] | +0.6 [+0.4, +0.9] | +0.8 [+0.6, +1.0] | 70.1 [68.3, 72.0] | 92.2 [91.8, 92.6] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_complement | alt_f1_not_f1_L2000 | 40000 | 85.9 [85.2, 86.5] | +2.3 [+1.8, +2.7] | +0.8 [+0.5, +1.2] | +0.9 [+0.6, +1.3] | 69.9 [68.2, 71.6] | 92.0 [91.6, 92.4] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_f2 | init | 0 | 51.6 [49.6, 53.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_f2 | alt_f1_f2_L2000 | 20000 | 78.2 [75.5, 81.0] | +1.7 [+1.3, +2.1] | +1.2 [+0.9, +1.5] | +1.1 [+0.9, +1.3] | 73.2 [72.0, 74.5] | 91.7 [91.4, 92.0] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_f2 | alt_f1_f2_L2000 | 40000 | 76.9 [74.2, 79.5] | +1.9 [+1.5, +2.3] | +1.4 [+1.0, +1.7] | +1.3 [+1.0, +1.5] | 73.0 [72.0, 74.0] | 91.5 [91.1, 91.8] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0pi | init | 0 | 50.1 [48.4, 51.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0pi | repeat_f1 | 20000 | 85.5 [84.1, 87.0] | +1.4 [+1.0, +1.8] | +0.7 [+0.4, +1.0] | +0.7 [+0.4, +1.0] | 74.1 [73.0, 75.2] | 92.2 [91.8, 92.6] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0pi | repeat_f1 | 40000 | 85.7 [84.4, 86.9] | +1.5 [+1.1, +1.9] | +0.7 [+0.4, +1.0] | +0.8 [+0.5, +1.1] | 73.8 [72.3, 75.4] | 92.1 [91.7, 92.5] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.125pi | init | 0 | 50.1 [48.7, 51.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.125pi | alt_f1_rot0.125pi_L2000 | 20000 | 87.7 [86.6, 88.8] | +1.5 [+1.1, +1.9] | +0.7 [+0.5, +1.0] | +0.8 [+0.5, +1.0] | 74.2 [72.7, 75.7] | 92.1 [91.8, 92.5] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.125pi | alt_f1_rot0.125pi_L2000 | 40000 | 87.3 [85.8, 88.8] | +1.4 [+1.0, +1.8] | +0.8 [+0.5, +1.1] | +0.8 [+0.6, +1.0] | 74.2 [72.8, 75.6] | 92.1 [91.7, 92.4] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.25pi | init | 0 | 50.6 [48.8, 52.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.25pi | alt_f1_rot0.25pi_L2000 | 20000 | 84.6 [83.7, 85.6] | +1.2 [+0.6, +1.9] | +0.8 [+0.4, +1.2] | +0.8 [+0.4, +1.2] | 73.9 [72.2, 75.5] | 92.1 [91.6, 92.5] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.25pi | alt_f1_rot0.25pi_L2000 | 40000 | 83.5 [82.2, 84.8] | +1.3 [+0.7, +2.0] | +1.0 [+0.6, +1.3] | +0.9 [+0.5, +1.3] | 73.8 [72.1, 75.4] | 91.9 [91.5, 92.3] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.5pi | init | 0 | 51.0 [49.7, 52.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.5pi | alt_f1_not_f1_L2000 | 20000 | 86.5 [85.4, 87.5] | +2.1 [+1.8, +2.4] | +0.6 [+0.4, +0.9] | +0.8 [+0.6, +1.0] | 70.1 [68.3, 72.0] | 92.2 [91.8, 92.6] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.5pi | alt_f1_not_f1_L2000 | 40000 | 85.9 [85.2, 86.5] | +2.3 [+1.8, +2.7] | +0.8 [+0.5, +1.2] | +0.9 [+0.6, +1.3] | 69.9 [68.2, 71.6] | 92.0 [91.6, 92.4] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |

## Family C

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| C_L100 | init | 0 | 51.6 [49.6, 53.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L100 | alt_f1_f2_L100 | 20000 | 72.5 [70.4, 74.7] | +1.2 [+0.8, +1.7] | +1.0 [+0.7, +1.3] | +0.9 [+0.6, +1.2] | 73.8 [72.9, 74.8] | 91.8 [91.7, 92.0] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L100 | alt_f1_f2_L100 | 40000 | 72.0 [70.7, 73.2] | +1.5 [+1.0, +2.0] | +1.3 [+0.9, +1.6] | +1.1 [+0.8, +1.4] | 73.6 [72.6, 74.5] | 91.6 [91.4, 91.8] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L500 | init | 0 | 51.6 [49.6, 53.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L500 | alt_f1_f2_L500 | 20000 | 76.5 [74.4, 78.6] | +1.7 [+1.3, +2.0] | +1.2 [+1.0, +1.4] | +1.1 [+1.0, +1.2] | 73.9 [73.1, 74.7] | 91.7 [91.4, 92.0] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L500 | alt_f1_f2_L500 | 40000 | 75.9 [73.6, 78.2] | +1.9 [+1.5, +2.3] | +1.4 [+1.2, +1.6] | +1.3 [+1.1, +1.4] | 73.5 [73.0, 74.0] | 91.5 [91.1, 91.8] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L2000 | init | 0 | 51.6 [49.6, 53.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L2000 | alt_f1_f2_L2000 | 20000 | 78.2 [75.5, 81.0] | +1.7 [+1.3, +2.1] | +1.2 [+0.9, +1.5] | +1.1 [+0.9, +1.3] | 73.2 [72.0, 74.5] | 91.7 [91.4, 92.0] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L2000 | alt_f1_f2_L2000 | 40000 | 76.9 [74.2, 79.5] | +1.9 [+1.5, +2.3] | +1.4 [+1.0, +1.7] | +1.3 [+1.0, +1.5] | 73.0 [72.0, 74.0] | 91.5 [91.1, 91.8] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L10000 | init | 0 | 51.6 [49.6, 53.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 75.3 [74.2, 76.4] | 92.9 [92.5, 93.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L10000 | alt_f1_f2_L10000 | 20000 | 82.9 [80.0, 85.8] | +1.8 [+1.4, +2.2] | +1.0 [+0.7, +1.2] | +1.0 [+0.7, +1.2] | 73.1 [71.8, 74.5] | 91.9 [91.6, 92.2] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |
| C_L10000 | alt_f1_f2_L10000 | 40000 | 81.0 [78.4, 83.6] | +2.2 [+1.8, +2.6] | +1.3 [+1.1, +1.6] | +1.3 [+1.1, +1.5] | 72.4 [71.4, 73.3] | 91.5 [91.2, 91.9] | 92.9 [92.5, 93.2] | full adaptation beats the tested frozen decoders: 18 |

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
| A | 20000 | A_persistent_rho0 | A_persistent_rho0.25 | -0.6 [-0.8, -0.4] |
| A | 20000 | A_persistent_rho0 | A_persistent_rho0.75 | -0.6 [-1.2, -0.0] |
| A | 20000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 20000 | A_persistent_rho0 | A_renewed_rho0.25 | -0.3 [-0.6, +0.0] |
| A | 20000 | A_persistent_rho0 | A_renewed_rho0.75 | -0.3 [-0.9, +0.3] |
| A | 20000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.0 [-0.5, +0.4] |
| A | 20000 | A_persistent_rho0.25 | A_renewed_rho0 | +0.6 [+0.4, +0.8] |
| A | 20000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +0.3 [+0.1, +0.5] |
| A | 20000 | A_persistent_rho0.25 | A_renewed_rho0.75 | +0.3 [-0.3, +0.9] |
| A | 20000 | A_persistent_rho0.75 | A_renewed_rho0 | +0.6 [+0.0, +1.2] |
| A | 20000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +0.3 [+0.0, +0.6] |
| A | 20000 | A_persistent_rho0.75 | A_renewed_rho0.75 | +0.3 [-0.3, +1.0] |
| A | 20000 | A_renewed_rho0 | A_renewed_rho0.25 | -0.3 [-0.6, +0.0] |
| A | 20000 | A_renewed_rho0 | A_renewed_rho0.75 | -0.3 [-0.9, +0.3] |
| A | 20000 | A_renewed_rho0.25 | A_renewed_rho0.75 | -0.0 [-0.5, +0.5] |
| A | 40000 | A_persistent_rho0 | A_persistent_rho0.25 | -0.6 [-0.8, -0.4] |
| A | 40000 | A_persistent_rho0 | A_persistent_rho0.75 | -0.6 [-1.2, +0.0] |
| A | 40000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 40000 | A_persistent_rho0 | A_renewed_rho0.25 | -0.2 [-0.5, +0.1] |
| A | 40000 | A_persistent_rho0 | A_renewed_rho0.75 | +0.6 [+0.0, +1.2] |
| A | 40000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.0 [-0.5, +0.4] |
| A | 40000 | A_persistent_rho0.25 | A_renewed_rho0 | +0.6 [+0.4, +0.8] |
| A | 40000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +0.4 [+0.2, +0.6] |
| A | 40000 | A_persistent_rho0.25 | A_renewed_rho0.75 | +1.2 [+0.6, +1.7] |
| A | 40000 | A_persistent_rho0.75 | A_renewed_rho0 | +0.6 [-0.0, +1.2] |
| A | 40000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +0.4 [+0.1, +0.8] |
| A | 40000 | A_persistent_rho0.75 | A_renewed_rho0.75 | +1.2 [+0.5, +1.9] |
| A | 40000 | A_renewed_rho0 | A_renewed_rho0.25 | -0.2 [-0.5, +0.1] |
| A | 40000 | A_renewed_rho0 | A_renewed_rho0.75 | +0.6 [+0.0, +1.2] |
| A | 40000 | A_renewed_rho0.25 | A_renewed_rho0.75 | +0.8 [+0.3, +1.2] |
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
| B | 20000 | B_complement | B_f2 | -0.4 [-1.0, +0.2] |
| B | 20000 | B_complement | B_repeat | -0.7 [-1.2, -0.2] |
| B | 20000 | B_complement | B_rot0.125pi | -0.6 [-1.0, -0.3] |
| B | 20000 | B_complement | B_rot0.25pi | -0.9 [-1.4, -0.3] |
| B | 20000 | B_complement | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 20000 | B_complement | B_rot0pi | -0.7 [-1.2, -0.2] |
| B | 20000 | B_f2 | B_repeat | -0.3 [-0.6, +0.1] |
| B | 20000 | B_f2 | B_rot0.125pi | -0.2 [-0.7, +0.3] |
| B | 20000 | B_f2 | B_rot0.25pi | -0.5 [-1.3, +0.4] |
| B | 20000 | B_f2 | B_rot0.5pi | +0.4 [-0.2, +1.0] |
| B | 20000 | B_f2 | B_rot0pi | -0.3 [-0.6, +0.1] |
| B | 20000 | B_repeat | B_rot0.125pi | +0.1 [-0.5, +0.6] |
| B | 20000 | B_repeat | B_rot0.25pi | -0.2 [-1.0, +0.6] |
| B | 20000 | B_repeat | B_rot0.5pi | +0.7 [+0.2, +1.2] |
| B | 20000 | B_repeat | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 20000 | B_rot0.125pi | B_rot0.25pi | -0.3 [-0.7, +0.2] |
| B | 20000 | B_rot0.125pi | B_rot0.5pi | +0.6 [+0.3, +1.0] |
| B | 20000 | B_rot0.125pi | B_rot0pi | -0.1 [-0.6, +0.5] |
| B | 20000 | B_rot0.25pi | B_rot0.5pi | +0.9 [+0.3, +1.4] |
| B | 20000 | B_rot0.25pi | B_rot0pi | +0.2 [-0.6, +1.0] |
| B | 20000 | B_rot0.5pi | B_rot0pi | -0.7 [-1.2, -0.2] |
| B | 40000 | B_complement | B_f2 | -0.4 [-1.2, +0.4] |
| B | 40000 | B_complement | B_repeat | -0.8 [-1.5, -0.0] |
| B | 40000 | B_complement | B_rot0.125pi | -0.8 [-1.3, -0.3] |
| B | 40000 | B_complement | B_rot0.25pi | -0.9 [-1.4, -0.5] |
| B | 40000 | B_complement | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 40000 | B_complement | B_rot0pi | -0.8 [-1.5, -0.0] |
| B | 40000 | B_f2 | B_repeat | -0.4 [-0.8, +0.0] |
| B | 40000 | B_f2 | B_rot0.125pi | -0.5 [-1.0, +0.1] |
| B | 40000 | B_f2 | B_rot0.25pi | -0.5 [-1.4, +0.3] |
| B | 40000 | B_f2 | B_rot0.5pi | +0.4 [-0.4, +1.2] |
| B | 40000 | B_f2 | B_rot0pi | -0.4 [-0.8, +0.0] |
| B | 40000 | B_repeat | B_rot0.125pi | -0.0 [-0.6, +0.5] |
| B | 40000 | B_repeat | B_rot0.25pi | -0.1 [-0.9, +0.7] |
| B | 40000 | B_repeat | B_rot0.5pi | +0.8 [+0.0, +1.5] |
| B | 40000 | B_repeat | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 40000 | B_rot0.125pi | B_rot0.25pi | -0.1 [-0.5, +0.3] |
| B | 40000 | B_rot0.125pi | B_rot0.5pi | +0.8 [+0.3, +1.3] |
| B | 40000 | B_rot0.125pi | B_rot0pi | +0.0 [-0.5, +0.6] |
| B | 40000 | B_rot0.25pi | B_rot0.5pi | +0.9 [+0.5, +1.4] |
| B | 40000 | B_rot0.25pi | B_rot0pi | +0.1 [-0.7, +0.9] |
| B | 40000 | B_rot0.5pi | B_rot0pi | -0.8 [-1.5, -0.0] |
| C | 0 | C_L100 | C_L10000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L100 | C_L2000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L100 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L2000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2000 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 20000 | C_L100 | C_L10000 | +0.6 [+0.0, +1.1] |
| C | 20000 | C_L100 | C_L2000 | +0.5 [-0.2, +1.1] |
| C | 20000 | C_L100 | C_L500 | +0.4 [-0.2, +1.0] |
| C | 20000 | C_L10000 | C_L2000 | -0.1 [-0.4, +0.2] |
| C | 20000 | C_L10000 | C_L500 | -0.1 [-0.5, +0.3] |
| C | 20000 | C_L2000 | C_L500 | -0.0 [-0.3, +0.3] |
| C | 40000 | C_L100 | C_L10000 | +0.7 [+0.0, +1.4] |
| C | 40000 | C_L100 | C_L2000 | +0.4 [-0.4, +1.2] |
| C | 40000 | C_L100 | C_L500 | +0.4 [-0.3, +1.1] |
| C | 40000 | C_L10000 | C_L2000 | -0.3 [-0.5, -0.1] |
| C | 40000 | C_L10000 | C_L500 | -0.3 [-0.7, +0.1] |
| C | 40000 | C_L2000 | C_L500 | -0.0 [-0.4, +0.3] |

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
