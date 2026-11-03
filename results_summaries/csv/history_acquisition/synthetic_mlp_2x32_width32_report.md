# Result 2: training histories and subsequent feature acquisition

Dataset synthetic, mlp_2x32, history lr 0.03, batch 256, T = 40000; protocol `726e7934832f`, analysis `c31a19c7ae2b`; seeds [0, 1, 2, 3, 4, 5].
Acquisition: targets ['f4', 'f5', 'f6'], 3 head replicas, full / fresh 2000 updates (continued to 8000) at lr 0.03, frozen decoders up to 8000 (lrs [0.01, 0.03, 0.1], weight decays [0.0, 0.0001, 0.001], nonlinear width 64). Plain SGD, momentum 0.

Measurement status: {'complete': 918}

Deficit = fresh - checkpoint test accuracy, update-weighted mean over the primary budget (positive: worse than fresh). Seed means [95% t interval over seeds], targets averaged within seed.

## Family A

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| A_persistent_rho0 | init | 0 | 51.7 [49.6, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0 | repeat_f1 | 20000 | 88.5 [86.9, 90.2] | +3.0 [+1.8, +4.1] | +0.9 [+0.3, +1.5] | +1.0 [+0.5, +1.5] | 63.7 [61.3, 66.0] | 93.1 [92.3, 94.0] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0 | repeat_f1 | 40000 | 88.5 [86.9, 90.1] | +3.1 [+1.9, +4.3] | +1.0 [+0.3, +1.6] | +1.1 [+0.5, +1.7] | 63.5 [61.2, 65.8] | 93.0 [92.1, 93.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0 | init | 0 | 51.7 [49.6, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0 | repeat_f1 | 20000 | 88.5 [86.9, 90.2] | +3.0 [+1.8, +4.1] | +0.9 [+0.3, +1.5] | +1.0 [+0.5, +1.5] | 63.7 [61.3, 66.0] | 93.1 [92.3, 94.0] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0 | repeat_f1 | 40000 | 88.5 [86.9, 90.1] | +3.1 [+1.9, +4.3] | +1.0 [+0.3, +1.6] | +1.1 [+0.5, +1.7] | 63.5 [61.2, 65.8] | 93.0 [92.1, 93.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.25 | init | 0 | 51.7 [49.6, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.25 | f1_persistent_rho0.25 | 20000 | 70.8 [69.2, 72.4] | +3.0 [+2.1, +4.0] | +2.4 [+2.1, +2.8] | +2.0 [+1.6, +2.3] | 66.6 [64.9, 68.4] | 91.6 [91.0, 92.1] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.25 | f1_persistent_rho0.25 | 40000 | 70.6 [68.8, 72.3] | +3.3 [+2.3, +4.2] | +2.7 [+2.4, +3.0] | +2.2 [+1.8, +2.5] | 66.6 [64.8, 68.5] | 91.3 [90.7, 91.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.25 | init | 0 | 51.7 [49.6, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.25 | f1_renewed_rho0.25_L2000 | 20000 | 70.1 [67.3, 72.9] | +3.9 [+3.0, +4.7] | +3.0 [+2.7, +3.2] | +2.4 [+2.2, +2.6] | 65.0 [62.9, 67.1] | 91.0 [90.4, 91.6] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.25 | f1_renewed_rho0.25_L2000 | 40000 | 69.2 [68.7, 69.8] | +4.5 [+3.6, +5.5] | +3.7 [+3.4, +4.0] | +2.9 [+2.6, +3.2] | 65.3 [63.1, 67.5] | 90.3 [89.7, 91.0] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.75 | init | 0 | 51.7 [49.6, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.75 | f1_persistent_rho0.75 | 20000 | 56.5 [55.1, 57.9] | +2.6 [+1.1, +4.2] | +2.8 [+2.0, +3.5] | +2.1 [+1.4, +2.9] | 66.1 [65.0, 67.2] | 91.2 [90.6, 91.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_persistent_rho0.75 | f1_persistent_rho0.75 | 40000 | 56.4 [55.2, 57.6] | +2.9 [+1.4, +4.5] | +3.1 [+2.3, +3.9] | +2.3 [+1.5, +3.2] | 66.1 [65.1, 67.1] | 90.9 [90.2, 91.7] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.75 | init | 0 | 51.7 [49.6, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.75 | f1_renewed_rho0.75_L2000 | 20000 | 57.1 [55.9, 58.3] | +6.2 [+5.2, +7.2] | +4.9 [+4.5, +5.4] | +3.8 [+3.5, +4.2] | 63.2 [62.0, 64.3] | 89.1 [88.8, 89.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| A_renewed_rho0.75 | f1_renewed_rho0.75_L2000 | 40000 | 55.0 [52.9, 57.1] | +9.1 [+6.7, +11.5] | +7.1 [+5.3, +8.8] | +5.3 [+4.2, +6.3] | 61.5 [59.0, 63.9] | 86.9 [85.2, 88.6] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |

## Family B

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| B_repeat | init | 0 | 51.7 [49.6, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_repeat | repeat_f1 | 20000 | 88.5 [86.9, 90.2] | +3.0 [+1.8, +4.1] | +0.9 [+0.3, +1.5] | +1.0 [+0.5, +1.5] | 63.7 [61.3, 66.0] | 93.1 [92.3, 94.0] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_repeat | repeat_f1 | 40000 | 88.5 [86.9, 90.1] | +3.1 [+1.9, +4.3] | +1.0 [+0.3, +1.6] | +1.1 [+0.5, +1.7] | 63.5 [61.2, 65.8] | 93.0 [92.1, 93.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_complement | init | 0 | 50.8 [48.0, 53.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_complement | alt_f1_not_f1_L2000 | 20000 | 89.6 [88.5, 90.6] | +7.8 [+4.4, +11.1] | +2.2 [+0.9, +3.5] | +2.4 [+1.4, +3.4] | 57.3 [55.0, 59.7] | 91.8 [90.5, 93.1] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_complement | alt_f1_not_f1_L2000 | 40000 | 88.7 [86.9, 90.4] | +7.3 [+5.0, +9.6] | +2.0 [+1.3, +2.7] | +2.3 [+1.7, +3.0] | 57.6 [55.7, 59.5] | 92.0 [91.1, 92.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_f2 | init | 0 | 51.6 [49.3, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_f2 | alt_f1_f2_L2000 | 20000 | 81.9 [78.8, 84.9] | +5.7 [+4.6, +6.8] | +2.3 [+1.8, +2.9] | +2.3 [+1.8, +2.8] | 61.2 [59.7, 62.7] | 91.7 [90.9, 92.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_f2 | alt_f1_f2_L2000 | 40000 | 80.5 [77.6, 83.4] | +6.3 [+5.0, +7.6] | +2.9 [+2.2, +3.6] | +2.8 [+2.2, +3.3] | 61.3 [59.6, 63.0] | 91.1 [90.2, 91.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0pi | init | 0 | 51.7 [49.6, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0pi | repeat_f1 | 20000 | 88.5 [86.9, 90.2] | +3.0 [+1.8, +4.1] | +0.9 [+0.3, +1.5] | +1.0 [+0.5, +1.5] | 63.7 [61.3, 66.0] | 93.1 [92.3, 94.0] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0pi | repeat_f1 | 40000 | 88.5 [86.9, 90.1] | +3.1 [+1.9, +4.3] | +1.0 [+0.3, +1.6] | +1.1 [+0.5, +1.7] | 63.5 [61.2, 65.8] | 93.0 [92.1, 93.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.125pi | init | 0 | 51.1 [48.7, 53.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.125pi | alt_f1_rot0.125pi_L2000 | 20000 | 90.6 [89.6, 91.5] | +3.4 [+2.9, +4.0] | +1.0 [+0.7, +1.3] | +1.2 [+0.9, +1.5] | 63.1 [61.0, 65.2] | 93.0 [92.4, 93.6] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.125pi | alt_f1_rot0.125pi_L2000 | 40000 | 89.4 [89.0, 89.9] | +3.6 [+2.7, +4.5] | +1.2 [+0.9, +1.6] | +1.4 [+1.0, +1.7] | 63.4 [62.0, 64.9] | 92.7 [92.2, 93.3] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.25pi | init | 0 | 50.2 [46.9, 53.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.25pi | alt_f1_rot0.25pi_L2000 | 20000 | 87.6 [86.7, 88.5] | +3.6 [+2.6, +4.7] | +1.4 [+1.0, +1.7] | +1.4 [+1.0, +1.9] | 63.1 [60.9, 65.3] | 92.6 [92.1, 93.2] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.25pi | alt_f1_rot0.25pi_L2000 | 40000 | 86.4 [85.7, 87.1] | +3.8 [+2.3, +5.4] | +1.6 [+1.1, +2.1] | +1.6 [+1.0, +2.2] | 63.1 [61.2, 64.9] | 92.4 [91.7, 93.1] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.5pi | init | 0 | 50.8 [48.0, 53.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.5pi | alt_f1_not_f1_L2000 | 20000 | 89.6 [88.5, 90.6] | +7.8 [+4.4, +11.1] | +2.2 [+0.9, +3.5] | +2.4 [+1.4, +3.4] | 57.3 [55.0, 59.7] | 91.8 [90.5, 93.1] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| B_rot0.5pi | alt_f1_not_f1_L2000 | 40000 | 88.7 [86.9, 90.4] | +7.3 [+5.0, +9.6] | +2.0 [+1.3, +2.7] | +2.3 [+1.7, +3.0] | 57.6 [55.7, 59.5] | 92.0 [91.1, 92.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |

## Family C

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| C_L100 | init | 0 | 51.6 [49.3, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L100 | alt_f1_f2_L100 | 20000 | 74.1 [70.4, 77.7] | +4.8 [+3.5, +6.1] | +2.5 [+2.1, +2.9] | +2.3 [+1.8, +2.8] | 62.9 [61.3, 64.4] | 91.5 [90.8, 92.2] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L100 | alt_f1_f2_L100 | 40000 | 74.4 [70.9, 78.0] | +5.7 [+4.5, +6.8] | +3.2 [+2.7, +3.8] | +2.8 [+2.4, +3.2] | 62.0 [60.9, 63.0] | 90.7 [89.9, 91.6] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L500 | init | 0 | 51.6 [49.3, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L500 | alt_f1_f2_L500 | 20000 | 78.8 [74.4, 83.3] | +5.4 [+4.7, +6.2] | +2.6 [+2.1, +3.1] | +2.5 [+2.0, +2.9] | 61.6 [60.3, 62.8] | 91.4 [90.7, 92.1] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L500 | alt_f1_f2_L500 | 40000 | 76.8 [72.5, 81.1] | +6.1 [+5.7, +6.6] | +3.2 [+2.8, +3.6] | +2.9 [+2.6, +3.3] | 60.8 [59.4, 62.1] | 90.8 [90.1, 91.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L2000 | init | 0 | 51.6 [49.3, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L2000 | alt_f1_f2_L2000 | 20000 | 81.9 [78.8, 84.9] | +5.7 [+4.6, +6.8] | +2.3 [+1.8, +2.9] | +2.3 [+1.8, +2.8] | 61.2 [59.7, 62.7] | 91.7 [90.9, 92.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L2000 | alt_f1_f2_L2000 | 40000 | 80.5 [77.6, 83.4] | +6.3 [+5.0, +7.6] | +2.9 [+2.2, +3.6] | +2.8 [+2.2, +3.3] | 61.3 [59.6, 63.0] | 91.1 [90.2, 91.9] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L10000 | init | 0 | 51.6 [49.3, 53.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 66.7 [65.1, 68.3] | 94.0 [93.6, 94.4] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L10000 | alt_f1_f2_L10000 | 20000 | 84.9 [82.0, 87.7] | +5.3 [+3.7, +7.0] | +2.1 [+1.1, +3.1] | +2.1 [+1.2, +3.0] | 61.0 [60.3, 61.6] | 91.9 [90.8, 93.0] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |
| C_L10000 | alt_f1_f2_L10000 | 40000 | 83.5 [79.7, 87.4] | +6.5 [+4.6, +8.4] | +3.0 [+1.8, +4.2] | +2.7 [+1.9, +3.6] | 59.6 [57.9, 61.3] | 91.0 [89.7, 92.3] | 94.0 [93.6, 94.4] | full adaptation beats the tested frozen decoders: 18 |

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
| A | 20000 | A_persistent_rho0 | A_persistent_rho0.25 | +0.0 [-1.2, +1.3] |
| A | 20000 | A_persistent_rho0 | A_persistent_rho0.75 | -0.3 [-2.9, +2.2] |
| A | 20000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 20000 | A_persistent_rho0 | A_renewed_rho0.25 | +0.9 [-0.8, +2.5] |
| A | 20000 | A_persistent_rho0 | A_renewed_rho0.75 | +3.3 [+1.4, +5.1] |
| A | 20000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.4 [-2.4, +1.6] |
| A | 20000 | A_persistent_rho0.25 | A_renewed_rho0 | -0.0 [-1.3, +1.2] |
| A | 20000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +0.8 [-0.1, +1.8] |
| A | 20000 | A_persistent_rho0.25 | A_renewed_rho0.75 | +3.2 [+1.7, +4.8] |
| A | 20000 | A_persistent_rho0.75 | A_renewed_rho0 | +0.3 [-2.2, +2.9] |
| A | 20000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +1.2 [-0.4, +2.9] |
| A | 20000 | A_persistent_rho0.75 | A_renewed_rho0.75 | +3.6 [+2.3, +4.9] |
| A | 20000 | A_renewed_rho0 | A_renewed_rho0.25 | +0.9 [-0.8, +2.5] |
| A | 20000 | A_renewed_rho0 | A_renewed_rho0.75 | +3.3 [+1.4, +5.1] |
| A | 20000 | A_renewed_rho0.25 | A_renewed_rho0.75 | +2.4 [+0.8, +3.9] |
| A | 40000 | A_persistent_rho0 | A_persistent_rho0.25 | +0.2 [-1.1, +1.4] |
| A | 40000 | A_persistent_rho0 | A_persistent_rho0.75 | -0.2 [-2.8, +2.5] |
| A | 40000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 40000 | A_persistent_rho0 | A_renewed_rho0.25 | +1.4 [-0.3, +3.0] |
| A | 40000 | A_persistent_rho0 | A_renewed_rho0.75 | +5.9 [+2.9, +9.0] |
| A | 40000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.3 [-2.4, +1.7] |
| A | 40000 | A_persistent_rho0.25 | A_renewed_rho0 | -0.2 [-1.4, +1.1] |
| A | 40000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +1.2 [+0.2, +2.2] |
| A | 40000 | A_persistent_rho0.25 | A_renewed_rho0.75 | +5.8 [+2.8, +8.8] |
| A | 40000 | A_persistent_rho0.75 | A_renewed_rho0 | +0.2 [-2.5, +2.8] |
| A | 40000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +1.6 [-0.3, +3.4] |
| A | 40000 | A_persistent_rho0.75 | A_renewed_rho0.75 | +6.1 [+3.6, +8.6] |
| A | 40000 | A_renewed_rho0 | A_renewed_rho0.25 | +1.4 [-0.3, +3.0] |
| A | 40000 | A_renewed_rho0 | A_renewed_rho0.75 | +5.9 [+2.9, +9.0] |
| A | 40000 | A_renewed_rho0.25 | A_renewed_rho0.75 | +4.6 [+1.5, +7.7] |
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
| B | 20000 | B_complement | B_f2 | -2.1 [-5.7, +1.5] |
| B | 20000 | B_complement | B_repeat | -4.8 [-8.0, -1.6] |
| B | 20000 | B_complement | B_rot0.125pi | -4.3 [-7.3, -1.3] |
| B | 20000 | B_complement | B_rot0.25pi | -4.1 [-6.9, -1.4] |
| B | 20000 | B_complement | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 20000 | B_complement | B_rot0pi | -4.8 [-8.0, -1.6] |
| B | 20000 | B_f2 | B_repeat | -2.7 [-4.4, -1.1] |
| B | 20000 | B_f2 | B_rot0.125pi | -2.2 [-3.6, -0.9] |
| B | 20000 | B_f2 | B_rot0.25pi | -2.1 [-3.7, -0.4] |
| B | 20000 | B_f2 | B_rot0.5pi | +2.1 [-1.5, +5.7] |
| B | 20000 | B_f2 | B_rot0pi | -2.7 [-4.4, -1.1] |
| B | 20000 | B_repeat | B_rot0.125pi | +0.5 [-0.5, +1.5] |
| B | 20000 | B_repeat | B_rot0.25pi | +0.6 [-0.1, +1.3] |
| B | 20000 | B_repeat | B_rot0.5pi | +4.8 [+1.6, +8.0] |
| B | 20000 | B_repeat | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 20000 | B_rot0.125pi | B_rot0.25pi | +0.2 [-0.4, +0.8] |
| B | 20000 | B_rot0.125pi | B_rot0.5pi | +4.3 [+1.3, +7.3] |
| B | 20000 | B_rot0.125pi | B_rot0pi | -0.5 [-1.5, +0.5] |
| B | 20000 | B_rot0.25pi | B_rot0.5pi | +4.1 [+1.4, +6.9] |
| B | 20000 | B_rot0.25pi | B_rot0pi | -0.6 [-1.3, +0.1] |
| B | 20000 | B_rot0.5pi | B_rot0pi | -4.8 [-8.0, -1.6] |
| B | 40000 | B_complement | B_f2 | -1.0 [-3.7, +1.7] |
| B | 40000 | B_complement | B_repeat | -4.2 [-6.4, -2.0] |
| B | 40000 | B_complement | B_rot0.125pi | -3.7 [-5.6, -1.8] |
| B | 40000 | B_complement | B_rot0.25pi | -3.5 [-4.9, -2.1] |
| B | 40000 | B_complement | B_rot0.5pi | +0.0 [+0.0, +0.0] |
| B | 40000 | B_complement | B_rot0pi | -4.2 [-6.4, -2.0] |
| B | 40000 | B_f2 | B_repeat | -3.2 [-5.0, -1.3] |
| B | 40000 | B_f2 | B_rot0.125pi | -2.7 [-4.4, -1.0] |
| B | 40000 | B_f2 | B_rot0.25pi | -2.5 [-4.5, -0.4] |
| B | 40000 | B_f2 | B_rot0.5pi | +1.0 [-1.7, +3.7] |
| B | 40000 | B_f2 | B_rot0pi | -3.2 [-5.0, -1.3] |
| B | 40000 | B_repeat | B_rot0.125pi | +0.5 [-0.8, +1.7] |
| B | 40000 | B_repeat | B_rot0.25pi | +0.7 [-0.3, +1.8] |
| B | 40000 | B_repeat | B_rot0.5pi | +4.2 [+2.0, +6.4] |
| B | 40000 | B_repeat | B_rot0pi | +0.0 [+0.0, +0.0] |
| B | 40000 | B_rot0.125pi | B_rot0.25pi | +0.2 [-0.8, +1.3] |
| B | 40000 | B_rot0.125pi | B_rot0.5pi | +3.7 [+1.8, +5.6] |
| B | 40000 | B_rot0.125pi | B_rot0pi | -0.5 [-1.7, +0.8] |
| B | 40000 | B_rot0.25pi | B_rot0.5pi | +3.5 [+2.1, +4.9] |
| B | 40000 | B_rot0.25pi | B_rot0pi | -0.7 [-1.8, +0.3] |
| B | 40000 | B_rot0.5pi | B_rot0pi | -4.2 [-6.4, -2.0] |
| C | 0 | C_L100 | C_L10000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L100 | C_L2000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L100 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L2000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2000 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 20000 | C_L100 | C_L10000 | +0.5 [-1.6, +2.6] |
| C | 20000 | C_L100 | C_L2000 | +0.9 [-0.2, +2.0] |
| C | 20000 | C_L100 | C_L500 | +0.6 [-0.7, +1.9] |
| C | 20000 | C_L10000 | C_L2000 | +0.3 [-1.6, +2.3] |
| C | 20000 | C_L10000 | C_L500 | +0.1 [-1.1, +1.3] |
| C | 20000 | C_L2000 | C_L500 | -0.3 [-1.8, +1.2] |
| C | 40000 | C_L100 | C_L10000 | +0.9 [-1.3, +3.0] |
| C | 40000 | C_L100 | C_L2000 | +0.6 [-0.9, +2.2] |
| C | 40000 | C_L100 | C_L500 | +0.5 [-0.9, +1.9] |
| C | 40000 | C_L10000 | C_L2000 | -0.2 [-2.5, +2.1] |
| C | 40000 | C_L10000 | C_L500 | -0.4 [-2.0, +1.2] |
| C | 40000 | C_L2000 | C_L500 | -0.2 [-1.6, +1.2] |

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
