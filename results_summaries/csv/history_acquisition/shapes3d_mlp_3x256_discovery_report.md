# Result 2: training histories and subsequent feature acquisition

Dataset shapes3d, mlp_3x256, history lr 0.01, batch 256, T = 400000; protocol `6a583ae81d27`, analysis `60d2d72208b7`; seeds [0, 1, 2, 3, 4, 5].
Acquisition: targets ['size', 'shape'], 3 head replicas, full / fresh 2000 updates (continued to 8000) at lr 0.01, frozen decoders up to 8000 (lrs [0.01, 0.03, 0.1], weight decays [0.0, 0.0001, 0.001], nonlinear width 256). Plain SGD, momentum 0.

Measurement status: {'complete': 768}

Deficit = fresh - checkpoint test accuracy, update-weighted mean over the primary budget (positive: worse than fresh). Seed means [95% t interval over seeds], targets averaged within seed.

## Family A

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| A_persistent_rho0 | init | 0 | 50.7 [42.9, 58.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0 | repeat_obj_hue | 200000 | 100.0 [100.0, 100.0] | -0.0 [-0.1, +0.1] | +0.0 [-0.0, +0.0] | -0.0 [-0.0, +0.0] | 97.7 [97.4, 98.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0 | repeat_obj_hue | 400000 | 100.0 [100.0, 100.0] | -0.0 [-0.1, +0.1] | +0.0 [-0.0, +0.0] | -0.0 [-0.0, +0.0] | 97.6 [97.4, 97.9] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0 | init | 0 | 50.7 [42.9, 58.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0 | repeat_obj_hue | 200000 | 100.0 [100.0, 100.0] | -0.0 [-0.1, +0.1] | +0.0 [-0.0, +0.0] | -0.0 [-0.0, +0.0] | 97.7 [97.4, 98.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0 | repeat_obj_hue | 400000 | 100.0 [100.0, 100.0] | -0.0 [-0.1, +0.1] | +0.0 [-0.0, +0.0] | -0.0 [-0.0, +0.0] | 97.6 [97.4, 97.9] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.25 | init | 0 | 50.7 [42.9, 58.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.25 | obj_hue_persistent_rho0.25 | 200000 | 92.7 [92.3, 93.1] | +0.2 [+0.1, +0.4] | +0.0 [+0.0, +0.0] | +0.1 [+0.0, +0.1] | 98.7 [98.5, 98.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.25 | obj_hue_persistent_rho0.25 | 400000 | 92.6 [92.1, 93.0] | +0.3 [+0.1, +0.4] | +0.0 [+0.0, +0.0] | +0.1 [+0.0, +0.1] | 98.7 [98.5, 98.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0.25 | init | 0 | 50.7 [42.9, 58.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0.25 | obj_hue_renewed_rho0.25_L4000 | 200000 | 91.0 [90.4, 91.6] | +1.1 [+0.9, +1.3] | +0.1 [+0.1, +0.1] | +0.3 [+0.3, +0.3] | 96.4 [95.3, 97.4] | 99.9 [99.9, 99.9] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 3; no difference beyond the thresholds: 2; readable from the frozen features (tested decoders): 7 |
| A_renewed_rho0.25 | obj_hue_renewed_rho0.25_L4000 | 400000 | 90.1 [88.7, 91.6] | +1.7 [+1.2, +2.2] | +0.2 [+0.1, +0.2] | +0.5 [+0.4, +0.6] | 94.7 [91.4, 97.9] | 99.8 [99.8, 99.9] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 7; readable from the frozen features (tested decoders): 5 |
| A_persistent_rho0.75 | init | 0 | 50.7 [42.9, 58.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.75 | obj_hue_persistent_rho0.75 | 200000 | 65.6 [64.3, 67.0] | +0.2 [+0.1, +0.3] | +0.0 [+0.0, +0.0] | +0.1 [+0.0, +0.1] | 98.6 [98.5, 98.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.75 | obj_hue_persistent_rho0.75 | 400000 | 65.6 [64.3, 66.9] | +0.2 [+0.1, +0.3] | +0.0 [+0.0, +0.0] | +0.1 [+0.0, +0.1] | 98.6 [98.5, 98.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0.75 | init | 0 | 50.7 [42.9, 58.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0.75 | obj_hue_renewed_rho0.75_L4000 | 200000 | 61.8 [53.6, 70.0] | +8.5 [+6.0, +11.1] | +0.5 [+0.0, +1.0] | +2.3 [+1.7, +2.8] | 57.9 [53.5, 62.3] | 99.5 [99.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| A_renewed_rho0.75 | obj_hue_renewed_rho0.75_L4000 | 400000 | 61.2 [53.6, 68.8] | +10.0 [+7.4, +12.6] | +0.6 [+0.0, +1.1] | +2.6 [+2.0, +3.3] | 56.0 [53.3, 58.6] | 99.4 [98.9, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |

## Family B

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| B_repeat | init | 0 | 50.7 [42.9, 58.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_repeat | repeat_obj_hue | 200000 | 100.0 [100.0, 100.0] | -0.0 [-0.1, +0.1] | +0.0 [-0.0, +0.0] | -0.0 [-0.0, +0.0] | 97.7 [97.4, 98.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_repeat | repeat_obj_hue | 400000 | 100.0 [100.0, 100.0] | -0.0 [-0.1, +0.1] | +0.0 [-0.0, +0.0] | -0.0 [-0.0, +0.0] | 97.6 [97.4, 97.9] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_complement | init | 0 | 53.0 [46.0, 60.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_complement | alt_obj_hue_not_obj_hue_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.5 [+0.4, +0.5] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 94.0 [93.3, 94.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 9; readable from the frozen features (tested decoders): 3 |
| B_complement | alt_obj_hue_not_obj_hue_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.5 [+0.4, +0.6] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.2] | 91.9 [90.6, 93.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| B_wall_hue | init | 0 | 49.7 [42.5, 56.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_wall_hue | alt_obj_hue_wall_hue_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.5 [+0.3, +0.6] | +0.0 [+0.0, +0.0] | +0.1 [+0.1, +0.1] | 95.9 [95.0, 96.9] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 3; readable from the frozen features (tested decoders): 9 |
| B_wall_hue | alt_obj_hue_wall_hue_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.6 [+0.5, +0.7] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.2] | 96.0 [95.1, 96.9] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 3; readable from the frozen features (tested decoders): 9 |
| B_shift1 | init | 0 | 52.2 [47.7, 56.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift1 | alt_obj_hue_obj_hue_shift1_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.1 [+0.0, +0.1] | -0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] | 97.5 [97.2, 97.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift1 | alt_obj_hue_obj_hue_shift1_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.2 [+0.0, +0.3] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.1] | 96.9 [96.5, 97.2] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 3; readable from the frozen features (tested decoders): 9 |
| B_shift2 | init | 0 | 53.3 [49.9, 56.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift2 | alt_obj_hue_obj_hue_shift2_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.1 [+0.0, +0.2] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.1] | 96.9 [96.3, 97.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 3; readable from the frozen features (tested decoders): 9 |
| B_shift2 | alt_obj_hue_obj_hue_shift2_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.2 [+0.1, +0.3] | +0.0 [+0.0, +0.0] | +0.1 [+0.0, +0.1] | 96.7 [96.1, 97.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 2; readable from the frozen features (tested decoders): 10 |
| B_shift5 | init | 0 | 53.0 [46.0, 60.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift5 | alt_obj_hue_not_obj_hue_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.5 [+0.4, +0.5] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 94.0 [93.3, 94.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 9; readable from the frozen features (tested decoders): 3 |
| B_shift5 | alt_obj_hue_not_obj_hue_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.5 [+0.4, +0.6] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.2] | 91.9 [90.6, 93.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |

## Family C

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| C_L2 | init | 0 | 49.7 [42.5, 56.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L2 | alt_obj_hue_wall_hue_L2 | 100000 | 74.7 [73.7, 75.6] | +0.2 [+0.1, +0.3] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 90.7 [89.0, 92.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L2 | alt_obj_hue_wall_hue_L2 | 200000 | 74.9 [74.6, 75.2] | +0.1 [+0.0, +0.2] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.1] | 89.2 [87.0, 91.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L2 | alt_obj_hue_wall_hue_L2 | 400000 | 74.9 [74.1, 75.7] | +0.2 [+0.1, +0.3] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 87.6 [85.4, 89.9] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L10 | init | 0 | 49.7 [42.5, 56.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L10 | alt_obj_hue_wall_hue_L10 | 100000 | 77.7 [76.3, 79.0] | +0.5 [+0.2, +0.7] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.2] | 82.2 [77.5, 86.9] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L10 | alt_obj_hue_wall_hue_L10 | 200000 | 77.1 [75.6, 78.5] | +0.5 [+0.2, +0.7] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.2] | 80.1 [75.2, 85.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L10 | alt_obj_hue_wall_hue_L10 | 400000 | 76.1 [74.0, 78.2] | +0.4 [+0.2, +0.6] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.2] | 78.5 [73.8, 83.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L50 | init | 0 | 49.7 [42.5, 56.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L50 | alt_obj_hue_wall_hue_L50 | 100000 | 87.3 [75.8, 98.7] | +1.2 [+1.2, +1.3] | +0.0 [-0.0, +0.0] | +0.3 [+0.3, +0.3] | 75.2 [72.9, 77.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L50 | alt_obj_hue_wall_hue_L50 | 200000 | 82.2 [68.3, 96.1] | +1.4 [+1.3, +1.4] | +0.0 [+0.0, +0.0] | +0.3 [+0.3, +0.4] | 72.5 [69.2, 75.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L50 | alt_obj_hue_wall_hue_L50 | 400000 | 81.3 [73.2, 89.5] | +1.4 [+1.3, +1.5] | +0.0 [+0.0, +0.0] | +0.4 [+0.3, +0.4] | 71.2 [68.4, 74.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L500 | init | 0 | 49.7 [42.5, 56.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L500 | alt_obj_hue_wall_hue_L500 | 100000 | 100.0 [100.0, 100.0] | +0.5 [+0.4, +0.7] | +0.0 [+0.0, +0.0] | +0.1 [+0.1, +0.2] | 95.2 [94.1, 96.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 7; readable from the frozen features (tested decoders): 5 |
| C_L500 | alt_obj_hue_wall_hue_L500 | 200000 | 100.0 [100.0, 100.0] | +0.7 [+0.6, +0.8] | +0.0 [+0.0, +0.0] | +0.2 [+0.1, +0.2] | 92.9 [90.7, 95.2] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 10; readable from the frozen features (tested decoders): 2 |
| C_L500 | alt_obj_hue_wall_hue_L500 | 400000 | 100.0 [100.0, 100.0] | +1.0 [+0.8, +1.3] | +0.0 [+0.0, +0.0] | +0.3 [+0.2, +0.3] | 88.5 [87.1, 89.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L2000 | init | 0 | 49.7 [42.5, 56.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L2000 | alt_obj_hue_wall_hue_L2000 | 100000 | 100.0 [100.0, 100.0] | +0.4 [+0.2, +0.5] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 96.2 [95.3, 97.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 3; readable from the frozen features (tested decoders): 9 |
| C_L2000 | alt_obj_hue_wall_hue_L2000 | 200000 | 100.0 [100.0, 100.0] | +0.5 [+0.3, +0.7] | +0.0 [+0.0, +0.0] | +0.1 [+0.1, +0.2] | 95.4 [93.7, 97.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 4; readable from the frozen features (tested decoders): 8 |
| C_L2000 | alt_obj_hue_wall_hue_L2000 | 400000 | 100.0 [100.0, 100.0] | +0.6 [+0.5, +0.7] | +0.0 [+0.0, +0.0] | +0.2 [+0.1, +0.2] | 94.3 [92.5, 96.1] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 8; readable from the frozen features (tested decoders): 4 |
| C_L10000 | init | 0 | 49.7 [42.5, 56.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L10000 | alt_obj_hue_wall_hue_L10000 | 100000 | 100.0 [100.0, 100.0] | +0.1 [+0.1, +0.2] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] | 96.6 [96.0, 97.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 3; readable from the frozen features (tested decoders): 9 |
| C_L10000 | alt_obj_hue_wall_hue_L10000 | 200000 | 100.0 [100.0, 100.0] | +0.3 [+0.2, +0.3] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 96.4 [95.8, 97.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 3; readable from the frozen features (tested decoders): 9 |
| C_L10000 | alt_obj_hue_wall_hue_L10000 | 400000 | 100.0 [100.0, 100.0] | +0.4 [+0.3, +0.5] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 95.9 [95.1, 96.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 4; readable from the frozen features (tested decoders): 8 |
| C_L50000 | init | 0 | 49.7 [42.5, 56.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 98.5 [98.5, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L50000 | alt_obj_hue_wall_hue_L50000 | 100000 | 100.0 [100.0, 100.0] | +0.1 [-0.1, +0.2] | +0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | 97.3 [96.7, 97.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 1; readable from the frozen features (tested decoders): 11 |
| C_L50000 | alt_obj_hue_wall_hue_L50000 | 200000 | 100.0 [100.0, 100.0] | +0.1 [-0.0, +0.2] | +0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.1] | 97.0 [96.5, 97.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 1; readable from the frozen features (tested decoders): 11 |
| C_L50000 | alt_obj_hue_wall_hue_L50000 | 400000 | 100.0 [100.0, 100.0] | +0.2 [+0.1, +0.3] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.1] | 96.8 [96.1, 97.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 2; readable from the frozen features (tested decoders): 10 |

## Memorisation in comparison A (history bank, each checkpoint against the label table of the last block it was trained on)

| condition | age | overwrite fraction | labels actually changed | assigned-label acc | overwritten subset: assigned / clean acc | kept subset clean acc | status |
|---|---|---|---|---|---|---|---|
| A_persistent_rho0.25 | 200000 | 0.250 | 0.125 | 100.0 | 100.0 / 50.1 | 100.0 | overwritten labels fitted |
| A_persistent_rho0.25 | 400000 | 0.250 | 0.125 | 100.0 | 100.0 / 50.1 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.25 | 200000 | 0.250 | 0.123 | 100.0 | 100.0 / 50.8 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.25 | 400000 | 0.250 | 0.125 | 100.0 | 99.9 / 50.1 | 100.0 | overwritten labels fitted |
| A_persistent_rho0.75 | 200000 | 0.750 | 0.374 | 100.0 | 100.0 / 50.1 | 100.0 | overwritten labels fitted |
| A_persistent_rho0.75 | 400000 | 0.750 | 0.374 | 100.0 | 100.0 / 50.1 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.75 | 200000 | 0.750 | 0.370 | 66.8 | 61.1 / 53.7 | 83.6 | overwritten labels NOT fitted (61% of them reproduced): no memorisation claimed |
| A_renewed_rho0.75 | 400000 | 0.750 | 0.372 | 63.8 | 56.4 / 53.0 | 85.9 | overwritten labels NOT fitted (56% of them reproduced): no memorisation claimed |

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
| A | 200000 | A_persistent_rho0 | A_persistent_rho0.25 | +0.2 [+0.2, +0.3] |
| A | 200000 | A_persistent_rho0 | A_persistent_rho0.75 | +0.2 [+0.1, +0.3] |
| A | 200000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 200000 | A_persistent_rho0 | A_renewed_rho0.25 | +1.1 [+0.9, +1.3] |
| A | 200000 | A_persistent_rho0 | A_renewed_rho0.75 | +8.6 [+5.9, +11.2] |
| A | 200000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.0 [-0.1, +0.0] |
| A | 200000 | A_persistent_rho0.25 | A_renewed_rho0 | -0.2 [-0.3, -0.2] |
| A | 200000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +0.9 [+0.7, +1.0] |
| A | 200000 | A_persistent_rho0.25 | A_renewed_rho0.75 | +8.3 [+5.7, +10.9] |
| A | 200000 | A_persistent_rho0.75 | A_renewed_rho0 | -0.2 [-0.3, -0.1] |
| A | 200000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +0.9 [+0.7, +1.1] |
| A | 200000 | A_persistent_rho0.75 | A_renewed_rho0.75 | +8.3 [+5.8, +10.9] |
| A | 200000 | A_renewed_rho0 | A_renewed_rho0.25 | +1.1 [+0.9, +1.3] |
| A | 200000 | A_renewed_rho0 | A_renewed_rho0.75 | +8.6 [+5.9, +11.2] |
| A | 200000 | A_renewed_rho0.25 | A_renewed_rho0.75 | +7.4 [+4.9, +10.0] |
| A | 400000 | A_persistent_rho0 | A_persistent_rho0.25 | +0.3 [+0.2, +0.4] |
| A | 400000 | A_persistent_rho0 | A_persistent_rho0.75 | +0.2 [+0.2, +0.3] |
| A | 400000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 400000 | A_persistent_rho0 | A_renewed_rho0.25 | +1.7 [+1.2, +2.3] |
| A | 400000 | A_persistent_rho0 | A_renewed_rho0.75 | +10.0 [+7.4, +12.6] |
| A | 400000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.0 [-0.1, +0.0] |
| A | 400000 | A_persistent_rho0.25 | A_renewed_rho0 | -0.3 [-0.4, -0.2] |
| A | 400000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +1.5 [+0.9, +2.0] |
| A | 400000 | A_persistent_rho0.25 | A_renewed_rho0.75 | +9.7 [+7.2, +12.3] |
| A | 400000 | A_persistent_rho0.75 | A_renewed_rho0 | -0.2 [-0.3, -0.2] |
| A | 400000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +1.5 [+0.9, +2.1] |
| A | 400000 | A_persistent_rho0.75 | A_renewed_rho0.75 | +9.8 [+7.2, +12.4] |
| A | 400000 | A_renewed_rho0 | A_renewed_rho0.25 | +1.7 [+1.2, +2.3] |
| A | 400000 | A_renewed_rho0 | A_renewed_rho0.75 | +10.0 [+7.4, +12.6] |
| A | 400000 | A_renewed_rho0.25 | A_renewed_rho0.75 | +8.3 [+5.5, +11.1] |
| B | 0 | B_complement | B_repeat | +0.0 [+0.0, +0.0] |
| B | 0 | B_complement | B_shift1 | +0.0 [+0.0, +0.0] |
| B | 0 | B_complement | B_shift2 | +0.0 [+0.0, +0.0] |
| B | 0 | B_complement | B_shift5 | +0.0 [+0.0, +0.0] |
| B | 0 | B_complement | B_wall_hue | +0.0 [+0.0, +0.0] |
| B | 0 | B_repeat | B_shift1 | +0.0 [+0.0, +0.0] |
| B | 0 | B_repeat | B_shift2 | +0.0 [+0.0, +0.0] |
| B | 0 | B_repeat | B_shift5 | +0.0 [+0.0, +0.0] |
| B | 0 | B_repeat | B_wall_hue | +0.0 [+0.0, +0.0] |
| B | 0 | B_shift1 | B_shift2 | +0.0 [+0.0, +0.0] |
| B | 0 | B_shift1 | B_shift5 | +0.0 [+0.0, +0.0] |
| B | 0 | B_shift1 | B_wall_hue | +0.0 [+0.0, +0.0] |
| B | 0 | B_shift2 | B_shift5 | +0.0 [+0.0, +0.0] |
| B | 0 | B_shift2 | B_wall_hue | +0.0 [+0.0, +0.0] |
| B | 0 | B_shift5 | B_wall_hue | +0.0 [+0.0, +0.0] |
| B | 200000 | B_complement | B_repeat | -0.5 [-0.6, -0.4] |
| B | 200000 | B_complement | B_shift1 | -0.4 [-0.5, -0.3] |
| B | 200000 | B_complement | B_shift2 | -0.3 [-0.4, -0.2] |
| B | 200000 | B_complement | B_shift5 | +0.0 [+0.0, +0.0] |
| B | 200000 | B_complement | B_wall_hue | -0.0 [-0.1, +0.1] |
| B | 200000 | B_repeat | B_shift1 | +0.1 [+0.0, +0.1] |
| B | 200000 | B_repeat | B_shift2 | +0.1 [-0.0, +0.3] |
| B | 200000 | B_repeat | B_shift5 | +0.5 [+0.4, +0.6] |
| B | 200000 | B_repeat | B_wall_hue | +0.5 [+0.3, +0.6] |
| B | 200000 | B_shift1 | B_shift2 | +0.1 [-0.1, +0.2] |
| B | 200000 | B_shift1 | B_shift5 | +0.4 [+0.3, +0.5] |
| B | 200000 | B_shift1 | B_wall_hue | +0.4 [+0.2, +0.5] |
| B | 200000 | B_shift2 | B_shift5 | +0.3 [+0.2, +0.4] |
| B | 200000 | B_shift2 | B_wall_hue | +0.3 [+0.2, +0.5] |
| B | 200000 | B_shift5 | B_wall_hue | -0.0 [-0.1, +0.1] |
| B | 400000 | B_complement | B_repeat | -0.6 [-0.6, -0.5] |
| B | 400000 | B_complement | B_shift1 | -0.4 [-0.4, -0.3] |
| B | 400000 | B_complement | B_shift2 | -0.3 [-0.4, -0.2] |
| B | 400000 | B_complement | B_shift5 | +0.0 [+0.0, +0.0] |
| B | 400000 | B_complement | B_wall_hue | +0.0 [-0.1, +0.1] |
| B | 400000 | B_repeat | B_shift1 | +0.2 [+0.1, +0.3] |
| B | 400000 | B_repeat | B_shift2 | +0.2 [+0.2, +0.3] |
| B | 400000 | B_repeat | B_shift5 | +0.6 [+0.5, +0.6] |
| B | 400000 | B_repeat | B_wall_hue | +0.6 [+0.4, +0.7] |
| B | 400000 | B_shift1 | B_shift2 | +0.1 [-0.1, +0.2] |
| B | 400000 | B_shift1 | B_shift5 | +0.4 [+0.3, +0.4] |
| B | 400000 | B_shift1 | B_wall_hue | +0.4 [+0.2, +0.5] |
| B | 400000 | B_shift2 | B_shift5 | +0.3 [+0.2, +0.4] |
| B | 400000 | B_shift2 | B_wall_hue | +0.3 [+0.2, +0.5] |
| B | 400000 | B_shift5 | B_wall_hue | +0.0 [-0.1, +0.1] |
| C | 0 | C_L10 | C_L10000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10 | C_L2 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10 | C_L2000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10 | C_L50 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10 | C_L50000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L2 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L2000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L50 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L10000 | C_L50000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2 | C_L2000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2 | C_L50 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2 | C_L50000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2000 | C_L50 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2000 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L2000 | C_L50000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L50 | C_L500 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L50 | C_L50000 | +0.0 [+0.0, +0.0] |
| C | 0 | C_L500 | C_L50000 | +0.0 [+0.0, +0.0] |
| C | 100000 | C_L10 | C_L10000 | -0.3 [-0.5, -0.1] |
| C | 100000 | C_L10 | C_L2 | -0.2 [-0.5, +0.1] |
| C | 100000 | C_L10 | C_L2000 | -0.1 [-0.3, +0.1] |
| C | 100000 | C_L10 | C_L50 | +0.8 [+0.6, +1.0] |
| C | 100000 | C_L10 | C_L500 | +0.1 [-0.0, +0.2] |
| C | 100000 | C_L10 | C_L50000 | -0.4 [-0.5, -0.2] |
| C | 100000 | C_L10000 | C_L2 | +0.1 [-0.0, +0.3] |
| C | 100000 | C_L10000 | C_L2000 | +0.2 [+0.1, +0.4] |
| C | 100000 | C_L10000 | C_L50 | +1.1 [+1.1, +1.2] |
| C | 100000 | C_L10000 | C_L500 | +0.4 [+0.3, +0.6] |
| C | 100000 | C_L10000 | C_L50000 | -0.1 [-0.1, +0.0] |
| C | 100000 | C_L2 | C_L2000 | +0.1 [-0.1, +0.4] |
| C | 100000 | C_L2 | C_L50 | +1.0 [+0.9, +1.1] |
| C | 100000 | C_L2 | C_L500 | +0.3 [+0.0, +0.6] |
| C | 100000 | C_L2 | C_L50000 | -0.2 [-0.4, +0.0] |
| C | 100000 | C_L2000 | C_L50 | +0.9 [+0.7, +1.0] |
| C | 100000 | C_L2000 | C_L500 | +0.2 [+0.0, +0.3] |
| C | 100000 | C_L2000 | C_L50000 | -0.3 [-0.4, -0.2] |
| C | 100000 | C_L50 | C_L500 | -0.7 [-0.8, -0.6] |
| C | 100000 | C_L50 | C_L50000 | -1.2 [-1.3, -1.1] |
| C | 100000 | C_L500 | C_L50000 | -0.5 [-0.5, -0.4] |
| C | 200000 | C_L10 | C_L10000 | -0.2 [-0.4, +0.0] |
| C | 200000 | C_L10 | C_L2 | -0.3 [-0.5, -0.2] |
| C | 200000 | C_L10 | C_L2000 | +0.1 [-0.2, +0.3] |
| C | 200000 | C_L10 | C_L50 | +0.9 [+0.7, +1.1] |
| C | 200000 | C_L10 | C_L500 | +0.2 [-0.1, +0.6] |
| C | 200000 | C_L10 | C_L50000 | -0.3 [-0.6, -0.1] |
| C | 200000 | C_L10000 | C_L2 | -0.1 [-0.3, +0.0] |
| C | 200000 | C_L10000 | C_L2000 | +0.3 [+0.1, +0.4] |
| C | 200000 | C_L10000 | C_L50 | +1.1 [+1.0, +1.2] |
| C | 200000 | C_L10000 | C_L500 | +0.4 [+0.3, +0.6] |
| C | 200000 | C_L10000 | C_L50000 | -0.1 [-0.3, -0.0] |
| C | 200000 | C_L2 | C_L2000 | +0.4 [+0.2, +0.6] |
| C | 200000 | C_L2 | C_L50 | +1.2 [+1.1, +1.3] |
| C | 200000 | C_L2 | C_L500 | +0.6 [+0.4, +0.7] |
| C | 200000 | C_L2 | C_L50000 | -0.0 [-0.2, +0.1] |
| C | 200000 | C_L2000 | C_L50 | +0.8 [+0.7, +1.0] |
| C | 200000 | C_L2000 | C_L500 | +0.2 [-0.1, +0.4] |
| C | 200000 | C_L2000 | C_L50000 | -0.4 [-0.6, -0.2] |
| C | 200000 | C_L50 | C_L500 | -0.7 [-0.8, -0.5] |
| C | 200000 | C_L50 | C_L50000 | -1.2 [-1.3, -1.1] |
| C | 200000 | C_L500 | C_L50000 | -0.6 [-0.8, -0.4] |
| C | 400000 | C_L10 | C_L10000 | -0.0 [-0.3, +0.2] |
| C | 400000 | C_L10 | C_L2 | -0.2 [-0.4, -0.1] |
| C | 400000 | C_L10 | C_L2000 | +0.2 [-0.1, +0.4] |
| C | 400000 | C_L10 | C_L50 | +1.0 [+0.8, +1.2] |
| C | 400000 | C_L10 | C_L500 | +0.6 [+0.2, +0.9] |
| C | 400000 | C_L10 | C_L50000 | -0.3 [-0.5, -0.1] |
| C | 400000 | C_L10000 | C_L2 | -0.2 [-0.4, +0.0] |
| C | 400000 | C_L10000 | C_L2000 | +0.2 [+0.1, +0.3] |
| C | 400000 | C_L10000 | C_L50 | +1.0 [+0.9, +1.1] |
| C | 400000 | C_L10000 | C_L500 | +0.6 [+0.3, +0.9] |
| C | 400000 | C_L10000 | C_L50000 | -0.2 [-0.3, -0.1] |
| C | 400000 | C_L2 | C_L2000 | +0.4 [+0.2, +0.6] |
| C | 400000 | C_L2 | C_L50 | +1.2 [+1.1, +1.3] |
| C | 400000 | C_L2 | C_L500 | +0.8 [+0.6, +1.0] |
| C | 400000 | C_L2 | C_L50000 | -0.0 [-0.2, +0.1] |
| C | 400000 | C_L2000 | C_L50 | +0.8 [+0.7, +0.9] |
| C | 400000 | C_L2000 | C_L500 | +0.4 [+0.1, +0.7] |
| C | 400000 | C_L2000 | C_L50000 | -0.4 [-0.5, -0.3] |
| C | 400000 | C_L50 | C_L500 | -0.4 [-0.7, -0.1] |
| C | 400000 | C_L50 | C_L50000 | -1.2 [-1.3, -1.2] |
| C | 400000 | C_L500 | C_L50000 | -0.8 [-1.1, -0.5] |

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
