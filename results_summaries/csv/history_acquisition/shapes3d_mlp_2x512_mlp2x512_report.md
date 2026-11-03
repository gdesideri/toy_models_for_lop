# Result 2: training histories and subsequent feature acquisition

Dataset shapes3d, mlp_2x512, history lr 0.01, batch 256, T = 400000; protocol `83a2786f6671`, analysis `60d2d72208b7`; seeds [0, 1, 2, 3, 4, 5].
Acquisition: targets ['size', 'shape'], 3 head replicas, full / fresh 2000 updates (continued to 8000) at lr 0.01, frozen decoders up to 8000 (lrs [0.01, 0.03, 0.1], weight decays [0.0, 0.0001, 0.001], nonlinear width 256). Plain SGD, momentum 0.

Measurement status: {'complete': 768}

Deficit = fresh - checkpoint test accuracy, update-weighted mean over the primary budget (positive: worse than fresh). Seed means [95% t interval over seeds], targets averaged within seed.

## Family A

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| A_persistent_rho0 | init | 0 | 48.5 [39.9, 57.1] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0 | repeat_obj_hue | 200000 | 100.0 [100.0, 100.0] | +0.1 [-0.0, +0.1] | -0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | 99.6 [99.6, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0 | repeat_obj_hue | 400000 | 100.0 [100.0, 100.0] | +0.1 [-0.0, +0.1] | -0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | 99.6 [99.6, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0 | init | 0 | 48.5 [39.9, 57.1] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0 | repeat_obj_hue | 200000 | 100.0 [100.0, 100.0] | +0.1 [-0.0, +0.1] | -0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | 99.6 [99.6, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0 | repeat_obj_hue | 400000 | 100.0 [100.0, 100.0] | +0.1 [-0.0, +0.1] | -0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | 99.6 [99.6, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.25 | init | 0 | 48.5 [39.9, 57.1] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.25 | obj_hue_persistent_rho0.25 | 200000 | 92.8 [92.3, 93.3] | +0.4 [+0.3, +0.5] | +0.0 [+0.0, +0.0] | +0.1 [+0.1, +0.1] | 99.6 [99.5, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.25 | obj_hue_persistent_rho0.25 | 400000 | 92.7 [92.2, 93.2] | +0.4 [+0.3, +0.5] | +0.0 [+0.0, +0.0] | +0.1 [+0.1, +0.1] | 99.6 [99.5, 99.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0.25 | init | 0 | 48.5 [39.9, 57.1] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0.25 | obj_hue_renewed_rho0.25_L4000 | 200000 | 91.8 [91.1, 92.5] | +0.7 [+0.6, +0.7] | +0.1 [+0.1, +0.1] | +0.2 [+0.2, +0.2] | 99.0 [99.0, 99.1] | 99.9 [99.9, 99.9] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0.25 | obj_hue_renewed_rho0.25_L4000 | 400000 | 91.0 [90.2, 91.7] | +0.9 [+0.8, +1.0] | +0.1 [+0.1, +0.1] | +0.2 [+0.2, +0.3] | 98.9 [98.8, 98.9] | 99.9 [99.9, 99.9] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.75 | init | 0 | 48.5 [39.9, 57.1] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.75 | obj_hue_persistent_rho0.75 | 200000 | 66.1 [64.9, 67.4] | +0.3 [+0.3, +0.4] | +0.0 [+0.0, +0.0] | +0.1 [+0.1, +0.1] | 99.6 [99.6, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_persistent_rho0.75 | obj_hue_persistent_rho0.75 | 400000 | 66.0 [64.8, 67.2] | +0.3 [+0.3, +0.4] | +0.0 [+0.0, +0.0] | +0.1 [+0.1, +0.1] | 99.6 [99.6, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0.75 | init | 0 | 48.5 [39.9, 57.1] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| A_renewed_rho0.75 | obj_hue_renewed_rho0.75_L4000 | 200000 | 62.0 [57.4, 66.6] | +8.3 [+7.2, +9.4] | +0.2 [+0.0, +0.4] | +2.1 [+1.9, +2.3] | 65.8 [63.4, 68.2] | 99.8 [99.6, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| A_renewed_rho0.75 | obj_hue_renewed_rho0.75_L4000 | 400000 | 59.9 [55.2, 64.6] | +11.0 [+9.4, +12.5] | +0.1 [-0.1, +0.4] | +2.8 [+2.4, +3.1] | 58.4 [57.0, 59.9] | 99.9 [99.6, 100.1] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |

## Family B

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| B_repeat | init | 0 | 48.5 [39.9, 57.1] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_repeat | repeat_obj_hue | 200000 | 100.0 [100.0, 100.0] | +0.1 [-0.0, +0.1] | -0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | 99.6 [99.6, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_repeat | repeat_obj_hue | 400000 | 100.0 [100.0, 100.0] | +0.1 [-0.0, +0.1] | -0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | 99.6 [99.6, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_complement | init | 0 | 50.8 [42.1, 59.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_complement | alt_obj_hue_not_obj_hue_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.3 [+0.2, +0.4] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 98.5 [97.9, 99.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_complement | alt_obj_hue_not_obj_hue_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.4 [+0.3, +0.5] | -0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 97.3 [96.7, 98.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_wall_hue | init | 0 | 52.6 [41.9, 63.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_wall_hue | alt_obj_hue_wall_hue_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.3 [+0.2, +0.4] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 99.4 [99.3, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_wall_hue | alt_obj_hue_wall_hue_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.4 [+0.3, +0.5] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 99.4 [99.3, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift1 | init | 0 | 47.9 [40.4, 55.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift1 | alt_obj_hue_obj_hue_shift1_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.1 [+0.1, +0.2] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.5 [99.4, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift1 | alt_obj_hue_obj_hue_shift1_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.2 [+0.1, +0.3] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.1] | 99.4 [99.4, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift2 | init | 0 | 48.1 [40.4, 55.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift2 | alt_obj_hue_obj_hue_shift2_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.1 [+0.0, +0.2] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.5 [99.4, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift2 | alt_obj_hue_obj_hue_shift2_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.1 [+0.1, +0.2] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.1] | 99.3 [99.2, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift5 | init | 0 | 50.8 [42.1, 59.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift5 | alt_obj_hue_not_obj_hue_L4000 | 200000 | 100.0 [100.0, 100.0] | +0.3 [+0.2, +0.4] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 98.5 [97.9, 99.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| B_shift5 | alt_obj_hue_not_obj_hue_L4000 | 400000 | 100.0 [100.0, 100.0] | +0.4 [+0.3, +0.5] | -0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 97.3 [96.7, 98.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |

## Family C

| condition | history | age | last-block rule val acc (%) | full deficit (U) | full deficit, end | full deficit (UE) | best frozen acc (UD, %) | full acc (U, %) | fresh acc (U, %) | qualification |
|---|---|---|---|---|---|---|---|---|---|---|
| C_L2 | init | 0 | 52.6 [41.9, 63.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L2 | alt_obj_hue_wall_hue_L2 | 100000 | 75.4 [74.8, 75.9] | +0.1 [-0.0, +0.2] | -0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | 98.3 [98.0, 98.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L2 | alt_obj_hue_wall_hue_L2 | 200000 | 75.2 [74.6, 75.8] | +0.1 [-0.0, +0.2] | -0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.0] | 97.8 [97.4, 98.2] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L2 | alt_obj_hue_wall_hue_L2 | 400000 | 74.8 [73.5, 76.1] | +0.1 [+0.0, +0.2] | -0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] | 96.9 [96.3, 97.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 1; readable from the frozen features (tested decoders): 11 |
| C_L10 | init | 0 | 52.6 [41.9, 63.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L10 | alt_obj_hue_wall_hue_L10 | 100000 | 76.8 [75.3, 78.3] | +0.3 [+0.2, +0.4] | -0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 89.8 [86.9, 92.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L10 | alt_obj_hue_wall_hue_L10 | 200000 | 75.7 [74.5, 76.9] | +0.3 [+0.2, +0.4] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 87.4 [84.5, 90.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L10 | alt_obj_hue_wall_hue_L10 | 400000 | 75.9 [73.9, 77.9] | +0.3 [+0.3, +0.4] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 85.9 [83.5, 88.3] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L50 | init | 0 | 52.6 [41.9, 63.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L50 | alt_obj_hue_wall_hue_L50 | 100000 | 91.7 [83.3, 100.0] | +0.7 [+0.6, +0.9] | +0.0 [-0.0, +0.0] | +0.2 [+0.1, +0.2] | 82.4 [81.0, 83.7] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L50 | alt_obj_hue_wall_hue_L50 | 200000 | 86.2 [75.0, 97.4] | +0.8 [+0.7, +1.0] | +0.0 [+0.0, +0.0] | +0.2 [+0.2, +0.2] | 80.3 [79.1, 81.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L50 | alt_obj_hue_wall_hue_L50 | 400000 | 79.5 [68.4, 90.5] | +0.8 [+0.7, +1.0] | +0.0 [+0.0, +0.0] | +0.2 [+0.2, +0.2] | 78.7 [77.7, 79.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | full adaptation beats the tested frozen decoders: 12 |
| C_L500 | init | 0 | 52.6 [41.9, 63.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L500 | alt_obj_hue_wall_hue_L500 | 100000 | 100.0 [100.0, 100.0] | +0.4 [+0.3, +0.5] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.1] | 99.4 [99.3, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L500 | alt_obj_hue_wall_hue_L500 | 200000 | 100.0 [100.0, 100.0] | +0.5 [+0.3, +0.6] | +0.0 [-0.0, +0.0] | +0.1 [+0.1, +0.2] | 98.5 [97.5, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L500 | alt_obj_hue_wall_hue_L500 | 400000 | 100.0 [100.0, 100.0] | +0.7 [+0.5, +0.9] | +0.0 [+0.0, +0.0] | +0.2 [+0.1, +0.2] | 97.4 [96.9, 98.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L2000 | init | 0 | 52.6 [41.9, 63.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L2000 | alt_obj_hue_wall_hue_L2000 | 100000 | 100.0 [100.0, 100.0] | +0.3 [+0.2, +0.4] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 99.4 [99.4, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L2000 | alt_obj_hue_wall_hue_L2000 | 200000 | 100.0 [100.0, 100.0] | +0.3 [+0.2, +0.4] | +0.0 [+0.0, +0.0] | +0.1 [+0.1, +0.1] | 99.4 [99.3, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L2000 | alt_obj_hue_wall_hue_L2000 | 400000 | 100.0 [100.0, 100.0] | +0.4 [+0.3, +0.5] | +0.0 [+0.0, +0.0] | +0.1 [+0.1, +0.1] | 99.3 [99.1, 99.4] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L10000 | init | 0 | 52.6 [41.9, 63.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L10000 | alt_obj_hue_wall_hue_L10000 | 100000 | 100.0 [100.0, 100.0] | +0.2 [+0.1, +0.3] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.1] | 99.5 [99.3, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L10000 | alt_obj_hue_wall_hue_L10000 | 200000 | 100.0 [100.0, 100.0] | +0.2 [+0.1, +0.4] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 99.4 [99.3, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L10000 | alt_obj_hue_wall_hue_L10000 | 400000 | 100.0 [100.0, 100.0] | +0.3 [+0.2, +0.4] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 99.3 [99.2, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L50000 | init | 0 | 52.6 [41.9, 63.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.8 [99.7, 99.8] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L50000 | alt_obj_hue_wall_hue_L50000 | 100000 | 100.0 [100.0, 100.0] | +0.1 [+0.0, +0.2] | +0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] | 99.5 [99.4, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L50000 | alt_obj_hue_wall_hue_L50000 | 200000 | 100.0 [100.0, 100.0] | +0.2 [-0.0, +0.3] | +0.0 [-0.0, +0.0] | +0.0 [-0.0, +0.1] | 99.5 [99.4, 99.6] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |
| C_L50000 | alt_obj_hue_wall_hue_L50000 | 400000 | 100.0 [100.0, 100.0] | +0.2 [+0.1, +0.3] | +0.0 [-0.0, +0.0] | +0.1 [+0.0, +0.1] | 99.4 [99.3, 99.5] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | readable from the frozen features (tested decoders): 12 |

## Memorisation in comparison A (history bank, each checkpoint against the label table of the last block it was trained on)

| condition | age | overwrite fraction | labels actually changed | assigned-label acc | overwritten subset: assigned / clean acc | kept subset clean acc | status |
|---|---|---|---|---|---|---|---|
| A_persistent_rho0.25 | 200000 | 0.250 | 0.125 | 100.0 | 100.0 / 50.1 | 100.0 | overwritten labels fitted |
| A_persistent_rho0.25 | 400000 | 0.250 | 0.125 | 100.0 | 100.0 / 50.1 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.25 | 200000 | 0.250 | 0.123 | 100.0 | 100.0 / 50.8 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.25 | 400000 | 0.250 | 0.125 | 100.0 | 100.0 / 50.0 | 100.0 | overwritten labels fitted |
| A_persistent_rho0.75 | 200000 | 0.750 | 0.374 | 100.0 | 100.0 / 50.1 | 100.0 | overwritten labels fitted |
| A_persistent_rho0.75 | 400000 | 0.750 | 0.374 | 100.0 | 100.0 / 50.1 | 100.0 | overwritten labels fitted |
| A_renewed_rho0.75 | 200000 | 0.750 | 0.370 | 73.5 | 68.9 / 53.2 | 87.3 | overwritten labels NOT fitted (69% of them reproduced): no memorisation claimed |
| A_renewed_rho0.75 | 400000 | 0.750 | 0.372 | 62.2 | 55.4 / 52.5 | 82.5 | overwritten labels NOT fitted (55% of them reproduced): no memorisation claimed |

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
| A | 200000 | A_persistent_rho0 | A_persistent_rho0.25 | +0.3 [+0.2, +0.4] |
| A | 200000 | A_persistent_rho0 | A_persistent_rho0.75 | +0.3 [+0.2, +0.3] |
| A | 200000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 200000 | A_persistent_rho0 | A_renewed_rho0.25 | +0.6 [+0.5, +0.7] |
| A | 200000 | A_persistent_rho0 | A_renewed_rho0.75 | +8.3 [+7.1, +9.4] |
| A | 200000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.1 [-0.1, +0.0] |
| A | 200000 | A_persistent_rho0.25 | A_renewed_rho0 | -0.3 [-0.4, -0.2] |
| A | 200000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +0.3 [+0.2, +0.4] |
| A | 200000 | A_persistent_rho0.25 | A_renewed_rho0.75 | +8.0 [+6.9, +9.0] |
| A | 200000 | A_persistent_rho0.75 | A_renewed_rho0 | -0.3 [-0.3, -0.2] |
| A | 200000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +0.3 [+0.2, +0.4] |
| A | 200000 | A_persistent_rho0.75 | A_renewed_rho0.75 | +8.0 [+6.9, +9.1] |
| A | 200000 | A_renewed_rho0 | A_renewed_rho0.25 | +0.6 [+0.5, +0.7] |
| A | 200000 | A_renewed_rho0 | A_renewed_rho0.75 | +8.3 [+7.1, +9.4] |
| A | 200000 | A_renewed_rho0.25 | A_renewed_rho0.75 | +7.7 [+6.6, +8.8] |
| A | 400000 | A_persistent_rho0 | A_persistent_rho0.25 | +0.3 [+0.2, +0.4] |
| A | 400000 | A_persistent_rho0 | A_persistent_rho0.75 | +0.3 [+0.2, +0.3] |
| A | 400000 | A_persistent_rho0 | A_renewed_rho0 | +0.0 [+0.0, +0.0] |
| A | 400000 | A_persistent_rho0 | A_renewed_rho0.25 | +0.8 [+0.7, +0.9] |
| A | 400000 | A_persistent_rho0 | A_renewed_rho0.75 | +10.9 [+9.3, +12.5] |
| A | 400000 | A_persistent_rho0.25 | A_persistent_rho0.75 | -0.0 [-0.1, +0.0] |
| A | 400000 | A_persistent_rho0.25 | A_renewed_rho0 | -0.3 [-0.4, -0.2] |
| A | 400000 | A_persistent_rho0.25 | A_renewed_rho0.25 | +0.5 [+0.4, +0.6] |
| A | 400000 | A_persistent_rho0.25 | A_renewed_rho0.75 | +10.6 [+9.1, +12.1] |
| A | 400000 | A_persistent_rho0.75 | A_renewed_rho0 | -0.3 [-0.3, -0.2] |
| A | 400000 | A_persistent_rho0.75 | A_renewed_rho0.25 | +0.6 [+0.4, +0.7] |
| A | 400000 | A_persistent_rho0.75 | A_renewed_rho0.75 | +10.7 [+9.1, +12.2] |
| A | 400000 | A_renewed_rho0 | A_renewed_rho0.25 | +0.8 [+0.7, +0.9] |
| A | 400000 | A_renewed_rho0 | A_renewed_rho0.75 | +10.9 [+9.3, +12.5] |
| A | 400000 | A_renewed_rho0.25 | A_renewed_rho0.75 | +10.1 [+8.5, +11.7] |
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
| B | 200000 | B_complement | B_repeat | -0.2 [-0.3, -0.1] |
| B | 200000 | B_complement | B_shift1 | -0.2 [-0.2, -0.1] |
| B | 200000 | B_complement | B_shift2 | -0.2 [-0.3, -0.1] |
| B | 200000 | B_complement | B_shift5 | +0.0 [+0.0, +0.0] |
| B | 200000 | B_complement | B_wall_hue | -0.0 [-0.1, +0.1] |
| B | 200000 | B_repeat | B_shift1 | +0.1 [+0.0, +0.1] |
| B | 200000 | B_repeat | B_shift2 | +0.0 [-0.1, +0.1] |
| B | 200000 | B_repeat | B_shift5 | +0.2 [+0.1, +0.3] |
| B | 200000 | B_repeat | B_wall_hue | +0.2 [+0.1, +0.3] |
| B | 200000 | B_shift1 | B_shift2 | -0.0 [-0.1, +0.0] |
| B | 200000 | B_shift1 | B_shift5 | +0.2 [+0.1, +0.2] |
| B | 200000 | B_shift1 | B_wall_hue | +0.1 [+0.1, +0.2] |
| B | 200000 | B_shift2 | B_shift5 | +0.2 [+0.1, +0.3] |
| B | 200000 | B_shift2 | B_wall_hue | +0.2 [+0.1, +0.3] |
| B | 200000 | B_shift5 | B_wall_hue | -0.0 [-0.1, +0.1] |
| B | 400000 | B_complement | B_repeat | -0.3 [-0.5, -0.2] |
| B | 400000 | B_complement | B_shift1 | -0.2 [-0.3, -0.1] |
| B | 400000 | B_complement | B_shift2 | -0.3 [-0.3, -0.2] |
| B | 400000 | B_complement | B_shift5 | +0.0 [+0.0, +0.0] |
| B | 400000 | B_complement | B_wall_hue | -0.0 [-0.2, +0.1] |
| B | 400000 | B_repeat | B_shift1 | +0.1 [+0.1, +0.2] |
| B | 400000 | B_repeat | B_shift2 | +0.1 [-0.0, +0.2] |
| B | 400000 | B_repeat | B_shift5 | +0.3 [+0.2, +0.5] |
| B | 400000 | B_repeat | B_wall_hue | +0.3 [+0.3, +0.3] |
| B | 400000 | B_shift1 | B_shift2 | -0.0 [-0.1, +0.0] |
| B | 400000 | B_shift1 | B_shift5 | +0.2 [+0.1, +0.3] |
| B | 400000 | B_shift1 | B_wall_hue | +0.2 [+0.1, +0.3] |
| B | 400000 | B_shift2 | B_shift5 | +0.3 [+0.2, +0.3] |
| B | 400000 | B_shift2 | B_wall_hue | +0.2 [+0.1, +0.4] |
| B | 400000 | B_shift5 | B_wall_hue | -0.0 [-0.2, +0.1] |
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
| C | 100000 | C_L10 | C_L10000 | -0.1 [-0.2, +0.0] |
| C | 100000 | C_L10 | C_L2 | -0.2 [-0.3, -0.1] |
| C | 100000 | C_L10 | C_L2000 | -0.0 [-0.1, +0.1] |
| C | 100000 | C_L10 | C_L50 | +0.5 [+0.3, +0.6] |
| C | 100000 | C_L10 | C_L500 | +0.1 [+0.0, +0.2] |
| C | 100000 | C_L10 | C_L50000 | -0.2 [-0.3, -0.0] |
| C | 100000 | C_L10000 | C_L2 | -0.1 [-0.3, +0.1] |
| C | 100000 | C_L10000 | C_L2000 | +0.1 [+0.0, +0.1] |
| C | 100000 | C_L10000 | C_L50 | +0.6 [+0.4, +0.8] |
| C | 100000 | C_L10000 | C_L500 | +0.2 [+0.2, +0.3] |
| C | 100000 | C_L10000 | C_L50000 | -0.1 [-0.1, -0.0] |
| C | 100000 | C_L2 | C_L2000 | +0.2 [+0.0, +0.3] |
| C | 100000 | C_L2 | C_L50 | +0.7 [+0.6, +0.8] |
| C | 100000 | C_L2 | C_L500 | +0.3 [+0.2, +0.5] |
| C | 100000 | C_L2 | C_L50000 | +0.0 [-0.1, +0.2] |
| C | 100000 | C_L2000 | C_L50 | +0.5 [+0.3, +0.7] |
| C | 100000 | C_L2000 | C_L500 | +0.1 [+0.1, +0.2] |
| C | 100000 | C_L2000 | C_L50000 | -0.2 [-0.2, -0.1] |
| C | 100000 | C_L50 | C_L500 | -0.4 [-0.5, -0.2] |
| C | 100000 | C_L50 | C_L50000 | -0.6 [-0.8, -0.4] |
| C | 100000 | C_L500 | C_L50000 | -0.3 [-0.3, -0.3] |
| C | 200000 | C_L10 | C_L10000 | -0.1 [-0.2, +0.1] |
| C | 200000 | C_L10 | C_L2 | -0.2 [-0.4, -0.1] |
| C | 200000 | C_L10 | C_L2000 | +0.0 [-0.1, +0.2] |
| C | 200000 | C_L10 | C_L50 | +0.5 [+0.4, +0.6] |
| C | 200000 | C_L10 | C_L500 | +0.2 [+0.0, +0.3] |
| C | 200000 | C_L10 | C_L50000 | -0.2 [-0.4, +0.0] |
| C | 200000 | C_L10000 | C_L2 | -0.2 [-0.3, +0.0] |
| C | 200000 | C_L10000 | C_L2000 | +0.1 [+0.0, +0.1] |
| C | 200000 | C_L10000 | C_L50 | +0.6 [+0.4, +0.7] |
| C | 200000 | C_L10000 | C_L500 | +0.2 [+0.1, +0.4] |
| C | 200000 | C_L10000 | C_L50000 | -0.1 [-0.2, -0.0] |
| C | 200000 | C_L2 | C_L2000 | +0.2 [+0.1, +0.4] |
| C | 200000 | C_L2 | C_L50 | +0.7 [+0.6, +0.8] |
| C | 200000 | C_L2 | C_L500 | +0.4 [+0.2, +0.6] |
| C | 200000 | C_L2 | C_L50000 | +0.1 [-0.1, +0.3] |
| C | 200000 | C_L2000 | C_L50 | +0.5 [+0.3, +0.7] |
| C | 200000 | C_L2000 | C_L500 | +0.2 [+0.0, +0.3] |
| C | 200000 | C_L2000 | C_L50000 | -0.2 [-0.3, -0.1] |
| C | 200000 | C_L50 | C_L500 | -0.3 [-0.5, -0.2] |
| C | 200000 | C_L50 | C_L50000 | -0.7 [-0.9, -0.4] |
| C | 200000 | C_L500 | C_L50000 | -0.3 [-0.5, -0.1] |
| C | 400000 | C_L10 | C_L10000 | -0.1 [-0.2, +0.1] |
| C | 400000 | C_L10 | C_L2 | -0.2 [-0.3, -0.2] |
| C | 400000 | C_L10 | C_L2000 | +0.1 [-0.1, +0.2] |
| C | 400000 | C_L10 | C_L50 | +0.5 [+0.3, +0.6] |
| C | 400000 | C_L10 | C_L500 | +0.4 [+0.1, +0.6] |
| C | 400000 | C_L10 | C_L50000 | -0.1 [-0.3, -0.0] |
| C | 400000 | C_L10000 | C_L2 | -0.2 [-0.3, -0.0] |
| C | 400000 | C_L10000 | C_L2000 | +0.1 [+0.1, +0.2] |
| C | 400000 | C_L10000 | C_L50 | +0.5 [+0.4, +0.7] |
| C | 400000 | C_L10000 | C_L500 | +0.4 [+0.3, +0.5] |
| C | 400000 | C_L10000 | C_L50000 | -0.1 [-0.2, -0.0] |
| C | 400000 | C_L2 | C_L2000 | +0.3 [+0.1, +0.4] |
| C | 400000 | C_L2 | C_L50 | +0.7 [+0.6, +0.9] |
| C | 400000 | C_L2 | C_L500 | +0.6 [+0.4, +0.8] |
| C | 400000 | C_L2 | C_L50000 | +0.1 [-0.1, +0.3] |
| C | 400000 | C_L2000 | C_L50 | +0.4 [+0.3, +0.6] |
| C | 400000 | C_L2000 | C_L500 | +0.3 [+0.2, +0.4] |
| C | 400000 | C_L2000 | C_L50000 | -0.2 [-0.3, -0.1] |
| C | 400000 | C_L50 | C_L500 | -0.1 [-0.3, +0.1] |
| C | 400000 | C_L50 | C_L50000 | -0.6 [-0.8, -0.5] |
| C | 400000 | C_L500 | C_L50000 | -0.5 [-0.6, -0.4] |

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
