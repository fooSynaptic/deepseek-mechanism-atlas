# E3 archive

Job `e3_gshard_x15`, 10B tokens, wall **32.5 h** (data-parallel archive).

| Arm | val BPB |
| --- | ---: |
| GShard 1.0× | 0.7970 |
| GShard×1.5 | **0.7860** |
| DeepSeekMoE | 0.7850 |

Gates: E3-A hit (−0.011 vs GShard); E3-B hit (|Δ|=0.001 vs DSMoE); E3-C hit (max_frac≈0.20).
