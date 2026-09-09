# E1 after load-balance fix (10B)

Primary metric: val BPB. Lower is better.

| job | val BPB | max_frac end | aux/CE end | C3 |
| --- | ---: | ---: | ---: | --- |
| e1_lb_gshard_w20 | 0.7970 | 0.195 | 3.33% | hit |
| e1_lb_dsmoe | **0.7850** | 0.204 | 3.38% | hit |

Δ (DSMoE − GShard) = **−0.012**. C2 (≥ 0.02) miss. Ranking: DeepSeekMoE.

Collapsed first E1 is not this table (`e1_archive/`).
