# fs_eval archive

Job `fs_eval`, data parallel (closed batch recipe), eval-only.
Ckpts: `e1_lb_dsmoe`, `e1_lb_gshard_w20`. Fixed val seed, `eval_batches=32`.

| Config | val BPB | note |
| --- | ---: | --- |
| GShard Top-2 (same batches) | 0.8318 | fair F reference |
| DSMoE 1+Top-3 | 0.8403 | worse than GShard |
| DSMoE 1+Top-4 | **0.8288** | beats GShard |
| DSMoE 1+Top-7 | **0.8204** | baseline; vs GShard −0.011 |
| Shared off + Top-8 | **0.9558** | +0.135 vs baseline |

P0: baseline 0.8204 within 0.05 of E1 0.7850 (different val draw). Relative F/S use one draw.
