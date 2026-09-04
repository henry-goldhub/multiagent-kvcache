| Setting | Accuracy | Prefill mean (s) | Decode mean (s) | Adapter mean (s) | Cache-hit rate | Adapter attempt rate | Adapter accept rate | Adapter fallback rate | Prefill speedup | Total speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_1 -> model_1 -> model_1 | 0.200 | 0.5731 | 9.0042 | 0.0000 | 0.805 | 0.000 | 0.000 | 0.000 | 2.43x | 1.08x |
| model_2 -> model_2 -> model_2 | 0.600 | 0.6375 | 8.3413 | 0.0000 | 0.785 | 0.000 | 0.000 | 0.000 | 2.44x | 1.10x |
| model_3 -> model_3 -> model_3 | 0.200 | 0.3272 | 4.6646 | 0.0000 | 0.785 | 0.000 | 0.000 | 0.000 | 2.49x | 1.09x |
| model_1 -> model_2 -> model_3 | 0.000 | 1.1904 | 6.7276 | 0.0008 | 0.000 | 0.667 | 0.000 | 1.000 | 1.00x | 1.01x |
