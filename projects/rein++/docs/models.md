# Models and results

The first release contains five Rein-G checkpoints and one Rein-A checkpoint. All six have completed full evaluation on their corresponding validation sets. Download the weights from [Hugging Face](https://huggingface.co/zhixiangwei/ReinPlusPlus).

Scores below are mIoU (%). C = Cityscapes, B = BDD100K, M = Mapillary Vistas v1.2, and A = ACDC validation. ACDC is the arithmetic mean over night, snow, fog, and rain. The score order follows the target datasets in each row.

| Model | Setting | Paper | Re-evaluated | Files |
|---|---|---|---|---|
| Rein-G / DINOv3-L | UrbanSyn + SYNTHIA + GTAV → C / B / M | 78.9 / 63.4 / 74.8 | 78.91 / 63.44 / 74.78 | [config](../configs/rdg_usg2cbm_dinov3_rein.py) · [weights](https://huggingface.co/zhixiangwei/ReinPlusPlus/resolve/f1547b68b1283add02425c2732eeb4dade42012d/rdg_usg2cbm_dinov3_rein/model.pth?download=true) |
| Rein-G / DINOv3-7B | UrbanSyn + SYNTHIA + GTAV → C / B / M | 80.0 / 66.3 / 75.9 | 80.02 / 66.32 / 75.91 | [config](../configs/rdg_usg2cbm_dinov3-7b_rein.py) · [weights](https://huggingface.co/zhixiangwei/ReinPlusPlus/resolve/f1547b68b1283add02425c2732eeb4dade42012d/rdg_usg2cbm_dinov3-7b_rein/model.pth?download=true) |
| Rein-G / DINOv3-L | Cityscapes → A / B / M | 72.5 / 65.0 / 77.4 | 72.52 / 65.04 / 77.41 | [config](../configs/rdg_c2abm_dinov3_rein.py) · [weights](https://huggingface.co/zhixiangwei/ReinPlusPlus/resolve/f1547b68b1283add02425c2732eeb4dade42012d/rdg_c2abm_dinov3_rein/model.pth?download=true) |
| Rein-G / DINOv3-7B | Cityscapes → A / B / M | 75.9 / 68.0 / 77.5 | 75.92 / 67.97 / 77.54 | [config](../configs/rdg_c2abm_dinov3-7b_rein.py) · [weights](https://huggingface.co/zhixiangwei/ReinPlusPlus/resolve/f1547b68b1283add02425c2732eeb4dade42012d/rdg_c2abm_dinov3-7b_rein/model.pth?download=true) |
| Rein-G / DINOv3-7B | GTAV → C / B / M | 73.9 / 65.5 / 72.6 | 73.89 / 65.49 / 72.58 | [config](../configs/rdg_g2cbm_dinov3-7b_rein.py) · [weights](https://huggingface.co/zhixiangwei/ReinPlusPlus/resolve/f1547b68b1283add02425c2732eeb4dade42012d/rdg_g2cbm_dinov3-7b_rein/model.pth?download=true) |
| Rein-A / DINOv3-L | GTAV → C | 79.0 | 79.03 | [config](../configs/rda_g2c_dinov3_rein_ta.py) · [weights](https://huggingface.co/zhixiangwei/ReinPlusPlus/resolve/f1547b68b1283add02425c2732eeb4dade42012d/rda_g2c_dinov3_rein_ta/model.pth?download=true) |

## Paper coverage

Table II (also reported in Table III):

| Source → target datasets | DINOv3-L | DINOv3-7B |
|---|---|---|
| GTAV → Cityscapes / BDD100K / Mapillary | Planned | Released |
| GTAV + UrbanSyn + SYNTHIA → Cityscapes / BDD100K / Mapillary | Released | Released |
| Cityscapes → ACDC(val) / BDD100K / Mapillary | Released | Released |

The Rein-A DINOv3-L checkpoint corresponds to Table VII, GTA → Cityscapes (Val.), with paper mIoU 79.0.

The remaining checkpoints and training code will be released progressively.

## Verification and metadata

The [validation report](validation.md) separates paper values, historical logs, and re-evaluated scores, and describes the forward-equivalence checks. Full metrics and per-model GPU reports are in [validation/](validation/).

[model-index.json](../model-index.json) records each model's configuration, student inference branch, checkpoint iteration, source-run identifier, required backbone, SHA-256 checksums, and validation status. The weight links above are pinned to the verified Hugging Face revision.
