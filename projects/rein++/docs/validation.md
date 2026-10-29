# Validation report

Completed on October 28, 2026 (Asia/Shanghai).

## Paper, historical, and re-evaluated results

The score order is the same as in the model card. All scores are validation mIoU (%); ACDC is the mean over four conditions.

| Model ID | Paper | Historical log | Re-evaluated |
|---|---|---|---|
| `rdg_usg2cbm_dinov3_rein` | 78.9 / 63.4 / 74.8 | 78.91 / 63.43 / 74.78 | 78.91 / 63.44 / 74.78 |
| `rdg_usg2cbm_dinov3-7b_rein` | 80.0 / 66.3 / 75.9 | 80.02 / 66.32 / 75.91 | 80.02 / 66.32 / 75.91 |
| `rdg_c2abm_dinov3_rein` | 72.5 / 65.0 / 77.4 | 72.52 / 65.04 / 77.42 | 72.52 / 65.04 / 77.41 |
| `rdg_c2abm_dinov3-7b_rein` | 75.9 / 68.0 / 77.5 | 75.92 / 67.98 / 77.54 | 75.92 / 67.97 / 77.54 |
| `rdg_g2cbm_dinov3-7b_rein` | 73.9 / 65.5 / 72.6 | 73.90 / 65.49 / 72.58 | 73.89 / 65.49 / 72.58 |
| `rda_g2c_dinov3_rein_ta` | 79.0 | 79.03 | 79.03 |

Twelve of the sixteen reported metrics match the historical logs at two decimal places. Four differ by 0.01 percentage points. All match the paper at one decimal place. The cause of the small differences has not been isolated. The complete per-domain metrics, including the Cityscapes source-domain diagnostic scores for `c2abm`, are in [evaluations.json](validation/evaluations.json).

## Forward equivalence

All six models were compared against the original experiment implementation in FP32 with TF32 disabled. Each comparison used a real image at 768 × 512 and a fixed random input at 736 × 528. In all twelve comparisons, the maximum absolute logit difference was 0.0 and predicted labels were identical. These are paired-input checks, separate from the full validation-set evaluations above.

The per-model JSON reports in `validation/` record input, source-code, configuration, and checkpoint hashes. All six full dataset evaluations completed successfully.

## Additional checks

- Configuration and parameter-loading preflight: passed for all six models.
- Automated tests: 19 passed.
- Single-image inference CLI: passed for `rdg_usg2cbm_dinov3_rein`; mask and overlay outputs checked.
- Missing dataset path: expected error confirmed.
- CUDA deformable attention against its PyTorch reference: maximum absolute error 1.1920928955078125e-07.
- Label conversion: 10 Cityscapes and 32 Mapillary masks matched the original evaluation masks pixel for pixel.
- Release gate: all six tested weight/configuration hashes match the staged files, and the Python source hash matches the code used in GPU validation.

## Public checkpoint verification

The first batch was published on October 29, 2026 at [zhixiangwei/ReinPlusPlus](https://huggingface.co/zhixiangwei/ReinPlusPlus), revision `f1547b68b1283add02425c2732eeb4dade42012d`.

All 33 release files, including six checkpoints totaling 1,017,967,434 bytes, were downloaded without authentication and matched their recorded SHA-256 checksums. The [download verification record](validation/public-download-verification.json) lists each file and result.
