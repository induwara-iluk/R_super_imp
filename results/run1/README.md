# Run 1 (Kaggle, 2× T4): first full run of the self-contained notebook

The notebook was commit `9c748ca`. All stages finished in 4.55 h of the 11.3 h budget.
Downloaded with:

```
kaggle kernels output induwarailukkumbura/r-super-1 -p results/run1 --file-pattern "<pattern>"
```

The pattern selected `results/*.csv|json`, `logs/`, `exp/*/history.jsonl`, the test predictions and the test labels.

To re-score and redraw: `python analysis/analyze_run.py results/run1`.

## What is in git and what is not
| In git (small) | Ignored (`.gitignore`, large) |
|---|---|
| `r-super-1.log` (notebook log), `logs/` (training and prediction logs, configs) | `exp/*/latest.pth` (checkpoints, 265 MB each) |
| `exp/*/history.jsonl` (losses per epoch) | `results/test/` (predictions, about 94 MB) |
| `results/*.csv, *.json` (the notebook's own metrics, status, time plan) | `rsuper_data/` (test labels and sample CTs) |
| `analysis/*.csv` (re-scored metrics), `figures/*.png` | |

## Settings
- **Baseline:** 24 epochs × 200 iterations, lr 1e-3.
- **Fine-tunes** (`control`, `volume`, `volume_ball`): 12 epochs each, lr 1e-4, starting from the baseline.
- **Warm-up:** 5 epochs in every run.
- **Patches:** 96³, batch 2.
- **Reports:** synthetic, σ = 0.10.

## Findings
1. **Evaluating on the whole CT was wrong for this model.**
   - Training crops come from the pancreas + 48 mm, and 90 % of them are centred on the tumour.
   - Over the whole CT, 81–83 % of the predicted tumour voxels lay more than 48 mm from the pancreas (`figures/06`), in anatomy the model never saw.
   - Result: about 40 false-positive blobs per CT, lesion Dice 0.04, pancreas Dice 0.17 for every model.
2. **Scored inside the pancreas region** (GT pancreas bbox + 48 mm, the training field of view):

   | model | lesion Dice | pancreas Dice | lesion recall | lesion precision | lesion F1 | FP blobs / CT |
   |---|---|---|---|---|---|---|
   | baseline | 0.112 | 0.475 | 0.65 (33/51) | 0.052 | 0.096 | 12.1 |
   | control | 0.094 | 0.436 | 0.67 (34/51) | 0.047 | 0.088 | 13.8 |
   | volume | 0.090 | 0.430 | 0.71 (36/51) | 0.045 | 0.084 | 15.4 |
   | volume_ball | 0.100 | 0.431 | 0.75 (38/51) | 0.049 | 0.092 | 14.7 |

   Every model is still weak. In the median, each predicts about 5–7 times more tumour than there is (25–33k vs 4.6k voxels).
   That fits a model that has seen only 9,600 patches, mostly centred on a tumour. nnU-Net-scale training is about 250k patches.
3. **Report losses compared with `control`** (same data, same start, same iterations, no report loss) — paired over the 50 test CTs (`figures/07`):
   - `volume_ball`: lesion Dice +0.006 (95 % CI +0.001 to +0.012). It is better in 25 CTs and worse in 12 (Wilcoxon p = 0.03).
   - `volume`: −0.004 (CI −0.009 to 0.000), p = 0.09.
4. **Detection by tumour size** (`figures/08`, pancreas ROI). Small means < 2.7 cm³:

   | tumour size | baseline | control | volume | volume_ball |
   |---|---|---|---|---|
   | small | 4/17 | 6/17 | 6/17 | 9/17 |
   | medium | 12/17 | 12/17 | 13/17 | 13/17 |
   | large | 17/17 | 16/17 | 17/17 | 16/17 |

   The paper also reports its gains mainly in detection. Here the counts are small (17 tumours per size group), so this is a hint, not a result.
5. **Fine-tuning itself lowered lesion Dice** compared with the baseline: −0.018 for `control`. The likely cause is that half of each fine-tune batch is report cases with no lesion labels.
   This is why `control` is the right comparison for the report losses.
6. **Time.** The probe timed a 30-iteration epoch including start-up, and overestimated seconds per iteration 1.5–3× (`figures/03`).
   The plan therefore chose 24 + 12 epochs, and 6.8 h went unused.
7. **Warm-up.** 5 epochs of warm-up took 5 of the 12 fine-tune epochs, during which the learning rate stayed below 1e-4 (`figures/01`).
8. **Crop statistics** in the log were all 0. The counter lived on the dataset object inside the DataLoader worker processes, so its counts never reached the main process.
   The report losses were non-zero (`figures/02`), so report crops did happen.
9. **Report volumes.** Even at σ = 0, the ellipsoid from two axial diameters is within ±10 % of the true volume for only 22 % of tumours (`figures/09`).

## Changes for run 2 (same PR)
- **Time plan.** The probe measures steady-state seconds per iteration (start-up excluded) for each training type. Epochs fill the budget, up to 80 for the baseline and 40 per fine-tune, and the fine-tune epochs are planned again after the baseline.
- **Warm-up** = 5 % of each run's epochs, the same fraction as the repo's 5 of 100.
- **Crop statistics** are counted in the main process from a per-sample tag, and written to `history.jsonl` together with the lr.
- **Evaluation** predicts inside the pancreas ROI (`EVAL_ROI_MARGIN = 48`). Set it to `None` to score the whole CT.
- **Analysis.** The notebook ends with `analysis/analyze_run.py`, so these figures are produced on Kaggle too.
