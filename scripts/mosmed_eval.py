"""MosMed evaluation: Dice / IoU between sampled predictions and GT masks.

``scripts/segmentation_env.py`` is hard-coded for ISIC (expects
``ISIC_<id>_Segmentation.png`` next to the predictions), so it cannot evaluate
MosMed, whose GT lives in 3D ``masks/study_XXXX_mask.nii.gz``. This script
fills that gap.

It scans a prediction directory for ``*_output_ens.jpg`` files produced by
``segmentation_sample.py`` (file names look like ``0255_slice10_output_ens.jpg``),
looks up the matching slice in the corresponding 3D mask volume, and reports
mean Dice / IoU at multiple binarisation thresholds.

Key fixes vs original version:
  1. NO per-image max-normalisation (pred/pred.max() was amplifying background
     noise on near-empty predictions). save_image already maps [0,1]->[0,255],
     so we divide by 255.0 to recover the probability in [0,1].
  2. Empty-set handling: when both pred and GT are empty, dice/iou = 0 (not 1).
     The old smooth=1e-6 made (0+1e-6)/(0+1e-6)=1.0, a false perfect score.
  3. Reports best-threshold Dice in addition to mean-over-thresholds, so a
     model that predicts the right region at one threshold isn't masked by
     others.

Usage (from project root):

    python scripts/mosmed_eval.py \
        --pred_dir ./results/mosmed_predictions \
        --data_root ./MosMedData-Chest-CT-Scans-with-COVID-19-Related-Findings-main \
        --image_size 256
"""
import argparse
import os
import re

import numpy as np
import nibabel as nib
import torch
import torchvision
from PIL import Image

# Cache of loaded mask volumes so we don't re-read the same nii.gz per slice.
_MASK_CACHE = {}


def _mask_volume(mask_path):
    if mask_path not in _MASK_CACHE:
        _MASK_CACHE[mask_path] = nib.load(mask_path).get_fdata().astype(np.float32)
    return _MASK_CACHE[mask_path]


def _resolve_mask_path(data_root, study_id):
    """Find masks/study_<id>_mask.nii.gz under data_root."""
    return os.path.join(data_root, "masks", f"study_{study_id}_mask.nii.gz")


def _iou(pred, gt):
    """IoU / Jaccard. Empty & empty -> 0.0 (not 1.0)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 0.0  # both empty: no lesion detected, no lesion present -> 0
    return float(inter / union)


def _dice(pred, gt):
    """Dice. Empty & empty -> 0.0 (not 1.0)."""
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    inter = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 0.0  # both empty -> 0
    return float((2 * inter) / denom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True,
                   help="dir containing *_output_ens.jpg from segmentation_sample.py")
    p.add_argument("--data_root", required=True,
                   help="MosMed dataset root (contains masks/study_XXXX_mask.nii.gz)")
    p.add_argument("--image_size", type=int, default=256,
                   help="size predictions / GT were resized to during training")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=(0.1, 0.3, 0.5, 0.7, 0.9),
                   help="binarisation thresholds (probability in [0,1]) to try")
    args = p.parse_args()

    # Pattern: <study_id>_slice<idx>_output_ens.jpg, study_id is digits.
    pat = re.compile(r"(\d+)_slice(\d+)_output_ens\.jpg$")

    files = sorted(f for f in os.listdir(args.pred_dir)
                   if f.endswith("_output_ens.jpg"))
    print(f"[mosmed_eval] found {len(files)} predictions in {args.pred_dir}")

    iou_sum = 0.0
    dice_sum = 0.0
    best_dice_sum = 0.0
    n = 0
    n_gt_empty = 0
    # Per-slice stats for non-empty-GT slices (the meaningful subset)
    iou_sum_ne = 0.0
    dice_sum_ne = 0.0
    best_dice_sum_ne = 0.0
    n_ne = 0
    resize = torchvision.transforms.Resize((args.image_size, args.image_size))

    for name in files:
        m = pat.match(name)
        if not m:
            print(f"  [skip] cannot parse slice id from {name}")
            continue
        study_id, slice_idx = m.group(1), int(m.group(2))

        mask_path = _resolve_mask_path(args.data_root, study_id)
        if not os.path.isfile(mask_path):
            print(f"  [skip] GT mask not found for study_{study_id}: {mask_path}")
            continue

        vol = _mask_volume(mask_path)
        if slice_idx >= vol.shape[-1]:
            print(f"  [skip] slice {slice_idx} out of range for study_{study_id} "
                  f"(has {vol.shape[-1]} slices)")
            continue
        gt2d = vol[..., slice_idx]

        # Resize GT to the same size as the prediction (256x256 by default).
        gt_t = torch.tensor(gt2d)[None, None, ...].float()
        gt_t = resize(gt_t)[0, 0].numpy()
        gt_bin = (gt_t > 0.5).astype(np.uint8)
        gt_empty = (gt_bin.sum() == 0)
        if gt_empty:
            n_gt_empty += 1

        # Load prediction. save_image maps tensor [0,1] -> [0,255] grayscale.
        # Recover probability by dividing by 255 (NOT by pred.max(), which
        # amplifies background noise on near-empty predictions).
        pred = Image.open(os.path.join(args.pred_dir, name)).convert("L")
        pred = torch.tensor(np.array(pred))[None, None, ...].float()
        pred = pred / 255.0
        pred = pred[0, 0].numpy()

        # Evaluate at multiple thresholds; report mean and best.
        iou_t = dice_t = 0.0
        best_dice = 0.0
        for th in args.thresholds:
            pb = (pred > th).astype(np.uint8)
            d = _dice(pb, gt_bin)
            iou_t += _iou(pb, gt_bin)
            dice_t += d
            if d > best_dice:
                best_dice = d
        iou_t /= len(args.thresholds)
        dice_t /= len(args.thresholds)

        iou_sum += iou_t
        dice_sum += dice_t
        best_dice_sum += best_dice
        n += 1
        if not gt_empty:
            iou_sum_ne += iou_t
            dice_sum_ne += dice_t
            best_dice_sum_ne += best_dice
            n_ne += 1
        flag = " [GT empty]" if gt_empty else ""
        print(f"  study_{study_id} slice{slice_idx}: "
              f"dice={dice_t:.4f} best={best_dice:.4f} iou={iou_t:.4f}{flag}")

    if n == 0:
        print("[mosmed_eval] no valid predictions evaluated. "
              "Check --pred_dir / --data_root / file naming.")
        return

    print(f"\n==== MosMed evaluation ====")
    print(f"  all slices      (n={n}, {n_gt_empty} GT-empty): "
          f"mean Dice={dice_sum / n:.4f} best={best_dice_sum / n:.4f} "
          f"IoU={iou_sum / n:.4f}")
    if n_ne > 0:
        print(f"  lesion slices   (n={n_ne}):                   "
              f"mean Dice={dice_sum_ne / n_ne:.4f} best={best_dice_sum_ne / n_ne:.4f} "
              f"IoU={iou_sum_ne / n_ne:.4f}")
    else:
        print("  lesion slices: none (all GT were empty)")


if __name__ == "__main__":
    main()
