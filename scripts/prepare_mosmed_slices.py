"""Pre-process MosMed into the flat 2D-slice layout expected by
``CustomDataset3D`` (images/*.nii.gz + masks/*.nii.gz).

This is *optional*. ``MosMedDataset`` already reads the raw 3D volumes on the
fly, so you can train without running this script. Pre-slicing just avoids
re-reading the same 3D nii.gz file for every slice during training, which is
useful when you train for many steps on a slow disk.

Usage (run from the project root):

    python scripts/prepare_mosmed_slices.py \
        --data_root MosMedData-Chest-CT-Scans-with-COVID-19-Related-Findings-main \
        --out_dir   data/MosMedSlices \
        --window_level -600 --window_width 1500 \
        --skip_empty

After this, train with the generic 3D loader instead of MOSMED:

    python scripts/segmentation_train.py \
        --data_dir data/MosMedSlices ...   # (use the CustomDataset3D fallback)

NOTE: training via ``--data_name MOSMED`` reads the raw volumes directly and
does NOT require running this script. Keep it simple: prefer MOSMED.
"""
import argparse
import csv
import os

import numpy as np
import nibabel as nib


def apply_window(arr, level, width):
    lo = level - width / 2.0
    hi = level + width / 2.0
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="MosMed dataset root (contains dataset_registry.csv)")
    p.add_argument("--out_dir", required=True,
                   help="Output directory; images/ and masks/ created inside")
    p.add_argument("--window_level", type=float, default=-600)
    p.add_argument("--window_width", type=float, default=1500)
    p.add_argument("--skip_empty", action="store_true",
                   help="Drop slices whose mask has no foreground pixels")
    args = p.parse_args()

    img_dir = os.path.join(args.out_dir, "images")
    mask_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    registry = os.path.join(args.data_root, "dataset_registry.csv")
    assert os.path.isfile(registry), f"missing {registry}"

    pairs = []
    with open(registry, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mask_rel = (row.get("mask_file") or "").strip()
            if not mask_rel:
                continue
            ct_rel = (row.get("study_file") or "").strip()
            ct_path = os.path.join(args.data_root,
                                   ct_rel.replace("/", os.sep).lstrip(os.sep))
            mask_path = os.path.join(
                args.data_root,
                mask_rel.replace("/", os.sep).lstrip(os.sep))
            if os.path.isfile(ct_path) and os.path.isfile(mask_path):
                pairs.append((ct_path, mask_path))

    print(f"[prepare] {len(pairs)} paired cases; "
          f"window L={args.window_level} W={args.window_width}; "
          f"skip_empty={args.skip_empty}")

    n_img = n_mask = n_empty = 0
    for case_idx, (ct_path, mask_path) in enumerate(pairs):
        study_id = os.path.basename(ct_path).split(".nii")[0]  # study_0255
        ct = nib.load(ct_path).get_fdata().astype(np.float32)
        seg = nib.load(mask_path).get_fdata().astype(np.float32)
        assert ct.shape == seg.shape, f"shape mismatch {ct_path} {mask_path}"
        num_slices = ct.shape[-1]
        for s in range(num_slices):
            if args.skip_empty and seg[..., s].sum() == 0:
                n_empty += 1
                continue
            img2d = apply_window(ct[..., s], args.window_level, args.window_width)
            msk2d = (seg[..., s] > 0).astype(np.float32)

            # Save as a single-slice nii.gz preserving the (H, W) shape with a
            # dummy z-axis so CustomDataset3D's [::-1] indexing still works.
            img_nii = nib.Nifti1Image(img2d[..., None], affine=np.eye(4))
            msk_nii = nib.Nifti1Image(msk2d[..., None], affine=np.eye(4))
            name = f"{study_id}_slice{s:03d}.nii.gz"
            nib.save(img_nii, os.path.join(img_dir, name))
            nib.save(msk_nii, os.path.join(mask_dir, name))
            n_img += 1
            n_mask += 1
        if (case_idx + 1) % 10 == 0:
            print(f"[prepare] case {case_idx + 1}/{len(pairs)} done")
    print(f"[prepare] wrote {n_img} image slices, {n_mask} mask slices, "
          f"skipped {n_empty} empty slices -> {args.out_dir}")


if __name__ == "__main__":
    main()
