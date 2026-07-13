import os
import csv
import numpy as np
import torch
import nibabel


class MosMedDataset(torch.utils.data.Dataset):
    """MosMed Chest CT (COVID-19) dataset loader for MedSegDiff.

    Reads 3D nii.gz volumes listed in ``dataset_registry.csv`` (only the rows
    that carry a ``mask_file``), slices them along the axial axis and returns
    2D (image, label, virtual_path) tuples. Mirrors the interface of
    ``BRATSDataset3D`` / ``CustomDataset3D`` so it can be plugged into the
    existing train/sample scripts without further changes.

    Unlike BRATS (MRI, already normalised), CT volumes are in Hounsfield Units
    (~ -1024 .. 3000). A configurable window (default: lung window
    L=-600 / W=1500) is applied per slice and mapped to [0, 1].
    """

    def __init__(self, data_root, transform, test_flag=False,
                 window_level=-600, window_width=1500, skip_empty=True):
        super().__init__()

        self.data_root = os.path.expanduser(data_root)
        self.transform = transform
        self.test_flag = test_flag
        self.window_level = window_level
        self.window_width = window_width
        self.skip_empty = skip_empty  # allow test mode to keep only lesion slices

        # Locate the registry CSV (sits next to studies/ and masks/).
        registry_path = os.path.join(self.data_root, "dataset_registry.csv")
        assert os.path.isfile(registry_path), (
            f"dataset_registry.csv not found under {self.data_root}"
        )

        # Build (ct_path, mask_path) pairs for every study that has a mask.
        self.valid_cases = []
        with open(registry_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mask_rel = (row.get("mask_file") or "").strip()
                if not mask_rel:
                    continue
                ct_rel = (row.get("study_file") or "").strip()
                ct_path = os.path.join(self.data_root,
                                       ct_rel.replace("/", os.sep).lstrip(os.sep))
                mask_path = os.path.join(
                    self.data_root,
                    mask_rel.replace("/", os.sep).lstrip(os.sep))
                if not (os.path.isfile(ct_path) and os.path.isfile(mask_path)):
                    # Registry mentions a file we do not have on disk yet.
                    continue
                self.valid_cases.append((ct_path, mask_path))

        assert len(self.valid_cases) > 0, (
            f"No paired CT/mask cases found under {self.data_root}. "
            "Check that studies/CT-*/study_*.nii.gz were downloaded."
        )

        # Pre-scan: load each mask once, record which slices carry a lesion.
        # We cache only the slice index list per case (not the volumes) so
        # memory stays flat; volumes are re-read per __getitem__.
        self.all_slices = []  # list of (case_idx, slice_idx)
        self._slice_cache = []  # per-case: list of kept slice indices
        print(f"[MosMed] scanning {len(self.valid_cases)} cases for "
              f"{'non-empty' if self.skip_empty else 'all'} slices...")
        for case_idx, (_, mask_path) in enumerate(self.valid_cases):
            seg = nibabel.load(mask_path).get_fdata()
            num_slices = seg.shape[-1]
            kept = []
            for s in range(num_slices):
                if (not self.skip_empty) or seg[..., s].sum() > 0:
                    kept.append(s)
            # A case may have zero positive slices if the lesion is tiny and
            # falls between integer slices; fall back to all slices so the
            # case is still represented instead of silently dropped.
            if len(kept) == 0:
                kept = list(range(num_slices))
            self._slice_cache.append(kept)
            self.all_slices.extend([(case_idx, s) for s in kept])
        print(f"[MosMed] kept {len(self.all_slices)} slices from "
              f"{len(self.valid_cases)} cases "
              f"(window L={self.window_level} W={self.window_width}).")

    def __len__(self):
        return len(self.all_slices)

    def _apply_window(self, arr):
        """Clip a HU array to [L-W/2, L+W/2] and rescale to [0, 1]."""
        lo = self.window_level - self.window_width / 2.0
        hi = self.window_level + self.window_width / 2.0
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        return arr.astype(np.float32)

    def __getitem__(self, x):
        case_idx, slice_idx = self.all_slices[x]
        ct_path, mask_path = self.valid_cases[case_idx]

        ct_vol = nibabel.load(ct_path).get_fdata().astype(np.float32)
        seg_vol = nibabel.load(mask_path).get_fdata().astype(np.float32)

        image2d = ct_vol[..., slice_idx]
        label2d = seg_vol[..., slice_idx]

        image = torch.tensor(self._apply_window(image2d))[None, ...]  # (1, H, W)
        label = torch.where(
            torch.tensor(label2d)[None, ...] > 0, 1, 0
        ).float()  # (1, H, W), binary {0, 1}

        # Virtual path identifies the slice uniquely; sample.py parses it
        # back into a slice_ID for naming output files.
        study_tag = os.path.basename(ct_path).split(".nii")[0]  # study_0255
        virtual_path = f"{study_tag}_slice{slice_idx}.nii"

        if self.test_flag:
            # Mirror BRATSDataset3D test-mode: return (image, image, path).
            if self.transform:
                image = self.transform(image)
            return (image, image, virtual_path)

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)
        return (image, label, virtual_path)
