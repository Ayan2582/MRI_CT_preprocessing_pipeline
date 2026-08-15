"""
dataset.py
──────────
PairedSliceDataset — the training index over the QC-accepted MRI/CT pairs.

The preprocessing pipeline deliberately stopped short of sizing the images.
pipeline_config.py:35 states it outright: "This pipeline does not crop, pad, or
resize... the GAN's dataloader owns the crop decision." So this is where that
decision gets made, and there are three of them worth explaining.

SIZE. The dataset holds 28 distinct image sizes from 180x180 to 430x430, all
square, at 1 mm/pixel. Nothing is resized: rescaling would break the 1 px = 1 mm
property, and once that is gone a Hounsfield-unit error at one patient's scale is
no longer comparable to another's, which quietly invalidates every metric split
by region. Instead:
  - train      pad up to at least crop_size, then take a random crop_size crop
  - val/test   pad up to the next multiple of pad_multiple, keep the whole slice

Note the padding is needed in BOTH directions: 180 < 256, so the smallest slices
must be padded before a 256 crop is even possible.

PADDING VALUE. Zero, not reflection. The slices already have zero backgrounds —
the QC tool's erase brush and the area outside the scanner FOV both write 0.0 —
so a zero pad is continuous with what the image already shows at its border.
Reflection
padding would fabricate mirrored anatomy, and the discriminator would happily
learn that fabricated texture as a real feature of CT. Every padded pixel is
recorded in a validity mask so it can be excluded from losses and metrics rather
than diluting them toward zero.

RANGE. Arrays are stored in [0,1]; the generator ends in tanh, whose range is
[-1,1]. The conversion happens here, once, and `to_unit_range` is the single
inverse used by both the metrics and the sample renderer.
"""

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def to_unit_range(x):
    """[-1,1] (network space) -> [0,1] (stored/normalised space), clamped."""
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def to_network_range(x):
    """[0,1] -> [-1,1]."""
    return x * 2.0 - 1.0


def _pad_to(arr, target_h, target_w):
    """
    Zero-pad a 2-D array up to (target_h, target_w), centring the content.

    Returns (padded, mask) where mask is 1.0 on original pixels and 0.0 on pad.
    Centring rather than bottom-right padding keeps the anatomy near the middle
    of the frame, which matters because a random crop is taken next and
    off-centre content would bias which anatomy the crops see.
    """
    h, w = arr.shape
    pad_h, pad_w = max(0, target_h - h), max(0, target_w - w)
    top, left = pad_h // 2, pad_w // 2

    out = np.zeros((h + pad_h, w + pad_w), dtype=np.float32)
    out[top:top + h, left:left + w] = arr

    mask = np.zeros_like(out, dtype=np.float32)
    mask[top:top + h, left:left + w] = 1.0
    return out, mask


def _ceil_multiple(value, multiple):
    return int(np.ceil(value / multiple) * multiple)


class PairedSliceDataset(Dataset):
    """
    One item = one QC-accepted MRI/CT slice pair.

    Parameters
    ----------
    manifest    : DataFrame from data.manifest (already filtered to this split)
    root        : dataset root that manifest paths are relative to
    mode        : 'train' (random crop + flip) or 'val'/'test' (full padded slice)
    crop_size   : training crop edge, in pixels == millimetres
    pad_multiple: val/test images are padded up to a multiple of this, so a
                  U-Net with 8 downsamplings can consume them without the
                  encoder/decoder shapes disagreeing
    hflip       : horizontal flip augmentation, train only
    """

    def __init__(self, manifest, root, mode="train", crop_size=256,
                 pad_multiple=256, hflip=True, num_downs=8):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be train/val/test, got {mode!r}")

        self.df = manifest.reset_index(drop=True)
        self.root = os.path.abspath(root)
        self.mode = mode
        self.crop_size = int(crop_size)
        self.hflip = bool(hflip) and mode == "train"
        self.num_downs = int(num_downs)

        # The U-Net halves the resolution num_downs times, so every side must be
        # divisible by 2**num_downs. Enforcing it here turns a confusing shape
        # mismatch deep inside the decoder into a clear message at construction.
        required = 2 ** self.num_downs
        self.pad_multiple = max(int(pad_multiple), required)
        if self.mode == "train" and self.crop_size % required:
            raise ValueError(
                f"crop_size {self.crop_size} is not divisible by 2**num_downs "
                f"({required}); the U-Net decoder shapes would not line up."
            )

        if len(self.df) == 0:
            raise ValueError(f"PairedSliceDataset('{mode}') got an empty manifest.")

        logger.info("dataset[%s]: %d pairs, %d patients", mode, len(self.df),
                    self.df["patient_id"].nunique())

    def __len__(self):
        return len(self.df)

    def _load(self, rel_path):
        path = os.path.join(self.root, rel_path.replace("/", os.sep))
        try:
            arr = np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing slice: {path}\n"
                f"The manifest and the data root disagree. Check data.root in "
                f"your config, or regenerate with model/scripts/make_split.py"
            ) from None
        return np.ascontiguousarray(arr, dtype=np.float32)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        mri = self._load(row["mri_path"])
        ct = self._load(row["ct_path"])

        if mri.shape != ct.shape:
            # metadata.csv reports 0 mismatches today, but a per-modality export
            # rect (the mri_height/mri_width columns) could introduce one, and a
            # silently misaligned pair is worse than a crashed epoch.
            raise ValueError(
                f"Shape mismatch for pair {index} ({row['patient_id']} "
                f"{row['ct_series']}): MRI {mri.shape} vs CT {ct.shape}"
            )

        if self.mode == "train":
            # Pad only where the slice is smaller than the crop (180x180 exists);
            # larger slices are left alone and the crop samples inside them.
            target_h = max(self.crop_size, mri.shape[0])
            target_w = max(self.crop_size, mri.shape[1])
            mri, mask = _pad_to(mri, target_h, target_w)
            ct, _ = _pad_to(ct, target_h, target_w)

            # One crop origin for both modalities. The pair is already only
            # approximately registered; cropping them independently would add a
            # second, larger misalignment on top.
            h, w = mri.shape
            top = np.random.randint(0, h - self.crop_size + 1)
            left = np.random.randint(0, w - self.crop_size + 1)
            sl = (slice(top, top + self.crop_size), slice(left, left + self.crop_size))
            mri, ct, mask = mri[sl], ct[sl], mask[sl]

            if self.hflip and np.random.rand() < 0.5:
                mri, ct, mask = mri[:, ::-1], ct[:, ::-1], mask[:, ::-1]
                mri, ct, mask = (np.ascontiguousarray(a) for a in (mri, ct, mask))
        else:
            target_h = _ceil_multiple(mri.shape[0], self.pad_multiple)
            target_w = _ceil_multiple(mri.shape[1], self.pad_multiple)
            mri, mask = _pad_to(mri, target_h, target_w)
            ct, _ = _pad_to(ct, target_h, target_w)

        item = {
            # A = source (MRI), B = target (CT), following the pix2pix convention.
            "A": to_network_range(torch.from_numpy(mri)).unsqueeze(0),
            "B": to_network_range(torch.from_numpy(ct)).unsqueeze(0),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0),
            # The HU window travels with the sample so metrics can invert to
            # Hounsfield units per-sample without a lookup table — and so the
            # training loop never needs Preprocessing/, which Kaggle lacks.
            "hu_min": torch.tensor(float(row["hu_min"])),
            "hu_max": torch.tensor(float(row["hu_max"])),
            "body_region": str(row["body_region"]),
            "patient_id": str(row["patient_id"]),
            "index": int(index),
        }
        return item


def build_datasets(cfg, manifest, split):
    """
    Construct the train/val/test datasets for a resolved config.

    Returns {split_name: PairedSliceDataset}. Splits with no patients are
    omitted rather than raising, so a smoke-test manifest with only training
    patients still works.
    """
    datasets = {}
    for name in ("train", "val", "test"):
        # Filter on subject_id, matching how the split was built: PA32 owns two
        # patient folders and both belong to whichever split PA32 landed in.
        subjects = split.get(name, [])
        rows = manifest[manifest["subject_id"].isin(subjects)]
        if len(rows) == 0:
            logger.warning("split '%s' has no slices; skipping", name)
            continue
        datasets[name] = PairedSliceDataset(
            manifest=rows,
            root=cfg.data.root,
            mode=name,
            crop_size=cfg.data.crop_size,
            pad_multiple=cfg.data.pad_multiple,
            hflip=cfg.data.get_path("augment.hflip", True),
            num_downs=cfg.model.generator.num_downs,
        )
    return datasets
