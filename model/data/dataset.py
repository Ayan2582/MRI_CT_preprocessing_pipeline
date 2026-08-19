"""
dataset.py
──────────
The training index over the QC-accepted MRI/CT slices.

PairedSliceDataset is the one every experiment up to exp7 uses: one item is one
QC-accepted pair, both modalities from the same manifest row.
UnpairedSliceDataset exists only for the CycleGAN baseline (exp8) and is
described at its own definition.

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

    ROI_MODES = ("none", "crop", "mask")

    @staticmethod
    def normalise_roi_mode(value):
        """
        Accept the booleans as well as the three names.

        `use_roi: true` is the spelling everyone reaches for first, and YAML
        parses it as a bool rather than a string. Rejecting it would mean
        discovering the mistake on Kaggle, after an upload — so true maps to
        'crop' (the recommended mode) and false to 'none'.
        """
        if isinstance(value, bool):
            return "crop" if value else "none"
        value = str(value).strip().lower()
        if value in ("true", "yes", "1"):
            return "crop"
        if value in ("false", "no", "0"):
            return "none"
        return value

    def __init__(self, manifest, root, mode="train", crop_size=256,
                 pad_multiple=256, hflip=True, num_downs=8, use_roi="none"):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be train/val/test, got {mode!r}")
        use_roi = self.normalise_roi_mode(use_roi)
        if use_roi not in self.ROI_MODES:
            raise ValueError(
                f"data.use_roi must be one of {self.ROI_MODES} (or true/false, "
                f"which map to 'crop'/'none'), got {use_roi!r}")

        self.df = manifest.reset_index(drop=True)
        self.root = os.path.abspath(root)
        self.mode = mode
        self.crop_size = int(crop_size)
        self.hflip = bool(hflip) and mode == "train"
        self.num_downs = int(num_downs)
        self.use_roi = use_roi

        if self.use_roi != "none":
            missing = [c for c in ("roi_x", "roi_y", "roi_w", "roi_h")
                       if c not in self.df.columns]
            if missing:
                raise ValueError(
                    f"data.use_roi='{use_roi}' needs {missing} in the manifest. "
                    f"Regenerate it: python model/scripts/make_split.py --force"
                )

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

        logger.info("dataset[%s]: %d pairs, %d patients, use_roi=%s", mode,
                    len(self.df), self.df["patient_id"].nunique(), self.use_roi)

    def __len__(self):
        return len(self.df)

    def _roi_rect(self, row, shape):
        """
        The reviewer's ROI as integer pixel bounds, clipped to the image.

        roi_mode is 'metric' throughout this dataset and the pipeline resamples
        to 1 mm/pixel, so millimetres and pixels are the same number and the
        stored values need no conversion.
        """
        h, w = shape
        x0 = max(0, int(round(float(row["roi_x"]))))
        y0 = max(0, int(round(float(row["roi_y"]))))
        x1 = min(w, int(round(float(row["roi_x"]) + float(row["roi_w"]))))
        y1 = min(h, int(round(float(row["roi_y"]) + float(row["roi_h"]))))
        if x1 - x0 < 8 or y1 - y0 < 8:
            # Degenerate box — fall back to the whole frame rather than hand
            # back an 8-pixel image that would fail padding in a confusing way.
            logger.warning("degenerate ROI (%d,%d)-(%d,%d) for %s; using full frame",
                           x0, y0, x1, y1, row.get("ct_path"))
            return 0, 0, w, h
        return x0, y0, x1, y1

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

        # ── Region of interest ───────────────────────────────────────────────
        # The QC reviewer drew a box around the anatomy, deliberately excluding
        # the scanner table rails — thin bright near-vertical lines that appear
        # in the CT and have no counterpart in the MRI. Left in, they teach the
        # model that "black at the frame edge in MRI" maps to "bright vertical
        # line in CT", and it duly synthesises rails into the air around the
        # patient. Measured over training slices, 87% carry some bright content
        # outside the box.
        #
        #   crop  physically discard everything outside the box. The rails never
        #         reach the generator OR the discriminator. Strongest, and the
        #         reason the box was drawn.
        #   mask  keep the frame, zero both modalities outside the box and mark
        #         those pixels invalid so they leave no gradient and score in no
        #         metric. Keeps geometry, but the generator's output there is
        #         then unconstrained rather than correct.
        # `valid` starts as all-ones over the original pixels and travels through
        # every geometric step alongside the images, so whatever ends up in the
        # returned mask is the truth about which output pixels are real.
        valid = np.ones_like(mri, dtype=np.float32)

        if self.use_roi != "none":
            x0, y0, x1, y1 = self._roi_rect(row, mri.shape)
            if self.use_roi == "crop":
                sl = (slice(y0, y1), slice(x0, x1))
                mri, ct, valid = mri[sl], ct[sl], valid[sl]
            else:
                keep = np.zeros_like(mri, dtype=np.float32)
                keep[y0:y1, x0:x1] = 1.0
                mri, ct, valid = mri * keep, ct * keep, valid * keep
            mri, ct, valid = (np.ascontiguousarray(a, dtype=np.float32)
                              for a in (mri, ct, valid))

        if self.mode == "train":
            # Pad only where the slice is smaller than the crop (180x180 exists);
            # larger slices are left alone and the crop samples inside them.
            target_h = max(self.crop_size, mri.shape[0])
            target_w = max(self.crop_size, mri.shape[1])
            mri, pad_mask = _pad_to(mri, target_h, target_w)
            ct, _ = _pad_to(ct, target_h, target_w)
            valid, _ = _pad_to(valid, target_h, target_w)
            mask = pad_mask * valid

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
            mri, pad_mask = _pad_to(mri, target_h, target_w)
            ct, _ = _pad_to(ct, target_h, target_w)
            valid, _ = _pad_to(valid, target_h, target_w)
            mask = pad_mask * valid

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


class UnpairedSliceDataset(Dataset):
    """
    One item = an MRI from one patient and a CT from a DIFFERENT patient.

    WHY THIS EXISTS AT ALL, GIVEN THE DATA IS PAIRED. CycleGAN is the standard
    unpaired comparator in the MRI->CT literature, and the question it answers
    here is "what does the pairing actually buy me?" — how much of exp1's
    accuracy comes from having 2161 QC-accepted correspondences rather than just
    two piles of images. It is expected to lose on mae_norm. The size of the gap
    is the result.

    THE SPLIT IS BY PATIENT, NOT BY ROW. The training subjects are partitioned
    into two disjoint halves and the MRI is drawn only from the first, the CT
    only from the second. Shuffling rows within one pool would be easier and is
    what most repurposed-paired-data CycleGAN setups do, but it leaves the model
    able to see both modalities of the same patient — so "unpaired" would be a
    claim the experiment does not support. The cost is that each domain sees
    about half the slices; that is the honest price of the claim.

    VALIDATION IS NEVER UNPAIRED. build_datasets swaps only the training set.
    Every metric in evaluation/metrics.py compares a prediction against ITS
    target, so an unpaired validation set would report numbers that mean nothing
    while still plotting a perfectly convincing curve.

    IMPLEMENTATION. Two PairedSliceDatasets over the two halves, from which only
    the wanted modality is kept. That loads one .npy more per item than strictly
    needed, which is a real but small cost (~300 KB), and buys exact reuse of the
    ROI handling, padding, crop and mask logic rather than a second copy of it
    that could drift.
    """

    def __init__(self, manifest, split_seed=1337, **kwargs):
        self._kwargs = dict(kwargs, mode="train")
        self._split_seed = int(split_seed)
        self._build(manifest)

    def _build(self, manifest):
        subjects = sorted(manifest["subject_id"].unique())
        if len(subjects) < 2:
            raise ValueError(
                f"UnpairedSliceDataset needs at least 2 training subjects to "
                f"split into disjoint domains, got {len(subjects)}."
            )

        # A dedicated RandomState, not the global one: which patients land in
        # which domain must depend on data.unpaired_split_seed alone, so the
        # partition is identical across runs, resumes and machines. Drawing from
        # the global RNG would make it depend on how many augmentation samples
        # had been taken before the dataset was constructed.
        order = list(subjects)
        np.random.RandomState(self._split_seed).shuffle(order)
        half = len(order) // 2
        subjects_a, subjects_b = set(order[:half]), set(order[half:])

        self.subjects_a, self.subjects_b = sorted(subjects_a), sorted(subjects_b)
        self.domain_A = PairedSliceDataset(
            manifest[manifest["subject_id"].isin(subjects_a)], **self._kwargs)
        self.domain_B = PairedSliceDataset(
            manifest[manifest["subject_id"].isin(subjects_b)], **self._kwargs)

        logger.info("dataset[train] UNPAIRED: MRI from %d subjects (%d slices), "
                    "CT from %d disjoint subjects (%d slices), split_seed=%d",
                    len(subjects_a), len(self.domain_A),
                    len(subjects_b), len(self.domain_B), self._split_seed)

    # `df` is a read/write view over both halves so that callers which subsample
    # a dataset by assigning to .df — scripts/smoke_test.py does — keep working.
    # Assignment re-derives the partition from the same seed rather than slicing
    # the existing halves, because an arbitrary subset of rows can easily miss
    # one domain entirely.
    @property
    def df(self):
        import pandas as pd
        return pd.concat([self.domain_A.df, self.domain_B.df], ignore_index=True)

    @df.setter
    def df(self, manifest):
        self._build(manifest.reset_index(drop=True))

    def __len__(self):
        # The larger domain defines an epoch; the smaller is drawn from with
        # replacement. Taking the min instead would silently discard slices from
        # whichever half happens to be bigger.
        return max(len(self.domain_A), len(self.domain_B))

    def __getitem__(self, index):
        a = self.domain_A[index % len(self.domain_A)]
        b = self.domain_B[np.random.randint(len(self.domain_B))]

        return {
            "A": a["A"],                 # MRI, patient from domain A
            "B": b["B"],                 # CT, patient from domain B
            # Two masks, because the two images are different sizes' worth of
            # different patients and share no geometry. Everything upstream of
            # exp8 assumes one shared mask, which is only meaningful for a
            # co-registered pair.
            "mask_A": a["mask"],
            "mask_B": b["mask"],
            # Kept so that anything reading the paired key still gets something
            # coherent: it is A's mask, which is the right one for the A-domain
            # terms.
            "mask": a["mask"],
            "hu_min": b["hu_min"],       # the CT window belongs to the CT
            "hu_max": b["hu_max"],
            "body_region": b["body_region"],
            "patient_id": a["patient_id"],
            "patient_id_B": b["patient_id"],
            "index": int(index),
        }

    def subsample_per_region(self, limit):
        """
        Keep roughly `limit` slices per domain, spread across body regions.

        Exists for the CPU smoke test. Subsampling each domain separately is what
        keeps both non-empty — taking `limit` rows from the concatenated frame
        can land them all on one side of the partition.
        """
        for domain in (self.domain_A, self.domain_B):
            frame = domain.df
            per_region = max(1, limit // frame["body_region"].nunique())
            domain.df = (frame.groupby("body_region", sort=True)
                         .head(per_region).reset_index(drop=True))
        return self


def build_datasets(cfg, manifest, split):
    """
    Construct the train/val/test datasets for a resolved config.

    Returns {split_name: dataset}. Splits with no patients are omitted rather
    than raising, so a smoke-test manifest with only training patients still
    works.

    With data.unpaired the TRAIN split becomes an UnpairedSliceDataset and val
    and test stay paired. That asymmetry is deliberate and is the only way the
    CycleGAN baseline can be scored against the other experiments at all — see
    UnpairedSliceDataset's docstring.
    """
    unpaired = bool(cfg.data.get("unpaired", False))
    datasets = {}
    for name in ("train", "val", "test"):
        # Filter on subject_id, matching how the split was built: PA32 owns two
        # patient folders and both belong to whichever split PA32 landed in.
        subjects = split.get(name, [])
        rows = manifest[manifest["subject_id"].isin(subjects)]
        if len(rows) == 0:
            logger.warning("split '%s' has no slices; skipping", name)
            continue
        common = dict(
            root=cfg.data.root,
            crop_size=cfg.data.crop_size,
            pad_multiple=cfg.data.pad_multiple,
            hflip=cfg.data.get_path("augment.hflip", True),
            num_downs=cfg.model.generator.num_downs,
            use_roi=cfg.data.get("use_roi", "none"),
        )
        if unpaired and name == "train":
            datasets[name] = UnpairedSliceDataset(
                manifest=rows,
                split_seed=cfg.data.get("unpaired_split_seed", 1337),
                **common)
        else:
            datasets[name] = PairedSliceDataset(manifest=rows, mode=name, **common)
    return datasets
