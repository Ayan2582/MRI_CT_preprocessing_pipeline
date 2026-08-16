# MRI → CT synthesis: pix2pix + PatchNCE

A conditional GAN that synthesises CT from MRI, trained on the 2161 QC-accepted
slice pairs produced by this repository's preprocessing pipeline.

```
L_total  =  L_cGAN  +  λ_L1 · L_L1  +  λ_NCE · L_PatchNCE
```

Every λ is a config value, and **setting one to zero removes that term's entire
code path** — `lambda_gan: 0` builds no discriminator, `lambda_nce: 0` builds no
projection heads and no third optimizer. So an ablation is genuinely cheaper than
the full model, and a checkpoint's structure records which terms a run used.

---

## Quickstart

```bash
pip install -r model/requirements.txt

# 1. Build the manifest and the subject-level split (once)
python model/scripts/make_split.py

# 2. Prove the wiring on CPU (~2 min)
python model/scripts/smoke_test.py

# 3. Train  (needs a GPU — local torch here is CPU-only)
python model/scripts/train.py --config exp3_nce_heavy.yaml

# 4. Compare runs
python model/evaluation/evaluate.py --compare model/runs/exp* --split val
```

Any λ is a one-line change:

```bash
python model/scripts/train.py --config exp2_paper.yaml --set loss.lambda_nce=0
```

---

## Documentation

| document | read it for |
|---|---|
| **[gan_evaluation_guide.md](docs/gan_evaluation_guide.md)** | **how to tell whether your GAN is improving** — start here |
| [notebook_walkthrough.md](docs/notebook_walkthrough.md) | **what the Kaggle notebook does, cell by cell** — and exactly what trained |
| [training_log_reference.md](docs/training_log_reference.md) | **every number the trainer prints**, and its healthy range |
| [loss_function_guide.md](docs/loss_function_guide.md) | what each term does, what each λ changes |
| [training_strategies.md](docs/training_strategies.md) | objective variants, and the six stabilisers |
| [kaggle_workflow.md](docs/kaggle_workflow.md) | package → upload → train → resume |

---

## The experiment ladder

Run in order. Each answers exactly one question; each is one config file.

| # | config | λ_GAN | λ_L1 | λ_NCE | architecture | question |
|---|---|---|---|---|---|---|
| 0 | `exp0_l1_only` | 0 | 100 | 0 | U-Net + PatchGAN | is the GAN earning its keep at all? |
| 1 | `exp1_pix2pix` | 1 | 100 | 0 | U-Net + PatchGAN | what does the standard recipe give? |
| 2 | `exp2_paper` | 1 | 100 | 1 | U-Net + PatchGAN | the target loss, textbook weights |
| 3 | `exp3_nce_heavy` | 1 | 50 | 2 | U-Net + PatchGAN | lean on NCE — open question |
| 4 | `exp4_nce_max` | 1 | 10 | 5 | U-Net + PatchGAN | where does hallucination start? |
| 5 | `exp5_stylegan2_vanilla` | 1 | **0** | **0** | StyleGAN2 | what does StyleGAN2's *own* loss give? |
| 6 | `exp6_stylegan2_fitted` | 1 | 100 | 1 | StyleGAN2 | does the architecture beat the U-Net? |

**0–4 vary the objective on one fixed architecture. 5 and 6 do the opposite.**

`exp0` is the floor and `exp5` is the ceiling — the same experiment run from
opposite ends. exp0 is a regression with no adversary; exp5 is StyleGAN2's
adversarial objective with no reconstruction term at all. Neither is a candidate:
if the GAN runs don't beat exp0 on anything but sharpness you are paying compute
and hallucination risk for a cosmetic change, and exp5 exists to show how much of
the texture is coming from the adversary alone.

**exp5 will score badly on `mae_norm`, and that is the expected result.** With
λ_L1 = 0 the only thing tying the synthesised CT to the input MRI is the
conditional discriminator seeing `cat[MRI, CT]` — pix2pix-without-L1, which the
pix2pix paper itself ablates and reports as substantially worse. Read exp5 for
texture and stability, exp6 for accuracy.

**exp6 is the one to compare against exp2.** Identical λs, identical data, split
and seed — so the only thing that moved is the architecture.

---

## Three facts about this dataset that shaped the design

### 1. Alignment was fixed by hand, so PatchNCE is a question — not a fix

`qc_app/registration_service.py:26-37` reports a median CT/MRI frame mismatch of
**5.2°**, and it is tempting to justify a misalignment-tolerant loss with it.
**That justification does not hold here.** The 5.2° describes the *raw DICOM
frames* — the input to quality control, not its output.

Every pair was then reviewed manually: **129 rejected**, **738 slices nudged
individually** (48 of 119 series carry per-slice nudges spanning up to 84 mm),
artifacts erased on **560**. Per-slice nudging is specifically the correction for
the slice-to-slice offset variation that out-of-plane tilt produces, and it was
applied across the dataset by eye.

What survives is **in-plane rotation**, which no translation fixes at any
granularity — and its magnitude here is **unmeasured**, because QC recorded what
was done to each slice rather than an alignment score afterwards.

So `exp3_nce_heavy` is an open experiment, not a predicted winner. PatchNCE may
still earn its weight: rotation is genuinely uncorrected, and a contrastive term
supplies structural signal that neither L1 nor a patch discriminator provides,
which matters at 1687 training slices. Run it to find out.

### 2. CT `[0,1]` → Hounsfield units is region-dependent, and brain saturates bone

| region | HU window | slices | bone (>150 HU) representable? |
|---|---|---|---|
| brain | 0 … 80 | 673 | **no — everything ≥80 HU clips to 1.0** |
| abdomen | −160 … 240 | 943 | yes, at norm 0.775 |
| MSK / spine | −200 … 300 | 545 | yes, at norm 0.70 |

Two consequences, both enforced in `evaluation/metrics.py`:

- **There is no valid pooled "MAE in HU".** The same normalised error is 80 HU on
  a brain slice and 500 HU on a spine slice. HU error is reported **per region**;
  the cross-region selection scalar is `mae_norm`.
- **Bone Dice and HU-band MAE skip brain slices** (computed on ~159 of 230
  validation slices), and every report states the count.

### 3. Pairs cannot be matched by path string

120 of 2161 MRI files sit in a folder whose name carries an orientation suffix the
CT folder lacks — CT `PA42_Poonam/ST0/SE1` pairs with MRI `.../SE1_axial`. A
loader that derives one path from the other silently drops 5.5% of the dataset
and reports no error. The manifest's `ct_path`/`mri_path` columns are the only
correct source.

---

## Layout

```
model/
├── configs/          base.yaml + the seven experiments (deltas only)
├── data/             manifest, subject-level split, PairedSliceDataset
├── networks/         builder (type dispatch), U-Net with tappable encoder,
│                     PatchGAN, StyleGAN2 G+D, PatchSampleF, tiled inference
├── losses/           GAN variants, PatchNCE, the loss plan
├── training/         composite model, trainer, EMA, DiffAugment
├── evaluation/       region-aware metrics, run comparison
├── scripts/          make_split, train, smoke_test, package_for_kaggle
├── notebooks/        kaggle_train.ipynb
└── docs/             the four guides above
```

`networks/builder.py` is the only module that knows more than one architecture
exists; `unet.py` and `patchgan.py` still know only about themselves. A third
architecture is one import and one branch there.

A StyleGAN2 generator emits **one fixed resolution**, but 45% of validation slices
(103 of 230 — every abdomen and every spine slice) pad to 512. `networks/tiling.py`
generates those through overlapping 256 windows blended with a raised cosine, so
all 230 slices are scored and exp5/exp6 stay comparable to exp0–exp4. Nothing in
the trainer or `evaluate.py` changes: the generator tiles internally.

`model/data/manifest.csv` and `model/data/splits.json` are **committed on
purpose**. Every experiment must train and validate on the same subjects or the
comparisons between them mean nothing. (`.gitignore` carries an explicit
negation, since `*.json` is ignored repo-wide.)

Nothing under `Preprocessing/` or `qc_app/` is modified. `bootstrap.py` reaches
into `pipeline_config.REGION_PROFILES` for the HU windows rather than re-typing
them, matching the pattern in `qc_app/bootstrap.py`.

---

## Data

| | |
|---|---|
| pairs | 2161 QC-accepted (2184 minus 23 flagged background) |
| subjects | 44 (45 patient folders; PA32 owns two) |
| split | 32 train / 6 val / 6 test subjects, region-stratified |
| format | `float32`, exactly `[0,1]`, CT shape == MRI shape |
| sizes | 28 distinct, 180×180 to 430×430, all square, 1 px = 1 mm |

**The split is by subject, not by slice or by folder.** Adjacent slices in a
series are near-duplicates, so a slice-level split validates the model on data it
has memorised. And PA32's ankle and knee folders are the same person — splitting
on folder name would put one in validation and the other in test.

Training takes a random 256×256 crop (padding first where a slice is smaller);
validation keeps the whole slice padded to a multiple of 256, with the padding
excluded from every metric.

---

## Verification

`smoke_test.py` runs on CPU in a few minutes and checks four things:

1. **Wiring** — the full objective trains; all three networks and optimizers exist.
2. **Modularity** — `lambda_nce=0` produces a checkpoint with *no* `netF` or
   `optimizer_F`; `lambda_gan=0` produces one with no `netD` or `optimizer_D`.
   Asserted on checkpoint structure, because a term multiplied by zero would look
   identical in the loss values.
3. **Resume** — 4 epochs straight equals 2 + resume + 2. Currently bit-exact
   (`delta = 0.00e+00`).
4. **StyleGAN2** — five assertions, four of which guard *silent* failures:
   - tiled inference of an identity function reconstructs its input exactly, which
     is only true if the blending weights sum to one at every pixel including the
     borders;
   - exp5's checkpoint holds `pl_mean` and **no** `netF`, exp6's the reverse — the
     same structural argument as check 2, proving exp5 really runs StyleGAN2's
     loss rather than a zero-weighted composite;
   - two eval forwards on one input are bit-identical, proving noise injection is
     off at evaluation and `mae_norm` is not a random variable;
   - StyleGAN2 conv weights have std ≈1 — if anything ever routes them through
     `init_weights` they become N(0,0.02) *and* runtime-scaled, roughly 50× too
     small, and the network trains happily while learning nothing;
   - GAN warm-up with no reconstruction term is refused at config time rather than
     crashing inside `backward_G` on a loss that never entered the graph.
