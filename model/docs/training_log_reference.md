# Reading the training log

Every number the trainer prints, what it means, and what range is healthy.

---

## The iteration line

```
e006  100/210  D_acc_fake=0.0012  D_acc_real=0.9992  D_total=0.2111
               G_GAN=0.2961  G_L1=0.1881  G_total=19.1050
```

`e006` = epoch 6. `100/210` = iteration 100 of 210 in this epoch (1687 training
slices ÷ batch 8). Values are for **that single batch**, not epoch averages — the
epoch average is what lands in `metrics.csv` under `train/*`.

During **GAN warm-up** (`train.gan_warmup_epochs`, default 5) only `G_L1` and
`G_total` appear. The discriminator is not being updated, so there is nothing to
report for it. Seeing `D_*` and `G_GAN` appear at epoch 5 is the warm-up ending.

### Generator terms

| field | what it is | direction |
|---|---|---|
| `G_L1` | mean abs. error vs the real CT, normalised units, padding excluded | ↓ |
| `G_GAN` | how convincingly D was fooled this batch | see below |
| `G_NCE` | PatchNCE loss, mean over tapped layers (absent when λ_NCE = 0) | ↓ slowly |
| `G_PL` | path-length penalty, StyleGAN2 runs only (absent unless enabled) | ↓ slowly |
| `G_PL_len` | the raw path length the penalty is pulling toward its running mean | settles |
| `G_total` | `λ_GAN·G_GAN + λ_L1·G_L1 + λ_NCE·G_NCE` | ↓ |

**A field is absent, not zero, when its term is off.** `exp5_stylegan2_vanilla`
writes no `G_L1` and no `G_NCE` at all, because λ_L1 = λ_NCE = 0 removes those
code paths rather than multiplying them by zero. If you are diffing two runs'
columns, a missing column is the objective differing, not a logging bug.

**`G_total` is dominated by L1.** With λ_L1 = 100 and G_L1 = 0.188, the L1 term
contributes 18.8 of a 19.1 total — 98%. `G_total` is essentially `100 × G_L1`
with a rounding error attached, so watching it tells you almost nothing that
`G_L1` doesn't. Watch the components.

**On `exp5_stylegan2_vanilla`, `G_total` *is* `G_GAN`.** There is nothing else in
that objective, so for once the total is worth reading directly — and `mae_norm`
in the validation table is a quantity the run never optimises. Expect it to be
poor and do not read that as a bug; see the ladder table in the README.

**`G_PL` has no published healthy range for this setting.** Path-length
regularization was characterised on unconditional face synthesis at 1024 px with
`w` drawn from a Gaussian prior. Here `w` is a real patient's encoding and the
penalty is estimated from the batch in flight, so the magnitude is not comparable
to anything in the literature. Watch `G_PL_len` for stabilisation rather than
`G_PL` for a target value: a path length that keeps climbing means the map is
getting more ill-conditioned, not better.

**`G_GAN` is not a quality score.** It measures G's performance against a
discriminator that changes every step. It falling can mean G improved *or* D got
worse. Flat and noisy in the 0.2–0.5 range is the healthy signature for LSGAN —
it means neither player is running away with it. A `G_GAN` that climbs steadily
while samples stop improving means D is winning; one that collapses toward 0
means D has stopped discriminating.

### Discriminator terms

| field | what it is | healthy |
|---|---|---|
| `D_total` | `0.5 × (loss on reals + loss on fakes)` | 0.1–0.4, roughly flat |
| `D_acc_real` | fraction of real patches called real | 0.7–1.0 |
| `D_acc_fake` | fraction of fake patches called fake | 0.6–0.95 |
| `D_real` / `D_fake` | the two halves of `D_total` separately | — |
| `D_score_real` / `D_score_fake` | D's mean raw output on each | see below |

Note `D_total` is a **loss**, so lower is D doing better. It is not accuracy.

`D_acc_*` are computed on the PatchGAN's 30×30 score grid, so they average over
900 patch judgements per image, not one verdict per image.

---

## ⚠️ `D_acc_*` before and after the threshold fix

If your log shows **`D_acc_fake` near 0.0000 while `D_acc_real` is near 1.0000**,
and everything else looks fine, you are seeing a metric bug that was fixed after
the first `exp1_pix2pix` run. **It did not affect training** — these values are
detached and used only for logging.

**The cause.** LSGAN does not classify; it *regresses*. D is trained to output
0.9 for reals and 0.0 for fakes, so the decision boundary is the midpoint
between them, **0.45**. The original code compared against **0**.

A healthy LSGAN discriminator at epoch 6 scores roughly:

```
D_score_real ≈ 1.43      well above 0.45  ->  correctly "real"
D_score_fake ≈ 0.35      below 0.45       ->  correctly "fake"
                         but ABOVE 0      ->  scored as "fooled" by the old code
```

So a discriminator classifying essentially perfectly reported `D_acc_fake ≈ 0`.
The corrected threshold reports `1.000` for the same scores.

The threshold is now objective-dependent (`GANLoss.decision_threshold`):

| `gan_mode` | boundary | why |
|---|---|---|
| `lsgan` | `(real_target + fake_target) / 2` = 0.45 | midpoint of the two regression targets |
| `hinge` | 0 | reals pushed above +1, fakes below −1 |
| `vanilla` | 0 | raw logit; 0 is probability 0.5 |

**How to tell a real collapse from the artifact:** with the artifact, `D_total`
stays stable (~0.2), `G_GAN` stays in range, and `val/mae_norm` keeps improving.
In a real collapse, `D_total` runs to 0 or the losses go flat and sample quality
stalls. If in doubt, `D_score_real` and `D_score_fake` are recorded in
`metrics.csv` — compare them against 0.45 yourself.

---

## The validation table

Printed after each epoch, computed on the **EMA generator** over the full
validation split at native resolution, with zero-padding excluded.

```
  metric             overall     macro   per region
  ----------------------------------------------------------------------
  mae_norm            0.0884    0.0905   abdomen=0.1004 brain=0.08864 ...
  psnr               15.7266   15.7565   abdomen=15.46 brain=14.61 ...
  ssim                0.5940    0.5781   abdomen=0.537 brain=0.6186 ...
  mae_hu             28.6072   33.4519   abdomen=40.17 brain=7.091 ...
  mae_band_soft      31.1764   34.4319   abdomen=35.05 musculoskeletal=22.31 ...
  mae_band_bone     164.6648  149.0449   abdomen=175.8 musculoskeletal=154 ...
  dice_bone           0.1697    0.1865   abdomen=0.1499 musculoskeletal=0.1961 ...
  ----------------------------------------------------------------------
  n_samples=230  bone/band metrics computed on 159 slices
  (regions excluded: brain — its HU window saturates bone)
```

### The three columns

- **overall** — pooled across every validation slice.
- **macro** — mean of the four per-region means. Abdomen is 41% of the
  validation set, so `overall` is weighted toward it; `macro` gives each region
  equal say. When the two disagree, one region is behaving differently.
- **per region** — the four body regions separately.

### The metrics

**`mae_norm`** — mean absolute error in normalised `[0,1]` units. **This is the
model-selection scalar**; `best.pt` is whichever epoch minimises it. It is the
only quantity here that is comparable across regions, because it carries no
region-dependent scale factor.

**`psnr`** — peak signal-to-noise ratio, dB, data range 1.0. A monotone function
of MSE, so it adds little over `mae_norm` except familiarity. Higher is better.

**`ssim`** — structural similarity, 0 to 1, Gaussian window 11, mask eroded by
the window radius so values straddling the padding boundary are excluded. Higher
is better. More sensitive to structure than MAE and less forgiving of blur.

**`mae_hu`** — error in Hounsfield units, per region. **Never read the `overall`
column of this row.** The HU window is region-dependent:

| region | HU window | 1.0 normalised = |
|---|---|---|
| brain | 0 … 80 | 80 HU |
| abdomen | −160 … 240 | 400 HU |
| MSK / spine | −200 … 300 | 500 HU |

In the table above, brain reports 7.09 HU and spine 54.18 HU for *near-identical*
`mae_norm` (0.0886 vs 0.1084). Brain is not eight times better; its window is
six times narrower. Pooling them produces a number that moves when the
composition of the validation set changes rather than when the model does. Quote
`mae_hu` per region, always with the region named.

**`mae_band_soft` / `mae_band_bone`** — MAE in HU, restricted to pixels whose
**true** value falls in that tissue band (soft −200…150 HU, bone >150 HU). Bands
are defined on the real CT, so they ask "how well does the model reproduce tissue
that genuinely is bone", not "how well does it agree with itself". An `air` band
also exists but is usually empty — the windows clip at −200 HU, above air's
−1000. A band outside a sample's window returns nothing rather than 0.

**`dice_bone`** — overlap of the thresholded bone mask (>150 HU), 0 to 1, higher
better. **This is the hallucination detector.** Bone is a small fraction of
pixels, so a model can fabricate or erase it while barely moving `mae_norm` or
`ssim`. If sharpness improves and `dice_bone` falls, the extra sharpness is
invention, not accuracy.

### Why brain is excluded from bone metrics

The brain window tops out at **80 HU**; cortical bone is 300–2000 HU. Every bone
voxel on a brain slice is clipped to exactly 1.0 and is indistinguishable from
any other bright tissue. Bone is not merely hard to measure there — it is not
present in the data as a distinguishable value. Scoring it would measure
agreement about a saturated constant.

So bone and band metrics run on the 159 non-brain validation slices of 230, and
the footer states that count every time so the exclusion is never invisible.

---

## What a healthy run looks like

Epochs 0–4 (warm-up, L1 only):

```
mae_norm   0.102 -> 0.091      falling fast
ssim       0.37  -> 0.55
dice_bone  0.0003             ~zero: a grey blur has nothing above threshold
```

Epoch 5 onward (discriminator on):

```
mae_norm   may WORSEN for 10-20 epochs, then resume falling
ssim       keeps climbing
dice_bone  climbs steadily as real bone texture appears
D_acc_*    settles into its band, D_total roughly flat
```

**The bump at epoch 5 is expected.** L1 alone is the MAE-optimal objective;
adding an adversarial term deliberately trades some pixel accuracy for texture.
A rise that recovers is normal. A rise that never recovers is a problem — check
the D curves.

Your actual epoch 0 → 6 progression:

| | epoch 0 | epoch 6 |
|---|---|---|
| `mae_norm` | 0.1024 | **0.0884** |
| `ssim` | 0.3704 | **0.5940** |
| `dice_bone` | 0.0003 | **0.1697** |
| `mae_band_bone` | 187.96 | **164.66** |

All four moving the right way, and `dice_bone` going from nothing to 0.17 means
the model has started producing genuine bone rather than a uniform grey.

---

## Symptom → cause

| what you see | meaning | action |
|---|---|---|
| `D_acc_fake` ≈ 0, everything else fine | the threshold bug above | none — cosmetic, fixed |
| `D_acc_*` → 1.0, `D_total` → 0, `G_GAN` climbing | D overpowering G | `stabilizers.ttur.enabled=true` |
| `D_acc_*` → 0.5, `G_GAN` → 0, samples worsen | D collapsed | raise `ndf`, lower `lambda_gan` |
| D perfect on train, `val/mae_norm` plateaus | D memorising | check DiffAugment on; try R1 |
| `mae_norm` improving, images blurry | λ_L1 dominating | lower `lambda_l1` |
| images sharp, `dice_bone` falling | hallucination | raise `lambda_l1` |
| `val/mae_norm` bounces wildly epoch to epoch | not using EMA | `eval.use_ema=true` |
| `G_NCE` flat from the start | NCE taps too deep | shallower `loss.nce.layers` |
| `G_GAN` flatlines on exp5/exp6 | R1 γ too high — D over-smoothed | `stabilizers.r1.gamma` — see below |
| exp5/exp6 samples look like plausible CT of the *wrong* anatomy | no encoder→decoder skips; the only route from MRI to output is the global `w` | expected for this architecture; compare against exp2 rather than tuning |
| a visible seam down a large abdomen slice | tiled inference — each 256 window gets its own `w` | lower `model.generator.tile_stride` for more overlap |

**On R1 γ for the StyleGAN2 runs.** `base.yaml` ships `gamma: 10.0`, StyleGAN2's
FFHQ-1024 value. The heuristic is `γ = 0.0002 · N² / M`, so at 256 px it is 3.3 at
batch 4 and 1.6 at batch 8 — which is what exp5 and exp6 set. Too high and D stops
discriminating, which looks exactly like the GAN not helping.
| `nan` anywhere | AMP + exploding gradient | `runtime.amp=false` to confirm, then `train.grad_clip=1.0` |

---

## Other lines you will see

```
epoch 6: new best mae_norm = 0.08841 (was 0.09076)
```
`best.pt` was just overwritten. If this stops appearing for 40+ epochs and the
LR decay phase has begun, the run has converged.

```
saved /kaggle/working/runs/exp1_pix2pix/checkpoints/last.pt (epoch 6)
```
Written every epoch (`logging.save_every`). This is what `--resume auto` reads.

```
epoch 1: GAN warm-up — L1/NCE only, D is not updated (5 warm-up epochs configured)
```
Expected for epochs 0–4.

```
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`
```
Benign. During warm-up D's optimizer takes no steps while its scheduler still
advances. The LR schedule is constant until epoch 100, so the scheduler is
multiplying by 1.0 the whole time it warns. It would only matter if
`gan_warmup_epochs` exceeded `lr_decay_start_frac × n_epochs`.

---

## See also

- `notebook_walkthrough.md` — the Kaggle notebook cell by cell, and what each run trained

- `gan_evaluation_guide.md` — how to decide whether a run is better than another
- `loss_function_guide.md` — the three loss terms and their λs
- `training_strategies.md` — the objective variants and the six stabilisers
