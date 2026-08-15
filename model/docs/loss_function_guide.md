# The loss function, term by term

```
L_total  =  L_cGAN  +  λ_L1 · L_L1  +  λ_NCE · L_PatchNCE
```

Everything in this file maps to a config key you can change from the command
line. The design rule throughout: **setting a λ to zero removes that term's
entire code path**, not just its contribution. `lambda_gan: 0` builds no
discriminator at all; `lambda_nce: 0` builds no projection heads and no third
optimizer. So an ablation is genuinely cheaper than the full model, and the
checkpoint structure proves which terms a run actually used.

---

## The three terms at a glance

| term | config | default | supplies | fails by |
|---|---|---|---|---|
| `L_cGAN` | `loss.lambda_gan` | 1.0 | local texture realism | hallucinating structure |
| `L_L1` | `loss.lambda_l1` | 100.0 | global anatomy, intensity | blurring |
| `L_PatchNCE` | `loss.lambda_nce` | 1.0 | structural correspondence | (weak on its own) |

Note the **100×** on L1. That is not a typo and it is the single most important
number in the file: L1 supplies roughly 99% of the gradient magnitude. The
adversarial term is a texture correction on top of what is fundamentally a
regression model — it is not the main event.

---

## `L_L1` — the anatomy anchor

```python
L_L1 = mean(|G(mri) - ct|)     # over valid (non-padded) pixels only
```

Straightforward pixel-wise error. It does almost all the work: it fixes global
structure, intensity calibration, and where organs are.

**Its failure mode is the reason the other two terms exist.** When the model is
uncertain about a detail, the L1-minimising answer is not to guess — it is to
predict the *average* of the possibilities. Averaged possibilities look like a
blur. So an L1-only model is reliably accurate and reliably soft.

**λ_L1 = 100** is the pix2pix default and a good starting point. Raise it for
safety, lower it for sharpness.

> **Implementation note:** L1 here is *masked*. Validation slices are zero-padded
> up to a multiple of 256, and those pixels are identical in prediction and
> target. Averaging over them would make a heavily-padded 180×180 slice look more
> accurate than a 430×430 one for reasons unrelated to the model.

---

## `L_cGAN` — the texture critic

A 70×70 PatchGAN discriminator scores overlapping patches of `cat[MRI, CT]`.
G tries to make those scores say "real".

**What it is for:** breaking the blur. L1 cannot escape the conditional mean
because the conditional mean *is* its optimum. An adversarial term makes blurry
output actively penalised — a blur does not look like real CT tissue at 70×70,
whatever its mean error.

**Why conditional** (`cat[MRI, CT]` rather than CT alone): an unconditional
critic only asks "is this a plausible CT?", which G could satisfy with a
convincing CT of the *wrong patient*. Feeding both modalities makes the question
"is this a plausible CT **of this MRI**", so the adversarial term reinforces
correspondence instead of competing with it.

**Why patches, not the whole image:** L1 already handles global structure, so a
full-image discriminator would be redundant there, and it would have far more
capacity to memorise 1687 training slices. 70 px = 70 mm at this dataset's 1
mm/pixel — roughly organ scale.

`loss.gan_mode` selects the objective: `lsgan` (default), `vanilla`, `hinge`.
See `training_strategies.md` for what distinguishes them.

---

## `L_PatchNCE` — structural correspondence

This is the term worth understanding properly, because it is the one added
specifically for **your** data.

### The mechanism

1. Encode the input MRI through the generator's encoder. Encode the generated CT
   through the *same* encoder.
2. Pick 256 random spatial locations. Sample both encodings at **exactly those
   locations**.
3. Project each sampled vector through a small MLP head and L2-normalise.
4. For each location: its own pair is the **positive**; the 255 other locations
   *in the same image* are the **negatives**. Classify with cross-entropy at
   temperature 0.07.

In effect: *a patch of the output should resemble the input patch it came from,
more than it resembles any other patch of the same image.* That is a statement
about structure being preserved in place — with no claim about what the pixel
values should be.

### Why it might matter here — and a correction

There is a tempting argument for this term that **does not hold on this dataset**,
and it is worth stating so nobody rebuilds it from the raw numbers.

The argument: `qc_app/registration_service.py:26-37` reports a median CT/MRI
frame mismatch of **5.2°**, with 61 of 120 series above 5°. L1 charges full price
for a displaced target even when the model is right, and its optimal response to
systematic displacement is to blur. So a misalignment-tolerant term should help.

**Why it fails.** The 5.2° describes the **raw DICOM frames** — the input to
quality control, not the output. Every pair here was then reviewed by hand:

| | |
|---|---|
| pairs rejected outright | 129 |
| slices nudged **individually** | 738 |
| series with per-slice nudges | 48 of 119 (ranges up to 84 mm) |
| slices with artifacts erased | 560 |

Those per-slice nudges matter most. Out-of-plane tilt shows up as a translation
that *varies from slice to slice*, and correcting each slice separately is
exactly the fix for it. That was done, by hand, across the dataset.

**What actually survives:** in-plane rotation, which no translation corrects at
any granularity. Its magnitude here is **unmeasured** — the QC process recorded
what was done to each slice, not an alignment score afterwards.

### So why keep the term?

Two reasons that don't depend on misalignment:

1. **In-plane rotation is still uncorrected**, even if we can't quantify it.
2. **Contrastive signal is useful in its own right.** PatchNCE constrains
   structure in a way neither L1 (pixel-wise) nor the PatchGAN (texture realism)
   does, and 1687 training slices is few enough that an extra structural
   constraint may earn its place regardless of alignment.

`exp3_nce_heavy` (λ_L1 = 50, λ_NCE = 2) tests one reweighting. **Run it as an
open question.** If it doesn't beat `exp2_paper`, that is a clean, publishable
result — not a failure.

### The three failure modes to know about

1. **Shared sample locations are mandatory.** The positive pair is "the same
   place, before and after". Sampling the two encodings independently makes every
   pair a negative — and the loss *still goes down*, which is what makes this bug
   so easy to ship. The code draws ids once and passes them to the second call.

2. **Taps must not be too deep.** `loss.nce.layers` indexes encoder depth: tap 0
   is the input image, tap *k* is the output of encoder block *k−1*, at
   `crop / 2^k` resolution. With `num_downs: 8` and a 256 crop, the bottleneck is
   **1×1** — one spatial location, when you asked for 256 patches. The defaults
   `[0,1,2,3,4]` give 65536 / 16384 / 4096 / 1024 / 256 locations. Deeper taps
   degrade silently into resampling the same patch; the model warns at startup.

3. **The heads need their own optimizer, built lazily.** The MLP widths are the
   encoder's channel counts at the tapped depths, which are unknown until a real
   tensor has flowed through. So `optimizer_F` is created after the first forward
   pass. Checkpoint save/load tolerates its absence.

---

## Tuning: what to change and when

### The ladder

| config | λ_GAN | λ_L1 | λ_NCE | question |
|---|---|---|---|---|
| `exp0_l1_only` | 0 | 100 | 0 | is the GAN earning its keep at all? |
| `exp1_pix2pix` | 1 | 100 | 0 | what does the standard recipe give? |
| `exp2_paper` | 1 | 100 | 1 | the target loss, textbook weights |
| `exp3_nce_heavy` | 1 | 50 | 2 | lean on NCE — open question, see above |
| `exp4_nce_max` | 1 | 10 | 5 | where does hallucination start? |

Run them in that order. Each answers exactly one question, and `exp0` is the
floor everything else is measured against.

### Symptom → knob

| symptom | change |
|---|---|
| output too blurry | ↓ `lambda_l1` (100 → 50 → 20) |
| structure drifting / wrong | ↑ `lambda_l1` |
| sharp but bone invented (`dice_bone` ↓) | ↑ `lambda_l1`, ↓ `lambda_gan` |
| structure misplaced, misregistration suspected | ↑ `lambda_nce` |
| NCE loss won't move | taps too deep — shallower `loss.nce.layers` |
| GAN unstable early | ↑ `train.gan_warmup_epochs` |

### Changing λ mid-run

**Don't.** Start a new run with a new `run.name`. The optimizer state, the LR
schedule and the EMA all carry history from the old objective, and the resulting
curve is not interpretable as either configuration. The config-hash warning on
resume exists to catch exactly this.

---

## Running an ablation

```bash
# plain pix2pix from the full config
python model/scripts/train.py --config exp2_paper.yaml --set loss.lambda_nce=0

# L1-only regression floor
python model/scripts/train.py --config exp2_paper.yaml --set loss.lambda_gan=0

# CUT-like: contrastive, no L1
python model/scripts/train.py --config exp2_paper.yaml --set loss.lambda_l1=0
```

Every run prints its objective at startup, so a run's identity is never in doubt:

```
L_total = 1*L_cGAN(lsgan)  +  100*L_L1  +  1*L_PatchNCE(layers=[0,1,2,3,4], patches=256)
          [pix2pix + PatchNCE]
```

and the checkpoint records the same, structurally — `netD` is simply absent from
an `lambda_gan=0` run. `scripts/smoke_test.py` asserts this rather than trusting
it.

---

## Full config reference

```yaml
loss:
  gan_mode: lsgan          # lsgan | vanilla | hinge
  lambda_gan: 1.0          # 0 => no discriminator is built
  lambda_l1: 100.0         # 0 => no pixel anchor
  lambda_nce: 1.0          # 0 => no MLP heads, no optimizer_F

  nce:
    layers: [0, 1, 2, 3, 4]  # encoder tap depths; 0 is the input image
    num_patches: 256         # sampled locations per layer per image
    temperature: 0.07        # changing this changes what lambda_nce means
    nce_dim: 256             # projection head width
    use_mlp: true            # false => sample raw features, no optimizer needed
    nce_idt: false           # identity NCE on real CT; L1 already covers it
```

> **On `temperature`:** it is not a free knob. It scales the logits, so it
> changes the loss magnitude, so it changes the effective weight of `lambda_nce`.
> Change one or the other, not both, or you will not know which caused what.
