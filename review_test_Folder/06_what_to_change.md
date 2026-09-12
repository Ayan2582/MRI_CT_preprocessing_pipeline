# 06 — What to change

Ranked by what actually gates the result. Nothing here has been applied — this document
is the recommendation, not the change.

Read §1 and §2 together: §2 is the one that decides whether exp8 can work at all, and
§1 is the one that decides how long it takes to get there.

---

## 1. Break the 15:1 ratio

**Fixes finding 1** (doc 02). Config only.

```yaml
loss:
  lambda_cycle: 5.0        # was 10.0
  lambda_identity: 0.1     # was 0.5  -> effective weight 0.5, not 5.0
```

Remember `lambda_identity` is a *fraction of* `lambda_cycle` (`cyclegan.py:263`), so
`0.1` here gives an effective 0.5. That moves the balance from `1 : 10 : 5` to
`1 : 5 : 0.5` — the identity term goes from ten times the adversarial weight to half of
it, which is what you want when the two domains are genuinely different modalities
rather than two renderings of one scene.

If you would rather keep the published λs, the alternative is to ramp `lambda_gan` from
1 up to ~5 over the first 20 epochs. Same effect on the ratio, more machinery. Cutting
the identity weight is the cheaper experiment and I would run that first.

**How you will know it worked:** the epoch-5 or epoch-10 panel is no longer a clean copy
of the MRI.

## 2. Tell the generator which region it is translating

**Fixes finding 2** (doc 03). **Without this, nothing else matters** — the target
relation is one-to-many and a deterministic generator cannot represent it.

`body_region` is already in the batch (`dataset.py:402`) and already carried through to
`hu_min`/`hu_max`. Two ways to use it:

**(a) One run per region — simplest, and probably the right first move.** A brain-only
exp8 has one HU window by construction, and a real question attached to it: *what is the
pairing worth for brain?* You get an answer per region instead of one uninterpretable
average, and per doc 04 you would drop spine anyway. Cost: three runs instead of one.

**(b) Condition the generator on region.** Concatenate a one-hot region plane (or the
normalised `hu_min`/`hu_max` pair) as extra input channels to `G_A2B`. `in_channels` is
already a config knob (`base.yaml`, `networks/unet.py:53-116`), so this is a small
change — but D must be conditioned on the same signal or it will still enforce the mixed
marginal, and D-conditioning is what `cyclegan.py:63-83` currently refuses outright.
More work, one run, and it keeps exp8 a single number comparable to exp1.

Start with (a). If per-region exp8 works, (b) is a follow-up; if it does not, (b) would
not have saved it either.

## 3. Fix the sampler, and drop spine

**Fixes finding 3** (doc 04).

Region-matched sampling in `UnpairedSliceDataset.__getitem__` (`dataset.py:383-406`) —
draw the CT from the subset of `domain_B` whose `body_region` matches the MRI's, instead
of `np.random.randint(len(self.domain_B))`. Note this changes what exp8 claims: the
model is then told the region implicitly through the batch pairing, which is a weaker
form of §2(b). Worth doing regardless, because right now an abdomen MRI is routinely
batched against a brain CT and the adversarial gradient is meaningless on those steps.

Redundant if you take §2(a), which gets region matching for free.

**Drop spine from exp8 either way.** 27 training slices, 9 on the CT side after the
split. There is no seed that fixes that. Say so in the write-up rather than reporting a
spine number that means nothing.

## 4. Give exp8 a ResNet generator

**Fixes finding 4** (doc 05 §5). The largest change, and the one with a real trade-off.

A 9-block ResNet with two downsamplings — reference CycleGAN's generator — removes the
1×1 bottleneck and the every-level skips that make `G = I` architecturally cheap.
`networks/builder.py` already dispatches on `generator.type`, and
`model/docs/learn/13_adding_your_own.md` is the guide for adding one.

The trade-off is explicit and you should decide it before writing any code:
`cyclegan.py:86-89` reuses the U-Net specifically so exp8 differs from exp1 "in its
objective and its data pairing, not in its capacity." Changing the generator gives
CycleGAN a fair shot and costs you the clean one-variable comparison. Both are
defensible; running both is better. If you only run one, keep the U-Net — the ladder's
question is what the pairing is worth, not whether CycleGAN can be made to work.

## 5. Two small ones

- **Drop `color` from the DiffAugment policy for any run with `lambda_l1: 0`** (exp5,
  exp7, exp8). Doc 05 §4: it blinds D to global tone, which is harmless when L1 pins
  intensity and harmful when D is the only cross-domain signal. `policy:
  translation,cutout`.
- **Make the panel show the ROI, not the padded frame.** `trainer.py:388` averages
  unmasked over the full frame, so the printed MAE is diluted by background and does not
  match `val/mae_norm`. Either apply `item["mask"]` in the panel MAE or crop the panel
  to the valid region. Cosmetic, but it is what let epoch 9's copy-the-MRI output score
  0.1006 and look unremarkable.

## 6. Re-run exp1 before quoting any gap

**Not a CycleGAN fix — a measurement fix, and it blocks the headline result.**

`exp1_pix2pix_results/config.resolved.yaml` has no `use_roi` key and
`augment.hflip: True`. ROI cropping landed in commit `4d98c6c`, after that run.
`exp2_paper_results/` has `use_roi: True, hflip: False` and is on the current pipeline.

So "the gap between exp8 and exp1 is what the pairing bought"
(`model/README.md:120-125`) does not currently hold: that comparison varies pairing, ROI
cropping, and hflip together. Either re-run exp1 on the current pipeline, or state the
gap against exp2 with the extra `lambda_nce: 1` noted as the known difference.

Worth doing before anything else in this list, because it is the only item that changes
what you can *claim*, and it needs no new code.

---

## Suggested order

| step | change | cost | unblocks |
|---|---|---|---|
| 1 | re-run exp1 on the current pipeline | 1 run, no code | the headline claim |
| 2 | brain-only exp8, λ_cycle 5 / λ_idt 0.1, no `color` aug | 1 run, config only | §1, §2, §3, §5 at once |
| 3 | read the epoch-10 panel | free | did §1 work? |
| 4 | repeat step 2 for abdomen and MSK | 2 runs | a per-region answer |
| 5 | ResNet generator, if you want CycleGAN to have a fair shot | code | §4 |

Step 2 is one config file and tests four of the five findings simultaneously. If the
epoch-10 panel is still a copy of the input, finding 1 was wrong and it is worth
re-reading doc 02 before spending more GPU time.

## What "success" would look like

Not a low MAE. exp8 is still expected to lose to exp1 — `model/README.md:127` is right
about that, and doc 03 does not change it.

Success is exp8 losing **for the stated reason**: an unpaired model that produces
anatomically plausible CT of the right region at the right window, and is beaten by the
paired model on fidelity to *this patient*. That is a measurement of what the QC work
bought. What you have now is a model that never got far enough to be measured.
