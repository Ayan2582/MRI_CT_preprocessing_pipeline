# 03 — What to change

This is the run to build on. Recommendations only; nothing applied.

---

## 1. Fix the name first

`run.name` in `config.resolved.yaml` says `exp3b_nce_4_pix32`; the weights are
`lambda_l1: 48, lambda_nce: 6`; the folder says `exp3b_nce_6_pix48`. Every sample panel is
titled with the wrong name.

Whichever way you resolve it, resolve it before this run appears in a writeup. A
recommended configuration that three sources disagree about the identity of is not a
recommendation anyone can act on — including you, in six months.

## 2. Keep the λs. Attack hallucination somewhere else

The λs are as close to right as this ladder can demonstrate: 8:1 is above the collapse
cliff and below the memorisation regime ([01](01_why_it_worked.md)). Moving them will make
one of the two failures worse.

The bone speckle in [02 §1](02_what_still_failed.md) is a texture-prior problem, so attack
it where it lives:

**Measure it before trying to fix it.** Add a false-positive bone rate —
`|pred_bone \ target_bone| / |target_bone|` — next to `bone_dice` in
`evaluation/metrics.py:212-236`. Right now `dice_bone` of 0.401 *rewards* the spraying, so
you have no number that would tell you whether a fix worked. Two lines, no retraining.

**Then try reducing λ_GAN.** The adversarial term is what supplies the invented texture. It
is at 1.0 and has never been varied in this ladder except by being switched off entirely
(`exp0`). A run at `λ_GAN: 0.5` with everything else fixed would show directly how much of
the speckle is adversarial and how much is the generator's own uncertainty. That is a
one-line, one-run experiment answering a question the ladder never asked.

**Consider a bone-band weighted L1.** Weight the L1 term by target intensity so bone pixels
are not drowned by the soft-tissue majority. This attacks the conditional-median argument in
[exp4/03 §2](../exp4_nce_max/03_the_mechanism.md) directly, and it should sharpen bone
placement rather than merely suppressing texture.

## 3. Take the cheap generalisation wins

From [exp2/03 §1](../exp2_paper/03_what_to_change.md), and they apply here even though this
run barely drifts:

- **`ngf: 32`** — quarters the generator from 54.4 M to ~14 M parameters. On 32 subjects
  this is the highest-value untried experiment in the ladder, it is one config line, and
  01 §2's argument says a smaller model should hold up *better* under this λ ratio, not
  worse.
- **`augment.hflip: true`** — anatomically legitimate for brain, spine and most
  musculoskeletal slices, roughly doubles the effective dataset, costs nothing.

Both are cheaper than any further λ search, and neither has been tried on the current
pipeline.

## 4. Split the reporting by region

[02 §4](02_what_still_failed.md) is the honest read: 10.3 HU on brain, 52.1 on abdomen,
90.3 on spine, 119.1 on bone. A single ladder-wide `mae_norm` averages a nearly-usable
brain model together with an underpowered spine one, and the per-region HU windows make
the normalised average actively misleading about which is which.

Report `mae_hu` per region as the primary table. Keep `mae_norm` for checkpoint selection,
where it is the right tool.

## 5. Then run the comparisons the ladder still owes

Two runs are missing before any of these numbers can carry the claims the ladder wants to
make:

- **exp1 on the current pipeline.** As it stands `exp1_pix2pix_results` is on a pre-ROI,
  `hflip: true` pipeline and is not comparable to anything here — see
  [exp1_pix2pix](../exp1_pix2pix/). Until it is re-run, exp3b_6_48 is the best *comparable*
  result and exp1's apparent lead is unverified.
- **exp7 and exp8 with their objectives repaired** ([exp7](../exp7_reggan/04_what_to_change.md)
  §2, [exp8](../exp8_cyclegan/)). Both currently answer no question at all, so the
  registration and pairing questions the ladder was built to settle remain open.

---

## What a successful next run looks like

| signal | target | current |
|---|---|---|
| `val/mae_hu/brain` | under 10 HU | 10.3 |
| `val/mae_band_bone` | under 100 HU | 119.1 |
| false-positive bone rate (new) | falling while Dice holds | unmeasured |
| drift best → final | under 2% | **0.9% — already met** |
| `is_best` | past epoch 50 | **121 — already met** |

Two of the five are already met, which is the point of this run. The remaining three are
about bone placement and absolute accuracy, and none of them is a λ problem.
