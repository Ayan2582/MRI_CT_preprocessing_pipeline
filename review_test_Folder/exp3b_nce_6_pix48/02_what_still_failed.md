# 02 — What still failed

Being the best run in the ladder is not the same as being a good one. Four things are still
wrong, and the last is the one that matters clinically.

---

## 1. It hallucinates bone, the same way exp2 does

`samples/epoch_0119.png` — near the best epoch.

**Brain axial: genuinely excellent.** Panel MAE 0.021. Bright continuous skull, correct
parenchymal grey, ventricles in the right place at the right density. If the whole panel
looked like this the model would be usable.

**Everything else has speckle.**

- **Abdomen coronal**: scattered bone-bright blobs through the gluteal and abdominal wall
  musculature, where the real CT is smooth soft tissue containing no bone at all.
- **Brain coronal**: the skull base and sinuses are a chaotic field of bright and dark
  granules against structured bone and air in the real CT.
- **Musculoskeletal**: the real CT has a clean bright cortical ring around a darker marrow
  cavity. The synth produces a large diffuse white mass covering much of the compartment —
  bone-density pixels spread across tissue that is not bone.
- **Spine**: bright granular texture filling the vertebral envelope, with disc spaces
  partly filled in.

This is the mechanism from [exp2/01 §2](../exp2_paper/01_the_idea.md), unchanged: the
adversarial term rewards images that are *distributionally* CT-like, and real CTs contain
bright granular structure. Where L1 is uncertain, `D`'s texture prior fills in. Reducing
λ_L1 from 100 to 48 did not help here — if anything it gave the prior slightly more room.

**The high `dice_bone` is partly this.** As derived in
[exp2/02 §4](../exp2_paper/02_what_happened.md), Dice's numerator gains twice what its
denominator does when you over-produce bone, so spraying raises the score. exp3b_6_48's
0.401 is the best in the ladder and is *not* evidence that its bone is correctly placed.

The number that sees it is `val/mae_band_bone` — error restricted to pixels that are bone
in the **target** — and it is **119.1 HU** at the best epoch. That is the honest bone
figure.

## 2. SSIM and MAE disagree about which epoch is best

| epoch | `val/mae_norm` | `val/ssim` |
|---|---|---|
| 121 (best MAE) | **0.1203** | 0.316 |
| 199 | 0.1214 | **0.355** |

SSIM improves by 12% over the stretch in which MAE gets slightly worse, and keeps rising to
the last epoch. Pick a checkpoint by SSIM and you get epoch 199; by MAE, epoch 121.

The same blind spot as in [exp3b_nce_20_5/01 §3](../exp3b_nce_20_5/01_what_happened.md):
SSIM's structural term is a *correlation of local variances*, normalised by those variances,
so it rewards structural agreement while being substantially insensitive to contrast scale.
Here the disagreement is small and either checkpoint is defensible. It is worth noting only
because in exp3b_nce_20_5 the same property makes SSIM rank a bone-free model above this
one.

## 3. Spine is still the worst region, by 2.2×

`val/mae_norm` at epoch 121:

| brain | abdomen | musculoskeletal | spine |
|---|---|---|---|
| 0.1284 | 0.1303 | 0.0834 | **0.1806** |

Better than exp2's 0.2045, but still the worst region — as it is in **all seven runs**
([`_comparison`](../_comparison/README.md)). Spine is 27 of 1687 training slices. No λ
setting in this ladder moved that, and none will.

## 4. The honest numbers are in Hounsfield units, and they are not good

This is the part to sit with.

`val/mae_norm` of 0.1203 sounds like a small error. Convert it using each region's own
window (`Preprocessing/pipeline_config.py:208-229`):

| region | `mae_norm` | `mae_hu` |
|---|---|---|
| brain | 0.128 | **10.3 HU** |
| musculoskeletal | 0.083 | **41.7 HU** |
| abdomen | 0.130 | **52.1 HU** |
| spine | 0.181 | **90.3 HU** |

And restricted to bone pixels: **119.1 HU**.

Two things follow.

**The ranking flips.** By `mae_norm`, musculoskeletal is the best region and brain is
third. In HU, brain is four times more accurate than musculoskeletal. This inversion holds
in all seven runs and is documented in [`_comparison`](../_comparison/README.md); it is a
property of dividing by a per-region window that varies 6.25× (brain 80 HU, MSK/spine
500 HU).

**Only brain is close to usable.** For synthetic-CT dose calculation the commonly cited
tolerance is on the order of tens of HU in soft tissue. 10.3 HU in brain is in that
territory. 52 HU in abdomen and 90 HU in spine are not, and 119 HU on bone — the tissue
whose density matters most for attenuation — is a long way off.

So the correct summary of the best run in this ladder is: **a promising brain model, and a
demonstration that the abdomen, spine and musculoskeletal cases need more data.** Not a
finished result.

---

## Summary

| problem | severity | fixable by λ? |
|---|---|---|
| bone speckle in soft tissue | real, and hidden by `dice_bone` | no — see [03](03_what_to_change.md) §2 |
| SSIM/MAE checkpoint disagreement | minor here | no, it is a metric property |
| spine worst by 2.2× | structural | no — 27 training slices |
| 119 HU bone error, 90 HU spine | the headline limitation | no |

Nothing in this list is a λ problem. The λs are, as far as this ladder can show, correct.
