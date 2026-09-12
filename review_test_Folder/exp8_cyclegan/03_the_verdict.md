# 03 — What exp8 measured

---

## 1. What it was supposed to measure

`model/README.md:120-125` is explicit, and the reasoning is sound:

> **exp8 is the only experiment that measures the QC effort itself.** Every other run is
> handed 2161 hand-checked correspondences — 129 pairs rejected, 738 slices nudged
> individually, artifacts erased on 560. exp8 throws that away […] The gap between exp8 and
> exp1 is what the pairing bought.

That is a real question and a well-posed experiment. The design is careful: the split is by
patient rather than by row, so no patient contributes both modalities; validation stays
paired so the numbers remain comparable; the same generator architecture is reused so
capacity is not a confound.

**exp8 was expected to lose.** `model/README.md:127` says so — "Report the size of the gap,
not the winner." Losing is not the failure.

## 2. What it actually measured

Nothing.

For the gap to mean "what the pairing bought", exp8 has to be a *competent unpaired model*.
A competent unpaired model that scores worse than a paired one tells you the pairing is
worth the difference. A model that never left its initialisation tells you only that this
particular configuration did not train.

exp8 did not train:

- Best epoch **7** of 200.
- `val/mae_norm` spread over the entire run: **0.2208 to 0.2283**. 3.4%.
- At its best epoch the generator was, per [01](../01_reading_the_panels.md), still visibly
  passing the MRI through.

And there are four independent reasons for that, none of which is "the data was unpaired":

| # | cause | where |
|---|---|---|
| 1 | λ ratio 15:1 makes the identity map an exact minimum of 94% of the objective | [02_the_lambda_arithmetic.md](../02_the_lambda_arithmetic.md) |
| 2 | the generator is never told which of four HU windows to target — **the target relation is not a function** | [03_the_undefined_mapping.md](../03_the_undefined_mapping.md) |
| 3 | the seed-1337 split makes the CT pool 42% brain against a 23%-brain input stream; spine gets **9** CT slices | [04_the_split_is_lopsided.md](../04_the_split_is_lopsided.md) |
| 4 | cycle and identity contribute zero gradient in the padding, so `D` alone governs it | [05_the_background_artifacts.md](../05_the_background_artifacts.md) |

Cause 2 is fatal at any λ. Causes 1, 3 and 4 are configuration.

**So the exp8-vs-exp1 gap currently measures: no pairing, plus a broken λ ratio, plus an
unspecifiable target, plus an anatomically mismatched split — against a baseline that was
itself trained on a different preprocessing pipeline** (see
[exp1_pix2pix](../exp1_pix2pix/)). That is at least five variables, and the ladder wanted
one.

## 3. What to change

Ranked. Full detail in [06_what_to_change.md](../06_what_to_change.md); this is the summary.

**1. Train per region.** Fixes cause 2, which nothing else fixes. A brain-only exp8 has one
HU window by construction and a well-posed question attached — *what is the pairing worth
for brain?* It also gets region-matched sampling for free, which addresses cause 3. Three
runs instead of one, and you get three interpretable answers instead of one
uninterpretable average.

**2. Break the 15:1 ratio.** `lambda_cycle: 5`, `lambda_identity: 0.1` (effective 0.5).
That moves the balance from 1 : 10 : 5 to 1 : 5 : 0.5. Config only. The test is immediate:
if the epoch-5 panel is no longer a copy of the input, it worked.

**3. Drop spine from exp8 entirely.** 27 training slices, 9 on the CT side after the split.
No seed fixes that. Reporting a spine number from this run is reporting noise.

**4. Remove `color` from the DiffAugment policy** for any run with `λ_L1: 0`. It blinds `D`
to global brightness and contrast, which is harmless when L1 pins intensity and harmful
when `D` is the only cross-domain signal.

**5. Re-run exp1 on the current pipeline** before quoting any gap at all. One run, no code
changes, and it unblocks three separate claims — see
[exp1_pix2pix/02](../exp1_pix2pix/02_what_to_change.md) §1.

## 4. What success would look like

Not a low `mae_norm`. exp8 should still lose to a paired model — `model/README.md:127` is
right about that and nothing here changes it.

Success is exp8 losing **for the stated reason**: an unpaired model that produces
anatomically plausible CT, of the right region, at the right window, and is beaten by the
paired model on fidelity to *this patient*. That difference is a measurement of what the QC
work bought.

| signal | target | current |
|---|---|---|
| `is_best` | past epoch 50 | **7** |
| `val/mae_norm` range across the run | a clear descent, not a plateau | **3.4% total** |
| `val/dice_bone` | above 0.2 — evidence bone exists at all | **0.080 peak** |
| epoch-5 panel | not a copy of the input | is a copy |
| the gap to exp1 | one variable | at least five |

## 5. The part worth keeping

The experiment design is good and should survive the rewrite. Three decisions in particular
are more careful than most published unpaired-translation setups:

- **Splitting by patient, not by row** (`data/dataset.py:309-315`). Row shuffling is easier
  and keeps twice the data, but lets the model see both modalities of the same patient —
  which would make "unpaired" a claim the experiment does not support. The docstring says
  so, and accepts the cost.
- **Keeping validation paired.** An unpaired validation set would report meaningless
  numbers while drawing a convincing curve.
- **Refusing to start with a conditional discriminator** (`training/cyclegan.py:63-83`),
  because the MRI and CT come from different patients and no correspondence exists to judge.

All three are right. The failure is in the weights, the region handling and the split
composition — not in the experiment's design. It is worth re-running properly.
