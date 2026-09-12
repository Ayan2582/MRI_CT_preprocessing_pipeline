# exp8_cyclegan

**Verdict: the best epoch was 7, and across all 200 epochs `val/mae_norm` moved by 3.4%
in total. The model never learned anything.**

This folder is the condensed, first-principles version plus the confirmation from the real
`metrics.csv`. The **detailed diagnosis is the five documents at the top level of
`review_test_Folder/`** — [01](../01_reading_the_panels.md),
[02](../02_the_lambda_arithmetic.md), [03](../03_the_undefined_mapping.md),
[04](../04_the_split_is_lopsided.md), [05](../05_the_background_artifacts.md) — written
from the sample panels before the metrics were available.

The metrics have now arrived. They confirm the central diagnosis and break one stated
prediction. Both are recorded in [02](02_what_the_metrics_confirmed.md).

---

## The numbers

| | value |
|---|---|
| best `val/mae_norm` | **0.2208 at epoch 7** |
| range across all 200 epochs | **0.2208 – 0.2283** — a 3.4% spread, total |
| `val/dice_bone` | 0.059 at best; peaks at **0.080**; never higher |
| `val/mae_hu/spine` | **195.2 HU** |
| `val/mae_band_bone` | **209.9 HU** |
| `train/G_cycle_A` | 0.170 → 0.036 — **4.7× better** |
| `train/G_idt_A` | 0.148 → 0.012 — **12× better** |

The training objective improved by up to 12×. The validation metric moved 3.4%, and its
best value was at epoch 7 — the epoch at which, per
[01](../01_reading_the_panels.md), the generator was still visibly copying the MRI through
unchanged.

**The best thing this model ever did was nothing.**

---

## Reading order

| doc | question it answers |
|---|---|
| [01_the_idea.md](01_the_idea.md) | Why unpaired translation is under-determined, what cycle consistency fixes, and what it provably does not |
| [02_what_the_metrics_confirmed.md](02_what_the_metrics_confirmed.md) | Which predictions from the panel-only analysis held, and which one was wrong |
| [03_the_verdict.md](03_the_verdict.md) | What exp8 was supposed to measure, what it actually measured, and what to do |

For the mechanism in full — the 15:1 λ arithmetic, the per-region HU windows the generator
is never told about, the anatomically lopsided seed-1337 split, and the unconstrained
background — read the five top-level documents. This folder does not repeat them.

---

## The one-paragraph version

CycleGAN replaces paired supervision with a round-trip constraint: `F(G(x)) ≈ x`. That
rules out the degenerate solutions an adversarial loss alone permits, but it is satisfied
*perfectly* by `G = F = identity` — and exp8 weighted the round-trip terms at 15 against
the adversarial term's 1, making the identity map an exact global minimum of 94% of the
objective. So the run spent its early epochs descending into the identity, which is where
its best validation score came from, and its later epochs escaping into a *wrong* CT rather
than a right one — because the generator is never told which of four per-region HU windows
it should be targeting, and because the CT pool it is asked to match is 42% brain while its
input stream is only 23% brain.
