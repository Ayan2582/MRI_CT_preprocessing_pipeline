# exp2_paper

**Verdict: peaked at epoch 18 and spent the next 181 epochs getting worse. The bone it
produces is bright, plentiful, and partly invented.**

`λ_GAN: 1`, `λ_L1: 100`, `λ_NCE: 1` — the textbook configuration, and on the current
pipeline the best-scoring run in the ladder that is honestly comparable to the others.

It did not fail catastrophically. It failed in the two ordinary ways that are easiest to
miss: it overfit, and it hallucinated.

---

## The numbers

| | value |
|---|---|
| best `val/mae_norm` | **0.1166 at epoch 18** |
| final `val/mae_norm` | 0.1264 — **8.4% worse**, the largest drift in the ladder |
| `val/dice_bone` | 0.369 (reached 0.39 by epoch 20 and never improved) |
| `val/ssim` | peaks 0.435 at epoch 20 → 0.386 at 199 |
| `train/G_L1` | 0.358 → **0.084**, a 4.3× reduction |
| `val/mae_norm/spine` | **0.2045** — 2.7× its own best region |

**The training loss improved 4.3× while the validation metric got worse.** That is the
textbook definition of overfitting, and 01 explains why it was close to inevitable here.

---

## Reading order

| doc | question it answers |
|---|---|
| [01_the_idea.md](01_the_idea.md) | What pix2pix's three terms each buy — and why an adversary invents texture |
| [02_what_happened.md](02_what_happened.md) | The evidence, and the mechanism behind each half of the failure |
| [03_what_to_change.md](03_what_to_change.md) | The fix, which is mostly not a λ change |

---

## The one-paragraph version

The generator has ~54 million parameters. The training set has 1687 slices — but they come
from **32 patients**, and slices 1 mm apart are nearly the same image, so the *effective*
number of independent examples is closer to 32 than to 1687. Under that ratio, a model that
keeps improving its training loss for 200 epochs is memorising, and epoch 18 is where
memorising overtook generalising. Separately, the adversarial term does not ask "is this
the right CT" — it asks "does this look like a CT", and a fast way to look like a CT is to
scatter bone-bright speckle around, because real CTs contain bright speckle. `dice_bone`
of 0.369 counts those invented pixels as successes wherever they happen to land on real
bone. The spine, with 27 training slices, gets the worst of both.
