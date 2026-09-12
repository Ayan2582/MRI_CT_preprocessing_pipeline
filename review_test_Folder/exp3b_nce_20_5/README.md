# exp3b_nce_20_5

**Verdict: the same failure as [exp4_nce_max](../exp4_nce_max/), one notch weaker — and it
exposes a second misleading metric on the way.**

`λ_L1: 20`, `λ_NCE: 5` — a 4:1 ratio, which is already below the cliff. `dice_bone` is
**exactly 0.0000 for epochs 6 through 102**, then crawls to 0.038.

---

## The numbers

| | exp3b_nce_20_5 | exp4_nce_max | exp3b_nce_6_pix48 |
|---|---|---|---|
| λ_L1 : λ_NCE | 20 : 5 = **4:1** | 10 : 5 = **2:1** | 48 : 6 = **8:1** |
| best `val/mae_norm` | 0.1585 @ ep199 | 0.1666 @ ep199 | 0.1203 @ ep121 |
| `val/dice_bone` | **0.038** | 0.029 | **0.385** |
| final `train/G_L1` | 0.297 | 0.322 | 0.097 |
| `val/ssim` | **0.367** | 0.296 | **0.355** |

Read the last row twice.

---

## Reading order

| doc | |
|---|---|
| [01_what_happened.md](01_what_happened.md) | the evidence, and the SSIM problem it exposes |
| [02_what_to_change.md](02_what_to_change.md) | the short version — the fix is exp4's fix |

The mechanism is not re-derived here. **Read
[exp4_nce_max/01_the_idea.md](../exp4_nce_max/01_the_idea.md) and
[03_the_mechanism.md](../exp4_nce_max/03_the_mechanism.md) first** — they contain the
derivation of why a contrastive loss cannot express a preference about intensity, and why
bone is the pixel that breaks first. This folder assumes both.

---

## What this run adds that exp4 does not

Two things.

**It locates the cliff.** exp4 at 2:1 could be dismissed as an extreme setting nobody
would ship. exp3b_20_5 at 4:1 is a perfectly reasonable-looking configuration, and it
fails just as completely — `dice_bone` 0.038 against 0.385 for the 8:1 run. So the boundary
is not somewhere out in the tail; it sits between 4:1 and 8:1, which is inside the range a
person would actually try.

**It breaks SSIM.** exp3b_20_5 scores `ssim` **0.367**; exp3b_nce_6_pix48 scores **0.355**.
So by SSIM, the run that produced essentially **no bone at all** ranks *above* the run with
the best bone reconstruction in the entire ladder. 01 §3 explains why, and it is a property
of SSIM you should expect to keep meeting.
