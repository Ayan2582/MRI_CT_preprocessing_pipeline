# exp4_nce_max

**Verdict: the anatomy is right and the intensities are gone. The model draws a correct
CT in the wrong greys, with no bone anywhere.**

`λ_L1: 10`, `λ_NCE: 5` — the most NCE-dominant setting in the ladder, and the answer to
the question `model/README.md:74` asks of it ("where does hallucination start?") turns
out to be: *not here*. Lowering λ_L1 does not cause hallucination. It causes the opposite
— the model stops committing to any extreme value at all.

This folder is the main lecture on PatchNCE. [exp3b_nce_20_5](../exp3b_nce_20_5/) is the
same failure one notch weaker and refers back here.

---

## The numbers

| | value |
|---|---|
| best `val/mae_norm` | 0.1666 at epoch **199** — still improving when the run ended |
| `val/dice_bone` | **0.0290** (and exactly **0.0000** for epochs 13–48) |
| `val/ssim` | 0.296 — the lowest of the five paired runs |
| final `train/G_L1` | **0.322** — barely moved from 0.479 at epoch 0 |
| `val/mae_band_bone` | 200+ HU |

Compare `exp3b_nce_6_pix48` (`λ_L1: 48`), which reaches `dice_bone` **0.385** and drives
`train/G_L1` down to 0.097. Same architecture, same data, same 200 epochs. The only
difference is how hard L1 pulls.

**Dice of 0.029 means the model produced almost no pixel anywhere above 150 HU.** Not
"placed the bone badly" — did not produce bone.

---

## Reading order

| doc | question it answers |
|---|---|
| [01_the_idea.md](01_the_idea.md) | What is PatchNCE, derived from scratch, and what can it possibly constrain? |
| [02_what_happened.md](02_what_happened.md) | The evidence — no explanation yet |
| [03_the_mechanism.md](03_the_mechanism.md) | Two proofs: why NCE cannot see intensity, and why weak L1 cannot reach bone |
| [04_what_to_change.md](04_what_to_change.md) | Where the cliff is, and which λ to use |

---

## The one-paragraph version

PatchNCE is built entirely from **cosine similarities between L2-normalised feature
vectors**, fed to a softmax that asks a ranking question: "does output location *i* match
input location *i* better than it matches location *j*?" A ranking question has no opinion
about absolute values — a CT with bone rendered black satisfies it exactly as well as one
with bone rendered white. So the *only* term in the objective that knows bone should be
bright is L1. Meanwhile L1's minimiser is the conditional **median**, which discards
minority modes outright, and bone is a small minority of pixels. At `λ_L1: 100` L1 is
strong enough to force the commitment anyway. At `λ_L1: 10` it is not, so the model
settles on soft-tissue grey everywhere — which is a genuinely good solution to the loss as
written, and scores `dice_bone = 0.029`.
