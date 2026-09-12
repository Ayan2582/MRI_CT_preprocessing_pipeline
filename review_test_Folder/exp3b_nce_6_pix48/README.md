# exp3b_nce_6_pix48

**Verdict: the best run in the ladder, and it still hallucinates bone. Also, it is not
called what its config says it is called.**

`λ_L1: 48`, `λ_NCE: 6` — an 8:1 ratio, just above the cliff documented in
[exp4_nce_max](../exp4_nce_max/). It produces the best bone reconstruction of any run here
and is the **only run in the ladder that did not degrade after its peak**.

This folder is about why that happened, and about the two things still wrong with it.

---

## Read this before anything else: the run is mislabelled

| source | says |
|---|---|
| folder name | `exp3b_nce_6_pix48` |
| `config.resolved.yaml` → `run.name` | **`exp3b_nce_4_pix32`** |
| sample panel titles | **`exp3b_nce_4_pix32`** |
| `config.resolved.yaml` → `loss` | **`lambda_l1: 48, lambda_nce: 6`** |

Only the folder name matches the weights that were actually used. The run name and every
panel title claim a different configuration (`nce 4 / pix 32`) that was not trained.

**The λs in `config.resolved.yaml` are authoritative** — they are what the trainer read.
This matters because this is the run being recommended, and a recommendation attached to
the wrong name is worse than no recommendation. Fix the `run.name` before this appears in
a writeup.

---

## The numbers

| | exp3b_nce_6_pix48 | exp2_paper | exp3b_nce_20_5 |
|---|---|---|---|
| λ_L1 : λ_NCE | **48 : 6** | 100 : 1 | 20 : 5 |
| best `val/mae_norm` | 0.1203 @ ep **121** | **0.1166** @ ep 18 | 0.1585 @ ep 199 |
| drift best → final | **+0.9%** | +8.4% | 0.0% |
| `val/dice_bone` | **0.401** | 0.391 | 0.038 |
| final `train/G_L1` | 0.097 | 0.084 | 0.297 |
| `val/mae_band_bone` | 119.1 HU | 117.5 HU | — |

It gives up 3% of peak accuracy against exp2 and buys the best bone in the set plus a run
you can train to completion.

---

## Reading order

| doc | question it answers |
|---|---|
| [01_why_it_worked.md](01_why_it_worked.md) | Why 8:1 is stable when 100:1 overfits and 4:1 collapses |
| [02_what_still_failed.md](02_what_still_failed.md) | Bone speckle, spine, and the honest HU numbers |
| [03_what_to_change.md](03_what_to_change.md) | Where to go from here |

Both [exp2_paper/01](../exp2_paper/01_the_idea.md) (why an adversary hallucinates, why
32 subjects overfit) and [exp4_nce_max/01](../exp4_nce_max/01_the_idea.md) (why PatchNCE
cannot see intensity) are assumed. This folder is the synthesis of those two.
