# 04 — What to change

Recommendations only; nothing here has been applied.

---

## 1. Do not run this setting again

`λ_L1: 10, λ_NCE: 5` is below the cliff. There is no schedule, no extra epoch count and no
stabiliser that recovers bone from it, because the problem is not optimisation — the run
was still improving monotonically at epoch 199 (02 §6). The problem is that the objective's
optimum does not contain bone.

exp4 was framed by `model/README.md:74` as "where does hallucination start?". It answered a
different question, and the answer is worth recording: **weakening λ_L1 does not produce
hallucination, it produces intensity collapse.** Hallucination shows up at the
*intermediate* settings instead — see [exp2](../exp2_paper/) and
[exp3b_nce_6_pix48](../exp3b_nce_6_pix48/), both of which scatter bone-bright speckle
through soft tissue while scoring the *best* Dice in the ladder.

That inversion is the real result of exp4, and it is more useful than the number it
produced.

## 2. Use λ_L1 48 / λ_NCE 6

The evidence is in [`_comparison`](../_comparison/README.md), but the short version:

| run | λ_L1 : λ_NCE | best `mae_norm` | `dice_bone` | drift from best to final |
|---|---|---|---|---|
| exp2_paper | 100 : 1 | **0.1166** | 0.369 | **+8.4%** (peaks at epoch 18) |
| exp3b_nce_6_pix48 | 48 : 6 | 0.1203 | **0.385** | **+0.9%** (peaks at epoch 121) |
| exp3b_nce_20_5 | 20 : 5 | 0.1585 | 0.038 | 0.0% |
| exp4_nce_max | 10 : 5 | 0.1666 | 0.029 | 0.0% |

`48 : 6` costs 3% of peak accuracy against exp2 and buys the best bone Dice in the set plus
the only run in the entire ladder that did not degrade after its peak. That is the setting
to build on.

If you want to probe the cliff more precisely, the interesting interval is **λ_L1 between
20 and 48** at λ_NCE 5–6. Two runs at 30 and 40 would locate it. That is a genuinely
publishable ablation — the boundary is sharp, it is measured in a metric designed to catch
exactly this, and it is specific to a task where the minority class is the clinically
important one.

## 3. Report `dice_bone` and `mae_hu` next to `mae_norm`, always

02 §5 is the lesson: `mae_norm` degraded 1.43× and `dice_bone` degraded 12.7×. Reading the
first alone files exp4 as "somewhat worse" when it is in fact unusable.

This is not a criticism of `mae_norm` — `evaluation/metrics.py:128-134` defends it as the
one scalar comparable across regions, which is right, and it is the correct choice for
`eval.selection_metric`. It is a criticism of quoting it alone. Any table in the writeup
that has a `mae_norm` column should have a `dice_bone` column beside it.

See [`_comparison`](../_comparison/README.md) for a second, sharper problem with
`mae_norm`: across all seven runs it ranks the regions in the *opposite* order from
`mae_hu`.

## 4. Two cheap diagnostics worth adding

**Log the output's dynamic range.** One line, and it would have caught this at epoch 10:

```python
stats["G_p99"] = fake_B.detach().float().quantile(0.99)
```

If the 99th percentile of the prediction never approaches the 99th percentile of the
target, the model is hedging, and you know it long before 200 epochs have run. For exp4
this number would have sat far below the 0.70 threshold from the start.

**Check the tap-0 hypothesis** from 03 §4 before trusting it. Run one short training with
`loss.nce.layers: [1, 2, 3, 4]` — dropping the raw-pixel tap — at the exp3b_6_48 weights.
If `dice_bone` improves and nothing else moves, tap 0 was costing you bone. If nothing
changes, the hypothesis is dead and 03 §4 should be struck. Either outcome is worth one
run, because tap 0 is currently contributing a quarter of nothing to the averaged term
regardless (`train/G_NCE_L0` sits at chance in all four runs).

## 5. If you specifically want NCE to carry more weight

The trade in 01 §5 is unavoidable *as long as L1 is the only intensity term*. If you want
to lean on NCE without losing bone, add intensity supervision back through a different
route rather than through λ_L1:

- **A histogram or quantile-matching loss** — penalise the difference between the
  prediction's intensity distribution and the target's. Cheap, and it directly attacks the
  hedging behaviour without reintroducing L1's blur.
- **A bone-band weighted L1** — weight the L1 term by target intensity so the small
  minority of bone pixels is not drowned by the soft-tissue majority. This attacks the
  median argument in 03 §2 head-on: it changes the conditional distribution's effective
  mass, which is the actual reason bone is unreachable.

The second is the more targeted fix and follows directly from the derivation. Neither has
been tried in this ladder.

---

## What a successful re-run looks like

| signal | target |
|---|---|
| `val/dice_bone` | above 0.35 — the ladder's demonstrated ceiling is 0.385 |
| `train/G_L1` (final) | below ~0.10 — evidence the term was actually driven, not just present |
| `G_p99` (if added) | tracks the target's 99th percentile within a few percent |
| the panel | white bone in the brain, MSK and spine slices, not grey |
| `is_best` | past epoch 50, and drift from best to final under ~2% |

And the honest framing for the writeup: exp4 is not a failed run to be hidden. It is the
run that establishes the boundary — it shows what happens when the only term carrying
Hounsfield units is turned down far enough, and it shows that the damage is invisible in
the headline metric.
