# 02 — What to change

Short, because the fix is the same as exp4's. Recommendations only; nothing applied.

---

## 1. Move to λ_L1 48 / λ_NCE 6

The full argument is in
[exp4_nce_max/04_what_to_change.md](../exp4_nce_max/04_what_to_change.md) §2. In one line:
the 8:1 run reaches `dice_bone` 0.385 against this run's 0.038, at a cost of nothing —
it is also the most stable run in the ladder.

Going the other way from here (toward exp4's 2:1) buys 0.009 of Dice and loses 0.05 of
`mae_norm`. There is no reason to.

## 2. Retire this setting, but keep the result

Both this run and exp4 land below the cliff, and together they establish where it is. That
is the useful output. Two more runs at **λ_L1 30 and 40**, λ_NCE 5, would pin it precisely —
and that is a genuinely interesting number, because the failure it marks is:

- **sharp** (a factor of ten in Dice across one halving of λ_L1),
- **invisible** in the headline metric (`mae_norm` moves 1.3×, Dice moves 10×),
- and **inverted** in two common metrics — SSIM prefers the failed model, and PatchNCE, if
  you were using it to select anything, would too.

That is worth writing up regardless of which λ you ship.

## 3. Stop using SSIM as a quality summary for this task

From 01 §3: SSIM ranks this run — which produced no bone — above the run with the best
bone reconstruction in the ladder. That is not a marginal disagreement; it is the opposite
ordering on the thing that matters.

Concretely:

- **Keep** SSIM. It is a reasonable measure of structural agreement and it is what people
  expect to see in a table.
- **Never** report it without `dice_bone` and `mae_hu` beside it. On this ladder, SSIM
  disagrees with `dice_bone` about which of two models is better, and `dice_bone` is right.
- **Do not** use it, or anything else built on locally-normalised correlations, to select
  a checkpoint. `eval.selection_metric` is `mae_norm`, which is the better choice — though
  [`_comparison`](../_comparison/README.md) documents a separate problem with that one too.

The general principle, and it is the transferable lesson from this run: **a
contrast-invariant metric cannot audit a contrast-destroying failure.** Check that at
least one metric in your table has an opinion about absolute values. Here that is
`mae_hu` and `dice_bone`, and nothing else.

## 4. Log the output's dynamic range

Same recommendation as [exp4 §4](../exp4_nce_max/04_what_to_change.md). One line in the
stats dict:

```python
stats["G_p99"] = fake_B.detach().float().quantile(0.99)
```

For this run it would have shown, at epoch 5, that the model's brightest pixels were
nowhere near the 0.70 threshold — and it would have kept showing it for the next ninety-five
epochs while `train/G_NCE` fell smoothly and `val/ssim` improved.
