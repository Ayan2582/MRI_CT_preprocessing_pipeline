# 01 — Reading the three panels

Before any explanation: what is actually on the page. Each epoch is evidence of a
different thing, and the sequence matters more than any single frame.

First, what you are looking at. `training/trainer.py:362-407` renders the panel every
`logging.sample_every: 5` epochs. Column 3 is `net(real_A)` where `net =
generator_for_eval()` — for CycleGAN that is the **EMA shadow of `netG_A2B`**
(`training/cyclegan.py:152-163`, `eval.use_ema: true`), i.e. MRI→CT. `netG_B2A`, `D_A`
and `D_B` never appear. The slices are a fixed set drawn from the **paired validation
set** (`trainer.py:335-359`), two per body region — so the "real CT" column is a
genuine correspondence even though training saw none.

---

## Epoch 9 — the generator is the identity function

The synth CT column is the MRI. Not "MRI-like": the brain axial still shows T2
cerebrospinal fluid bright in the ventricles, which in a real CT is dark. The
musculoskeletal slices still show MRI fat/muscle contrast. Bone — the one thing CT is
*for* — is dark, exactly as it is in the MRI, and exactly opposite to the real CT
beside it.

Compare `exp1_pix2pix` at the same epoch 9: already a bright skull, already a grey
brain, already recognisable as CT. exp1 had the answer keyed in by `lambda_l1: 100`.
exp8 has `lambda_l1: 0` and after 9 epochs has learned to do nothing.

**This is not slow convergence. It is a stable solution to the loss as written**, and
02 does the arithmetic.

## Epoch 109 — it escaped, into the wrong answer

Now it looks like CT: bone is bright, soft tissue is grey. Three things went wrong at
the same time.

**The tissue classes are wrong.** Look at the musculoskeletal pair. The real CT shows a
bright cortical bone ring around a darker marrow cavity, surrounded by mid-grey muscle
with a dark fat plane. The synth CT paints most of the muscle compartment bone-white.
It has learned "CT means large saturated bright regions" rather than "muscle is ~50 HU."
04 shows where that prior came from — it is measurable, not a guess.

**A blob appeared in the corner.** Top-left, roughly the same place, in the brain
coronal, the brain axial, and both musculoskeletal slices — four different anatomies,
four different patients, one artifact in one position. Something input-independent is
being stamped onto the output. 05 explains what and why.

**The MAE went up.** Brain coronal 0.171 → 0.189, both MSK slices up. The panel got
*more* CT-looking and *less* correct simultaneously. This is the failure mode
`exp8_cyclegan.yaml` warned about in its own header comment — "a plausible CT that is
not this patient's" — and it is why the config tells you to read the panels rather than
the metric.

## Epoch 199 — frozen, not converged

Against epoch 109: 0.108/0.070/0.189/0.103/0.069/0.078/0.108/0.109 becomes
0.108/0.071/0.169/0.082/0.070/0.069/0.102/0.103. Two slices improved meaningfully, one
got worse, five did not move.

Some of that stillness is the schedule. `lr_policy: linear_decay` with
`lr_decay_start_frac: 0.5` over `n_epochs: 200` (`base.yaml`, implemented at
`training/pix2pix_nce.py:575-601`) holds lr at 2e-4 for epochs 0–99 and then ramps it
linearly to zero. By epoch 150 it is halved; by 199 it is 2e-6. The model was being
annealed into whatever state it was in.

But the state it was annealed into was reached at **full learning rate**, by epoch 109
at the latest. The schedule froze the failure; it did not cause it.

And the spine slices never left epoch 9. Both are still essentially the MRI with a tone
shift — which is precisely what 04 predicts, because the discriminator that was
supposed to teach the generator what a spine CT looks like was shown nine of them.

---

## The one number that summarises the run

Mean panel MAE went 0.1006 → 0.0968 across 190 epochs. **The epoch-9 output was a copy
of the input.** So the entire training run bought 3.8% over the identity function, and
lost to it outright on both musculoskeletal slices.

That is the result. Everything after this document is why.

---

## Checked against the real `metrics.csv`

**This section was written before `results/exp8_cyclegan_results/` existed**, as six
explicit predictions. The metrics have since arrived. The predictions are kept below so the
calls can be scored — five held, one was wrong.

| predicted | actual | |
|---|---|---|
| `val/mae_norm` roughly flat from ~ep110 | **flat the whole run** — total range 0.2208–0.2283 across 200 epochs | **held, and understated** |
| spine the worst region by a wide margin | 0.3903 vs 0.1641 musculoskeletal — 2.4× | **held** |
| `val/dice_bone` stays low | peaks at **0.080**; 0.059 at the best epoch | **held** |
| `train/G_cycle` falls smoothly while the panels get worse | `G_cycle_A` 0.170 → 0.036 (4.7×) | **held** |
| musculoskeletal regresses after ~ep50 | those two *panel slices* regressed; validation MSK improved slightly | **partly wrong** |
| `train/D_acc_fake` high and stable — D winning comfortably | **0.543 → 0.869, never saturates** | **wrong** |

Two things to add, both of which change how this document should be read.

**The discriminator never won.** `D_acc_fake_B` climbs slowly from chance to about 87% and
stops; `D_acc_real_B` does the same. By every standard diagnostic exp8's adversarial game
is *healthy* — balanced, no collapse, no oscillation, no vanishing gradient. Compare
[exp7_reggan](exp7_reggan/), where `D_acc` is exactly 1.0000 in most epochs past 20 and the
generator's adversarial gradient dies outright.

That correction is worth more than the prediction was. **A balanced adversarial equilibrium
is not evidence of a working run** — it only says the two networks are matched. Two
well-matched networks sat in a stable equilibrium for 200 epochs and produced nothing,
because the terms meant to force correctness do not measure correctness. There is a second
detail in the same columns: `G_GAN_A2B` *rises* 0.302 → 0.430, so the generator got
steadily worse at fooling D while its cycle loss fell 4.7×. The reconstruction terms, at 15
against 1, won.

**The best epoch was 7.** `is_best` is True at epoch 7 and never again — the epoch at which
the generator was still visibly copying the MRI through. The entire useful progress of the
run happened in seven epochs and consisted of learning to leave the image roughly alone.

**And the caveat below was right, by a factor of 2.3.** The panel MAE is unmasked
(`trainer.py:388`) and `val/mae_norm` is masked (`evaluation/metrics.py:127-135`): the
panel mean at epoch 199 is 0.0968 against a real `val/mae_norm` of **0.2235**. Every panel
number in these documents is roughly half the true error. Compare trends, not levels.

Full scorecard and the Hounsfield-unit figures:
[exp8_cyclegan/02_what_the_metrics_confirmed.md](exp8_cyclegan/02_what_the_metrics_confirmed.md).
