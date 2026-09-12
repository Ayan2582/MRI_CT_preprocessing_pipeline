# Across all seven runs

Findings that exist only in the comparison. Every table here is reproduced by:

```
python review_test_Folder/_comparison/ladder_audit.py
```

which is read-only and writes nothing.

---

## The scoreboard

All runs: 200 epochs, identical validation set (230 samples, 159 bone-eligible).

| run | λ_L1 | λ_NCE | best `mae_norm` | @ep | final | drift | `dice_bone` | `ssim` |
|---|---|---|---|---|---|---|---|---|
| exp1_pix2pix ¹ | 100 | 0 | **0.0763** | 72 | 0.0780 | +2.2% | 0.364 | **0.615** |
| exp2_paper | 100 | 1 | 0.1166 | **18** | 0.1264 | **+8.4%** | 0.369 | 0.386 |
| exp3b_nce_6_pix48 | 48 | 6 | 0.1203 | 121 | 0.1214 | **+0.9%** | **0.385** | 0.355 |
| exp3b_nce_20_5 | 20 | 5 | 0.1585 | 199 | 0.1585 | 0.0% | **0.038** | 0.367 |
| exp4_nce_max | 10 | 5 | 0.1666 | 199 | 0.1666 | 0.0% | **0.029** | 0.296 |
| exp8_cyclegan | 0 | 0 | 0.2208 | **7** | 0.2235 | +1.2% | 0.078 | 0.278 |
| exp7_reggan | 0 | 0 | 0.2897 | **0** | 0.3071 | +6.0% | 0.196 | 0.222 |

¹ **not comparable** — trained pre-ROI with `hflip: true`. See
[exp1_pix2pix](../exp1_pix2pix/).

Two runs peaked at epoch **0** and epoch **7**: they never learned. Two produced
essentially **no bone**. One peaked at epoch 18 and then degraded for 181 epochs. **One run
in seven — exp3b_nce_6_pix48 — behaved like a model you could ship.**

---

## Finding 1: spine is the worst region in all seven runs

`val/mae_norm` per region, at each run's best epoch:

| run | brain | abdomen | musculoskeletal | **spine** | spine / best |
|---|---|---|---|---|---|
| exp1_pix2pix | 0.0686 | 0.0967 | 0.0473 | **0.1053** | 2.2× |
| exp2_paper | 0.1177 | 0.1317 | 0.0755 | **0.2045** | 2.7× |
| exp3b_nce_6_pix48 | 0.1284 | 0.1303 | 0.0834 | **0.1806** | 2.2× |
| exp3b_nce_20_5 | 0.2016 | 0.1509 | 0.1070 | **0.2193** | 2.1× |
| exp4_nce_max | 0.2131 | 0.1573 | 0.1154 | **0.2160** | 1.9× |
| exp8_cyclegan | 0.2702 | 0.2011 | 0.1641 | **0.3903** | 2.4× |
| exp7_reggan | 0.3115 | 0.2901 | 0.2455 | **0.3877** | 1.6× |

Seven out of seven, without exception, at roughly the same ratio.

**Spine is 27 of 1687 training slices — 1.6%.** Across this ladder the objective was
changed five ways, the architecture once, and the data pairing once. None of it moved
spine. That is not a finding about any model; it is a finding about the dataset, and the
only fix is more spine data.

Until there is more, spine should be reported separately and labelled as underpowered.
Including it in a whole-ladder average makes every model look worse for the same reason,
and obscures which differences between models are real.

---

## Finding 2: the selection metric inverts the clinical one

`eval.selection_metric` is `mae_norm`, so this is the number that picks `best.pt`.

| run | best region by `mae_norm` | best region by `mae_hu` |
|---|---|---|
| exp1_pix2pix | musculoskeletal | **brain** |
| exp2_paper | musculoskeletal | **brain** |
| exp3b_nce_6_pix48 | musculoskeletal | **brain** |
| exp3b_nce_20_5 | musculoskeletal | **brain** |
| exp4_nce_max | musculoskeletal | **brain** |
| exp8_cyclegan | musculoskeletal | **brain** |
| exp7_reggan | musculoskeletal | **brain** |

**7 of 7 disagree.** Side by side, at each run's best epoch:

| run | brain | abdomen | musculoskeletal | spine |
|---|---|---|---|---|
| exp1_pix2pix | 0.069 = **5.5 HU** | 0.097 = 38.7 HU | 0.047 = 23.6 HU | 0.105 = 52.7 HU |
| exp2_paper | 0.118 = **9.4 HU** | 0.132 = 52.7 HU | 0.076 = 37.8 HU | 0.204 = 102.2 HU |
| exp3b_nce_6_pix48 | 0.128 = **10.3 HU** | 0.130 = 52.1 HU | 0.083 = 41.7 HU | 0.181 = 90.3 HU |
| exp3b_nce_20_5 | 0.202 = 16.1 HU | 0.151 = 60.4 HU | 0.107 = 53.5 HU | 0.219 = 109.6 HU |
| exp4_nce_max | 0.213 = 17.0 HU | 0.157 = 62.9 HU | 0.115 = 57.7 HU | 0.216 = 108.0 HU |
| exp8_cyclegan | 0.270 = 21.6 HU | 0.201 = 80.4 HU | 0.164 = 82.1 HU | 0.390 = 195.2 HU |
| exp7_reggan | 0.311 = 24.9 HU | 0.290 = 116.0 HU | 0.246 = 122.8 HU | 0.388 = 193.8 HU |

**Why.** CT is normalised per region (`Preprocessing/pipeline_config.py:208-229`): brain
0–80 HU, abdomen −160–240, musculoskeletal and spine −200–300. So one normalised unit is
**80 HU in brain and 500 HU in musculoskeletal** — a factor of 6.25. `mae_norm` divides
that out.

Take exp1: `mae_norm` says musculoskeletal (0.047) is 32% better than brain (0.069). In HU,
brain (5.5) is **4.3× better** than musculoskeletal (23.6). The two metrics do not disagree
about detail; they rank the regions in the opposite order.

**This is not an argument against `mae_norm`.** `evaluation/metrics.py:128-134` defends it
as the one scalar comparable across regions, and for *ranking checkpoints of a single
model* — which is its job — that is correct, since the window is constant within a run.

It is an argument about **reporting**. `mae_norm` is a selection statistic, not an accuracy
statistic. Any table that reports per-region accuracy should report `mae_hu`, because that
is the number with a physical meaning, and it says something the normalised number hides:
**only brain is anywhere near clinically usable.** 5.5–10.3 HU in brain is in the right
territory for synthetic-CT dose calculation. 52 HU in abdomen and 90 HU in spine, from the
best comparable run, are not.

---

## Finding 3: the L1:NCE cliff

The four paired U-Net runs, ordered by how hard L1 pulls:

| run | λ_L1 : λ_NCE | ratio | final `train/G_L1` | best `mae_norm` | `dice_bone` | outcome |
|---|---|---|---|---|---|---|
| exp2_paper | 100 : 1 | 100:1 | 0.084 | **0.1166** | 0.369 | overfits — peaks ep18, +8.4% drift |
| exp3b_nce_6_pix48 | 48 : 6 | **8:1** | 0.097 | 0.1203 | **0.385** | **stable — peaks ep121, +0.9%** |
| exp3b_nce_20_5 | 20 : 5 | 4:1 | **0.297** | 0.1585 | **0.038** | bone gone |
| exp4_nce_max | 10 : 5 | 2:1 | **0.322** | 0.1666 | **0.029** | bone gone |

Two clusters, nothing between them. Above the cliff `train/G_L1` lands near 0.09 and Dice
near 0.38; below it `G_L1` stalls near 0.30 and Dice collapses tenfold. Halving λ_L1 from
48 to 20 costs **0.347 of Dice**; halving it again from 20 to 10 costs 0.009.

Two mechanisms meet at 8:1, and they fail in opposite directions:

- **Too much L1 → memorisation.** L1 is a per-pixel regression against a fixed target, so
  it *can* be memorised — 54.4 M parameters against **32 independent subjects**. exp2
  drives `train/G_L1` to 0.084 and its validation error rises for 181 epochs while doing
  it. Derived in [exp2/01 §4](../exp2_paper/01_the_idea.md).
- **Too little L1 → intensity collapse.** PatchNCE is built from cosine similarities
  between L2-normalised vectors, so it is provably invariant to intensity remaps that
  preserve ranking — bone rendered black satisfies it exactly as well as bone rendered
  white. L1 is the *only* term carrying Hounsfield values. Weaken it and the model settles
  on the conditional median, which never crosses the 0.70 normalised bone threshold.
  Derived in [exp4/03](../exp4_nce_max/03_the_mechanism.md).

So **λ_L1 : λ_NCE is a regularisation hyperparameter**, not only a question about what to
constrain — because one of the two terms is memorisable and the other is not. The ladder
posed it as "lean on NCE — open question" (`model/README.md:73`); this is the answer, and
the recommended setting is **48 : 6**.

The interval worth probing next is λ_L1 between 20 and 48.

### A side observation: NCE layer 0 never learns

`train/G_NCE_L0`, against a chance level of `ln(257) = 5.549` for the 257-way
classification (`num_patches: 256`):

| run | epoch 0 | epoch 199 |
|---|---|---|
| exp2_paper | 5.548 | 5.352 |
| exp3b_nce_6_pix48 | 5.460 | 5.203 |
| exp3b_nce_20_5 | 5.327 | 5.164 |
| exp4_nce_max | 5.342 | 5.089 |

Starts at chance, ends near chance, in every run — while layers 1–4 fall 60–75%. Tap 0 is
the **raw input image** (`networks/unet.py:166-168`) through an `nn.Linear(1, 256)`, so a
one-dimensional feature cannot separate 256 locations. One fifth of the averaged NCE term
is contributing nothing. Dropping it (`loss.nce.layers: [1,2,3,4]`) is a one-line
experiment.

---

## Finding 4: every metric here has a blind spot, and two of them share one

| metric | blind to | shown by |
|---|---|---|
| `mae_norm` | field of view; the per-region HU window | exp1 (§ [01](../exp1_pix2pix/01_the_measurement_problem.md)); Finding 2 |
| `ssim` | absolute contrast scale | exp3b_nce_20_5 (no bone) **outranks** exp3b_nce_6_pix48 (best bone): 0.367 vs 0.355 |
| `dice_bone` | over-production of bone | exp2 sprays bone through soft tissue and scores 0.369 |
| panel MAE | it is **unmasked** — diluted by background | exp8: panel 0.097 vs `val/mae_norm` 0.2235, a 2.3× gap |
| `train/*` | everything that matters | exp7: `G_corr` improves 22× while validation degrades |

The dangerous pair is **SSIM and PatchNCE**. Both are built on locally-normalised,
ranking-style comparisons, so both are largely indifferent to absolute intensity. Use
PatchNCE as a loss and SSIM as a metric and you have chosen a training signal and a scoring
signal with an identical blind spot — and the thing they are jointly blind to is the one
thing a synthetic CT exists to get right.

The only metrics in this set with an opinion about absolute values are **`mae_hu`** and
**`dice_bone`**. At least one of them belongs in every table.

---

## Finding 5: four of seven runs would have been caught in under 20 epochs

| run | the tell | visible by |
|---|---|---|
| exp7_reggan | `is_best` still at epoch 0; `G_corr` falling while `val/mae_norm` rises | epoch 2 |
| exp8_cyclegan | `is_best` still at epoch 7; `val/mae_norm` flat | epoch 15 |
| exp2_paper | `val/mae_norm` turned upward | epoch 25 |
| exp3b_nce_20_5 / exp4 | `val/dice_bone` at or below 0.0002 | epoch 5 |

All four signals are already in `metrics.csv`. Nothing new needs measuring — the runs
finished 200 epochs each because nobody was looking at `is_best` or `dice_bone` while they
ran. Two cheap rules would have saved most of that compute:

- **Stop if `is_best` is still below epoch 5 once you pass epoch 20.** A run whose best
  checkpoint is its first is not converging slowly; it is not converging.
- **Alert if `val/dice_bone` is exactly 0 after epoch 5.** That is not a bad score, it is
  a model producing no bone at all.

---

## Where the ladder stands

| question the ladder asks | answered? |
|---|---|
| "What does the standard recipe give?" (exp1) | **no** — measured on a different pipeline |
| "The target loss, textbook weights" (exp2) | yes — 0.1166, but overfits at epoch 18 |
| "Lean on NCE — open question" (exp3) | **yes** — 48:6, and the cliff is between 8:1 and 4:1 |
| "Where does hallucination start?" (exp4) | **inverted** — low λ_L1 gives collapse, not hallucination; hallucination is at the *intermediate* settings |
| "How much residual misalignment is there?" (exp7) | **no** — the run measured an interpolation artifact, not anatomy |
| "What is the pairing worth?" (exp8) | **no** — the model never trained |

Two of six answered, one answered in a way that inverts the question, three open. The three
open ones need: one re-run of exp1 on the current pipeline, and repaired objectives for
exp7 and exp8 — all detailed in the respective folders.
