# 02 — What to change

Recommendations only; nothing applied.

---

## 1. Re-run exp1 on the current pipeline

One run, no code changes — `configs/exp1_pix2pix.yaml` already inherits `use_roi: true` and
`hflip: false` from the current `base.yaml`. Just run it.

**This is the highest-value single run available**, and not because exp1 is interesting in
itself. It is because three separate claims in this project are currently blocked on it:

- **"What does the standard recipe give?"** (`model/README.md:71`) — the ladder's baseline.
  Every other run is implicitly measured against it, and it is measured differently.
- **"The gap between exp8 and exp1 is what the pairing bought"** (`model/README.md:120-125`)
  — exp8's entire purpose. As it stands that comparison varies pairing **and** ROI cropping
  **and** hflip, so it measures nothing. See
  [exp8_cyclegan](../exp8_cyclegan/03_the_verdict.md).
- **"exp7 is the one to compare against exp1"** (`model/README.md:101`) — same problem, and
  exp7 has its own ([exp7_reggan](../exp7_reggan/)).

One run unblocks all three.

## 2. Meanwhile, what you may and may not say

**May:**

- exp1 achieves **5.5 HU mean error on brain**, the best in the ladder. HU is absolute and
  window-independent, so this is comparable to anything.
- exp1 achieves **SSIM 0.615**, far above every other run, and SSIM is local enough to be
  largely insensitive to the field-of-view change.
- exp1 has the **healthiest validation curve** in the ladder — best at epoch 72, 2.2% drift.
  With identical λs to exp2 (best at epoch 18, 8.4% drift), that difference is attributable
  to the `hflip` augmentation, which is itself a useful finding: see §3.

**May not:**

- "exp1 is the best model" — not established. Its `mae_norm` is measured on an easier task
  ([01 §3-4](01_the_measurement_problem.md)).
- Any exp1-vs-anything `mae_norm` gap, in either direction.
- The estimate 0.109 from [01 §4](01_the_measurement_problem.md), as a number. It is useful
  for judging whether the lead is likely real (it is not obviously so); it is not a
  substitute for the measurement.

If exp1 has to appear in a table before it is re-run, put it in a separate block with the
pipeline difference stated in the caption. Do not put 0.0763 in the same column as 0.1166.

## 3. Turn `hflip` back on

The current `base.yaml` has `augment.hflip: false`. exp1 is the only run with it on, and it
is the only run that does not overfit early.

Given 54.4 M parameters against 32 independent subjects
([exp2/01 §4](../exp2_paper/01_the_idea.md)), effectively doubling the dataset for free is
not a small thing, and the exp1-vs-exp2 comparison is direct evidence it works: identical
λs, and best-epoch 72 versus 18.

Left-right flip is anatomically legitimate for brain, spine and most musculoskeletal
slices. It is defensible for abdomen with the caveat that it mirrors organ laterality —
liver to the left, and so on — which a model could in principle learn as real anatomy. If
that concerns you, enable it per region rather than globally.

**This is one config line and it is supported by evidence already in hand.** It should be
in the re-run of §1.

## 4. Record the pipeline version in the run directory

The only reason this was catchable is that `config.resolved.yaml` is written per run, and
`use_roi` is simply *absent* from exp1's — an absence someone has to notice while diffing
two YAML files.

Write the git SHA into the run directory at startup. `bootstrap.py` already resolves the
config; adding the commit hash to `config.resolved.yaml` is a few lines, and it turns "why
does this run's config not have the key I expected?" into "this run predates commit
`4d98c6c`."

More generally, this failure is worth naming: **a config that inherits its defaults cannot
record a default that did not exist yet.** Absent keys are silent. If you compare runs by
diffing resolved configs, an absent key is exactly the difference you will not see.

---

## What a successful re-run looks like

| signal | expectation |
|---|---|
| `val/mae_norm` | somewhere near **0.10–0.12** — beating exp3b_6_48's 0.1203 would be a genuine result |
| `val/mae_hu/brain` | at or below 5.5 HU; that number should be roughly pipeline-independent |
| `is_best` | past epoch 50 **if** `hflip` is on; near epoch 20 if not |
| vs exp3b_nce_6_pix48 | the real question — does adding NCE at 8:1 beat plain pix2pix, on the same pipeline? |

That last row is what the ladder was built to answer, and right now it has not been asked.
