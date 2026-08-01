# MRI→CT Registration: Findings and Recommended Approach

Conclusions from the registration investigation (`registration_demo*.py`), written to be
actioned. Every recommendation below is followed by the evidence for it, so anything that
looks wrong can be re-checked rather than taken on trust.

> **New to this?** Read [`registration_docs.md`](registration_docs.md) first. It teaches the same
> material from first principles — what registration is, why CT and MRI are hard to align, what
> mutual information measures, and why each decision below was made. This document assumes you
> already know that and gets straight to the recommendations.

**Scope of the evidence.** 4 patients (brain `PA0`, shoulder `PA6`, spine `PA18`, knee `PA32_knee`),
11 CT/MRI series pairs, 33 slice-pairs across all available orientations at first/middle/last
positions. That is enough to identify failure modes and fix clear bugs. It is **not** enough to
claim any of this generalises to the other ~41 patients — treat the thresholds below as
starting points to be re-validated, not settled constants.

---

## TL;DR

1. **`resample_mri_to_ct_grid` is broken and should be replaced.** It never corrects
   translation, which left 7 of 33 test slices with a completely blank MRI.
2. **Any box you compute must be in the same coordinate frame as the content that fills it.**
   Computing the crop in the pre-alignment frame and filling it post-alignment silently discarded
   14 of 18 slices on one series, and looked exactly like a crop working correctly.
3. **Force single-threaded registration and use multi-start**, and **gate transforms during
   selection, not after** — otherwise a rejected seed that merely outscored an admissible one
   takes the whole transform down with it.
4. **Always keep the unregistered baseline as the last fallback.** A registration that makes a
   slice worse must not ship; on this data that rung fires on 3 of 33 slices.
5. **Score with normalized MI, not raw MI.** Raw MI grows with how much image is in the frame,
   which makes it useless for the one comparison that matters here — full canvas vs cropped.
6. **Crop to the FOV intersection only as a fallback**, for slices where full-canvas registration
   was actively *worse* than doing nothing — not merely where it failed to improve. Once that
   distinction is made, 2 of 33 slices qualify and 1 benefits.
7. **Exclude spine** until landmark validation exists. The metric cannot verify it in two of three axes.
8. **Never treat a higher score as proof of better alignment.** It is a screening signal, nothing more.

---

## Recommended pipeline

Order matters — each step assumes the previous one.

```
load series  →  [QC 0] FOV overlap gate  →  N4 (MRI)  →  resample CT in-plane 1mm
             →  [1] volume align: GEOMETRY translation      (replaces the direction hack)
             →  [2] per-slice 2D on the FULL canvas: multi-start, gated during selection,
                    best admissible affine → rigid → unregistered
             →  [3] IF that slice REGRESSED: retry on the FOV-intersection crop (computed
                    in the post-alignment frame), keep whichever of crop / full /
                    unregistered wins on the common region             (fallback only)
             →  [4] record QC columns  →  normalise + export
```

Note the order changed from an earlier draft: cropping used to sit *before* registration as a
conditional preprocessing step. It is now strictly a per-slice retry, because the decision to
crop is only sound once you have seen the full-canvas result fail — and "failed" has to mean
*worse than doing nothing*, not *did not improve*.

### QC 0 — Field-of-view overlap gate

Before any processing, compute the world-space overlap of the two volumes and reject or flag
pairs that barely intersect. Reference implementation: `registration_demo_fov.py`.

Transform all 8 corners of each volume to physical space, take the axis-aligned bounding box of
each, and report the per-axis overlap as a fraction of the smaller field of view.

**Compute it twice: before and after alignment.** These are different numbers answering different
questions, and conflating them caused the worst bug in this investigation (see step 3).

| Number | Frame | What it tells you |
|---|---|---|
| `fov_overlap_raw` | scanner coordinates, as acquired | How far apart the two acquisitions were positioned |
| `fov_overlap_aligned` | after the step-1 GEOMETRY translation | How much shared canvas registration actually has |

| Action | Overlap on worst axis, **raw** |
|---|---|
| Proceed normally | > 70% |
| Proceed, but expect the crop fallback (step 3) to be exercised | 30–70% |
| **Flag for review** | < 30% |

This is a *reporting* gate only. It does not decide whether to crop — step 3 does, per slice,
from the registration result — and it should not by itself exclude a pair, because step 1 may
close the gap entirely.

**Evidence.** Raw overlap ranged from 86–100% for brain, shoulder and spine, but knee/axial was
72.5% and **knee/sagittal only 12.5%** — CT covering `X ∈ [53.3, 145.4]` against MRI
`X ∈ [−39.9, 64.8]`. That is a real and worth-flagging positioning difference. But it is **not** a
shortage of shared anatomy: the step-1 translation of −86.9mm in X closes it, and the aligned
overlap for the same series is **100%**, with real MRI content on all 18 CT slices.

An earlier draft of this document claimed that series "yields 4 usable slices out of 18." That
was wrong, and it was wrong because the crop was being computed in the raw frame — the bug
described in step 3. A low raw overlap means *check this pair*, not *discard most of it*.

### Metric: normalized MI, not raw MI

Every score reported from here on is Studholme normalized mutual information

```
NMI = ( H(fixed) + H(moving) ) / H(fixed, moving)
```

implemented as `nmi_score` in `registration_demo.py` (64-bin joint histogram, intensities
clipped — not discarded — to their 0.5/99.5 percentiles so CT metal or MRI spikes cannot
collapse the tissue range into two bins). It is bounded in `[1, 2]`: 1 is statistical
independence, 2 is identical partitions.

**Why the change.** Raw Mattes MI is unbounded and rises with the marginal entropies, so its
value tracks *how much image is in the frame* nearly as strongly as *how well the two images
line up*. That is tolerable when comparing two alignments of the same slice and fatal when
comparing two different canvas sizes — which is exactly the question the crop fallback has to
answer. NMI divides that dependence out. It also makes numbers roughly comparable across
regions, which raw MI never was.

**What did not change: the optimizer.** SimpleITK 2.5 exposes no NMI metric to
`ImageRegistrationMethod` — the MI-family options are Mattes and JointHistogram only — so
gradient descent still minimizes Mattes MI internally. Multi-start makes that a smaller problem
than it sounds: **MI proposes candidate alignments, NMI decides which one is kept.**

**It changed a conclusion, not just a scale.** Under raw MI, knee/sagittal — the 12.5%-overlap
series — appeared to collapse on the full canvas (~0.31) and to need cropping. Under NMI, two of
its three test slices register perfectly well on the full canvas and never trigger the fallback
at all. The apparent collapse was largely raw MI penalising the larger frame, not a registration
failure. That is the concrete reason this matters.

**Consequence for old numbers.** Every MI figure in this document below predates the switch and
is on the old scale. Do not compare an MI of 0.61 with an NMI of 1.21 — they are different
quantities, not a change in quality. Anything re-run under NMI is labelled as such.

### 1 — Volume alignment: replace the direction hack

Do **not** use the current production function. It does two harmful things and one useless one:

```python
# image_processing.py — DO NOT USE AS-IS
mri_aligned = mri_image                              # aliases, does not copy
mri_aligned.SetDirection(ct_image.GetDirection())    # destroys real MRI orientation
resampler.SetTransform(sitk.Transform())             # identity: translation never corrected
```

Replace with an explicitly computed translation that leaves the MRI's real geometry intact
(`resample_mri_to_ct_grid_v2` in `registration_demo.py`):

```python
initial_transform = sitk.CenteredTransformInitializer(
    sitk.Cast(ct_image, sitk.sitkFloat32),
    sitk.Cast(mri_image, sitk.sitkFloat32),
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,
)
resampler.SetReferenceImage(ct_image)
resampler.SetTransform(initial_transform)
```

Rotation is deliberately left at identity: a genuine 3D rotation estimate needs more
through-plane sampling than 9–21 anisotropic slices can support. Rotation is handled per-slice
in 2D at step 3, where it is well constrained.

**Evidence.** The old baseline left translations of **−84mm (knee)**, **−56mm (shoulder)** entirely
uncorrected — it only ever adjusted rotation. Blank-MRI slices went **7/33 → 0/33**. Rigid
registration beat the old baseline on **33/33** slices.

**Known limitation.** GEOMETRY aligns bounding-box *centres*, not anatomy. Where the old hack
happened to be near-correct already (brain/axial, a centred symmetric structure) this can be
slightly worse — brain baseline MI fell 0.451 → 0.348. It made no difference downstream there
(rigid/affine converged to ~0.61 either way), but it is a real caveat when the two volumes
cover different physical extents.

### 2 — Per-slice 2D registration (on the full canvas)

**Determinism first.** Set this once at import, before any registration:

```python
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
```

**Then multi-start.** Run N independent attempts with different sampling seeds and keep the best
scoring one that also passes the scale gate.

```python
N_STARTS             = 5
SCALE_TOL            = 0.60    # outer sanity bound on |scale − 1|
CANVAS_FIT_TOL       = 0.04    # veto scales sitting on the canvas ratio (0 disables)
MIN_TRANSFORM_MARGIN = 0.005   # floor on the margin affine must beat rigid by
```

Prefer `AffineTransform(2)`, or `Similarity2DTransform` if you want isotropic scale only.

**Gate inside the selection loop, not after it.** This is the part that is easy to get wrong and
was wrong here for a full round of results. The gate is an admissibility predicate on each seed,
so multi-start returns the best *admissible* attempt:

```python
best = multistart(fixed, moving, "affine", accept_fn=make_scale_gate(canvas_ratio))   # right
```

The predicate itself is now two tests, not one band — see the UPDATE note in the open items
below, and `registration_docs.md` §5.3.

Scoring all seeds, taking the highest, and *then* testing it discards a perfectly good in-tolerance
alignment whenever an out-of-tolerance one happens to outscore it — and it discards the whole
affine, not just that seed. Record the per-seed scales (`full_affine_scales`), otherwise the gate
is unauditable: you cannot tell "every seed wanted to resize" from "one bad seed won the ballot."

**Then the ladder of fallbacks, in order:** best admissible affine → best rigid → unregistered.
The last rung is not optional. Without it a slice can ship a result measurably worse than doing
nothing (knee/sagittal `last` shipped 1.081 against a 1.127 baseline before this was added).

**Prefer affine over rigid only by more than the seed noise.** A zero margin means the choice is
made by noise: on **18 of 33** slices the affine–rigid gap was smaller than the multi-start seed
spread on that same slice. Use the **affine's own** spread as the margin, floored at
`MIN_TRANSFORM_MARGIN`. Do *not* use `max(rigid_spread, affine_spread)` — that lets an unstable
rigid raise the bar for a stable affine, so the worse rigid behaves the harder it becomes to
replace.

**Evidence — determinism.** Identical code, identical slice, identical `seed=42` produced MI of
0.131, 0.132 and **0.289** across process runs. Forcing one thread made it exactly repeatable
(0.125 on 3/3 runs). Note the reproducible answer was *worse* than the lucky one — determinism
and quality are separate problems, which is why multi-start is still needed.

**Evidence — multi-start.** Knee/rigid seeds ranged **0.100–0.323** before the volume alignment
was fixed. After fixing it the same spread narrowed to 0.060, but wide spreads persist
elsewhere (12/33 slices exceed 0.10 for rigid), so multi-start remains worth its cost.

**Evidence — affine over rigid (superseded, raw MI).** Affine beat the baseline on **31/33**
slices with mean gain **+0.169**, against **24/33** and **+0.060** for rigid. That comparison used
raw MI, no scale gate during selection, and no noise-floor margin. Under NMI with both in place,
**rigid ships on 26 of 33 slices and affine on 4** — most affines are either out of tolerance or
ahead of rigid by less than the seed spread. Treat the original 2.5× claim as an artifact of the
metric and the ungated selection, not as a reason to prefer affine.

**Evidence — scale gate.** On spine, unconstrained affine recovered scale **0.825** while the
MRI/CT FOV ratio is **0.824** — a match to 0.3%. It was resizing the image to match the canvas,
not aligning anatomy, and doing so *reproducibly* (all five seeds landed in 0.787–0.847). A
patient's spine does not shrink 17% between scans. The gate rejected all five, which is the
correct verdict rather than a failure.

### 3 — Intersection crop, as a per-slice fallback only

Reference: `registration_demo_sweep_v3.py` (`resample_mri_to_ct_grid_v3` plus the fallback logic
in `process_slice`).

Register every slice on the full CT canvas first, then classify what happened:

| `full_outcome` | Meaning | Fallback? |
|---|---|---|
| `all_seeds_failed` | Every multi-start attempt crashed — usually "images do not sufficiently overlap" | yes |
| `nothing_admissible` | Seeds ran but none survived the scale gate on either transform | yes |
| `unevaluable_nmi` | The winning result cannot be scored (constant/blank slice) | yes |
| `sparse_overlap_<f>` | The MRI fills less than `MIN_MRI_COVERAGE` (25%) of the canvas | yes |
| `regressed` | The result is **worse** than the unregistered baseline by more than `MIN_GAIN` | yes |
| `marginal` | Result within ±`MIN_GAIN` of baseline — registration had nothing to add | **no** |
| `improved` | Result beats baseline by more than `MIN_GAIN` | no |

**`marginal` is not a failure, and this distinction matters more than it looks.** An earlier
version used a single `no_gain` criterion — anything that failed to beat the baseline by the
margin — and it routed 14 of 33 slices to the fallback. **Twelve of those fourteen had actually
improved**, just by less than 0.010. Cropping is not a remedy for "the baseline was already
good." Only an actively worse result is a registration failure.

Test causes before symptoms. `sparse_overlap` is checked before `regressed`, so a slice that is
starved of MRI *and* regresses is recorded as the former; the reverse order labels every such
slice with its downstream effect and hides why.

When the fallback runs, all candidates are **re-scored over the identical cropped region** — a
full-canvas score and a cropped-canvas score are not comparable numbers even for the same
anatomy. The choice is then **three-way**, not two-way:

```
keep crop   if crop beats BOTH the full-canvas result AND the unregistered baseline by MIN_GAIN
keep none   if the baseline beats the full-canvas result on the common region
keep full   otherwise                                      (recorded as crop_did_not_help)
```

Comparing the crop only against the full result — the earlier logic — lets a crop that is worse
than doing nothing win whenever the full result is worse still.

Three geometry requirements, all of which were bugs in earlier implementations:

- **Compute the intersection in the same frame as the content that fills it.** This was the
  costliest error here. A resampler's transform maps *output* points to *moving* points, so an
  output point `p` sees real MRI exactly when `T(p)` is inside the MRI — that is, when `p` is
  inside `T⁻¹(MRI extent)`. Intersecting the CT box with the MRI's **raw** box answers a
  different question: where the two volumes overlapped *before* the alignment that exists to
  make them overlap. On knee/sagittal, with a −86.9mm translation, MRI content lands on all 18
  CT slices while the raw-frame ROI kept only slices 14–17 — **the fallback discarded 14 of 18
  slices holding real MRI**, and the failure was invisible because a smaller crop is exactly
  what you expect a crop to produce. Push the MRI's 8 corners through
  `transform.GetInverse()` before intersecting.
- `RegionOfInterest` takes a voxel **count**, so an inclusive `[start, stop]` span is
  `stop - start + 1`. Omitting the `+1` shaves a voxel off every axis and looks exactly like a
  small genuine crop.
- **Fit the transform against the full CT, never the cropped one.** Cropping selects which part
  of physical space to keep; it must not move the MRI. Fitting against the crop re-centres the
  MRI on the crop's centre instead — a ~40mm displacement in testing, which was the entire
  reason cropping first appeared to destroy registration quality.

**Why fallback rather than unconditional.** Applied to every slice, cropping was a wash overall —
mean MI change **+0.012 (rigid)**, **−0.013 (affine)** across 33 slices — but strongly bimodal per
series. It rescued knee/sagittal (rigid held ~0.61 where the full canvas collapsed to 0.31) and
clearly harmed well-matched pairs (spine/coronal affine **0.63 → 0.48**, shoulder/axial ~−0.09 on
both transforms) by trimming away context the optimiser was using. An average over those two
populations answers nothing; the useful question is which slices belong to which population, and
that is what the failure criteria above decide.

**Evidence — the corrected fallback run (33 slices, NMI, `sweep_v3_summary.csv`).**

| Full-canvas outcome | Slices |
|---|---|
| `improved` | **18 / 33** |
| `marginal` | **13 / 33** |
| `regressed` | **2 / 33** |
| Crop attempted (both `regressed`) | **2 / 33** |
| Crop actually kept | **1 / 33** |

The headline is how little cropping is actually needed once the frame bug and the failure
definition are fixed: **2 attempts and 1 keep, against 14 and 4 before.** The one keep is
knee/coronal `first`, 1.093 → 1.124 on the common region.

The four earlier "rescues" did not survive scrutiny. The largest, knee/sagittal `last` at
1.091 → 1.197, was measured against a full-canvas result of 1.091 that *should never have
shipped* — it was worse than that slice's own unregistered baseline of 1.127. With the
unregistered rung in place the full canvas ships 1.110, the crop manages 1.116, and the
difference no longer clears `MIN_GAIN`. The apparent gain was mostly the badness of the thing it
was being compared against.

Shipped results now beat the unregistered baseline by **+0.021 NMI** on average, with a floor of
exactly **0.000** — by construction, since nothing worse than the baseline can ship. That rung
fired on **3 slices**.

Two things this run did *not* show:

- **Three failure criteria still never fired.** All fallbacks were `regressed`. Minimum MRI
  coverage was **27.8%**, just above the 25% `sparse_overlap` threshold, and no slice was
  unscorable. `all_seeds_failed`, `nothing_admissible`, `unevaluable_nmi` and `sparse_overlap`
  remain **untriggered and unvalidated** on this sample.
- **Gating during selection helped, but did not explain the rejections.** I expected most of the
  gate's 19/33 rejection rate to be an artifact of gating the winner instead of the candidates.
  It was not. Gating during selection recovers a usable affine on **10 of 33** slices that
  post-hoc gating would have discarded outright — a real improvement — but **12 of 33** slices
  still lose *every* affine seed, and 11 have no rejection at all. Across the run **52 of 99
  affine seeds** were out of tolerance. The affine genuinely wants to resize on this data; that
  is a finding about the data, not a bug in the selection.

### 4 — QC columns to record per slice

None of this is auditable without it. Add to `metadata.csv`:

| Column | Why |
|---|---|
| `nmi_baseline`, `nmi_best`, `nmi_shipped` | Three different numbers. `best` is what registration achieved (drives the diagnosis); `shipped` is what was used (never worse than baseline) |
| `seed_spread` | **`None`, not `0.0`, when fewer than 2 admissible seeds survive** |
| `n_seeds_failed`, `n_rejected` | Crashed and gate-rejected are different failures; keep them apart |
| `affine_scales` | **Per seed, not just the winner** — otherwise the gate cannot be audited |
| `noise_floor` | The margin the rigid/affine choice had to clear |
| `full_outcome` | One of the seven classes above |
| `crop_attempted`, `crop_used` | Different questions — a slice can fall back and still keep the full canvas |
| `crop_skipped_reason` / `crop_rejected_reason` | "Why we didn't try" vs "why we tried and discarded". One column cannot carry both |
| `final_source`, `final_kind`, `final_scoring_region` | Which run it came from, which transform, and **which region the score refers to** — without the last one `final_nmi` is not comparable row to row |
| `mri_coverage` | Fraction of the canvas the MRI fills; feeds `sparse_overlap` |
| `fov_overlap_raw`, `fov_overlap_aligned` | The two QC-0 numbers, per pair |

The `seed_spread` note is from a real bug: computing `max − min` over a single surviving value
yields `0.0`, which reads as "every seed agreed" when it means the opposite.

---

## What not to do

- **Do not trust MI or NMI as a validity measure.** Spine's highest-scoring result was
  anatomically wrong. Both compare intensity distributions; neither has any concept of *which*
  structure it matched. Normalizing fixed a comparability problem, not a correctness one.
- **Do not compare a raw-MI number with an NMI number.** Different quantities, different scales.
- **Do not apply intersection cropping unconditionally.** It costs about as much as it gains.
- **Do not compute a crop box in one coordinate frame and fill it from another.** A crop that is
  too small looks exactly like a crop that is working.
- **Do not treat "registration did not improve this slice" as "registration failed."** Most of
  the time it means the volume alignment in step 1 had already done the job.
- **Do not gate a transform after picking the winner.** Gate during selection, or you throw away
  admissible alignments that merely scored second.
- **Do not use unconstrained affine.** Without a scale gate it will resize to fit the canvas.
- **Do not attempt 3D rotation estimation** on this data. 9–21 slices at 5–9mm spacing with
  non-uniform sampling cannot constrain it; the optimiser will fit noise.
- **Do not assume a sharp optimum in one axis validates the alignment.** See below.

---

## Known limits and open items

**Spine is not usable as-is — recommend exclusion.** Lumbar vertebrae are near-periodic, so MI
cannot localise along the spine axis: the gap between the baseline position and the metric's
"best" was **0.013** against an optimiser noise floor of **0.047** (signal-to-noise **0.28**), with a
**~78mm** window in which every position scores within noise. Anchoring the metric on the sacrum
sharpens that axis (ambiguity 45mm → 20mm) but **widens anterior-posterior** ambiguity
(10mm → 20mm), and leaves left-right — which is *through-plane* for a sagittal series, so 2D
registration cannot correct it at all — indeterminate (MI varies 0.014–0.028 across ±25mm,
below the noise floor). The scale gate can prove a spine result wrong; nothing available proves
one right. With only 4 spine patients, exclusion is cheap.

**Brain/axial regression** from GEOMETRY centring (see step 1) is unexplained and worth a look
if brain data matters — likely a real rotation the translation-only fix cannot absorb.

**UPDATE — the scale gate has been redesigned; the paragraph below is the pre-change analysis.**
`SCALE_TOL` is now a generous outer sanity bound (0.60) plus a *canvas-fit veto*: reject scales
that coincide with the MRI's bounding-box-to-canvas ratio or its reciprocal, allow everything
else. Rationale: after `resample_mri_to_ct_grid_v2` both images are on the same 1mm world grid,
so a field-of-view difference leaves **no magnification gap for affine to correct** — a large
recovered scale is the optimiser fitting the frame border, and it is identifiable because it
lands on the canvas ratio. Effect over 33 slices: affine seed rejections **52/99 → 18/99**,
affine shipped on **4 → 12** slices, `regressed` **2 → 0**. Separately, the affine-vs-rigid margin
was `max(rigid_spread, affine_spread)`, which let an *unstable rigid* raise the bar for affine;
it now uses the affine's own reproducibility. **Caveat:** the recovered scale is not a stable
measurement under `RANDOM` metric sampling (it varies by up to ±17% between identical runs — see
`registration_docs.md` §6.2), so any scale threshold is gating on a noisy quantity until the
sampler is made deterministic.

**The 5% scale tolerance rejects a lot, and it is not a selection artifact.** **52 of 99** affine
seeds were out of tolerance, and **12 of 33** slices had no admissible affine at all, with
recovered scales spanning roughly 0.81–1.30. Moving the gate inside the selection loop recovered
an affine on 10 slices but did not change this picture, so the remaining question is about the
data, not the code. Three readings remain and this sample does not separate them: the tolerance
is too tight for genuine inter-scan change; affine really is resizing to the canvas here; or the
2D-per-slice model is absorbing through-plane mismatch as in-plane scale. Spine is the strongest
hint — its rejected scales cluster near the MRI/CT FOV ratio, which is the signature of fitting
the canvas. Sweep the tolerance and compare each recovered scale against that pair's FOV ratio
before changing it.

**Rigid now ships on 26 of 33 slices, affine on 4.** That is a direct consequence of requiring
affine to beat rigid by more than the seed spread. Whether the noise floor is *too* strict is
open: it is the right rule in principle, but with only 3 seeds the spread estimate is itself
noisy. Re-check with more seeds before concluding affine is not worth using.

**Most of the fallback criteria are untested.** Only `regressed` ever fired, on 2 slices.
`MIN_GAIN` (0.010 NMI) and `MIN_MRI_COVERAGE` (25%) are guesses; coverage came within 2.8 points
of its threshold without crossing it, so that branch has still never executed.

**Thresholds are unvalidated.** The 70%/30% overlap bands and the 5% scale tolerance were chosen
from 4 patients. Re-check against a larger sample before relying on them.

**The sacrum anchor is a heuristic**, not sacrum detection — it takes the inferior 45% of image
rows and would select the wrong region on a differently framed series.

---

## How to validate a registration result

Since MI cannot be trusted as a correctness measure, use these instead.

1. **Sweep the parameter and look at the landscape shape.** Translate the moving image across a
   range and plot MI. A sharp peak means the metric can localise; a plateau means it cannot, and
   any "optimum" inside that plateau is arbitrary. Costs seconds. Should be routine.
2. **Sweep every degree of freedom, not one.** A sharp peak in one axis says nothing about the
   others — the sacrum anchor produced exactly that and still left two axes unresolved.
3. **Compare recovered scale against the FOV ratio.** If they match, the optimiser is fitting the
   canvas, not the anatomy.
4. **Compare the signal against the optimiser's own noise.** If the MI difference between
   candidate alignments is smaller than the seed-to-seed spread, the result is not meaningful.
5. **Look at the images.** The spine failure was found by eye after surviving three rounds of
   numeric reporting.

---

## Demo scripts

| Script | What it does |
|---|---|
| `registration_demo.py` | Per-region raw→registered walkthrough; holds `resample_mri_to_ct_grid_v2`, multi-start, `score_spread` |
| `registration_demo_sweep.py` | 33 slice-pairs, all orientations, first/middle/last |
| `registration_demo_sweep_v3.py` | Full-canvas registration with the intersection crop as a per-slice fallback; NMI scoring |
| `registration_demo_fov.py` | Field-of-view overlap diagnostic (QC 0) |
| `registration_demo_spine_fix.py` | Scale gate and sacrum anchor on spine |
| `registration_demo_spine_axes.py` | Multi-axis landscape showing what the anchor does and does not fix |

None are part of the production pipeline; they are diagnostics and reference implementations.

Full visual reports, with every figure and the per-slice tables behind the numbers above:

- [Registration QC — per-region walkthrough and spine diagnosis](https://claude.ai/code/artifact/24212b86-9ec6-47f0-9a31-a1d3973192f1)
- [Registration Sweep — 33 slices, every orientation, and the cropping experiment](https://claude.ai/code/artifact/f4725048-f5b3-45cf-ae72-6763c3b3aee2)
