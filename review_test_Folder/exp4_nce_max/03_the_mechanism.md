# 03 — The mechanism

Two proofs and one measurement. 01 argued from the shape of a contrastive loss; this
document shows that the implementation has exactly that shape, then does the threshold
arithmetic in the units this repo actually uses.

---

## Proof 1: PatchNCE cannot express a preference about intensity

Not "does not in practice". Cannot, from the form of the loss.

### The features are unit vectors

`model/networks/patch_sampler.py:133`, the last thing that happens to every patch before
it reaches the loss:

```python
# L2-normalise so the dot product in the InfoNCE numerator is a
# cosine similarity and the temperature has a consistent meaning.
sampled.append(F.normalize(patch, p=2, dim=1))
```

`F.normalize(·, p=2)` maps every feature vector to the unit sphere. **All magnitude
information is destroyed here, deliberately**, for the reason the comment gives.

### The loss is a ranking over those unit vectors

`model/losses/patch_nce.py:98-119`:

```python
l_pos = torch.bmm(feat_q.view(n_total, 1, -1),
                  feat_k.view(n_total, -1, 1)).view(n_total, 1)
...
l_neg = torch.bmm(q, k.transpose(2, 1))            # [B, n, n]
l_neg.masked_fill_(diagonal[None, :, :], -10.0)
...
logits = torch.cat((l_pos, l_neg), dim=1) / self.temperature
loss = self.cross_entropy(logits, torch.zeros(n_total, ...))
```

Every entry of `logits` is an inner product of two unit vectors — a cosine. The loss is
cross-entropy over a softmax of cosines, with the positive pinned at column 0. Its value
depends only on the **ordering and spacing of angles**.

### The consequence

Let `Φ` be the composition of the encoder tap and the projection head, and let `m` be any
remapping of image intensities. If `Φ(m(patch))` preserves the angular ordering that
`Φ(patch)` had, then every logit ordering is preserved, every softmax probability is
identical, and

```
L_NCE( m ∘ G )  =  L_NCE( G )
```

exactly. Not approximately.

Compress the whole image toward mid-grey and, so long as bone remains angularly
distinguishable from muscle, **PatchNCE returns the same number.** It has no mechanism by
which to prefer bright bone over dark bone.

### And the negatives make it strictly local

`patch_nce.py:104-113` builds negatives from `q` and `k` *within the same image*
(`[B, n, n]`, block-diagonal by construction). So the question the loss asks is not even
"is this a plausible CT" — it is "within this slice, is location *i* more like input
location *i* than like input location *j*". A purely internal, purely relative statement.

### Two details that matter for reading the numbers

- **The layers are averaged, not summed** — `patch_nce.py:144-156` returns
  `total / max(1, len(self.losses))`. So `λ_NCE: 5` really is 5. There is no hidden 5×
  multiplier from the five taps, and the λ ratios in the config read as written.
- **`feat_k` is detached** (`patch_nce.py:95`). The keys carry no gradient, so the term
  can only push the generated CT toward the MRI's feature geometry, never the reverse.

**Conclusion.** Of the three terms in exp4's objective, exactly one carries a claim about
absolute Hounsfield values: `λ_L1 · L₁`, at weight **10**. The conditional discriminator
carries a weak indirect claim at weight 1. `λ_NCE · L_NCE`, at weight 5, carries none —
by construction, and correctly so.

## Proof 2: bone is unreachable from the conditional median

### The threshold, in this repo's units

`evaluation/metrics.py:212-226`:

```python
def bone_dice(pred, target, mask, hu_min, hu_max, threshold_hu=150.0):
    threshold = hu_to_norm(float(threshold_hu), hu_min, hu_max)
    if not 0.0 < threshold < 1.0:
        return None
    pred_bone = ((pred >= threshold).float() * mask)
```

The CT windows are per-region (`Preprocessing/pipeline_config.py:208-229`), so the
normalised threshold differs by region:

| region | HU window | 150 HU in normalised units |
|---|---|---|
| musculoskeletal / spine | −200 … 300 | **0.700** |
| abdomen | −160 … 240 | **0.775** |
| brain | 0 … 80 | 1.875 → out of range, returns `None` |

(That last row is why brain is excluded from bone metrics, matching
`eval.bone_metrics_exclude_regions: [brain]`.)

**So for a single pixel to count as bone, the generator must output ≥ 0.70 — that is, it
must commit to the top 30% of the dynamic range.**

### Why the median will not go there

From 01 §1, the L1-optimal output at a location is the conditional **median** of
`p(y | x)`. The median has a property the mean does not, and it is decisive here:

> **The median is not a weighted average.** If bone accounts for less than half the
> conditional mass at a location, the median does not move part-way toward bone. It sits
> entirely inside the soft-tissue mode.

A weighted mean would at least drift upward with the bone fraction. The median is a step
function of it. Below 50% bone mass, the L1-optimal answer is *pure soft tissue* — around
0.4–0.5 in normalised units — and stays there.

And cortical bone is a thin shell, a few percent of the pixels inside the body. At almost
every location where the network is uncertain, bone is a minority mode.

So under L1 alone the generator would never cross 0.70. Something has to *overcome* the
median, and what overcomes it is having enough weight on L1 that the loss difference
between "commit to 0.85 and be right 70% of the time" and "sit at 0.45 and be safe" is
large in absolute terms — plus the discriminator, which penalises the flat texture that
hedging produces.

At `λ_L1: 100`, that is enough. At `λ_L1: 10`, it is not. `pred_bone.sum()` goes to zero,
the Dice numerator goes to zero, and `dice_bone` reports **0.0000** — while the model's
anatomy, governed by the untouched NCE term, remains excellent.

**That is the panel in 02 §4, derived.**

## 3. Why there is a cliff and not a slope

The four paired runs, ordered by how hard L1 pulls:

| run | λ_L1 | λ_NCE | ratio | final `train/G_L1` | `dice_bone` |
|---|---|---|---|---|---|
| exp2_paper | 100 | 1 | 100:1 | 0.084 | 0.369 |
| exp3b_nce_6_pix48 | 48 | 6 | 8:1 | 0.097 | **0.385** |
| exp3b_nce_20_5 | 20 | 5 | 4:1 | **0.297** | **0.038** |
| exp4_nce_max | 10 | 5 | 2:1 | **0.322** | **0.029** |

Two clusters, and nothing in between. Above the cliff, `train/G_L1` lands near 0.09 and
Dice near 0.38. Below it, `G_L1` stalls near 0.30 and Dice collapses by a factor of ten.
Halving λ_L1 from 20 to 10 changes Dice by 0.009; halving it from 48 to 20 changes Dice by
**0.347**.

The reason it is a cliff is that crossing 0.70 is a **threshold decision**, not a
continuous quantity. Bone Dice does not measure how bright the bone is. It counts pixels
on one side of a line. A model whose bone sits at 0.65 and a model whose bone sits at 0.20
score the same — zero — even though one is obviously closer. So the *smooth* degradation
in the underlying image shows up in the *discontinuous* metric as a collapse the moment
the median-plus-adversarial-pressure stops clearing 0.70.

That is a property of the metric, not an artifact of it: for a synthetic CT the threshold
is the thing that matters. A bone-window reconstruction that never reaches bone density is
not 80% useful.

## 4. One loose end: layer 0 never learns

Flagged as an observation with a hypothesis attached, not as an established cause.

`train/G_NCE_L0` across all four NCE runs:

| run | epoch 0 | epoch 199 |
|---|---|---|
| exp2_paper | 5.548 | 5.352 |
| exp3b_nce_6_pix48 | 5.460 | 5.203 |
| exp3b_nce_20_5 | 5.327 | 5.164 |
| exp4_nce_max | 5.342 | 5.089 |

Chance level for a 257-way softmax (1 positive + 256 negatives, `num_patches: 256`) is
`ln(257) = 5.549`. **Layer 0 starts at chance and ends near chance, in every run** — while
layers 1–4 fall by 60–75%. One fifth of the averaged NCE term is contributing essentially
no gradient signal.

A plausible reason, worth checking before relying on it: tap 0 is the **raw input image**
(`networks/unet.py:166-168` appends `x` itself), so each "feature" is a single scalar
pushed through `nn.Linear(1, 256)` (`patch_sampler.py:67-71`). A one-dimensional input
cannot distinguish 256 locations that share an intensity, so near-chance is expected.

The speculative part: `patch_sampler.py:129` selects the head by tap index only —
`getattr(self, f"mlp_{i}")` — so the **same** projection is applied to MRI features and
generated-CT features. At tap 0 that makes the easy optimum of
`cos(Φ(fake_CT_i), Φ(MRI_i))` sit at `fake_CT_i ≈ MRI_i`. In MRI, cortical bone is signal
void — black. If that reading is right, tap 0 is a weak *anti-bone* term.

It would be a small effect next to Proof 2, and it is not needed to explain the result.
The check is in [04](04_what_to_change.md) §4.

---

## Summary

| claim | status |
|---|---|
| PatchNCE is invariant to intensity remaps that preserve ranking | **proved** from `patch_sampler.py:133` + `patch_nce.py:98-119` |
| `λ_NCE: 5` is really 5, not 25 | **verified** — layers averaged, `patch_nce.py:144-156` |
| Bone requires clearing 0.70 (MSK/spine) or 0.775 (abdomen) normalised | **verified** — `metrics.py:212-226` + the region windows |
| L1's optimum is the conditional median, which discards minority modes outright | **derived** |
| Weak L1 → never crosses the threshold → Dice 0 with anatomy intact | **derived, and matches the panel and the numbers** |
| Tap 0 acts as an anti-bone term | **hypothesis** — consistent with the logs, not needed for the explanation, check first |
