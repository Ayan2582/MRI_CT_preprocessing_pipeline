# 04 — The unpaired split is anatomically lopsided

**Finding 3.** CycleGAN's justification requires the two domains to be two styles of
the *same* content. Seed 1337 gave you two halves with materially different anatomy
mixes, and one region with nine target images.

Everything below is measured, not argued. Reproduce it with:

```
python review_test_Folder/evidence/split_audit.py
```

---

## The failure

`data/dataset.py:334-361` partitions the 32 training subjects into two disjoint halves
by patient, and draws MRI only from the first and CT only from the second:

```python
order = list(subjects)
np.random.RandomState(self._split_seed).shuffle(order)
half = len(order) // 2
subjects_a, subjects_b = set(order[:half]), set(order[half:])
```

That design decision is the *good* part of exp8 and the docstring defends it well — row
shuffling would leave the model able to see both modalities of the same patient, which
would make "unpaired" a claim the experiment cannot support. Splitting by patient is
right.

The problem is what the split happened to contain.

| region | MRI pool (A) | CT pool (B) | skew | 1 unit = |
|---|---|---|---|---|
| abdomen | 464 (49.8%) | 283 (37.4%) | 0.75× | 400 HU |
| **brain** | 213 (22.9%) | **319 (42.2%)** | **1.84×** | 80 HU |
| musculoskeletal | 236 (25.3%) | 145 (19.2%) | 0.76× | 500 HU |
| **spine** | 18 (1.9%) | **9 (1.2%)** | 0.62× | 500 HU |
| **total** | **931** | **756** | | |

## Consequence 1: brain is 23% of the input and 42% of the target

`D_B` is trained to recognise the CT pool's marginal distribution — and that pool is
**42% brain**, against a generator input stream that is only 23% brain. There is no
per-region matching to correct this; `dataset.py:385` draws the CT uniformly at random:

```python
a = self.domain_A[index % len(self.domain_A)]
b = self.domain_B[np.random.randint(len(self.domain_B))]
```

So an abdomen MRI is routinely batched against a brain CT, and `D_B` — unconditional,
by the refusal at `cyclegan.py:63-83` — is only ever asked "is this a real CT?", never
"is this a real CT *of that*".

Now combine with 03. Brain CT is windowed 0–80 HU, the narrowest window in the set, so
brain slices are the ones where cortical bone **saturates flat white**. Forty-two
percent of what `D_B` calls "real" is an image with a large hard-saturated bright region
and a low-texture grey interior. The cheapest gradient available to the generator, on
*any* input, is to produce more saturated bright area.

That is the epoch-109 musculoskeletal panel, exactly: the muscle compartment painted
bone-white, with the marrow cavity left dark. The generator learned the texture prior of
the majority region and applied it to a minority region where it is anatomically absurd.
Both MSK slices got *worse* as this happened (0.059 → 0.070, 0.059 → 0.069), which is
what "learning the wrong prior harder" looks like in a metric.

**Where CycleGAN's assumption breaks.** The method assumes `p(A)` and `p(B)` differ in
*style*, not in *content*. Here they differ in both, and there is no term in the
objective that can tell the difference — the adversarial loss on the marginal is
satisfied by matching the content mix as much as by matching the modality.

## Consequence 2: spine has nine target images

The CT pool contains **9 spine slices**. Not nine patients — nine 2-D images, 1.2% of
the pool.

`D_B` therefore has, to a good approximation, never seen a spine CT. It cannot tell the
generator what one looks like, because it does not know. And with no L1 term, `D_B` is
the *only* thing that could.

The panels confirm it. Both spine slices at epoch 199 are still recognisably the input
MRI with a global tone shift — the vertebral bodies are still MRI-bright-marrow rather
than CT-bright-cortex, the discs are still visible as discs. Their MAE moved 0.119 →
0.102 and 0.118 → 0.103, which is about what a global brightness/contrast fit alone
would buy. Nothing anatomical happened to the spine in 200 epochs.

This is not really a CycleGAN failure. Spine is 55 of 2161 manifest rows (2.5%) and 27
of 1687 training rows; halving it leaves too little to train anything unpaired. exp8
should not have been asked to do spine.

## Consequence 3: the exp1 comparison has a confound the ladder does not mention

exp1 trains on all **1687** paired slices. exp8's epoch is `max(931, 756) = 931` items
(`dataset.py:377-381`), and the CT half contributes 756 distinct images sampled with
replacement.

`exp8_cyclegan.yaml` is upfront about this — *"The halved data is the honest price"* —
and that is fair. But it means the exp8-vs-exp1 gap is **pairing plus a 45% cut in MRI
slices plus a 55% cut in CT slices**, on a dataset small enough (33 subjects) that data
volume is plausibly the binding constraint. The gap is real; attributing all of it to
the pairing is not supported.

## The part that is not luck

Re-running the audit with other seeds moves these numbers around, and a kinder seed
exists. But the structural problems do not depend on the seed:

- **Spine cannot be rescued by any seed.** 27 training slices halved is 13 or 14 on the
  CT side at best.
- **Some skew is guaranteed.** With 16 subjects per side and 4 regions of very unequal
  size, an even anatomy split would be a coincidence.
- **The sampler is anatomy-blind by construction** (`dataset.py:385`), so whatever skew
  the seed produces is applied uniformly to every input.

Re-rolling the seed until the table looks balanced would also be seed-shopping on the
training set, which is not a result. 06 §3 fixes the sampler instead.
