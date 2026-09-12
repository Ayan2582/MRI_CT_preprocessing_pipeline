# PatchNCE — teaching correspondence without pixel alignment

The third term in the objective is a contrastive loss. This document explains what
contrastive learning is, derives InfoNCE from scratch, then walks
`losses/patch_nce.py` and `networks/patch_sampler.py` — including the one trick the
whole thing depends on, and the bug that trick prevents, which is invisible if you
get it wrong.

`loss_function_guide.md` covers *when to change* `lambda_nce`. This covers what the
term is.

---

## 1. The failure it addresses

L1 compares pixel `(i, j)` in the prediction to pixel `(i, j)` in the target. That
comparison is only meaningful if the two are the same anatomical point.

In this dataset they are approximately, not exactly. The MRI/CT pairs were aligned by
hand, and a residual misregistration remains. Where the pair is off by a pixel or
two, L1 punishes the generator for putting the edge in the right place — and the
generator's cheapest way to reduce that punishment is to blur the edge until it is
wrong everywhere and catastrophically wrong nowhere.

> **A caveat this project is explicit about.** `model/README.md` §1 retracts the
> simple version of this argument — alignment *was* fixed by hand, so PatchNCE here
> is a hypothesis under test, not an established fix. `exp3_nce_heavy` exists to find
> out whether the term earns its cost. Read that section before treating the
> motivation above as settled.

What you want is a term that says "the generated image should have the *same
structure* as the input at each location" without requiring pixel-exact agreement.
Structure, not intensity. That is what a contrastive loss on features does.

---

## 2. What contrastive learning is

Contrastive learning turns "these two things should be similar" into a
**classification problem**, which is trainable in a way that a raw similarity is not.

Take one location in the input — call it the **anchor**. Take the same location in the
output — the **positive**. Take a batch of other locations — the **negatives**.

Then ask the network: *given the anchor, which of these candidates is the positive?*

If the network can pick it out, the anchor and its positive share something the
negatives do not. That is what "corresponding structure" means, operationally, and
crucially the network is free to represent that structure however it likes — it never
has to agree pixel for pixel.

### Deriving InfoNCE

Score each candidate by similarity to the anchor. If features are L2-normalised, the
dot product *is* the cosine similarity:

```
s_i = q · k_i          where |q| = |k_i| = 1,  so  s_i ∈ [−1, 1]
```

Now treat those scores as classification logits and apply softmax cross-entropy, with
the positive as the correct class:

```
L = −log(  exp(s_pos / τ)  /  ( exp(s_pos / τ) + Σ_neg exp(s_neg / τ) )  )
```

That is InfoNCE. It is exactly `nn.CrossEntropyLoss` applied to a vector of cosine
similarities. **The whole implementation is "build that logit vector correctly."**

`τ` is the temperature. It divides the logits before the softmax:

| τ | effect |
|---|---|
| low (0.07) | sharpens the distribution; the hardest negatives dominate the gradient |
| high | flattens it; all negatives contribute roughly equally |

0.07 is the CUT/SimCLR value and this project's default. It is **not a free
parameter** — the source is emphatic (`losses/patch_nce.py:62-67`) that it interacts
with the effective loss scale, so changing it changes what `lambda_nce` means.

---

## 3. Minimal version — build it yourself

```python
import torch, torch.nn.functional as F

def info_nce(q, k, tau=0.07):
    """q, k: [n, C] L2-normalised. k[i] is the positive for q[i]."""
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1).detach()          # keys carry no gradient

    l_pos = (q * k).sum(dim=1, keepdim=True)    # [n, 1]  each with its own key
    l_neg = q @ k.t()                           # [n, n]  each with every key
    l_neg.masked_fill_(torch.eye(len(q), dtype=torch.bool), -10.0)   # kill diagonal

    logits = torch.cat([l_pos, l_neg], dim=1) / tau                  # [n, 1+n]
    target = torch.zeros(len(q), dtype=torch.long)                   # positive at col 0
    return F.cross_entropy(logits, target)

q = torch.randn(64, 32)
print(float(info_nce(q, q.clone())))     # ~0.0  identical -> trivially solvable
print(float(info_nce(q, torch.randn(64, 32))))   # large  unrelated -> unsolvable
```

Two sanity checks worth running: identical features give near-zero loss, unrelated
features give a large one. If your implementation does not show that gap, the logit
layout is wrong.

---

## 4. The real one: `losses/patch_nce.py:80-120`

Line for line, it is the minimal version with the batch dimension handled:

```python
feat_k = feat_k.detach()                                          # :95

l_pos = torch.bmm(feat_q.view(n_total, 1, -1),
                  feat_k.view(n_total, -1, 1)).view(n_total, 1)   # :98-99

q = feat_q.view(batch, n_patches, dim)
k = feat_k.view(batch, n_patches, dim)
l_neg = torch.bmm(q, k.transpose(2, 1))                           # :106-108  [B,n,n]

diagonal = torch.eye(n_patches, device=..., dtype=torch.bool)
l_neg.masked_fill_(diagonal[None, :, :], -10.0)                   # :112-113
l_neg = l_neg.view(-1, n_patches)

logits = torch.cat((l_pos, l_neg), dim=1) / self.temperature      # :117
loss = self.cross_entropy(logits, torch.zeros(n_total, dtype=torch.long, ...))
```

### Four decisions

**Keys are detached** (`:95`). The contrastive signal should shape the *generated*
output, not drag the encoding of the fixed input around to make the problem easier.
Skipping this lets the model reduce the loss by degrading its own representation —
it makes the classification task easier instead of making the image better.

**Negatives come only from the same image** (`:106-108`). Reshaping `[B*n, C]` to
`[B, n, C]` and doing a batched matmul means an anchor's negatives are other
*locations in its own slice*, never locations in another patient's slice. That is
what makes the loss a statement about **spatial correspondence** rather than about
patient identity — a much easier and much less useful question.

**The diagonal is masked to `−10.0`, not removed** (`:112-113`). The `[i, i]` entry of
`l_neg` is the positive again; leaving it would ask the model to be dissimilar from
itself. Filling with `−10.0` before dividing by `τ = 0.07` gives a logit of about
`−143`, whose softmax weight is effectively zero. It stays in the denominator with
negligible weight — cheaper than a gather, and numerically equivalent.

**The positive sits at column 0**, so the cross-entropy target is a vector of zeros
(`:119`). No index bookkeeping.

### `self.batch_size` is set from outside, and has to be

```python
# losses/patch_nce.py:70-78
self.batch_size = 1     # set by the caller before each forward
```

A `[B*n, C]` tensor is genuinely ambiguous between `(B=2, n=256)` and `(B=1, n=512)`.
`PatchNCECollection.forward` assigns `crit.batch_size = batch_size` before each call
(`:147`).

> ⚠️ **Guessing wrong silently mixes negatives across images — and the loss still goes
> down.** You would be training a working model on the wrong question, with no symptom.
> This is the second time in this codebase a design exists purely to make a silent
> failure impossible; it will not be the last.

### Layers are averaged, not summed (`:156`)

```python
return total / max(1, len(self.losses)), per_layer
```

Mean over taps, so the magnitude of the term — and therefore the meaning of
`lambda_nce` — is **independent of how many taps are configured**. Changing
`loss.nce.layers` from five taps to three does not silently rescale the loss you spent
five experiments tuning.

Each tap also gets its own `PatchNCELoss` instance so per-layer values can be logged
separately (`:123-136`). A layer whose loss refuses to move is a useful signal that
the tap is too deep — see section 6.

---

## 5. `PatchSampleF` — sampling and projection

`networks/patch_sampler.py` does the other half: pick locations, project them, and
normalise. It lives in `networks/` rather than `losses/` because it has trainable
parameters.

### The MLP head (`:61-77`)

```python
nn.Sequential(nn.Linear(channels, nce_dim), nn.ReLU(inplace=True),
              nn.Linear(nce_dim, nce_dim))
```

One head per tap, registered as `mlp_0`, `mlp_1`, … via `setattr`. Two linear layers
projecting each tap's raw channel count to a common `nce_dim: 256`.

**Why project at all?** The taps have different widths (1, 64, 128, 256, 512), and the
contrastive comparison needs a consistent space with a consistent notion of cosine
similarity. The head also gives the loss somewhere to put information that is useful
for the contrastive task but that you do not want forced into the encoder's features.

### The sampling loop (`:108-134`)

```python
flat = feat.permute(0, 2, 3, 1).flatten(1, 2)        # [B,C,H,W] -> [B,H*W,C]

if patch_ids is not None:
    ids = patch_ids[i].to(flat.device)               # REUSE
else:
    n = min(int(num_patches), h * w)                 # clamp
    ids = torch.randperm(h * w, device=feat.device)[:n]

patch = flat[:, ids, :].flatten(0, 1)                # [B*n, C]
if self.use_mlp:
    patch = getattr(self, f"mlp_{i}")(patch)
sampled.append(F.normalize(patch, p=2, dim=1))       # L2 -> dot == cosine
```

The permute-then-flatten turns a feature map into one row per spatial location. One
`randperm` is drawn and **shared across the batch** — matching reference CUT, and the
mechanism behind "negatives come from the same image".

`F.normalize` here is what licenses treating the dot products in `patch_nce.py` as
cosine similarities. The two files are one algorithm split across two modules.

---

## 6. The trick everything depends on: `compute_nce`

```python
# training/pix2pix_nce.py:274-278
feat_k = self.netG(source, tap_layers=layers, encode_only=True)   # real MRI
feat_q = self.netG(target, tap_layers=layers, encode_only=True)   # generated CT

k_pool, sample_ids = self.netF(feat_k, self.plan.num_patches, None)
q_pool, _          = self.netF(feat_q, self.plan.num_patches, sample_ids)
```

**The first call returns its `patch_ids`. The second call is handed them back.** Both
encodings therefore sample the *identical* spatial locations, which is the only reason
`feat_q[i]` and `feat_k[i]` are a positive pair at all.

> ⚠️ **Sample independently and every pair becomes a negative — and the loss still
> decreases.** The docstring at `patch_sampler.py:12-18` says exactly this. The model
> learns to make patches distinguishable from each other, which is a coherent task
> with a falling loss curve and no relationship to the thing you wanted. There is no
> symptom in the logs.

Note also that **both images go through the same encoder** — the generator's own.
There is no separate feature extractor to keep in sync.

### The lazy construction chain

Three objects cannot exist until the first NCE forward has run, because their shapes
depend on the encoder's channel counts:

```
first compute_nce call
  → netF.create_mlp(feats)          heads sized from the tapped widths
  → _ensure_optimizer_F()           optimizer over parameters that now exist
  → backward_G collects optimizers  AFTER the autocast block
```

`_ensure_optimizer_F` (`training/pix2pix_nce.py:181-193`) is idempotent and returns
early when `use_mlp: false`, where `PatchSampleF` is a pure sampler with nothing to
train. `05_the_training_step.md` section 5 covers why the optimizer list is collected
where it is.

`PatchSampleF` also has to handle checkpoints written before the heads existed: its
`load_state_dict` (`:148-164`) **reconstructs the head widths from the checkpoint's key
names** when the local instance has not built them yet.

### The tap-depth warning

`_validate_nce_taps` (`training/pix2pix_nce.py:159-179`) runs at construction:

```
NCE tap 5 yields 8x8 = 64 locations at crop 256, fewer than num_patches=256.
Sampling will be clamped to 64, which weakens this tap's contrastive signal.
```

A tap deeper than the image can support degenerates into sampling the same handful of
locations repeatedly. The loss still decreases. Checking at startup converts a silent
quality bug into a visible warning. The arithmetic is in `03_unet_generator.md`
section 5.

---

## 7. What actually changes in the objective

Everything above adds one term:

```
L_total = λ_gan · L_cGAN + λ_L1 · L_L1 + λ_NCE · L_PatchNCE
```

The ladder that tests whether it earns its place:

| config | λ_L1 | λ_NCE | question |
|---|---|---|---|
| `exp1_pix2pix` | 100 | 0 | the baseline with no contrastive term |
| `exp2_paper` | 100 | 1 | the textbook weights |
| `exp3_nce_heavy` | 25 | 5 | does shifting weight from L1 to NCE help? |
| `exp4_nce_max` | 10 | 5 | how far can L1 be backed off before anatomy drifts? |

`loss_function_guide.md` has the tuning guidance and `model/README.md` has the
hypothesis these were built to test.

---

## See also

- `03_unet_generator.md` — the tap protocol these features come from
- `05_the_training_step.md` — where `compute_nce` is called in the G step
- `loss_function_guide.md` — what changing `lambda_nce` does, and when to
- `10_stylegan2_adapted.md` — the StyleGAN2 encoder implements the same tap contract
