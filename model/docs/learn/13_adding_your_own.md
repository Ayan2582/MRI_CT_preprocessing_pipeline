# Adding your own architecture or model

You have read how the five existing models are built. This document is the other
direction: where to plug a new one in, which of the two dispatch points you need,
which contract you have to satisfy, and how to decide between extending
`Pix2PixNCEModel` and writing a sibling. It ends with a worked example.

---

## 1. The two dispatch points

There are exactly two, at different levels:

| level | file | switches on | choices today |
|---|---|---|---|
| architecture | `networks/builder.py` | `model.generator.type`, `model.discriminator.type` | `unet`, `stylegan2` / `patchgan`, `stylegan2` |
| training model | `training/builder.py` | `model.name` | `pix2pix_nce`, `reggan`, `cyclegan` |

**Which one you need depends on what is different about your idea:**

```
A new network shape, trained the same way            → networks/builder.py only
A new loss term on the existing networks             → a lambda + a hook, no dispatch at all
A different training procedure                       → training/builder.py
```

`exp6_stylegan2_fitted` is the first case, `exp7_reggan` the second, `exp8_cyclegan`
the third.

### Both dispatchers are deliberately dumb

```python
# networks/builder.py:25-40
def build_generator(cfg_generator):
    gen_type = cfg_generator.get("type", "unet")

    if gen_type == "unet":
        from .unet import build_generator as build_unet     # import INSIDE the branch
        return build_unet(cfg_generator)

    if gen_type == "stylegan2":
        from .stylegan2 import build_stylegan2_generator
        return build_stylegan2_generator(cfg_generator)

    raise NotImplementedError(
        f"generator type '{gen_type}' is not implemented. Available: "
        f"{', '.join(GENERATOR_TYPES)}.")
```

Three things to preserve if you add a branch:

**Imports go inside the branch.** So `unet.py` never imports `stylegan2.py`, and an
unused architecture is never even loaded. Before StyleGAN2 arrived,
`unet.build_generator` doubled as dispatcher and constructor; adding a second
architecture there would have meant `unet.py` importing `stylegan2.py`, which is
backwards.

**The `*_TYPES` tuples are for error messages, not a registry.** There is no decorator
magic to learn. `NotImplementedError` names what is available, so a typo in a config
produces a useful message rather than a `KeyError`.

**Leaf builders stay in their own module** and know only about their own network. This
module is the only thing that knows a choice exists.

### The argument shapes are asymmetric, on purpose

```python
build_generator(cfg.model.generator)                       # just the sub-dict
build_discriminator(cfg.model, spectral, batch_size)       # the whole model block
```

The discriminator builder needs the *generator's* channel counts to derive its
conditional input width (`04_patchgan_discriminator.md` section 7), so it takes the
whole `model` block. `spectral` and `batch_size` are passed separately because they are
training-stability facts, not architectural ones.

---

## 2. The contract a training model must satisfy

Written down at `training/builder.py:19-25`:

```
optimize_parameters(batch, epoch, scaler, amp_ctx) -> {str: float}
train_mode()
gan_active(epoch) -> bool
generator_for_eval() -> nn.Module        # takes real_A, returns fake_B
set_eval_mode(net) -> nn.Module
state_dict() / load_state_dict(state)
.optimizer_G, .optimizer_D (may be None), .plan, .warmup_epochs
```

> **This is the whole reason the trainer is indifferent to what it drives.**
> `Pix2PixNCEModel` defines the surface, `RegGANModel` inherits it, and
> `CycleGANModel` implements it independently — so **the epoch loop drives four
> networks and one network through the same calls.**

Two entries repay attention:

**`generator_for_eval()`** returns a module that takes `real_A` and returns `fake_B` —
which for CycleGAN is `netG_A2B`'s EMA shadow, and for everything else is `netG`'s. The
trainer never has to know there was a second generator.

**`.optimizer_G` / `.optimizer_D`** exist because `training/trainer.py:93-95` builds
exactly two LR schedulers from them. CycleGAN chains two networks into each optimizer
(`11_cyclegan.md` section 4) specifically so those attributes keep meaning what the
trainer expects.

---

## 3. The two extension hooks — the cheap path

Before writing a new model class, check whether these are enough. They were added for
RegGAN and they cover "same step, extra term and maybe an extra network":

```python
# training/pix2pix_nce.py:236-248
def extra_G_terms(self, stats):
    """Additional generator loss terms. Returns a zero scalar in the base class."""
    return torch.zeros((), device=self.device)

# training/pix2pix_nce.py:250-261
def g_step_optimizers(self):
    """Optimizers stepped by the generator's backward."""
    return [opt for opt in (self.optimizer_G, self.optimizer_F) if opt is not None]
```

`extra_G_terms` is called inside the autocast block at `:406`, so anything it returns
joins `loss_G` and is backpropagated in the same pass. `g_step_optimizers` is collected
after the block at `:410` and every entry is stepped.

**RegGAN's entire training-step difference is those two overrides.** It does not touch
`backward_G`, `backward_D` or `optimize_parameters` (`12_reggan.md` section 6).

### When the hooks are not enough

CycleGAN needed a different `forward` (four passes), a different `backward_D` (two
discriminators), an extra device on the D side (the image pool), and a different set of
loss terms. `CycleGANModel` is therefore a **sibling** of `Pix2PixNCEModel`, not a
subclass (`training/cyclegan.py:55`), and it reuses everything at the layer below —
the builders, `GANLoss`, `masked_l1`, `DiffAugment`, `ModelEMA`, the whole trainer.

The rule of thumb:

```
The step order is the same, you are adding a term      →  extra_G_terms
The step order is the same, you are adding a network   →  + g_step_optimizers
The step order itself changes                          →  sibling class
```

---

## 4. Prefer satisfying an existing contract over branching

Two examples in this repo are worth copying as a habit.

**`StyleEncoder` implements the U-Net's tap protocol** (`networks/stylegan2.py:349-351`)
— `n_taps`, `tap_channels`, `tap_spatial`, `forward(x, tap_layers, encode_only)` — so
`compute_nce` needs no branch. PatchNCE works on StyleGAN2 with zero changes to
`training/pix2pix_nce.py`.

**`CycleGANModel` chains its two generators into one `optimizer_G`** so the trainer's
scheduler code is untouched.

In both cases the alternative was an `isinstance` check in the training loop. The
question to ask when adding something is not "where do I branch?" but **"what contract
already exists that I could satisfy?"**

---

## 5. Worked example: adding a ResNet generator

CycleGAN's paper generator is a ResNet with 9 residual blocks. Suppose you want
`generator.type: resnet` to compare against the U-Net.

### Step 1 — write the network and its leaf builder

`model/networks/resnet.py`:

```python
import torch.nn as nn
from .init import get_norm_layer, init_weights, uses_bias


class ResnetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ngf=64, n_blocks=9,
                 norm="instance", use_dropout=False):
        super().__init__()
        self.in_channels = in_channels          # the tap protocol needs this
        norm_layer, bias = get_norm_layer(norm), uses_bias(norm)
        ...
        init_weights(self)                       # N(0, 0.02) — see 01_building_blocks.md

    # --- tap protocol, so PatchNCE works without a branch -------------------
    @property
    def n_taps(self): ...
    def tap_channels(self, tap): ...
    def tap_spatial(self, tap, input_size): ...

    def forward(self, x, tap_layers=None, encode_only=False):
        """out | (out, feats) | feats — the contract in unet.py:145-165."""
        ...


def build_resnet_generator(cfg_generator):
    return ResnetGenerator(
        in_channels=cfg_generator.get("in_channels", 1),
        out_channels=cfg_generator.get("out_channels", 1),
        ngf=cfg_generator.get("ngf", 64),
        n_blocks=cfg_generator.get("n_blocks", 9),
        norm=cfg_generator.get("norm", "instance"),
        use_dropout=cfg_generator.get("dropout", False),
    )
```

Read `cfg_generator` with `.get(key, default)` throughout, so every key is optional and
`base.yaml` remains the single statement of the defaults.

### Step 2 — one import and one branch

```python
# networks/builder.py
GENERATOR_TYPES = ("unet", "stylegan2", "resnet")       # for the error message

    if gen_type == "resnet":
        from .resnet import build_resnet_generator
        return build_resnet_generator(cfg_generator)
```

### Step 3 — declare the new keys in `configs/base.yaml`

```yaml
model:
  generator:
    # ResNet only. Number of residual blocks at the bottleneck resolution.
    n_blocks: 9
```

> **This step is not optional.** `apply_overrides` rejects any dotted path not already
> present in `base.yaml` (`model/config.py:163-168`), so an undeclared key cannot be set
> with `--set` — and, more importantly, `base.yaml` is where the *why* of every knob is
> recorded. Match the surrounding comment style: what it does, and what breaks if you
> change it.

### Step 4 — write the experiment config

```yaml
# configs/exp9_resnet.yaml
_base_: base.yaml
run: { name: exp9_resnet }
model:
  generator: { type: resnet, n_blocks: 9 }
# Everything else inherited, so this is a one-variable comparison against exp2_paper.
```

Change exactly one thing. An experiment that moves the architecture *and* the loss
answers no question — `model/README.md` notes that the `exp1`↔`exp2` gap is already
confounded for exactly this reason, and it cost a run to find out.

### Step 5 — verify before spending a GPU

```bash
python model/scripts/smoke_test.py
python model/scripts/train.py --config exp9_resnet.yaml \
    --set train.n_epochs=1 data.crop_size=64
```

`smoke_test.py` exercises the builders, tiling and the registration path on CPU. Then a
one-epoch run at a small crop proves the whole loop — forward, both backwards,
validation, logging, checkpoint, sample panel — before anything queues on Kaggle. Local
torch is CPU-only, which is enough for both.

---

## 6. A checklist for anything you add

- [ ] Leaf builder reads config with `.get(key, default)` for every key
- [ ] New keys declared in `configs/base.yaml`, with a comment saying why
- [ ] `init_weights` applied — **unless** it is equalized-LR, in which case never
      (`01_building_blocks.md` section 8)
- [ ] Tap protocol implemented if PatchNCE should work with it
- [ ] Invariants checked at construction with an error naming both sides of the
      disagreement (`networks/stylegan2.py:590-596` is the model to copy)
- [ ] Anything that can be wrong while the loss curve still looks healthy is
      constrained structurally, not left to inspection
- [ ] Every logged stat `.detach()`ed at creation
- [ ] Any penalty needing `create_graph=True` gets its own unscaled fp32 step
      (`08_stabilisers_in_code.md` section 6)
- [ ] `state_dict` / `load_state_dict` cover the new state, and `load` **raises** if the
      checkpoint is missing something the live model has
- [ ] `smoke_test.py` passes, and a one-epoch run at a small crop completes

---

## See also

- `05_the_training_step.md` — the step the hooks plug into
- `12_reggan.md` — the extension case, done via the hooks
- `11_cyclegan.md` — the sibling case, and why the hooks were not enough
- `model/README.md` — the experiment ladder, and the discipline of one variable per rung
- `kaggle_workflow.md` — packaging and running a new experiment end to end
