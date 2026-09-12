# Learning to build these models

Thirteen documents that build every GAN in `model/` from the ground up. The existing
guides in `model/docs/` tell you how to *run* and *judge* these models; this set tells
you how they *work*, and how you would write them yourself.

Every document follows the same three beats:

1. **The failure it prevents** — what goes wrong without this piece, stated before the fix.
2. **Minimal version — build it yourself** — 15 to 30 lines of runnable PyTorch, deliberately simplified. Every snippet in this set has been executed; where one prints a shape or a number, that is the real output.
3. **The real one** — the actual repo file walked in reading order, cited as `file.py:line`.

---

## Reading order

The order is pedagogical, not the order of the experiment ladder. The ladder is an
*ablation* order and it starts from a fully built system; this starts from nothing.

### Foundations

| # | document | you will be able to |
|---|---|---|
| 01 | [01_building_blocks.md](01_building_blocks.md) | compute a receptive field, explain InstanceNorm vs BatchNorm, and say why the weight init is load-bearing |
| 02 | [02_gan_from_zero.md](02_gan_from_zero.md) | derive why L1 blurs, and why a discriminator fixes it |

### pix2pix, piece by piece

| # | document | you will be able to |
|---|---|---|
| 03 | [03_unet_generator.md](03_unet_generator.md) | write a U-Net, and explain what its skip connections buy |
| 04 | [04_patchgan_discriminator.md](04_patchgan_discriminator.md) | derive the 70x70 receptive field, and say why D emits a grid |
| 05 | [05_the_training_step.md](05_the_training_step.md) | write the D-then-G step, and explain every `detach()` in it |
| 06 | [06_the_trainer.md](06_the_trainer.md) | build an epoch loop that survives a killed session |

### The other loss terms

| # | document | you will be able to |
|---|---|---|
| 07 | [07_patchnce.md](07_patchnce.md) | write InfoNCE from scratch, and name the bug that makes it silently meaningless |
| 08 | [08_stabilisers_in_code.md](08_stabilisers_in_code.md) | implement EMA, DiffAugment, R1 and path-length — and get the mixed precision right |

### A second architecture

| # | document | you will be able to |
|---|---|---|
| 09 | [09_stylegan2_core.md](09_stylegan2_core.md) | explain W space, equalized LR, and write a modulated convolution |
| 10 | [10_stylegan2_adapted.md](10_stylegan2_adapted.md) | say what changes when StyleGAN2 translates instead of samples, and what each change costs |

### Two other framings of the problem

| # | document | you will be able to |
|---|---|---|
| 11 | [11_cyclegan.md](11_cyclegan.md) | explain why unpaired translation is underdetermined, and what cycle consistency does and does not fix |
| 12 | [12_reggan.md](12_reggan.md) | write a spatial transformer, and say why its head must start at zero |

### Building on it

| # | document | you will be able to |
|---|---|---|
| 13 | [13_adding_your_own.md](13_adding_your_own.md) | add an architecture or a training model without touching the trainer |

---

## If you are short on time

| you want to | read |
|---|---|
| understand the model that is currently training | 02, 03, 04, 05 |
| understand what `lambda_nce` actually computes | 07 |
| debug a run that trains but produces nothing | 01 §7-8, 08 §6, 12 §4 |
| add a new architecture | 13, then 03 as the template |
| know why `exp7` is worth running before `exp3` | 12 §8 |

---

## Where each source file is explained

| file | document |
|---|---|
| `networks/init.py` | 01 |
| `networks/unet.py` | 03 |
| `networks/patchgan.py` | 04 |
| `networks/patch_sampler.py` | 07 |
| `networks/stylegan2.py` | 09, 10 |
| `networks/tiling.py` | 10 §6 |
| `networks/registration.py` | 12 |
| `networks/builder.py` | 13 §1 |
| `losses/gan_loss.py` | 02 |
| `losses/patch_nce.py` | 07 |
| `losses/registration.py` | 12 §5 |
| `losses/builder.py` | 05 §7-8 |
| `training/pix2pix_nce.py` | 05, and 08 for the regularisers |
| `training/trainer.py` | 06 |
| `training/ema.py`, `training/diffaug.py` | 08 |
| `training/cyclegan.py`, `training/image_pool.py` | 11 |
| `training/reggan.py` | 12 |
| `training/builder.py` | 13 §2 |

---

## A theme worth naming up front

The same design principle recurs in almost every file, and it is the most transferable
thing in this codebase:

> **When a component can be wrong in a way that leaves the loss curve looking healthy,
> constrain it structurally rather than trusting yourself to notice.**

You will meet it as the StyleGAN2 init warning (01 §8), the shared `patch_ids` in
PatchNCE (07 §6), `LossPlan`'s three startup errors (05 §7), the zero-initialised
registration head (12 §4), and the CSV that is rebuilt rather than appended (06 §5). A
crash is cheap. A run that completes 200 epochs and produces nothing is not.

---

## See also

- `../gan_evaluation_guide.md` — how to tell whether your GAN is improving
- `../loss_function_guide.md` — what each λ changes, and when to change it
- `../training_strategies.md` — the objective variants and the stabilisers, argued as choices
- `../training_log_reference.md` — every number the trainer prints
- `../kaggle_workflow.md` — package, upload, train, resume
- `../../README.md` — the experiment ladder and the dataset facts that shaped the design
