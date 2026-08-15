# The Kaggle notebook, cell by cell

What `kaggle_train.ipynb` does, what each cell is for, and — the question this
document exists to answer — **exactly which model trained and with what
parameters**.

---

## The short answer for the run you just finished

You trained **plain pix2pix**. No PatchNCE.

| | |
|---|---|
| **Generator** | U-Net, 8 down / 8 up, 64 base filters, InstanceNorm, tanh output — **54.4M params** |
| **Discriminator** | 70×70 conditional PatchGAN, 3 layers, 64 filters, spectral norm — **2.8M params** |
| **Loss** | `1.0 × L_cGAN(lsgan) + 100.0 × L_L1` — **λ_NCE was 0**, so no contrastive term and no projection heads were built |
| **Optimiser** | Adam, lr 2e-4, β=(0.5, 0.999), constant to epoch 100 then linear decay to 0 |
| **Batch / epochs** | 8 × 200 epochs, 210 iterations/epoch |
| **Input** | random 256×256 crop, random h-flip, scaled to [−1, 1] |
| **Stabilisers on** | spectral norm, EMA (0.999), DiffAugment, label smoothing (0.9) |
| **Stabilisers off** | TTUR, R1 |
| **Warm-up** | 5 epochs L1-only before the discriminator switched on |
| **Selected by** | lowest `val/mae_norm` on the **EMA** generator |
| **Config hash** | `47e03aded350` |

The full parameter list, with the origin of every value, is in §3.

---

## 1. Where "what trained" is recorded

**You never have to guess.** Every run writes the fully-resolved configuration to

```
runs/<run_name>/config.resolved.yaml
```

That file is the complete, flattened truth — base defaults, experiment file and
notebook overrides already merged. It is the first thing to open when you come
back to a run.

Three other places carry the same information:

- **`train.log`** — line 1 prints the objective, e.g.
  `L_total = 1*L_cGAN(lsgan) + 100*L_L1   [pix2pix]`
- **the checkpoint** — `state['model']['loss_plan']` holds the three λs and the
  nickname, so a stray `.pt` still identifies itself
- **the config hash** — a 12-character fingerprint of everything semantically
  meaningful. Paths, device, worker count and AMP are excluded, so the same
  experiment run locally and on Kaggle produces the *same* hash. Yours printed
  `47e03aded350` in the sanity cell and again in the checkpoint.

---

## 2. Cell by cell

### Cell 2 — get the code

Either clones the GitHub repo or points `REPO_DIR` at the uploaded
`mri-ct-model-code` dataset, and puts it on `sys.path`. Nothing else.

### Cell 3 — environment check

Prints the torch version, whether CUDA is available, and the GPU name. This is
where a **P100 would fail** later with `no kernel image is available for
execution on the device` — recent PyTorch builds dropped Pascal (sm_60). Use T4.

### Cell 5 — find the dataset

Globs `/kaggle/input/*/manifest.csv` and derives `DATA_ROOT` from wherever it
lands, so you never hardcode a mount path. Prints the pair and subject counts —
**it must say 2161 pairs, 44 subjects**. Anything else means an incomplete upload.

### Cell 6 — ⭐ THE CELL THAT DECIDES WHAT TRAINS

This is the one that answers your question, and the only one you normally edit.

```python
CONFIG = 'exp1_pix2pix.yaml'      # <- WHICH EXPERIMENT
RESUME = 'auto'
OVERRIDES = [ ... ]               # <- WHAT IS CHANGED FOR KAGGLE
```

`CONFIG` selects one of five experiment files. Each is a **delta** on
`base.yaml`, which holds every default:

| config | λ_GAN | λ_L1 | λ_NCE | what it is |
|---|---|---|---|---|
| `exp0_l1_only` | 0 | 100 | 0 | L1 regression, no discriminator built |
| `exp1_pix2pix` | 1 | 100 | 0 | **← what you ran** |
| `exp2_paper` | 1 | 100 | 1 | + PatchNCE at textbook weight |
| `exp3_nce_heavy` | 1 | 50 | 2 | shifts weight onto PatchNCE |
| `exp4_nce_max` | 1 | 10 | 5 | weak L1 anchor; hallucination probe |

`OVERRIDES` are Kaggle-specific: where the data is, where output goes, batch
size, epochs, workers, AMP. **They do not change the model.**

You can override any config key from here, which is how ablations are run
without touching a file:

```python
OVERRIDES += ['loss.lambda_nce=0']    # turn exp2 into plain pix2pix
OVERRIDES += ['train.lr=1e-4']        # different learning rate
```

A key not declared in `base.yaml` is **rejected**, so a typo stops the run
rather than silently training something else.

### Cell 8 — sanity check

Builds the config, manifest, split and datasets, then loads one sample. Run it
before the training cell every time — it costs seconds and catches a bad mount
before you spend GPU hours. Expect:

```
A (1, 256, 256) B (1, 256, 256) range (-1.0, 1.0)
config hash 47e03aded350
{'train': 1687, 'val': 230, 'test': 244}
```

`A` is the MRI (input), `B` is the CT (target) — the pix2pix convention.

### Cell 10 — train

```python
trainer = Trainer(cfg, datasets)
trainer.maybe_resume(RESUME)
trainer.train()
```

Three lines. Everything — networks, optimisers, stabilisers, schedules,
checkpointing, validation, sample panels — comes from the config.

### Cell 12 — curves

Reads `train_log.jsonl` and plots the selection metric, D accuracy, and the
loss terms. See `training_log_reference.md` for how to read them.

### Cell 13 — sample panel

Displays the latest `samples/epoch_XXXX.png`: MRI | real CT | synth CT | error,
on the **same fixed validation slices** every epoch so progress is comparable
rather than anecdotal.

### Cells 15–17 — export

Lists the run's files, then bundles the logs and sample panels into one zip,
writes an inference-only copy of the EMA generator, and produces download links.

### Cell 19 — resume

Commented out. Uncomment in a **follow-up** session, after adding the previous
run as a Notebook Output input, to restore `last.pt` before training.

### Cell 21 — test-set evaluation

Commented out on purpose. Run **once**, at the very end, after the configuration
is frozen. Six test subjects will not survive being looked at repeatedly.

---

## 3. Every parameter of the run you finished, and where it came from

Resolved from `exp1_pix2pix.yaml` + your notebook overrides. Hash `47e03aded350`.

### Loss — what made this "pix2pix"

| parameter | value | source |
|---|---|---|
| `loss.gan_mode` | `lsgan` | base.yaml |
| `loss.lambda_gan` | `1.0` | base.yaml |
| `loss.lambda_l1` | `100.0` | base.yaml |
| **`loss.lambda_nce`** | **`0.0`** | **exp1_pix2pix.yaml** |

That single overridden value is the entire difference between exp1 and exp2. At
zero, no projection heads and no third optimizer are built at all — the run is
structurally pix2pix, not pix2pix-with-a-disabled-term. All the `loss.nce.*`
settings below it are inert.

**Note the 100×.** L1 contributes ~99% of the gradient magnitude. The
discriminator is a texture correction on a regression model, not the main event.

### Generator

| parameter | value | source |
|---|---|---|
| `model.generator.type` | `unet` | base.yaml |
| `model.generator.in_channels` / `out_channels` | `1` / `1` | base.yaml |
| `model.generator.ngf` | `64` | base.yaml |
| `model.generator.num_downs` | `8` | base.yaml |
| `model.generator.norm` | `instance` | base.yaml |
| `model.generator.dropout` | `True` | base.yaml |
| `model.generator.dropout_at_eval` | `False` | base.yaml |

`num_downs: 8` takes a 256×256 crop to 1×1 at the bottleneck. Channels run
64 → 128 → 256 → 512 → 512 → 512 → 512 → 512.

### Discriminator

| parameter | value | source |
|---|---|---|
| `model.discriminator.type` | `patchgan` | base.yaml |
| `model.discriminator.n_layers` | `3` | base.yaml |
| `model.discriminator.ndf` | `64` | base.yaml |
| `model.discriminator.conditional` | `True` | base.yaml |

`n_layers: 3` gives the 70×70 receptive field and a 30×30 output grid.
`conditional: True` means D sees `cat[MRI, CT]`, so it judges correspondence,
not just plausibility.

### Stabilisers

| parameter | value | source |
|---|---|---|
| `stabilizers.spectral_norm_d` | `True` | base.yaml |
| `stabilizers.ema.enabled` / `.decay` / `.start_epoch` | `True` / `0.999` / `1` | base.yaml |
| `stabilizers.diffaug.enabled` / `.policy` | `True` / `color,translation,cutout` | base.yaml |
| `stabilizers.label_smoothing.enabled` / `.real_target` | `True` / `0.9` | base.yaml |
| `stabilizers.ttur.enabled` | `False` | base.yaml |
| `stabilizers.r1.enabled` | `False` | base.yaml |

What each one fixes is in `training_strategies.md`, Part 3.

### Training schedule

| parameter | value | source |
|---|---|---|
| `train.n_epochs` | `200` | **notebook OVERRIDES** |
| `train.batch_size` | `8` | **notebook OVERRIDES** |
| `train.lr` | `0.0002` | base.yaml |
| `train.beta1` / `beta2` | `0.5` / `0.999` | base.yaml |
| `train.lr_policy` | `linear_decay` | base.yaml |
| `train.lr_decay_start_frac` | `0.5` | base.yaml |
| `train.gan_warmup_epochs` | `5` | base.yaml |
| `train.grad_clip` | `0.0` (off) | base.yaml |

Constant lr to epoch 100, then linear decay to zero by epoch 200. Epochs 0–4 ran
L1-only with the discriminator frozen.

### Data

| parameter | value | source |
|---|---|---|
| `data.crop_size` | `256` | base.yaml |
| `data.pad_multiple` | `256` | base.yaml |
| `data.drop_background` | `True` | base.yaml |
| `data.augment.hflip` | `True` | base.yaml |
| `data.root` / `manifest` / `splits` | Kaggle mount paths | **notebook OVERRIDES** |

Train: pad if smaller than 256, random 256×256 crop, random h-flip, → [−1, 1].
Val: whole slice zero-padded to a multiple of 256, padding masked out of metrics.

### Evaluation

| parameter | value | source |
|---|---|---|
| `eval.selection_metric` / `selection_mode` | `mae_norm` / `min` | base.yaml |
| `eval.use_ema` | `True` | base.yaml |
| `eval.bone_threshold_hu` | `150.0` | base.yaml |
| `eval.bone_metrics_exclude_regions` | `['brain']` | base.yaml |
| `eval.every` / `batch_size` | `1` / `1` | base.yaml |

### Runtime — excluded from the config hash

| parameter | value | source |
|---|---|---|
| `runtime.device` | `auto` | base.yaml |
| `runtime.num_workers` | `2` | **notebook OVERRIDES** |
| `runtime.amp` | `True` | **notebook OVERRIDES** |
| `run.seed` | `1337` | base.yaml |
| `run.out_dir` | `/kaggle/working/runs` | **notebook OVERRIDES** |

These describe *where* a run happens, not *what* it computes, so changing them
does not change the hash. That is why moving from your laptop to Kaggle keeps
the fingerprint identical and the runs stay comparable.

---

## 4. Answering "what did I train?" for any run

```python
# from the resolved config — the complete truth
print(open('runs/exp1_pix2pix/config.resolved.yaml').read())

# or just the objective, from the checkpoint
import torch
ck = torch.load('runs/exp1_pix2pix/checkpoints/best.pt',
                map_location='cpu', weights_only=False)
print(ck['model']['loss_plan'])
# {'lambda_gan': 1.0, 'lambda_l1': 100.0, 'lambda_nce': 0.0, 'nickname': 'pix2pix'}
print('epoch', ck['epoch'], '| best', ck['best_value'], '| hash', ck['config_hash'])
```

The same `loss_plan` is embedded in the inference-only export, so even a bare
`*_generator.pt` tells you what produced it.

---

## 5. Running the next experiment

Change **one line** in cell 6, then re-run cells **6 → 8 → 10**:

```python
CONFIG = 'exp0_l1_only.yaml'
```

Leave `OVERRIDES`, the split and `run.seed` alone — same data, same split, same
seed is what makes two runs comparable. Outputs land under a different
`run.name`, so nothing collides, and `RESUME='auto'` safely starts fresh
because the new run directory has no `last.pt`.

---

## See also

- `training_log_reference.md` — every number the trainer prints
- `gan_evaluation_guide.md` — deciding whether a run is better than another
- `loss_function_guide.md` — the three loss terms and their λs
- `training_strategies.md` — objective variants and the six stabilisers
- `kaggle_workflow.md` — package, upload, train, resume
