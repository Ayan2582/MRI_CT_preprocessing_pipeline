# Kaggle workflow

Package → upload → train → resume → bring the results home.

Local torch here is **CPU-only**, so this machine runs smoke tests and Kaggle
runs the real training. The code is identical on both; only `data.root` differs.

---

## 1. Prepare the data (once, locally)

```bash
python model/scripts/make_split.py            # if you haven't already
python model/scripts/package_for_kaggle.py --dry-run
python model/scripts/package_for_kaggle.py
```

Expected output:

```
  pairs        2161
  subjects     44
  files        4322
  total size   1.7 GB
```

1.7 GB is well inside Kaggle's 20 GB dataset limit, so the arrays stay `float32`
— halving to `float16` is not worth the precision loss on `[0,1]` data.

### What the packager fixes

Three things are true of `qc_workspace/output` that must not be true of the
uploaded copy:

1. **Absolute Windows paths.** `metadata.csv` holds
   `C:\Users\moham\...\IM0.npy`, meaningless on Linux. The packaged manifest
   carries paths relative to the dataset root, forward-slashed.
2. **A dependency on `Preprocessing/`.** The per-region HU windows that make the
   CT arrays invertible to Hounsfield units live in `pipeline_config.py`, which
   will not exist on Kaggle. They are baked into the manifest as `hu_min` /
   `hu_max` columns.
3. **A split that could drift.** `splits.json` is **copied, not recomputed**. A
   regenerated split would validate on different subjects, and the remote run
   would not be comparable to the local one.

Only the 2161 QC-accepted pairs are copied — not previews, not the `.npz` cache,
not rejected slices.

---

## 2. Authenticate the Kaggle CLI (once)

**Phone-verify your Kaggle account first** — Settings → Phone Verification.
**GPU access is locked until you do.**

Then Settings → API → **Create New Token**.

> Depending on the UI version, Kaggle either downloads a `kaggle.json` or just
> **displays the key inline**. They are the same thing: `kaggle.json` is a
> two-line file wrapping that key. If you were only shown a string, create the
> file yourself:
>
> ```json
> {"username":"your_kaggle_username","key":"the_long_hex_string"}
> ```
>
> `username` is the one in your profile URL (`kaggle.com/<username>`), lowercase
> — not your display name. Only the newest token is valid; creating a second
> invalidates the first.

```bash
pip install kaggle
mkdir -p ~/.kaggle          # resolves to C:\Users\<you>\.kaggle on Windows
# create/move kaggle.json into it, then:
chmod 600 ~/.kaggle/kaggle.json
python -m kaggle config view      # prints your username, never the key
```

`config view` showing your username means you are authenticated.

## 3. Upload the dataset

The packager reads your username from `~/.kaggle/kaggle.json` (or
`$KAGGLE_USERNAME`, or `--username`) and writes the dataset id for you, so
`dataset-metadata.json` normally needs no editing. It prints the id it used.

```bash
python -m kaggle datasets create -p model/kaggle_dataset --dir-mode zip
```

To update later:

```bash
python -m kaggle datasets version -p model/kaggle_dataset -m "regenerated split" --dir-mode zip
```

> ⚠️ **`--dir-mode zip` is required.** Without it the CLI silently skips
> subdirectories — `CT/` and `MRI/` would not be uploaded at all, and you would
> only find out when a training run failed on a missing slice.

> ⚠️ **Keep the dataset Private.** These are patient images. The CLI creates
> datasets private by default; confirm on the dataset page under Settings.

**Verify before going further:** open the dataset page and check the file browser
shows `CT/` and `MRI/` as browsable folders, not as two unextracted `.zip` files.

---

## 4. Get the code onto Kaggle

**A — clone from GitHub** (needs Settings → Internet **On**, and a public repo).

Push `model/` first:

```bash
git add .gitignore model/
git commit -m "Add pix2pix + PatchNCE model"
git push origin main
```

That pushes the 44 code files plus `manifest.csv` and `splits.json` (both
intentionally tracked); `runs/` and `kaggle_dataset/` are gitignored. The repo
contains no patient data — `qc_workspace/` and `Raw_data_mri_ct/` are ignored —
so making it public exposes nothing.

Then in the notebook:

```python
REPO_URL = 'https://github.com/Ayan2582/MRI_CT_preprocessing_pipeline.git'
```

**B — upload `model/` as a second Kaggle dataset.** Works with internet off, and
with a private GitHub repo. Nothing outside `model/` is needed for training:
`bootstrap.py` only reaches into `Preprocessing/` for the HU windows, and those
are already baked into the manifest. Point `REPO_DIR` at the mounted path and
skip the clone cell.

---

## 5. Upload the notebook and configure the session

kaggle.com/code → **New Notebook** → **File** → **Import Notebook** → **Upload**
→ pick `model/notebooks/kaggle_train.ipynb`.

Then in the right-hand panel:

| setting | value | why |
|---|---|---|
| Accelerator | **GPU T4 x2** | **not P100** — see below. The code is single-GPU, so the second T4 sits idle; that is fine |
| Internet | **On** | needed for `git clone` (not needed with option B) |
| Persistence | Files only | keeps `/kaggle/working` between runs — but see the stale-clone warning below |
| Input | **+ Add Input** → your dataset | mounts it under `/kaggle/input/` |

> **Do not pick P100.** It is the faster card on paper, and this table used to
> recommend it. Recent PyTorch builds dropped Pascal (sm_60), so a P100 session
> fails with `no kernel image is available for execution on the device` — not at
> startup, but at the first CUDA op, after you have already spent the setup time.
> `docs/notebook_walkthrough.md` records the same finding.

> **Persistence + the clone guard is a trap.** Cell 2 clones only
> `if not os.path.isdir(REPO_DIR)`. With persistence on, `/kaggle/working` survives
> between sessions, so an existing clone is *never* refreshed and code you pushed
> since then never arrives — the run trains happily with the old code. After any
> push, either delete `/kaggle/working/MRI_CT_preprocessing_pipeline` or run
> `git -C /kaggle/working/MRI_CT_preprocessing_pipeline pull` before training.

Quota is **30 GPU-hours/week**, reset Saturday. One 200-epoch experiment is
roughly 3-5 hours, so budget ~2 sessions per experiment.

---

## 6. Train

Run the cells top to bottom. The important one:

```python
CONFIG = 'model/configs/exp3_nce_heavy.yaml'
RESUME = 'auto'

OVERRIDES = [
    f'data.root={DATA_ROOT}',
    f'data.manifest={DATA_ROOT}/manifest.csv',
    f'data.splits={DATA_ROOT}/splits.json',
    'run.out_dir=/kaggle/working/runs',
    'train.batch_size=8',
    'train.n_epochs=200',
    'runtime.num_workers=2',
    'runtime.amp=true',
]
```

### ⚠ Keep `n_epochs` at the target, not at what fits today

The LR schedule is defined against the **total** epoch count: constant for the
first half, linear decay to zero over the second. Setting `n_epochs=60` because
that is what fits in one session would start the decay at epoch 30 — so the
epochs you already ran would have used a different learning rate than they
should have, and the run is no longer the experiment you meant to run.

Leave it at 200 and let the session die. That is what resume is for.

---

## 7. Resume across sessions

Every epoch writes `runs/<name>/checkpoints/last.pt`, containing all three
networks, all three optimizer states, both schedulers, the EMA shadow, the epoch
and step counters, the python/numpy/torch RNG states, and a hash of the config.

**This is verified, not assumed.** `smoke_test.py` trains 4 epochs straight and
2+resume+2, and asserts the results match — currently to `delta = 0.00e+00`,
bit-exact.

### The loop

1. Session ends (time limit, or you stop it). Output is saved automatically.
2. **Save Version** on the notebook, so `/kaggle/working` is preserved.
3. In the next session, add the previous run's output as an input dataset.
4. Copy `last.pt` back into place:

```python
import shutil, os
os.makedirs('/kaggle/working/runs/exp3_nce_heavy/checkpoints', exist_ok=True)
shutil.copy('/kaggle/input/<previous-run>/runs/exp3_nce_heavy/checkpoints/last.pt',
            '/kaggle/working/runs/exp3_nce_heavy/checkpoints/last.pt')
```

5. Run with `RESUME = 'auto'`. It logs:

```
resumed from .../last.pt at epoch 61 (best mae_norm = 0.0834)
```

### The config-hash warning

If you changed something meaningful between sessions:

```
CONFIG MISMATCH: checkpoint was written with config hash a3f2..., this run
resolves to 9c1e.... Resuming will continue training with DIFFERENT
hyper-parameters than the earlier epochs used...
```

It warns rather than blocks, because *extending* `n_epochs` is legitimate. If you
did not intend a change, stop and find it — a run whose first half used different
λs is not interpretable as either configuration.

Paths, device, worker count and AMP are excluded from the hash, so moving between
this machine and Kaggle does not trigger it.

---

## 8. Bring the results home

The *Output* tab has everything under `/kaggle/working`:

```
runs/exp3_nce_heavy/
├── config.resolved.yaml     exactly what ran
├── metrics.csv              one row per epoch — this is what you plot
├── train_log.jsonl          same, plus nested per-region results
├── train.log
├── checkpoints/
│   ├── best.pt              best val/mae_norm on the EMA generator
│   └── last.pt              for the next session
└── samples/epoch_XXXX.png   the fixed comparison panels
```

Checkpoints are ~650 MB each (G + D + F + three optimizer states + EMA). Only
`best.pt` and `last.pt` are kept. `exp7_reggan` adds R and its optimizer, but R
is small by design (nrf=32, ~2 M params) so the total moves by well under 10%.
`exp8_cyclegan` carries **four** networks and is roughly **1.2 GB** — budget for
that when a session's output quota matters. Its image pools are deliberately not
saved, so a resumed exp8 is not bit-identical to an uninterrupted one; every
weight is restored exactly, but D sees a different sample of past fakes for the
first epoch back.

For inference you only need `state['model']['netG']`, or the EMA weights at
`state['model']['ema']['ema']`, which is what you should actually deploy.

---

## 9. The run plan

Each is a separate Kaggle run with its own `run.name`:

```python
CONFIG = 'model/configs/exp1_pix2pix.yaml'     # baseline first
CONFIG = 'model/configs/exp7_reggan.yaml'      # then measure the misalignment
CONFIG = 'model/configs/exp3_nce_heavy.yaml'   # then the hypothesis it informs
CONFIG = 'model/configs/exp0_l1_only.yaml'     # the floor (fast, no D)
CONFIG = 'model/configs/exp8_cyclegan.yaml'    # last: what was the pairing worth?
```

Run **exp7 before exp3**, not after. exp3 reweights its objective as a hedge
against a residual misalignment nobody has measured; exp7 measures it. If
`R_flow_px` converges to ~0 there is nothing for exp3 to hedge against and you
can skip it, which is a whole Kaggle session saved.

Then locally:

```bash
python model/evaluation/evaluate.py --compare model/runs/exp* --split val
```

Budget roughly: ~210 iterations/epoch at batch 8; 200 epochs is a few hours on a
P100, so 1–2 sessions per experiment. `exp0` is noticeably faster — no
discriminator is built at all. `exp8` is the slowest: four networks, four
generator passes per step, and its own batch size of 4, so budget 2–3 sessions.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `No dataset with a manifest.csv found` | dataset not attached | *Add Input* → your dataset |
| `Missing slice: /kaggle/input/...` | `data.root` wrong | point it at the dir containing `CT/` and `MRI/` |
| CUDA out of memory | batch too large | `train.batch_size=4` |
| Dataloader hangs | too many workers | `runtime.num_workers=0` |
| `nan` losses | AMP + exploding gradient | `runtime.amp=false` to confirm, then `train.grad_clip=1.0` |
| Resume starts from epoch 0 | `last.pt` not in place | check the path in step 7.4 |
| CONFIG MISMATCH warning | config changed | intended? fine. Not intended? find the change |
| Very slow first epoch | dataset unzipping | normal, once |

---

## See also

- `notebook_walkthrough.md` — the Kaggle notebook cell by cell, and what each run trained

- `gan_evaluation_guide.md` — how to tell whether it is working
- `loss_function_guide.md` — the three terms and their λs
- `training_strategies.md` — objective variants and the six stabilisers
