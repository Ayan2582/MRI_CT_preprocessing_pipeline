# The trainer — everything around the training step

`optimize_parameters` handles one batch. Something has to loop over epochs, decide
when to validate, step the learning-rate schedules, write logs you can plot,
checkpoint safely enough to survive a killed Kaggle session, and render sample
panels you can compare across runs. That is `training/trainer.py`, 491 lines, and
almost every design decision in it is a scar from a specific failure.

---

## 1. The entry point

```bash
python model/scripts/train.py --config exp2_paper.yaml
python model/scripts/train.py --config exp2_paper.yaml --set loss.lambda_nce=0
python model/scripts/train.py --config exp3_nce_heavy.yaml --resume auto
```

`scripts/train.py:61-74` is the whole assembly:

```python
manifest = load_manifest(cfg.data.manifest)
split    = load_split(cfg.data.splits)
datasets = build_datasets(cfg, manifest, split)

trainer = Trainer(cfg, datasets)
trainer.maybe_resume(args.resume)
trainer.train()
```

> **`--set` rejects any dotted path not already present in `base.yaml`**
> (`model/config.py:163-168`). A mistyped override stops the run instead of quietly
> training something other than the experiment you meant. `base.yaml` is the schema,
> not just the defaults.

---

## 2. The epoch loop (`training/trainer.py:142-200`)

```python
for epoch in range(self.start_epoch, last_epoch):
    train_stats = self.train_one_epoch(epoch)                 # timed

    record = {"epoch": epoch, "train_time_s": ..., "lr_G": ...}
    record.update({f"train/{k}": v for k, v in train_stats.items()})

    if "val" in self.loaders and (epoch + 1) % eval.every == 0:
        val_results = self.validate(epoch)
        record.update({f"val/{k}": v for k, v in val_results.items()})
        record["is_best"] = bool(self._update_best(val_results, epoch))

    self.sched_G.step()
    if self.sched_D is not None:
        self.sched_D.step()

    self._log_epoch(record)

    if (epoch + 1) % logging.save_every  == 0: self.save_checkpoint(epoch, tag="last")
    if record.get("is_best"):                  self.save_checkpoint(epoch, tag="best")
    if (epoch + 1) % logging.sample_every == 0: self.render_samples(epoch)
```

Order: **train → validate → step schedulers → log → checkpoint → samples.** Logging
happens after validation so one record holds both halves of the epoch, and
checkpointing happens after logging so a checkpoint never exists without its log line.

### Two subtleties in this loop

**`stop_after_epoch` does not change `train.n_epochs`** (`:142-152`). The learning-rate
schedule is defined against the *total*: lowering `n_epochs` to stop sooner would also
move the decay onset, so the epochs you did run would have used a different learning
rate than they should have. On Kaggle you keep `n_epochs` at the full target and let
the session die — `kaggle_workflow.md` §6 says the same thing more forcefully.

**The `sched_D` warning is suppressed, not skipped** (`:175-187`):

```python
with warnings.catch_warnings():
    if not self.model.gan_active(epoch):
        warnings.filterwarnings("ignore", message=r".*lr_scheduler\.step\(\).*")
    self.sched_D.step()
```

During GAN warm-up `optimizer_D` is never stepped, so PyTorch warns that the
scheduler ran before the optimizer. The warning is correct and harmless — D's
learning rate is not consulted while D is frozen. The important part is that
`sched_D.step()` still runs: **stepping both schedulers in lockstep is what makes
them reach `lr = 0` at `n_epochs` together.** Skipping it during warm-up would leave
D's schedule permanently five epochs behind G's.

---

## 3. `train_one_epoch` (`:202-230`)

```python
self.model.train_mode()
amp_ctx = torch.autocast(device_type=self.device.type, enabled=self.amp)
for batch in self.loaders["train"]:
    stats = self.model.optimize_parameters(batch, epoch, self.scaler, amp_ctx)
    # accumulate into totals
return {k: v / count for k, v in totals.items()}
```

Three things:

**`self.model.train_mode()`, never `self.model.netG.train()`.** The trainer asks the
model to put itself in training mode and never reaches for a network by name. That is
what lets CycleGAN — which has four networks — use this same trainer unchanged
(`training/cyclegan.py:145-147`).

**The autocast context is built once per epoch** (`:207`) and passed into every step,
rather than being constructed per batch.

**The return value is the epoch *mean* of every stat.** So `train/G_L1` in the log is
an average over ~210 iterations, not the last batch's value. That is what makes the
curves readable.

`training_log_reference.md` decodes every field this produces.

---

## 4. `validate` (`:235-259`)

```python
net = self.model.generator_for_eval()
self.model.set_eval_mode(net)
...
fake_B = net(real_A)
self.metrics.update(to_unit_range(fake_B.float()), to_unit_range(real_B.float()),
                    mask, batch["hu_min"], batch["hu_max"], batch["body_region"])
```

**`generator_for_eval()` returns the EMA shadow**, not the live generator, when
`eval.use_ema` is true — which it is by default (`training/pix2pix_nce.py:472-482`).
Every validation number in this project is measured on the averaged weights. See
`08_stabilisers_in_code.md`.

**The `.float()` calls are required, not defensive** (`:250-251`): SSIM's convolutions
are numerically fragile in fp16.

Validation loaders use `batch_size = eval.batch_size`, **default 1** (`:119-138`),
because validation slices keep their native size and cannot be stacked into a batch.

---

## 5. Logging: one JSONL, and a CSV rebuilt from it every epoch

This is the best-documented bug in the file, and it is worth reading as a case study.

**The failure.** An earlier version froze the CSV header from the first epoch's keys
and appended thereafter. Epoch 0 is GAN warm-up, so `D_acc_real`, `D_acc_fake`,
`D_total` and `G_GAN` **do not exist yet**. The header was written without them, and
every later epoch's values were dropped on the way out. Nothing complained. The CSV
simply had no D columns for a whole 200-epoch run, and the plotting cell drew an
empty axis.

**The fix** (`:281-331`): append one JSON record per epoch to `train_log.jsonl`, then
re-read the whole file and rebuild `metrics.csv` from scratch:

```python
columns = []
for row in rows:
    for key, value in row.items():
        if (isinstance(value, (int, float, bool)) or value is None) and key not in columns:
            columns.append(key)
```

The column set is the **union of all keys ever seen**, in first-appearance order (so
`epoch` stays leftmost). A column that appears at epoch 5 is backfilled with blanks
for epochs 0–4. At a few hundred epochs the cost is microseconds, and the JSONL stays
the authoritative record.

Two more details:

- **Malformed trailing lines are skipped, not fatal** (`:307-312`). A session killed
  mid-write leaves one truncated line; losing the whole history to it would be absurd.
- **Only flat scalars go to the CSV.** Nested per-region results stay in the JSONL.

The naming convention is `train/<stat>` and `val/<metric>`, plus `epoch`,
`train_time_s`, `lr_G` and `is_best`.

**The general lesson:** the authoritative log should be append-only and
schema-free; the convenient log should be *derived* from it. Anything that requires
knowing the full set of columns before the run starts will be wrong in a run where
some columns appear late.

---

## 6. Sample panels are fixed on purpose (`:335-407`)

```python
def _pick_sample_batch(self):
    # chosen once, cached in self._sample_batch
```

**The failure this prevents:** if the panel showed a different random batch each time,
you could not tell whether epoch 50 is better than epoch 45 or just luckier, and two
runs could not be compared image-for-image. So the slices are chosen once and reused
for the life of the run.

Selection (`:344-359`) takes rows *evenly spaced* within each `body_region`:

```python
step = max(1, len(rows) // (per_region + 1))
chosen = rows[step::step][:per_region]
```

Taking the first N would draw them all from one patient's first series.

The grid is `n x 4` — **MRI | real CT | synth CT | |error|**. The first three use
`cmap="gray", vmin=0, vmax=1`. The error map uses `cmap="inferno"` with a **fixed**
`vmin=0, vmax=0.5` (`:394-395`), and the fixed scale is the point:

> An autoscaled error map looks equally red at every epoch and hides the improvement
> it is supposed to show.

matplotlib is imported with the `Agg` backend inside a try/except (`:368-374`), so a
missing matplotlib skips samples with a warning instead of killing the run.

---

## 7. Checkpoints have to survive a killed session

`save_checkpoint` (`:411-439`) stores rather more than the weights:

| key | why |
|---|---|
| `model` | the model's own `state_dict` — G, D, F, EMA, optimizers, whatever exists |
| `sched_G`, `sched_D` | so the LR resumes at the right point on the decay ramp |
| `scaler` | the AMP scale factor, which is state |
| `best_value`, `selection_metric` | so "is this the best epoch" survives a restart |
| `config_hash`, `run_name` | so you can tell what produced it |
| `rng` | python, numpy, torch and cuda RNG states (`:423-429`) |

Saving the RNG states is what makes a resumed run continue the *same* data ordering
and augmentation sequence rather than starting a fresh one.

**Writes are atomic** (`:436-438`):

```python
torch.save(state, tmp)          # "<tag>.pt.tmp"
os.replace(tmp, path)           # atomic rename
```

`os.replace` is atomic on both POSIX and Windows, so a session killed mid-write leaves
the *previous* checkpoint intact rather than a truncated file. `_log_epoch` uses the
same tmp-then-replace for the CSV (`:324-331`). On a platform where the session can be
killed at any moment, this is not paranoia.

**A config-hash mismatch warns rather than fails** (`:444-452`) — the common case is
extending `n_epochs`, which is legitimate. `kaggle_workflow.md` §7 covers when to
worry about it.

`maybe_resume` (`:478-491`): falsy is a no-op; `"auto"` uses `checkpoints/last.pt` if
it exists and starts fresh otherwise; anything else is treated as a path and raises if
missing.

---

## 8. Two small things worth stealing

**`worker_init_fn` (`:55-73`).** PyTorch seeds only *its own* generator per DataLoader
worker. Anything using numpy's global RNG — which the augmentation pipeline does —
gets the **same sequence in every worker**, so four workers hand back four correlated
crops. The fix derives a per-worker numpy seed from `torch.initial_seed()`. This bug is
invisible: training still works, it just has a quarter of the augmentation diversity
you think it has.

**Device resolution (`:42-46`, `:99`).**

```python
self.amp = bool(cfg.get_path("runtime.amp", True)) and self.device.type == "cuda"
```

`runtime.amp` defaults to true, and a CPU box silently disables it rather than
erroring. That is what lets `scripts/smoke_test.py` run the same code path locally
that Kaggle runs on a GPU.

---

## See also

- `05_the_training_step.md` — what `optimize_parameters` does with one batch
- `training_log_reference.md` — every field this trainer prints, and its healthy range
- `gan_evaluation_guide.md` — how to decide whether the numbers are improving
- `notebook_walkthrough.md` — the Kaggle notebook that drives all of this, cell by cell
- `kaggle_workflow.md` — packaging, resuming across sessions, retrieving results
