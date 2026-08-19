"""
trainer.py
──────────
The epoch loop, validation, logging, sample rendering and checkpoint/resume.

CHECKPOINTING IS NOT AN AFTERTHOUGHT HERE. Kaggle kills a session at its time
limit, and a 200-epoch run does not fit in one. So a checkpoint stores enough to
continue as if nothing happened: all three networks, all three optimizer states,
both schedulers, the EMA shadow, the epoch and step counters, the python/numpy/
torch RNG states, and a hash of the config that produced it. Resuming reseeds
the generators from the saved RNG state, so the second half of a split run sees
the same augmentation stream it would have seen had it never stopped.

WHAT GETS LOGGED, AND WHY IT IS SPLIT IN TWO.
  metrics.csv   one row per epoch, every scalar. This is what you plot.
  train_log.jsonl  one record per epoch including nested per-region results,
                   which do not fit a flat CSV cell without losing structure.
Both are appended and both survive a resume, so a run split across three Kaggle
sessions still produces one continuous history.
"""

import json
import logging
import os
import random
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import dump_config
from ..data.dataset import to_unit_range
from ..evaluation.metrics import MetricAccumulator
from .builder import build_model
from .pix2pix_nce import build_lr_scheduler

logger = logging.getLogger(__name__)


def resolve_device(spec):
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """
    Re-seed numpy and python RNGs inside each dataloader worker.

    PyTorch seeds only its OWN generator per worker; numpy's global state is
    inherited from the parent (fork) or freshly default-initialised (spawn,
    which is what Windows uses). Either way every worker ends up drawing the
    SAME numpy sequence — and PairedSliceDataset picks its crop offsets and
    flips from numpy. With num_workers=2 that means the two workers hand back
    correlated crops, quietly halving the augmentation diversity.

    torch.initial_seed() inside a worker is the DataLoader's per-worker,
    per-epoch seed, so deriving from it keeps workers independent of each other
    while staying reproducible from the run seed.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class Trainer:

    def __init__(self, cfg, datasets):
        self.cfg = cfg
        self.device = resolve_device(cfg.get_path("runtime.device", "auto"))
        seed_everything(int(cfg.run.seed))

        self.run_dir = os.path.join(cfg.run.out_dir, cfg.run.name)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.sample_dir = os.path.join(self.run_dir, "samples")
        for path in (self.run_dir, self.ckpt_dir, self.sample_dir):
            os.makedirs(path, exist_ok=True)
        dump_config(cfg, os.path.join(self.run_dir, "config.resolved.yaml"))

        self.model = build_model(cfg, self.device)
        self.datasets = datasets
        self.loaders = self._build_loaders(datasets)

        self.sched_G = build_lr_scheduler(self.model.optimizer_G, cfg)
        self.sched_D = (build_lr_scheduler(self.model.optimizer_D, cfg)
                        if self.model.optimizer_D is not None else None)

        # AMP is a GPU feature; on the CPU box this project develops on it is a
        # no-op and enabling it would only add overhead.
        self.amp = bool(cfg.get_path("runtime.amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.metrics = MetricAccumulator(cfg)
        self.start_epoch = 0
        self.best_value = None
        self.selection_metric = cfg.get_path("eval.selection_metric", "mae_norm")
        self.selection_mode = cfg.get_path("eval.selection_mode", "min")

        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self.jsonl_path = os.path.join(self.run_dir, "train_log.jsonl")
        self._csv_columns = None
        self._sample_batch = None

        logger.info("run dir: %s", self.run_dir)
        logger.info("device: %s | amp: %s | config hash: %s",
                    self.device, self.amp, cfg.get("_hash"))

    # ── Setup ────────────────────────────────────────────────────────────────

    def _build_loaders(self, datasets):
        cfg = self.cfg
        workers = int(cfg.get_path("runtime.num_workers", 4))
        pin = bool(cfg.get_path("runtime.pin_memory", True)) and self.device.type == "cuda"

        loaders = {}
        if "train" in datasets:
            loaders["train"] = DataLoader(
                datasets["train"], batch_size=int(cfg.train.batch_size),
                shuffle=True, num_workers=workers, pin_memory=pin,
                drop_last=True, persistent_workers=workers > 0,
                worker_init_fn=worker_init_fn if workers > 0 else None)
        for name in ("val", "test"):
            if name in datasets:
                # batch_size 1: validation slices keep their native size and
                # differ from each other, so they cannot be stacked.
                loaders[name] = DataLoader(
                    datasets[name], batch_size=int(cfg.get_path("eval.batch_size", 1)),
                    shuffle=False, num_workers=workers, pin_memory=pin)
        return loaders

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, stop_after_epoch=None):
        """
        Run the epoch loop.

        `stop_after_epoch` stops early WITHOUT changing train.n_epochs, which
        matters because the LR schedule is defined against the total: lowering
        n_epochs to stop sooner would also move the decay onset, so the epochs
        you did run would have used a different learning rate than they should
        have. Use this to simulate a Kaggle session limit — and on Kaggle, keep
        n_epochs at the full target and simply let the session die.
        """
        n_epochs = int(self.cfg.train.n_epochs)
        last_epoch = n_epochs if stop_after_epoch is None else min(n_epochs,
                                                                  stop_after_epoch)
        logger.info("training epochs %d..%d (target total %d)",
                    self.start_epoch, last_epoch - 1, n_epochs)

        for epoch in range(self.start_epoch, last_epoch):
            t0 = time.time()
            train_stats = self.train_one_epoch(epoch)
            train_time = time.time() - t0

            record = {"epoch": epoch, "train_time_s": round(train_time, 1),
                      "lr_G": self.model.optimizer_G.param_groups[0]["lr"]}
            record.update({f"train/{k}": v for k, v in train_stats.items()})

            if "val" in self.loaders and (epoch + 1) % int(self.cfg.get_path("eval.every", 1)) == 0:
                val_results = self.validate(epoch)
                record.update({f"val/{k}": v for k, v in val_results.items()})
                improved = self._update_best(val_results, epoch)
                record["is_best"] = bool(improved)

            self.sched_G.step()
            if self.sched_D is not None:
                # During GAN warm-up the discriminator optimizer is never
                # stepped, so PyTorch warns that the scheduler ran first. The
                # warning is correct but harmless here: D's learning rate is not
                # consulted while D is frozen, and stepping its scheduler in
                # lockstep with G's is what makes both reach lr=0 at n_epochs
                # together. Suppressed rather than skipped, so the two schedules
                # cannot drift apart by the length of the warm-up.
                with warnings.catch_warnings():
                    if not self.model.gan_active(epoch):
                        warnings.filterwarnings(
                            "ignore", message=r".*lr_scheduler\.step\(\).*")
                    self.sched_D.step()

            self._log_epoch(record)

            if (epoch + 1) % int(self.cfg.get_path("logging.save_every", 1)) == 0:
                self.save_checkpoint(epoch, tag="last")
            if record.get("is_best"):
                self.save_checkpoint(epoch, tag="best")
            if (epoch + 1) % int(self.cfg.get_path("logging.sample_every", 5)) == 0:
                self.render_samples(epoch)

        logger.info("training complete. best %s = %s",
                    self.selection_metric, self.best_value)
        return self.best_value

    def train_one_epoch(self, epoch):
        self.model.train_mode()

        loader = self.loaders["train"]
        print_every = int(self.cfg.get_path("logging.print_every", 50))
        amp_ctx = torch.autocast(device_type=self.device.type, enabled=self.amp)

        totals, count = {}, 0
        warming = epoch < self.model.warmup_epochs and self.model.plan.use_gan
        if warming:
            logger.info("epoch %d: GAN warm-up — L1/NCE only, D is not updated "
                        "(%d warm-up epochs configured)",
                        epoch, self.model.warmup_epochs)

        for i, batch in enumerate(loader):
            stats = self.model.optimize_parameters(batch, epoch, self.scaler, amp_ctx)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1

            if print_every and i % print_every == 0:
                summary = "  ".join(f"{k}={v:.4f}" for k, v in sorted(stats.items())
                                    if k in ("G_total", "G_L1", "G_GAN", "G_NCE",
                                             "G_corr", "G_smooth", "R_flow_px",
                                             "G_cycle_A", "G_cycle_B", "G_GAN_A2B",
                                             "D_total", "D_acc_real", "D_acc_fake"))
                logger.info("  e%03d %4d/%d  %s", epoch, i, len(loader), summary)

        return {k: v / max(1, count) for k, v in totals.items()}

    # ── Validation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch, split="val"):
        net = self.model.generator_for_eval()
        self.model.set_eval_mode(net)
        self.metrics.reset()

        amp_ctx = torch.autocast(device_type=self.device.type, enabled=self.amp)
        for batch in self.loaders[split]:
            real_A = batch["A"].to(self.device, non_blocking=True)
            real_B = batch["B"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            with amp_ctx:
                fake_B = net(real_A)

            # Metrics are defined on the stored [0,1] scale, which is also the
            # scale the HU windows invert from. float() because the SSIM
            # convolutions are numerically fragile in fp16.
            self.metrics.update(
                to_unit_range(fake_B.float()), to_unit_range(real_B.float()),
                mask, batch["hu_min"], batch["hu_max"], batch["body_region"])

        results = self.metrics.compute()
        logger.info("epoch %d  %s metrics:\n%s", epoch, split,
                    self.metrics.format_table(results))
        return results

    def _update_best(self, results, epoch):
        value = results.get(self.selection_metric)
        if value is None:
            logger.warning("selection metric '%s' absent from validation results",
                           self.selection_metric)
            return False

        better = (self.best_value is None
                  or (value < self.best_value if self.selection_mode == "min"
                      else value > self.best_value))
        if better:
            previous = self.best_value
            self.best_value = value
            logger.info("epoch %d: new best %s = %.5f (was %s)", epoch,
                        self.selection_metric, value,
                        f"{previous:.5f}" if previous is not None else "none")
        return better

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_epoch(self, record):
        """
        Append to the JSONL, then rebuild metrics.csv from it.

        WHY REBUILD RATHER THAN APPEND. An earlier version froze the CSV header
        from the first epoch's keys and appended thereafter. That silently lost
        every GAN column for a whole 200-epoch run: epoch 0 is GAN warm-up, so
        D_acc_real, D_acc_fake, D_total and G_GAN do not exist yet, the header
        was written without them, and each later epoch's values were dropped on
        the way out. Nothing complained — the CSV just had no D columns, and the
        plotting cell drew an empty axis.

        Rebuilding from the JSONL each epoch takes the union of all keys seen so
        far, so a column that appears at epoch 5 is backfilled with blanks for
        epochs 0-4 and populated from then on. At a few hundred epochs the cost
        is microseconds, and the JSONL stays the authoritative record.
        """
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=float) + "\n")

        rows = []
        with open(self.jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A session killed mid-write can leave one truncated line.
                    # Skip it rather than lose the whole history.
                    logger.warning("skipping malformed line in %s", self.jsonl_path)

        # Flat scalars only; nested per-region values stay in the JSONL. Column
        # order follows first appearance, so 'epoch' stays leftmost.
        columns = []
        for row in rows:
            for key, value in row.items():
                if (isinstance(value, (int, float, bool)) or value is None) \
                        and key not in columns:
                    columns.append(key)
        self._csv_columns = columns

        tmp = self.csv_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(",".join(columns) + "\n")
            for row in rows:
                fh.write(",".join(
                    "" if row.get(c) is None else str(row.get(c, ""))
                    for c in columns) + "\n")
        os.replace(tmp, self.csv_path)

    # ── Samples ──────────────────────────────────────────────────────────────

    def _pick_sample_batch(self):
        """
        Choose a FIXED panel of validation slices, one set per body region.

        Fixed on purpose. Comparing epoch 40's random slices against epoch 45's
        random slices tells you about the slices, not the model. The same panel
        every time makes visual progress legible, and makes two different runs
        directly comparable image-for-image.
        """
        if self._sample_batch is not None or "val" not in self.datasets:
            return self._sample_batch

        dataset = self.datasets["val"]
        per_region = int(self.cfg.get_path("logging.n_sample_slices_per_region", 2))
        chosen = []
        for region in sorted(dataset.df["body_region"].unique()):
            rows = dataset.df.index[dataset.df["body_region"] == region].tolist()
            # Evenly spaced through the region's slices rather than the first N,
            # which would all come from one patient's first series.
            step = max(1, len(rows) // (per_region + 1))
            chosen.extend(rows[step::step][:per_region])

        self._sample_batch = [dataset[dataset.df.index.get_loc(i)] for i in chosen]
        logger.info("sample panel: %d fixed validation slices", len(self._sample_batch))
        return self._sample_batch

    @torch.no_grad()
    def render_samples(self, epoch):
        """Write an MRI | real CT | synth CT | error panel as a PNG."""
        samples = self._pick_sample_batch()
        if not samples:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping sample panel")
            return

        net = self.model.generator_for_eval()
        self.model.set_eval_mode(net)

        n = len(samples)
        fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
        for r, item in enumerate(samples):
            real_A = item["A"].unsqueeze(0).to(self.device)
            fake_B = net(real_A)

            mri = to_unit_range(item["A"]).squeeze().cpu().numpy()
            real = to_unit_range(item["B"]).squeeze().cpu().numpy()
            fake = to_unit_range(fake_B.float()).squeeze().cpu().numpy()
            err = np.abs(fake - real)

            for c, (img, title, kwargs) in enumerate([
                (mri, f"MRI ({item['body_region']})", dict(cmap="gray", vmin=0, vmax=1)),
                (real, "real CT", dict(cmap="gray", vmin=0, vmax=1)),
                (fake, "synth CT", dict(cmap="gray", vmin=0, vmax=1)),
                # Fixed error scale: an autoscaled error map looks equally red at
                # every epoch and hides exactly the improvement it should show.
                (err, f"|error| (MAE {err.mean():.3f})", dict(cmap="inferno", vmin=0, vmax=0.5)),
            ]):
                axes[r][c].imshow(img, **kwargs)
                axes[r][c].set_title(title, fontsize=9)
                axes[r][c].axis("off")

        fig.suptitle(f"{self.cfg.run.name} — epoch {epoch}", fontsize=11)
        fig.tight_layout()
        path = os.path.join(self.sample_dir, f"epoch_{epoch:04d}.png")
        fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", path)

    # ── Checkpoints ──────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch, tag="last"):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "sched_G": self.sched_G.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_value": self.best_value,
            "selection_metric": self.selection_metric,
            "config_hash": self.cfg.get("_hash"),
            "run_name": self.cfg.run.name,
            # RNG state, so a resumed run continues the same augmentation and
            # patch-sampling stream instead of restarting it.
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
        }
        if self.sched_D is not None:
            state["sched_D"] = self.sched_D.state_dict()

        path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)          # atomic: a session killed mid-write must
                                       # not leave a truncated checkpoint behind
        logger.info("saved %s (epoch %d)", path, epoch)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)

        saved_hash = state.get("config_hash")
        if saved_hash and saved_hash != self.cfg.get("_hash"):
            logger.warning(
                "CONFIG MISMATCH: checkpoint was written with config hash %s, "
                "this run resolves to %s. Resuming will continue training with "
                "DIFFERENT hyper-parameters than the earlier epochs used, and "
                "the metrics.csv will mix the two. Continuing anyway — this is "
                "correct if you only extended train.n_epochs.",
                saved_hash, self.cfg.get("_hash"))

        self.model.load_state_dict(state["model"])
        self.sched_G.load_state_dict(state["sched_G"])
        if self.sched_D is not None and "sched_D" in state:
            self.sched_D.load_state_dict(state["sched_D"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        self.best_value = state.get("best_value")
        self.start_epoch = int(state["epoch"]) + 1

        rng = state.get("rng") or {}
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])

        logger.info("resumed from %s at epoch %d (best %s = %s)",
                    path, self.start_epoch, self.selection_metric, self.best_value)
        return self.start_epoch

    def maybe_resume(self, resume):
        """`resume` is a path, 'auto' (use last.pt if present), or falsy."""
        if not resume:
            return
        if resume == "auto":
            candidate = os.path.join(self.ckpt_dir, "last.pt")
            if not os.path.isfile(candidate):
                logger.info("--resume auto: no checkpoint at %s, starting fresh",
                            candidate)
                return
            resume = candidate
        if not os.path.isfile(resume):
            raise FileNotFoundError(f"Checkpoint not found: {resume}")
        self.load_checkpoint(resume)
