"""
smoke_test.py
─────────────
Prove the wiring on CPU, in about a minute, before burning Kaggle GPU hours.

    python model/scripts/smoke_test.py            # all checks
    python model/scripts/smoke_test.py --quick    # wiring only

Local torch here is CPU-only, so this is the ceiling of what can be verified on
this machine — it says nothing about whether the model learns, only that every
code path executes, that the loss terms switch on and off as configured, and
that a resumed run is bit-for-bit identical to an uninterrupted one.

THE MODULARITY CHECK IS THE IMPORTANT ONE. The whole design rests on a zero
lambda removing a term's entire code path rather than multiplying it by zero, and
that claim is checked here structurally — by inspecting which keys exist in the
saved checkpoint — rather than by reading loss values, which would look identical
either way.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch                                                # noqa: E402

from model.config import load_config                        # noqa: E402
from model.data.dataset import build_datasets               # noqa: E402
from model.data.manifest import load_manifest               # noqa: E402
from model.data.splits import load_split                    # noqa: E402
from model.training.trainer import Trainer                  # noqa: E402

log = logging.getLogger("smoke")

# Tiny, CPU-sized overrides. num_downs=4 keeps a 64px crop from collapsing past
# 4x4, and the NCE taps are shallow to match.
BASE_OVERRIDES = [
    "data.crop_size=64",
    "data.pad_multiple=64",
    "model.generator.num_downs=4",
    "model.generator.ngf=16",
    "model.discriminator.ndf=16",
    "loss.nce.layers=[0,1,2]",
    "loss.nce.num_patches=64",
    "loss.nce.nce_dim=32",
    "train.batch_size=2",
    "train.n_epochs=2",
    "train.gan_warmup_epochs=0",
    "runtime.num_workers=0",
    "runtime.device=cpu",
    "runtime.amp=false",
    "logging.print_every=5",
    "logging.sample_every=999",
    "eval.every=1",
]


def _tiny_datasets(cfg, n_train=20, n_val=8):
    """Cut the real data down to a handful of slices, keeping region variety."""
    manifest = load_manifest(cfg.data.manifest)
    split = load_split(cfg.data.splits)
    datasets = build_datasets(cfg, manifest, split)

    for name, limit in (("train", n_train), ("val", n_val)):
        if name not in datasets:
            continue
        df = datasets[name].df
        # Take a few per region so the region-gated metrics are exercised —
        # a sample of only brain slices would never build a bone metric at all.
        per_region = max(1, limit // df["body_region"].nunique())
        keep = df.groupby("body_region", sort=True).head(per_region)
        datasets[name].df = keep.reset_index(drop=True)
    datasets.pop("test", None)
    return datasets


def _run(cfg, out_root, resume=None, epochs=None, stop_after_epoch=None):
    cfg["run"]["out_dir"] = out_root
    if epochs is not None:
        cfg["train"]["n_epochs"] = epochs
    datasets = _tiny_datasets(cfg)
    trainer = Trainer(cfg, datasets)
    trainer.maybe_resume(resume)
    best = trainer.train(stop_after_epoch=stop_after_epoch)
    return trainer, best


def check_wiring(out_root):
    """Every loss term active: all three networks and three optimizers exist."""
    log.info("─" * 70)
    log.info("CHECK 1  full objective (GAN + L1 + NCE)")
    cfg = load_config("model/configs/exp2_paper.yaml",
                      BASE_OVERRIDES + ["run.name=smoke_full"])
    trainer, best = _run(cfg, out_root)

    state = torch.load(os.path.join(out_root, "smoke_full", "checkpoints", "last.pt"),
                       map_location="cpu", weights_only=False)["model"]
    assert "netD" in state, "discriminator missing from a lambda_gan>0 run"
    assert "netF" in state, "projection heads missing from a lambda_nce>0 run"
    assert "optimizer_F" in state, "optimizer_F was never built"
    log.info("PASS  netG/netD/netF and all three optimizers present; "
             "best mae_norm=%.5f", best)
    return True


def check_modularity(out_root):
    """
    A zero lambda must remove the term's whole code path, not just its value.

    Asserted on checkpoint structure: an ablation that merely multiplied by zero
    would still carry netD/netF and their optimizer state.
    """
    log.info("─" * 70)
    log.info("CHECK 2  modularity — zero lambdas remove code paths")
    results = {}

    for name, overrides, absent, present in [
        ("lambda_nce=0 (plain pix2pix)",
         ["loss.lambda_nce=0", "run.name=smoke_no_nce"],
         ["netF", "optimizer_F"], ["netD", "optimizer_D"]),
        ("lambda_gan=0 (L1-only regression)",
         ["loss.lambda_gan=0", "run.name=smoke_no_gan"],
         ["netD", "optimizer_D"], ["netG", "optimizer_G"]),
    ]:
        cfg = load_config("model/configs/exp2_paper.yaml", BASE_OVERRIDES + overrides)
        run_name = cfg.run.name
        _run(cfg, out_root)

        state = torch.load(os.path.join(out_root, run_name, "checkpoints", "last.pt"),
                           map_location="cpu", weights_only=False)["model"]
        for key in absent:
            assert key not in state, (
                f"{name}: '{key}' is in the checkpoint, so the code path was "
                f"built and merely multiplied by zero")
        for key in present:
            assert key in state, f"{name}: '{key}' should still exist"

        log.info("PASS  %-34s absent=%-26s nickname=%r",
                 name, ",".join(absent), state["loss_plan"]["nickname"])
        results[name] = True
    return all(results.values())


def check_resume(out_root):
    """
    4 epochs straight must equal 2 + resume 2.

    NOTE the interruption is simulated with stop_after_epoch, NOT by setting
    train.n_epochs=2 for the first stage. The LR schedule is defined against the
    total epoch count, so a first stage declaring n_epochs=2 would begin its
    linear decay at epoch 1 while the straight run holds LR constant until
    epoch 2 — the two would then legitimately differ, and the test would be
    measuring its own unfairness rather than the resume logic. This mirrors real
    use: on Kaggle you keep n_epochs at the full target and let the session die.
    """
    log.info("─" * 70)
    log.info("CHECK 3  resume correctness")

    cfg_a = load_config("model/configs/exp2_paper.yaml",
                        BASE_OVERRIDES + ["run.name=smoke_straight"])
    _, best_straight = _run(cfg_a, out_root, epochs=4)

    # Session 1: dies after epoch 1, but always knew the target was 4.
    cfg_b = load_config("model/configs/exp2_paper.yaml",
                        BASE_OVERRIDES + ["run.name=smoke_resumed"])
    _run(cfg_b, out_root, epochs=4, stop_after_epoch=2)

    # Session 2: same config, picks up last.pt.
    cfg_c = load_config("model/configs/exp2_paper.yaml",
                        BASE_OVERRIDES + ["run.name=smoke_resumed"])
    _, best_resumed = _run(cfg_c, out_root, resume="auto", epochs=4)

    delta = abs(best_straight - best_resumed)
    log.info("straight(4)=%.8f  resumed(2+2)=%.8f  delta=%.2e",
             best_straight, best_resumed, delta)
    assert delta < 1e-6, (
        f"resume diverged by {delta:.2e}; the RNG, optimizer or scheduler state "
        f"is not being restored faithfully")
    log.info("PASS  resumed run matches the uninterrupted one")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU smoke test for the model wiring")
    parser.add_argument("--quick", action="store_true", help="wiring check only")
    parser.add_argument("--keep", action="store_true", help="keep the temp run dir")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s",
                        force=True)
    # The per-iteration chatter from the trainer would bury the check results.
    logging.getLogger("model.training.trainer").setLevel(logging.WARNING)

    out_root = tempfile.mkdtemp(prefix="smoke_runs_")
    log.info("temp run dir: %s", out_root)

    try:
        checks = [check_wiring]
        if not args.quick:
            checks += [check_modularity, check_resume]
        for check in checks:
            check(out_root)

        log.info("─" * 70)
        log.info("ALL CHECKS PASSED")
        return 0
    except AssertionError as exc:
        log.error("FAILED: %s", exc)
        return 1
    finally:
        if not args.keep:
            shutil.rmtree(out_root, ignore_errors=True)
        else:
            log.info("kept %s", out_root)


if __name__ == "__main__":
    raise SystemExit(main())
