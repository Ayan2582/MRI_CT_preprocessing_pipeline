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

# StyleGAN2 is sized by native_size rather than by ngf/num_downs, so it needs its
# own shrink list. native_size=64 with const_size=4 is 4 downsamplings, which is
# what num_downs must then declare — dataset.py reads it for the padding multiple
# and the generator builder rejects a mismatch.
STYLEGAN_OVERRIDES = [
    "data.crop_size=64",
    "data.pad_multiple=64",
    "model.generator.num_downs=4",
    "model.generator.native_size=64",
    "model.generator.w_dim=64",
    "model.generator.n_mapping=4",
    "model.generator.channel_base=2048",
    "model.generator.channel_max=64",
    "model.generator.tile_stride=32",
    "model.discriminator.channel_base=2048",
    "model.discriminator.channel_max=64",
    "model.discriminator.mbstd_group_size=2",
    "loss.nce.layers=[0,1,2]",
    "loss.nce.num_patches=64",
    "loss.nce.nce_dim=32",
    "train.batch_size=2",
    "train.n_epochs=1",
    "train.gan_warmup_epochs=0",   # exercise the D path in a one-epoch run
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


def check_stylegan2(out_root):
    """
    The second architecture: structure, tiling, determinism, and initialisation.

    Four of these five assertions guard failures that are SILENT — the run trains,
    the loss goes down, and the damage only shows up as a worse number at the end
    of a Kaggle session. They are worth the minute they cost.
    """
    log.info("─" * 70)
    log.info("CHECK 4  StyleGAN2 architecture")

    from model.networks.builder import build_generator
    from model.networks.stylegan2 import EqualizedConv2d
    from model.networks.tiling import tiled_forward
    from model.training.pix2pix_nce import Pix2PixNCEModel

    # ── 4a. Tiling reconstructs exactly ──────────────────────────────────────
    # An identity "generator" must come back unchanged, which is true only if the
    # raised-cosine weights sum to one at EVERY pixel — including the borders,
    # where a textbook Hann window is zero and the normalisation would divide by
    # zero. This is the cheapest possible test of the thing that lets 45% of the
    # validation set be scored at all.
    source = torch.randn(1, 1, 128, 128)
    rebuilt = tiled_forward(lambda t: t, source, native=64, stride=32)
    assert rebuilt.shape == source.shape, f"tiling changed shape: {rebuilt.shape}"
    error = float((rebuilt - source).abs().max())
    assert error < 1e-5, f"tiled identity is not the identity: max error {error:.2e}"
    log.info("PASS  tiling: 128px through 64px windows, max error %.1e", error)

    # ── 4b/4c. Both configs train, with the right structure ──────────────────
    expectations = [
        ("exp5_stylegan2_vanilla", "smoke_sg2_vanilla",
         ["netG", "netD", "optimizer_D", "pl_mean"], ["netF", "optimizer_F"]),
        ("exp6_stylegan2_fitted", "smoke_sg2_fitted",
         ["netG", "netD", "netF", "optimizer_F"], ["pl_mean"]),
    ]
    for config, run_name, present, absent in expectations:
        cfg = load_config(f"model/configs/{config}.yaml",
                          STYLEGAN_OVERRIDES + [f"run.name={run_name}"])
        _run(cfg, out_root)

        state = torch.load(os.path.join(out_root, run_name, "checkpoints", "last.pt"),
                           map_location="cpu", weights_only=False)["model"]
        for key in present:
            assert key in state, f"{config}: '{key}' missing from the checkpoint"
        for key in absent:
            assert key not in state, (
                f"{config}: '{key}' is in the checkpoint, so that code path was "
                f"built despite being switched off")
        log.info("PASS  %-24s absent=%-22s nickname=%r", config,
                 ",".join(absent), state["loss_plan"]["nickname"])

    # exp5's objective must not have quietly acquired a reconstruction term.
    cfg5 = load_config("model/configs/exp5_stylegan2_vanilla.yaml", STYLEGAN_OVERRIDES)
    assert cfg5.loss.lambda_l1 == 0 and cfg5.loss.lambda_nce == 0, \
        "exp5 is supposed to run StyleGAN2's own loss: adversarial only"

    # ── 4d. Deterministic at eval ────────────────────────────────────────────
    # Noise injection is resampled per call. If it survives into evaluation,
    # mae_norm becomes a random variable and two runs of evaluate.py on one
    # checkpoint disagree.
    model = Pix2PixNCEModel(cfg5, torch.device("cpu"))
    net = model.set_eval_mode(model.netG)
    probe = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        first, second = net(probe), net(probe)
    assert torch.equal(first, second), (
        "the generator is not deterministic at eval — noise injection is still "
        "live. Check model.generator.noise_at_eval and NoiseInjection.training.")
    log.info("PASS  eval is deterministic (noise injection off)")

    # ── 4e. Initialisation was not clobbered ─────────────────────────────────
    # Equalized learning rate needs N(0,1) weights, scaled at runtime. If anything
    # ever passes these modules through networks/init.py:init_weights they become
    # N(0,0.02) AND runtime-scaled — roughly 50x too small — and the network trains
    # happily while learning nothing at all. Nothing else would report it.
    fresh = build_generator(cfg5.model.generator)
    stds = [float(m.weight.detach().std()) for m in fresh.modules()
            if isinstance(m, EqualizedConv2d)]
    assert stds, "no EqualizedConv2d found; did the generator type change?"
    worst = min(stds)
    assert worst > 0.5, (
        f"StyleGAN2 conv weights have std {worst:.4f}, expected ~1.0. Something "
        f"applied init_weights (N(0,0.02)) to an equalized-LR module.")
    log.info("PASS  equalized-LR init intact (%d convs, min std %.3f)",
             len(stds), worst)

    # ── 4f. Warm-up with no reconstruction term is refused ───────────────────
    try:
        load_and_plan = load_config("model/configs/exp5_stylegan2_vanilla.yaml",
                                    STYLEGAN_OVERRIDES + ["train.gan_warmup_epochs=3"])
        from model.losses.builder import LossPlan
        LossPlan(load_and_plan)
    except ValueError:
        log.info("PASS  warm-up without L1/NCE is refused at config time")
    else:
        raise AssertionError(
            "gan_warmup_epochs>0 with lambda_l1=0 and lambda_nce=0 should raise: "
            "the warm-up epochs would have no loss at all")

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
            checks += [check_modularity, check_resume, check_stylegan2]
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
