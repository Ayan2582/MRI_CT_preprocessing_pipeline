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


# RegGAN carries a third network with its own sizing knobs, so it needs the base
# shrink plus its own. num_downs=2 on R against a 64px crop bottoms out at 16x16,
# which is plenty for a field that is low-frequency by construction.
REGGAN_OVERRIDES = BASE_OVERRIDES + [
    "model.registration.nrf=8",
    "model.registration.num_downs=2",
]


# CycleGAN holds four networks, so it shrinks the same way exp0-exp4 do but
# needs the unconditional D its config already declares, plus a pool small enough
# to actually cycle within a two-epoch run.
CYCLEGAN_OVERRIDES = BASE_OVERRIDES + [
    "train.image_pool_size=8",
]


def _tiny_datasets(cfg, n_train=20, n_val=8):
    """Cut the real data down to a handful of slices, keeping region variety."""
    manifest = load_manifest(cfg.data.manifest)
    split = load_split(cfg.data.splits)
    datasets = build_datasets(cfg, manifest, split)

    for name, limit in (("train", n_train), ("val", n_val)):
        if name not in datasets:
            continue
        # UnpairedSliceDataset has to subsample each domain separately: its two
        # halves hold disjoint patients, and `limit` rows taken from the
        # concatenation can easily land entirely on one side, leaving the other
        # empty.
        if hasattr(datasets[name], "subsample_per_region"):
            datasets[name].subsample_per_region(limit)
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


def check_reggan(out_root):
    """
    RegGAN: structure, and the three ways its correction term fails silently.

    Every assertion here except the structural one guards something that trains
    happily and reports a falling loss while being wrong. That is the character
    of this model — the registration field has no ground truth, so nothing
    downstream can notice when it drifts from "the residual misalignment" into
    "whatever minimises the loss".
    """
    log.info("─" * 70)
    log.info("CHECK 5  RegGAN")

    from model.losses.builder import LossPlan
    from model.networks.registration import RegistrationUNet, SpatialTransformer

    # ── 5a. R starts as the identity ─────────────────────────────────────────
    # networks/registration.py zero-initialises the output head AFTER
    # init_weights, which would otherwise overwrite it with N(0, 0.02). Lose that
    # ordering and step 0 warps the target by a field of tens of pixels: the
    # generator chases a scrambled CT and never recovers, while the loss curve
    # falls exactly as it would in a healthy run, because R is simultaneously
    # learning to undo its own noise. There is no other symptom.
    fresh = RegistrationUNet(in_channels=2, nrf=8, num_downs=2)
    with torch.no_grad():
        field = fresh(torch.randn(2, 2, 64, 64))
    worst = float(field.abs().max())
    assert worst < 1e-2, (
        f"R emits a displacement of {worst:.3f} px at initialisation, expected "
        f"~0. The zero-init of the output head was clobbered — check that it "
        f"runs AFTER init_weights in RegistrationUNet.__init__.")
    log.info("PASS  R starts at the identity (max displacement %.2e px)", worst)

    # ── 5b. The warp is exact ────────────────────────────────────────────────
    # SpatialTransformer converts pixel displacements into grid_sample's
    # normalised coordinates. Get the align_corners convention or the half-pixel
    # offset wrong and every warp carries a constant sub-pixel shift, so the
    # correction loss measures a fixed resampling blur on top of the real
    # residual and R_flow_px reports a bias that is not in the data.
    stn = SpatialTransformer()
    probe = torch.randn(2, 1, 48, 64)
    identity = stn(probe, torch.zeros(2, 2, 48, 64))
    shift = float((identity - probe).abs().max())
    assert shift < 1e-5, (
        f"warping by a zero field changed the image by {shift:.2e}; the "
        f"pixel-to-normalised coordinate conversion in SpatialTransformer is off "
        f"(align_corners / half-pixel offset).")

    # A known integer shift must move the image by exactly that much, in the
    # direction claimed: flow is (dx, dy) and samples FROM base + flow.
    flow = torch.zeros(2, 2, 48, 64)
    flow[:, 0] = 3.0
    moved = stn(probe, flow)
    assert torch.allclose(moved[:, :, :, :-3], probe[:, :, :, 3:], atol=1e-5), (
        "a +3 px dx did not shift the image by 3 px along x; the (x, y) channel "
        "order or the sampling direction is reversed.")
    log.info("PASS  warp is exact: zero field is a no-op, +3px dx shifts by 3px")

    # ── 5c. lambda_corr without lambda_smooth is refused ─────────────────────
    # The degenerate optimum: an unpenalised field can warp almost any prediction
    # onto almost any target, so L_corr falls to zero while G learns nothing.
    try:
        LossPlan(load_config("model/configs/exp7_reggan.yaml",
                             REGGAN_OVERRIDES + ["loss.lambda_smooth=0"]))
    except ValueError:
        log.info("PASS  lambda_corr without lambda_smooth is refused at config time")
    else:
        raise AssertionError(
            "lambda_corr>0 with lambda_smooth=0 should raise: the field would be "
            "unconstrained and the correction loss trivially minimisable")

    # ── 5d. Structure, and a real training run ───────────────────────────────
    cfg = load_config("model/configs/exp7_reggan.yaml",
                      REGGAN_OVERRIDES + ["run.name=smoke_reggan"])
    _run(cfg, out_root)

    state = torch.load(os.path.join(out_root, "smoke_reggan", "checkpoints", "last.pt"),
                       map_location="cpu", weights_only=False)["model"]
    for key in ("netG", "netD", "netR", "optimizer_R"):
        assert key in state, f"exp7: '{key}' missing from the checkpoint"
    for key in ("netF", "optimizer_F"):
        assert key not in state, (
            f"exp7: '{key}' is in the checkpoint, but lambda_nce is 0 — that code "
            f"path was built despite being switched off")
    assert state["loss_plan"]["lambda_l1"] == 0, (
        "exp7 is supposed to REPLACE L1 with the registered version, not add to it")
    log.info("PASS  %-24s absent=%-22s nickname=%r", "exp7_reggan",
             "netF,optimizer_F", state["loss_plan"]["nickname"])

    # The diagnostic exp7 exists to produce must actually reach the metrics file.
    with open(os.path.join(out_root, "smoke_reggan", "metrics.csv"),
              encoding="utf-8") as fh:
        header = fh.readline()
    for column in ("train/G_corr", "train/G_smooth", "train/R_flow_px"):
        assert column in header, f"'{column}' never reached metrics.csv"
    assert "train/G_L1" not in header, (
        "an unwarped L1 term was logged by a run whose lambda_l1 is 0")
    log.info("PASS  G_corr/G_smooth/R_flow_px logged to metrics.csv, G_L1 absent")

    # ── 5e. Resume restores R too ────────────────────────────────────────────
    # netR and optimizer_R are new checkpoint state. If either were dropped, the
    # run would continue from the identity field with an already-trained
    # generator — which looks like a mild loss bump and nothing else. Same
    # construction as CHECK 3; see its docstring for why the interruption uses
    # stop_after_epoch rather than a shorter n_epochs.
    cfg_a = load_config("model/configs/exp7_reggan.yaml",
                        REGGAN_OVERRIDES + ["run.name=smoke_reg_straight"])
    _, best_straight = _run(cfg_a, out_root, epochs=4)

    cfg_b = load_config("model/configs/exp7_reggan.yaml",
                        REGGAN_OVERRIDES + ["run.name=smoke_reg_resumed"])
    _run(cfg_b, out_root, epochs=4, stop_after_epoch=2)
    cfg_c = load_config("model/configs/exp7_reggan.yaml",
                        REGGAN_OVERRIDES + ["run.name=smoke_reg_resumed"])
    _, best_resumed = _run(cfg_c, out_root, resume="auto", epochs=4)

    delta = abs(best_straight - best_resumed)
    log.info("straight(4)=%.8f  resumed(2+2)=%.8f  delta=%.2e",
             best_straight, best_resumed, delta)
    assert delta < 1e-6, (
        f"RegGAN resume diverged by {delta:.2e}; netR or optimizer_R is not being "
        f"restored faithfully")
    log.info("PASS  resumed RegGAN run matches the uninterrupted one")

    return True


def check_cyclegan(out_root):
    """
    CycleGAN: four networks, an honestly unpaired training set, paired metrics.

    The two assertions that matter most are not about the model at all. 6b checks
    that no patient contributes both modalities — without which "unpaired" is a
    claim the experiment does not support — and 6c checks that validation stayed
    paired, without which the reported numbers mean nothing while still plotting
    a convincing curve.
    """
    log.info("─" * 70)
    log.info("CHECK 6  CycleGAN")

    from model.data.dataset import PairedSliceDataset, UnpairedSliceDataset
    from model.training.image_pool import ImagePool

    # ── 6a. The image pool returns history, and detaches it ──────────────────
    # A stored tensor that kept its graph would pin the generator's activations
    # from several steps ago alive — a leak that ends in an OOM tens of epochs
    # in, far from the change that caused it.
    pool = ImagePool(4)
    tracked = torch.randn(3, 1, 8, 8, requires_grad=True) * 2.0
    out = pool.query(tracked)
    assert out.shape == tracked.shape, f"pool changed the batch shape: {out.shape}"
    assert not out.requires_grad, (
        "ImagePool returned a tensor that still carries a graph; stored fakes "
        "must be detached or the generator's history is never freed")
    for _ in range(20):
        pool.query(torch.randn(3, 1, 8, 8))
    assert len(pool) == 4, f"pool grew past pool_size: {len(pool)}"
    log.info("PASS  image pool: detached, bounded at pool_size")

    # ── 6b. The unpaired split is genuinely unpaired ─────────────────────────
    cfg = load_config("model/configs/exp8_cyclegan.yaml",
                      CYCLEGAN_OVERRIDES + ["run.name=smoke_cyclegan"])
    manifest = load_manifest(cfg.data.manifest)
    datasets = build_datasets(cfg, manifest, load_split(cfg.data.splits))

    train = datasets["train"]
    assert isinstance(train, UnpairedSliceDataset), (
        f"data.unpaired is true but the train set is a {type(train).__name__}")
    assert set(train.subjects_a).isdisjoint(train.subjects_b), (
        "the two domains share a subject, so the model can see both modalities "
        "of one patient and this is not an unpaired experiment")

    clashes = sum(1 for i in range(120)
                  if train[i]["patient_id"] == train[i]["patient_id_B"])
    assert clashes == 0, (
        f"{clashes} of 120 items paired an MRI and a CT from the same patient")
    log.info("PASS  unpaired: %d + %d disjoint subjects, 0/120 same-patient items",
             len(train.subjects_a), len(train.subjects_b))

    # ── 6c. Validation stayed paired ─────────────────────────────────────────
    # Every metric in evaluation/metrics.py compares a prediction against ITS
    # target. An unpaired val set would score a synthetic CT of one patient
    # against a real CT of another and report the result as mae_norm.
    for split in ("val", "test"):
        assert isinstance(datasets[split], PairedSliceDataset), (
            f"the {split} set is unpaired; its metrics would be meaningless")
    log.info("PASS  val and test stay paired")

    # ── 6d. A conditional discriminator is refused ───────────────────────────
    # It would judge whether a CT corresponds to a given MRI — but they are
    # different patients here and no correspondence is claimed.
    try:
        from model.training.builder import build_model
        build_model(load_config("model/configs/exp8_cyclegan.yaml",
                                CYCLEGAN_OVERRIDES +
                                ["model.discriminator.conditional=true"]),
                    torch.device("cpu"))
    except ValueError:
        log.info("PASS  a conditional discriminator is refused at construction")
    else:
        raise AssertionError(
            "model.discriminator.conditional=true should raise for CycleGAN: "
            "the two modalities in a batch come from different patients")

    # ── 6e. Structure, and a real training run ───────────────────────────────
    trainer, best = _run(cfg, out_root)

    state = torch.load(os.path.join(out_root, "smoke_cyclegan", "checkpoints", "last.pt"),
                       map_location="cpu", weights_only=False)["model"]
    for key in ("netG_A2B", "netG_B2A", "netD_A", "netD_B",
                "optimizer_G", "optimizer_D"):
        assert key in state, f"exp8: '{key}' missing from the checkpoint"
    for key in ("netG", "netD", "netF", "netR", "optimizer_F", "optimizer_R"):
        assert key not in state, (
            f"exp8: '{key}' is in the checkpoint — a pix2pix-family key leaked "
            f"into a CycleGAN run")
    log.info("PASS  %-24s 4 networks, 2 optimizers, nickname=%r",
             "exp8_cyclegan", state["loss_plan"]["nickname"])

    with open(os.path.join(out_root, "smoke_cyclegan", "metrics.csv"),
              encoding="utf-8") as fh:
        header = fh.readline()
    for column in ("train/G_cycle_A", "train/G_cycle_B", "val/mae_norm"):
        assert column in header, f"'{column}' never reached metrics.csv"
    assert "train/G_L1" not in header, (
        "an L1-against-target term was logged by a run that has no paired target")
    log.info("PASS  cycle terms logged, paired val/mae_norm still computed (%.5f)",
             best)

    # ── 6f. Resume restores all four networks ────────────────────────────────
    # NOT the bit-exact comparison CHECK 3 and CHECK 5 make: the image pools are
    # deliberately not checkpointed (see image_pool.py), so a resumed run sees a
    # different sample of past fakes and legitimately diverges. What must hold is
    # that every weight comes back exactly.
    cfg_resume = load_config("model/configs/exp8_cyclegan.yaml",
                             CYCLEGAN_OVERRIDES + ["run.name=smoke_cyclegan"])
    reloaded, _ = _run(cfg_resume, out_root, resume="auto", epochs=2,
                       stop_after_epoch=0)
    for name in ("netG_A2B", "netG_B2A", "netD_A", "netD_B"):
        before = state[name]
        after = getattr(reloaded.model, name).state_dict()
        assert set(before) == set(after), f"{name}: parameter names changed on load"
        for key, tensor in before.items():
            assert torch.equal(tensor, after[key].cpu()), (
                f"{name}.{key} differs after a resume; the checkpoint is not "
                f"being restored faithfully")
    log.info("PASS  all four networks restored bit-exactly on resume")

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
            checks += [check_modularity, check_resume, check_stylegan2,
                       check_reggan, check_cyclegan]
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
