"""
export_run.py
─────────────
Bundle the artifacts worth keeping from a finished run.

    # on Kaggle, from a notebook cell
    !python /kaggle/input/mri-ct-model-code/model/scripts/export_run.py \
        --run /kaggle/working/runs/exp1_pix2pix

    # or, importable, which is tidier in a notebook
    from model.scripts.export_run import export_run
    export_run(trainer.run_dir)

    # locally
    python model/scripts/export_run.py --run model/runs/exp1_pix2pix

WHAT IT PRODUCES

  <name>_results.zip     logs, sample panels, resolved config.  A few MB.
                         THIS IS THE ONE THAT MATTERS. Losing best.pt costs a
                         re-run; losing the logs costs the experiment, because
                         a run with no history cannot be compared to anything.

  <name>_generator.pt    the EMA generator alone, for inference.  ~215 MB.
                         A full checkpoint also carries the discriminator, the
                         projection heads and three optimizer states — about
                         two-thirds of the file, all of which exist only to
                         resume training. This keeps the EMA weights, which are
                         what you should deploy anyway (see docs/training_
                         strategies.md on why the shadow beats the live model).

Full checkpoints are listed but not copied. Pass --include-checkpoints to fold
last.pt into the zip when you intend to resume in another session.
"""

import argparse
import json
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logger = logging.getLogger("export")

# Files copied into the results bundle, if present.
LOG_FILES = ("metrics.csv", "train_log.jsonl", "train.log", "config.resolved.yaml")


def _mb(path):
    return os.path.getsize(path) / 1e6


def export_generator(run_dir, out_dir, checkpoint="best"):
    """
    Write an inference-only file containing just the generator weights.

    Prefers the EMA shadow. If a run had `stabilizers.ema.enabled: false` there
    is no shadow, and the live weights are used instead — reported either way so
    it is never ambiguous which one you are holding.
    """
    import torch

    src = os.path.join(run_dir, "checkpoints", f"{checkpoint}.pt")
    if not os.path.isfile(src):
        logger.warning("no %s.pt in %s; skipping generator export",
                       checkpoint, run_dir)
        return None

    state = torch.load(src, map_location="cpu", weights_only=False)
    model = state["model"]

    ema = model.get("ema") or {}
    weights = ema.get("ema")
    source = "EMA"
    if weights is None:
        weights = model["netG"]
        source = "live (no EMA in this checkpoint)"

    payload = {
        "netG": weights,
        "weights_source": source,
        "epoch": state.get("epoch"),
        "best_value": state.get("best_value"),
        "selection_metric": state.get("selection_metric"),
        "config_hash": state.get("config_hash"),
        "run_name": state.get("run_name"),
        "loss_plan": model.get("loss_plan"),
    }

    name = os.path.basename(os.path.normpath(run_dir))
    dst = os.path.join(out_dir, f"{name}_generator.pt")
    torch.save(payload, dst)
    logger.info("generator: %s weights from epoch %s (%s)",
                source, state.get("epoch"), checkpoint)
    return dst


def export_results(run_dir, out_dir, include_checkpoints=False):
    """Zip the logs, sample panels and resolved config."""
    name = os.path.basename(os.path.normpath(run_dir))
    staging = os.path.join(out_dir, f"{name}_results")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)

    copied = []
    for filename in LOG_FILES:
        src = os.path.join(run_dir, filename)
        if os.path.isfile(src):
            shutil.copy(src, staging)
            copied.append(filename)
        else:
            logger.warning("missing %s", filename)

    samples = os.path.join(run_dir, "samples")
    n_panels = 0
    if os.path.isdir(samples):
        shutil.copytree(samples, os.path.join(staging, "samples"))
        n_panels = len(os.listdir(samples))

    if include_checkpoints:
        ckpt = os.path.join(run_dir, "checkpoints")
        if os.path.isdir(ckpt):
            shutil.copytree(ckpt, os.path.join(staging, "checkpoints"))
            copied.append("checkpoints/")

    archive = shutil.make_archive(staging, "zip", staging)
    shutil.rmtree(staging, ignore_errors=True)
    logger.info("results: %s + %d sample panels", ", ".join(copied), n_panels)
    return archive


def summarise(run_dir):
    """Pull the headline numbers out of the run's own log."""
    jsonl = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.isfile(jsonl):
        return None
    rows = []
    with open(jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    scored = [r for r in rows if "val/mae_norm" in r]
    if not scored:
        return None
    best = min(scored, key=lambda r: r["val/mae_norm"])
    return {
        "epochs": len(rows),
        "best_epoch": best.get("epoch"),
        "best_mae_norm": best.get("val/mae_norm"),
        "final_ssim": scored[-1].get("val/ssim"),
        "final_dice_bone": scored[-1].get("val/dice_bone"),
    }


def export_run(run_dir, out_dir=None, checkpoint="best",
               include_checkpoints=False, generator=True):
    """Export one run. Returns the paths written."""
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Not a run directory: {run_dir}")

    # Default beside the run, except on Kaggle where only /kaggle/working is
    # writable and is also what the Output panel exposes for download.
    if out_dir is None:
        out_dir = ("/kaggle/working" if os.path.isdir("/kaggle/working")
                   else os.path.dirname(run_dir))
    os.makedirs(out_dir, exist_ok=True)

    written = []
    archive = export_results(run_dir, out_dir, include_checkpoints)
    written.append(archive)

    if generator:
        gen = export_generator(run_dir, out_dir, checkpoint)
        if gen:
            written.append(gen)

    stats = summarise(run_dir)

    print()
    print("=" * 72)
    print(f"exported from  {run_dir}")
    if stats:
        print(f"  {stats['epochs']} epochs | best mae_norm "
              f"{stats['best_mae_norm']:.5f} @ epoch {stats['best_epoch']}"
              f" | final ssim {stats['final_ssim']:.4f}"
              f" | final dice_bone {stats['final_dice_bone']:.4f}")
    print("-" * 72)
    for path in written:
        print(f"  {_mb(path):8.1f} MB  {os.path.basename(path)}")

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(ckpt_dir) and not include_checkpoints:
        print("  --- not exported (resume only, pass --include-checkpoints) ---")
        for f in sorted(os.listdir(ckpt_dir)):
            print(f"  {_mb(os.path.join(ckpt_dir, f)):8.1f} MB  {f}")

    print("=" * 72)
    print("Take *_results.zip always — it is what makes this run comparable to")
    print("the others. Take *_generator.pt if you want to run inference.")
    print()
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bundle the artifacts worth keeping from a run")
    parser.add_argument("--run", required=True, help="A run directory")
    parser.add_argument("--out", default=None,
                        help="Where to write (default: /kaggle/working, else beside the run)")
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"],
                        help="Which checkpoint the generator export comes from")
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="Fold the full checkpoints into the zip (adds ~1.3 GB)")
    parser.add_argument("--no-generator", action="store_true",
                        help="Skip the inference-only weights export")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s",
                        force=True)
    export_run(args.run, args.out, args.checkpoint,
               args.include_checkpoints, not args.no_generator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
