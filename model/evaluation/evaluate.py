"""
evaluate.py
───────────
Score a saved checkpoint, and compare runs against each other.

    # one checkpoint on the validation split
    python model/evaluation/evaluate.py --run model/runs/exp3_nce_heavy

    # the final number, on the held-out test split
    python model/evaluation/evaluate.py --run model/runs/exp3_nce_heavy \
        --checkpoint best --split test

    # the comparison table across the whole experiment ladder
    python model/evaluation/evaluate.py --compare model/runs/exp*

A WORD ON THE TEST SPLIT. It exists to be looked at once, at the end, after the
configuration is frozen. Every time you read a test number and then change
something in response, the test set becomes a second validation set and its
ability to estimate generalisation degrades. Six subjects is not enough to
survive that being done repeatedly. The default split here is 'val' deliberately.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch                                                # noqa: E402

from model.config import load_config                        # noqa: E402
from model.data.dataset import build_datasets               # noqa: E402
from model.data.manifest import load_manifest               # noqa: E402
from model.data.splits import load_split                    # noqa: E402
from model.training.trainer import Trainer                  # noqa: E402

logger = logging.getLogger(__name__)

# Metrics worth putting side by side when comparing runs, and which direction
# is better for each.
COMPARE_METRICS = [
    ("mae_norm", "min"),
    ("psnr", "max"),
    ("ssim", "max"),
    ("dice_bone", "max"),
    ("mae_band_bone", "min"),
]


def evaluate_run(run_dir, checkpoint="best", split="val", overrides=None):
    """Load a run's resolved config and checkpoint, and score it."""
    config_path = os.path.join(run_dir, "config.resolved.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"No resolved config at {config_path}. Point --run at a directory "
            f"produced by train.py (it contains checkpoints/ and metrics.csv)."
        )

    cfg = load_config(config_path, overrides)
    ckpt_path = checkpoint
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(run_dir, "checkpoints", f"{checkpoint}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    manifest = load_manifest(cfg.data.manifest)
    split_map = load_split(cfg.data.splits)
    datasets = build_datasets(cfg, manifest, split_map)
    if split not in datasets:
        raise ValueError(f"split '{split}' has no data")

    trainer = Trainer(cfg, datasets)
    trainer.load_checkpoint(ckpt_path)
    results = trainer.validate(trainer.start_epoch - 1, split=split)

    results["_run"] = os.path.basename(os.path.normpath(run_dir))
    results["_checkpoint"] = os.path.basename(ckpt_path)
    results["_split"] = split
    return results, trainer.metrics


def compare_runs(run_dirs, checkpoint="best", split="val"):
    """Score several runs and print one table."""
    rows = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        try:
            results, _ = evaluate_run(run_dir, checkpoint, split)
            rows.append(results)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping %s: %s", run_dir, exc)

    if not rows:
        print("No runs could be scored.")
        return rows

    name_width = max(len(r["_run"]) for r in rows) + 2
    header = f"{'run':<{name_width}}" + "".join(
        f"{name:>15}" for name, _ in COMPARE_METRICS)
    print()
    print(f"Comparison on the '{split}' split, checkpoint '{checkpoint}'")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Mark the best value per column so the winner is visible at a glance.
    best = {}
    for name, mode in COMPARE_METRICS:
        values = [r[name] for r in rows if name in r]
        if values:
            best[name] = min(values) if mode == "min" else max(values)

    for row in rows:
        line = f"{row['_run']:<{name_width}}"
        for name, _ in COMPARE_METRICS:
            if name not in row:
                line += f"{'-':>15}"
                continue
            marker = "*" if abs(row[name] - best.get(name, 1e9)) < 1e-9 else " "
            line += f"{row[name]:>14.4f}{marker}"
        print(line)
    print("=" * len(header))
    print("* = best in column.  mae_norm is the selection metric; the others are "
          "diagnostics.")
    print("Read dice_bone and mae_band_bone together with mae_norm: a run that "
          "wins on\nmae_norm while losing on dice_bone is trading real bone "
          "accuracy for average-case\nsmoothness, which is the failure mode "
          "these metrics exist to catch.")
    print()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate or compare trained runs")
    parser.add_argument("--run", help="A run directory under model/runs/")
    parser.add_argument("--compare", nargs="*", help="Several run directories")
    parser.add_argument("--checkpoint", default="best",
                        help="'best', 'last', or a path (default: best)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--set", nargs="*", dest="overrides", default=[])
    parser.add_argument("--json-out", help="Write the results as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s",
                        force=True)
    logging.getLogger("model.training.trainer").setLevel(logging.WARNING)

    if args.compare:
        rows = compare_runs(args.compare, args.checkpoint, args.split)
        payload = rows
    elif args.run:
        results, accumulator = evaluate_run(args.run, args.checkpoint, args.split,
                                            args.overrides)
        print()
        print(f"{results['_run']}  [{results['_checkpoint']}]  split={args.split}")
        print(accumulator.format_table(results))
        print()
        payload = results
    else:
        parser.error("pass --run or --compare")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=float)
        logger.info("wrote %s", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
