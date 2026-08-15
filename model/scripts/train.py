"""
train.py
────────
Training entry point.

    python model/scripts/train.py --config exp2_paper.yaml
    python model/scripts/train.py --config exp2_paper.yaml --set loss.lambda_nce=0
    python model/scripts/train.py --config exp3_nce_heavy.yaml --resume auto

--set takes dotted paths and rejects any key not declared in configs/base.yaml,
so a mistyped override stops the run instead of quietly training something other
than the experiment you meant.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from model.config import load_config                        # noqa: E402
from model.data.dataset import build_datasets               # noqa: E402
from model.data.manifest import load_manifest               # noqa: E402
from model.data.splits import load_split                    # noqa: E402
from model.training.trainer import Trainer                  # noqa: E402


def setup_logging(run_dir=None, level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(run_dir, "train.log"),
                                            encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train pix2pix + PatchNCE (MRI -> CT)")
    parser.add_argument("--config", required=True,
                        help="Path, or a filename inside model/configs/")
    parser.add_argument("--set", nargs="*", dest="overrides", default=[],
                        metavar="KEY=VALUE",
                        help="Dotted config overrides, e.g. loss.lambda_nce=0")
    parser.add_argument("--resume", default=None,
                        help="'auto' to continue this run's last.pt, or a path")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run validation once and exit (needs --resume)")
    parser.add_argument("--split", default="val",
                        help="Which split --eval-only scores")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = os.path.join(cfg.run.out_dir, cfg.run.name)
    setup_logging(run_dir)
    log = logging.getLogger("train")

    log.info("config: %s  (hash %s)", cfg.get("_config_path"), cfg.get("_hash"))

    manifest = load_manifest(cfg.data.manifest)
    split = load_split(cfg.data.splits)
    datasets = build_datasets(cfg, manifest, split)

    trainer = Trainer(cfg, datasets)
    trainer.maybe_resume(args.resume)

    if args.eval_only:
        if args.split not in trainer.loaders:
            parser.error(f"split '{args.split}' has no data")
        trainer.validate(trainer.start_epoch - 1, split=args.split)
        return 0

    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
