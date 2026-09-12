"""
ladder_audit.py - reproduce every table in the review from results/.

Read-only. Reads results/*/metrics.csv and results/*/config.resolved.yaml, prints,
writes nothing.

    python review_test_Folder/_comparison/ladder_audit.py
    python review_test_Folder/_comparison/ladder_audit.py --table cliff
"""

import argparse
import pathlib

import pandas as pd
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
REGIONS = ["brain", "abdomen", "musculoskeletal", "spine"]

# Preprocessing/pipeline_config.py:208-229 - frozen per row into the manifest.
HU_SPAN = {"brain": 80.0, "abdomen": 400.0, "musculoskeletal": 500.0, "spine": 500.0}

# Reading order: worst last, so the eye lands on the failures.
ORDER = ["exp1_pix2pix", "exp2_paper", "exp3b_nce_6_pix48", "exp3b_nce_20_5",
         "exp4_nce_max", "exp8_cyclegan", "exp7_reggan"]


def load():
    runs = {}
    for d in RESULTS.iterdir():
        if not (d / "metrics.csv").is_file():
            continue
        name = d.name.replace("_results", "")
        m = pd.read_csv(d / "metrics.csv")
        cfg = yaml.safe_load((d / "config.resolved.yaml").read_text())
        runs[name] = dict(m=m, cfg=cfg, best=m.loc[m["val/mae_norm"].idxmin()],
                          last=m.iloc[-1])
    return [(n, runs[n]) for n in ORDER if n in runs] + \
           [(n, r) for n, r in sorted(runs.items()) if n not in ORDER]


def scoreboard(runs):
    print("== SCOREBOARD " + "=" * 62)
    print("all runs: 200 epochs, same val set (230 samples / 159 bone-eligible)\n")
    print(f"{'run':<20}{'L1':>5}{'NCE':>5}{'best':>8}{'@ep':>5}{'final':>8}"
          f"{'drift':>8}{'dice':>7}{'ssim':>7}")
    print("-" * 76)
    for n, r in runs:
        L = r["cfg"]["loss"]
        b, f = r["best"]["val/mae_norm"], r["last"]["val/mae_norm"]
        print(f"{n:<20}{L.get('lambda_l1', 0):>5.0f}{L.get('lambda_nce', 0):>5.0f}"
              f"{b:>8.4f}{int(r['best']['epoch']):>5}{f:>8.4f}"
              f"{100 * (f - b) / b:>7.1f}%{r['last']['val/dice_bone']:>7.3f}"
              f"{r['last']['val/ssim']:>7.3f}")
    print("\ndrift = how far the final epoch sits above the best one.")
    print("A best epoch of 0 or 7 means the run never learned anything.\n")


def per_region(runs):
    print("== PER-REGION, at each run's best epoch " + "=" * 36)
    print("\nmae_norm (the model-selection metric):")
    print(f"{'run':<20}" + "".join(f"{r[:9]:>10}" for r in REGIONS) + f"{'worst':>18}")
    print("-" * 76)
    for n, r in runs:
        v = {g: r["best"][f"val/mae_norm/{g}"] for g in REGIONS}
        w, b = max(v, key=v.get), min(v, key=v.get)
        print(f"{n:<20}" + "".join(f"{v[g]:>10.4f}" for g in REGIONS)
              + f"{w + f' ({v[w] / v[b]:.1f}x)':>18}")
    print("\nSpine is the worst region in EVERY run. It is 27 of 1687 training")
    print("slices (1.6%). No model or lambda change in this ladder touched it.\n")


def inversion(runs):
    print("== THE METRIC INVERSION " + "=" * 52)
    print("\nSame predictions, two metrics. mae_norm divides out the region's HU")
    print("window; mae_hu does not. The windows differ 6.25x:")
    for g in REGIONS:
        print(f"    {g:<18}{HU_SPAN[g]:>6.0f} HU across the full [0,1] range")
    print(f"\n{'run':<20}{'best by mae_norm':>26}{'best by mae_hu':>26}")
    print("-" * 76)
    flipped = 0
    for n, r in runs:
        bn = min(REGIONS, key=lambda g: r["best"][f"val/mae_norm/{g}"])
        bh = min(REGIONS, key=lambda g: r["best"][f"val/mae_hu/{g}"])
        flipped += bn != bh
        flag = "  <- disagree" if bn != bh else ""
        print(f"{n:<20}{bn:>26}{bh + flag:>26}")
    print(f"\n{flipped} of {len(runs)} runs disagree.")
    print("mae_norm calls musculoskeletal the best region; in HU it is among the")
    print("worst, because one normalised unit is 500 HU there and 80 HU in brain.")
    print("eval.selection_metric is mae_norm, so this is what picks best.pt.\n")
    print(f"{'run':<20}" + "".join(f"{g[:9]:>14}" for g in REGIONS))
    print("-" * 76)
    for n, r in runs:
        cells = [f"{r['best'][f'val/mae_norm/{g}']:.3f}={r['best'][f'val/mae_hu/{g}']:5.1f}HU"
                 for g in REGIONS]
        print(f"{n:<20}" + "".join(f"{c:>14}" for c in cells))
    print()


def cliff(runs):
    print("== THE L1:NCE CLIFF " + "=" * 56)
    print("\nThe five paired U-Net runs, ordered by how much L1 outweighs NCE.")
    print("train/G_L1 is the final-epoch training L1 - whether the term was ever")
    print("actually driven down.\n")
    print(f"{'run':<20}{'L1':>5}{'NCE':>5}{'ratio':>8}{'final G_L1':>12}"
          f"{'best mae':>10}{'dice':>8}")
    print("-" * 76)
    for n, r in runs:
        L = r["cfg"]["loss"]
        l1, nce = L.get("lambda_l1", 0), L.get("lambda_nce", 0)
        if not l1 or not nce:
            continue
        print(f"{n:<20}{l1:>5.0f}{nce:>5.0f}{l1 / nce:>7.0f}:1"
              f"{r['last']['train/G_L1']:>12.3f}{r['best']['val/mae_norm']:>10.4f}"
              f"{r['last']['val/dice_bone']:>8.3f}")
    print("\nThe cliff sits between 8:1 and 4:1. At 8:1 and above the L1 term is")
    print("still driven down (~0.09) and bone survives. At 4:1 and below it stalls")
    print("near 0.30 and bone Dice collapses by an order of magnitude.")
    print("\nNCE layer 0 never learns, in any run (chance = ln(257) = 5.549):")
    for n, r in runs:
        if "train/G_NCE_L0" in r["m"].columns:
            c = r["m"]["train/G_NCE_L0"]
            print(f"    {n:<20}{c.iloc[0]:.3f} -> {c.iloc[-1]:.3f}")
    print()


TABLES = {"scoreboard": scoreboard, "region": per_region,
          "inversion": inversion, "cliff": cliff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=list(TABLES) + ["all"], default="all")
    args = ap.parse_args()
    runs = load()
    if not runs:
        raise SystemExit(f"no runs with a metrics.csv under {RESULTS}")
    for name, fn in TABLES.items():
        if args.table in (name, "all"):
            fn(runs)


if __name__ == "__main__":
    main()
