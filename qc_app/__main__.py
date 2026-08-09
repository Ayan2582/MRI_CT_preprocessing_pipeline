"""
Entry point:  python -m qc_app

Scans the dataset on first start, then serves the review UI on localhost. The
scan is skipped when the manifest already has rows, so restarting is instant;
--rescan forces it, which is also what the "Rescan" button in the UI does.
"""

import argparse
import sys
import webbrowser

import uvicorn

from . import config as app_config
from . import db as db_mod, registration_service as regsvc, scanner


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m qc_app",
        description="Local CT/MRI registration QC tool, built on the existing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root", default=app_config.DEFAULT_DATA_ROOT,
                   help="Dataset root containing CT/ and MRI/ (read-only)")
    p.add_argument("--workspace", default=app_config.DEFAULT_WORKSPACE,
                   help="Where the manifest, cache and outputs are written")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address. Leave on loopback unless you mean otherwise")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reg_search_mm", type=float, default=None,
                   help="Registration search half-width in mm (default: pipeline_config)")
    p.add_argument("--n4_shrink", type=int, default=None,
                   help="N4 in-plane shrink factor (default: pipeline_config)")
    p.add_argument("--prefetch", type=int, default=2,
                   help="Series registered ahead of the reviewer")
    p.add_argument("--rescan", action="store_true",
                   help="Re-walk the dataset even if the manifest is populated")
    p.add_argument("--scan_only", action="store_true",
                   help="Scan, print the report, and exit without serving")
    p.add_argument("--no_browser", action="store_true",
                   help="Do not open a browser window on start")
    return p.parse_args(argv)


def build_config(args) -> app_config.AppConfig:
    cfg = app_config.AppConfig(data_root=args.data_root, workspace=args.workspace)
    if args.reg_search_mm is not None:
        cfg.reg_search_mm = args.reg_search_mm
    if args.n4_shrink is not None:
        cfg.n4_shrink = args.n4_shrink
    cfg.prefetch_depth = args.prefetch
    cfg.ensure_dirs()
    return cfg


def run_scan(cfg, verbose: bool = True):
    manifest = db_mod.Manifest(cfg.db_path)
    report = scanner.scan_dataset(cfg)
    added = manifest.sync_scan(report, regsvc.region_for_patient)
    manifest.set_setting("scan_problems", report.problems)
    counts = manifest.counts()
    manifest.close()

    if verbose:
        print(f"\n  Dataset      : {cfg.data_root}")
        print(f"  Series pairs : {len(report.pairs)}")
        print(f"  Slice pairs  : {report.n_slice_pairs}")
        print(f"  New this scan: {added['added_series']} series, {added['added_pairs']} pairs")
        if report.problems:
            print(f"\n  {len(report.problems)} problem(s):")
            for p in report.problems[:25]:
                print(f"    [{p['level']:7s}] {p['where']}: {p['message']}")
            if len(report.problems) > 25:
                print(f"    ... and {len(report.problems) - 25} more")
        print(f"\n  Accepted {counts['accepted']} | Rejected {counts['rejected']} | "
              f"Pending {counts['pending']} | Errors {counts['errors']}\n")
    return report, counts


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    manifest = db_mod.Manifest(cfg.db_path)
    populated = manifest.query_one("SELECT COUNT(*) AS n FROM pairs")["n"] > 0
    manifest.close()

    if args.scan_only or args.rescan or not populated:
        run_scan(cfg)
        if args.scan_only:
            return 0

    from .server import create_app
    app = create_app(cfg)

    url = f"http://{args.host}:{args.port}/"
    print("=" * 64)
    print("  CT/MRI Registration QC")
    print("=" * 64)
    print(f"  Dataset (read-only) : {cfg.data_root}")
    print(f"  Workspace           : {cfg.workspace}")
    print(f"  Outputs             : {cfg.output_dir}")
    print(f"  Geometry            : 2D, 1 mm/px from PixelSpacing "
          f"(registration_idea method)")
    print(f"  Serving             : {url}")
    print("=" * 64)

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
