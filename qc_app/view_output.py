"""
view_output.py
──────────────
Look at the .npy pairs this tool wrote.

    python -m qc_app.view_output                       browse everything
    python -m qc_app.view_output --patient PA0_Ranjeet
    python -m qc_app.view_output --series PA0_Ranjeet/ST0/SE0
    python -m qc_app.view_output --info                summary, no window
    python -m qc_app.view_output --export pngs/        write PNGs instead

Pairs are found by mirroring the output tree: every CT/<patient>/<study>/<series>/<file>.npy
is matched with the MRI file of the same relative path. The MRI series folder can
be named differently (SE0 vs SE0_axial), so that case is resolved by looking up
the sibling directory rather than assuming the names match.

Navigation
    right / left, or n / p    next / previous pair
    up / down                 next / previous SERIES
    1 2 3 4 5                 CT | MRI | overlay | difference | all four
    o                         cycle overlay opacity
    g                         toggle a 20 mm grid
    r                         re-read crops and nudges from the manifest
    s                         save the current view as a PNG
    q                         quit

Crop rectangles are read from the LIVE manifest (qc_workspace/qc.db), not from
metadata.csv, so a box drawn in the review app shows up here on the next `r`
without anyone having to press Export CSV. Three are drawn, in their own
colours: the amber metric ROI, and the blue/green per-modality export rects.

Read-only. It opens .npy files and writes nothing unless you ask for --export
or press `s`.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


# ── discovery ─────────────────────────────────────────────────────────────────

def find_pairs(output_dir, patient=None, series=None):
    """
    Every CT/MRI .npy pair under output_dir, in review order.

    Returns a list of dicts: patient, study, series, name, ct, mri.
    """
    ct_root = os.path.join(output_dir, "CT")
    mri_root = os.path.join(output_dir, "MRI")
    if not os.path.isdir(ct_root):
        raise SystemExit(f"no CT directory under {output_dir} - nothing has been accepted yet")

    pairs = []
    for dirpath, _dirnames, filenames in os.walk(ct_root):
        npys = [f for f in filenames if f.endswith(".npy")]
        if not npys:
            continue
        rel = os.path.relpath(dirpath, ct_root)              # PA0_Ranjeet/ST0/SE0
        parts = rel.split(os.sep)
        if len(parts) < 3:
            continue
        pat, study, ser = parts[0], parts[1], parts[2]

        if patient and pat != patient:
            continue
        if series and rel.replace(os.sep, "/") != series.replace("\\", "/"):
            continue

        # The MRI series folder may carry a suffix the CT one does not
        # (SE0 vs SE0_axial), so resolve it rather than assuming.
        mri_study_dir = os.path.join(mri_root, pat, study)
        mri_dir = os.path.join(mri_study_dir, ser)
        if not os.path.isdir(mri_dir) and os.path.isdir(mri_study_dir):
            cands = [d for d in sorted(os.listdir(mri_study_dir))
                     if d.split("_")[0] == ser.split("_")[0]]
            if cands:
                mri_dir = os.path.join(mri_study_dir, cands[0])

        for f in sorted(npys, key=_num_key):
            mri_path = os.path.join(mri_dir, f)
            if not os.path.exists(mri_path):
                # CT and MRI stems can differ; fall back to position in the folder.
                alt = sorted((x for x in os.listdir(mri_dir) if x.endswith(".npy")),
                             key=_num_key) if os.path.isdir(mri_dir) else []
                idx = sorted(npys, key=_num_key).index(f)
                if idx < len(alt):
                    mri_path = os.path.join(mri_dir, alt[idx])
                else:
                    continue
            pairs.append({
                "patient": pat, "study": study, "series": ser,
                "name": os.path.splitext(f)[0],
                "ct": os.path.join(dirpath, f), "mri": mri_path,
            })

    pairs.sort(key=lambda p: (p["patient"], p["study"], p["series"], _num_key(p["name"])))
    return pairs


def _num_key(s):
    """Sort IM2 before IM10 - numeric where there is a number, else alphabetic."""
    digits = "".join(c for c in os.path.splitext(str(s))[0] if c.isdigit())
    return (int(digits) if digits else 0, str(s))


def _key(patient, study, series, name):
    """Identity of a pair, independent of where its files ended up."""
    return (patient, study, series, os.path.splitext(str(name))[0])


def load_meta(output_dir, db_path=None):
    """
    Current state for every pair, keyed by (patient, study, series, stem).

    Read from the MANIFEST, not from metadata.csv. The CSV is a snapshot taken
    the last time Export CSV was pressed, and it only holds accepted pairs - so
    a crop drawn or adjusted since then is invisible in it. The manifest is
    what the review app is actually writing to, which is what makes a box you
    just dragged show up here on the next keypress.

    Falls back to the CSV when there is no database (someone handed you an
    output folder on its own).
    """
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "qc.db")

    if os.path.exists(db_path):
        import sqlite3
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        out = {}
        try:
            rows = con.execute("""
                SELECT p.patient, p.study, p.ct_series, p.ct_file, p.qc_status,
                       p.roi_json, p.roi_mode, p.ct_rect_json, p.mri_rect_json,
                       p.nudge_dx, p.nudge_dy, p.erase_json, p.is_background,
                       s.orientation, s.body_region, s.reg_applied,
                       s.reg_dx_mm, s.reg_dy_mm, s.reg_nmi_gain, s.reg_reason,
                       s.auto_erase
                FROM pairs p JOIN series s ON s.key = p.series_key""").fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            con.close()

        for r in rows:
            j = lambda v: json.loads(v) if v else None
            out[_key(r["patient"], r["study"], r["ct_series"], r["ct_file"])] = {
                "roi": j(r["roi_json"]), "roi_mode": r["roi_mode"],
                "ct_rect": j(r["ct_rect_json"]), "mri_rect": j(r["mri_rect_json"]),
                "manual_dx_mm": r["nudge_dx"] or 0, "manual_dy_mm": r["nudge_dy"] or 0,
                "erased": bool(r["erase_json"]), "auto_erase": bool(r["auto_erase"]),
                "is_background": bool(r["is_background"]),
                "qc_status": r["qc_status"], "orientation": r["orientation"],
                "body_region": r["body_region"],
                "reg_applied": bool(r["reg_applied"]),
                "reg_dx_mm": r["reg_dx_mm"] or 0.0, "reg_dy_mm": r["reg_dy_mm"] or 0.0,
                "reg_nmi_gain": r["reg_nmi_gain"], "reg_reason": r["reg_reason"],
                "source": "manifest",
            }
        if out:
            return out

    # ── fallback: the CSV snapshot ───────────────────────────────────────────
    path = os.path.join(output_dir, "metadata.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            num = lambda k: (float(row[k]) if row.get(k) not in (None, "") else None)
            roi = None
            if row.get("roi_w"):
                try:
                    roi = [float(row["roi_x"]), float(row["roi_y"]),
                           float(row["roi_w"]), float(row["roi_h"])]
                except ValueError:
                    roi = None
            out[_key(row["patient_id"], row.get("study", "ST0"),
                     row["ct_series"], row.get("ct_file", ""))] = {
                "roi": roi, "roi_mode": row.get("roi_mode"),
                "ct_rect": None, "mri_rect": None,
                "manual_dx_mm": num("manual_dx_mm") or 0,
                "manual_dy_mm": num("manual_dy_mm") or 0,
                "erased": row.get("erased") == "True", "auto_erase": False,
                "is_background": row.get("is_background") == "True",
                "qc_status": row.get("qc_status"), "orientation": row.get("orientation"),
                "body_region": row.get("body_region"),
                "reg_applied": row.get("reg_applied") == "True",
                "reg_dx_mm": num("reg_dx_mm") or 0.0,
                "reg_dy_mm": num("reg_dy_mm") or 0.0,
                "reg_nmi_gain": num("reg_nmi_gain"), "reg_reason": row.get("reg_note"),
                "source": "metadata.csv",
            }
    return out


def meta_for(meta, p):
    """The record for one discovered pair, or an empty dict."""
    return meta.get(_key(p["patient"], p["study"], p["series"], p["name"]), {})


# ── summary ───────────────────────────────────────────────────────────────────

def print_info(pairs, meta):
    by_series = {}
    for p in pairs:
        by_series.setdefault(f"{p['patient']}/{p['study']}/{p['series']}", []).append(p)

    print(f"\n  {len(pairs)} pairs in {len(by_series)} series\n")
    src = next((v.get("source") for v in meta.values()), "none")
    print(f"  crop / nudge state read from: {src}\n")
    print(f"  {'series':28s} {'n':>4s}  {'shape':>11s}  {'shift mm':>11s} "
          f"{'manual':>9s}  {'cropped':>14s}  reg")
    for key, items in sorted(by_series.items()):
        a = np.load(items[0]["ct"], mmap_mode="r")
        m = meta_for(meta, items[0])
        shift = (f"{m.get('reg_dx_mm', 0):+.0f},{m.get('reg_dy_mm', 0):+.0f}" if m else "-")
        man = (f"{m.get('manual_dx_mm', 0):+.0f},{m.get('manual_dy_mm', 0):+.0f}" if m else "-")
        rois = [meta_for(meta, it).get("roi") for it in items]
        n_roi = sum(1 for r in rois if r)
        uniq = len({tuple(r) for r in rois if r})
        crop = (f"{n_roi}/{len(items)}" + (f" ({uniq} boxes)" if uniq > 1 else "")) if n_roi else "-"
        print(f"  {key:28s} {len(items):4d}  {a.shape[0]:4d}x{a.shape[1]:<4d}  "
              f"{shift:>11s} {man:>9s}  {crop:>14s}  "
              f"{'applied' if m.get('reg_applied') else 'no shift'}")

    shapes = {}
    for p in pairs:
        s = np.load(p["ct"], mmap_mode="r").shape
        shapes[s] = shapes.get(s, 0) + 1
    print(f"\n  distinct CT shapes: {len(shapes)}")
    for s, n in sorted(shapes.items(), key=lambda kv: -kv[1])[:8]:
        print(f"     {s[0]}x{s[1]}: {n} slices")
    print()


# ── export ────────────────────────────────────────────────────────────────────

def export_pngs(pairs, dest):
    from PIL import Image
    os.makedirs(dest, exist_ok=True)
    for i, p in enumerate(pairs, 1):
        ct = np.load(p["ct"]).astype(np.float32)
        mri = _match(np.load(p["mri"]).astype(np.float32), ct.shape)
        strip = np.hstack([ct, np.full((ct.shape[0], 2), 0.7, np.float32), mri])
        img = (np.clip(strip, 0, 1) * 255).astype(np.uint8)
        out = os.path.join(dest, f"{p['patient']}_{p['series']}_{p['name']}.png")
        Image.fromarray(img, mode="L").save(out)
        if i % 100 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}")
    print(f"\n  wrote {len(pairs)} PNGs to {dest}")


def _draw_roi(ax, m):
    """
    The crop rectangle, if this pair has one.

    Drawn as an outline rather than applied, because a `metric` ROI never
    crops the exported pixels - it only restricts what the registration metric
    looked at. Showing it here is what makes that visible instead of leaving
    you wondering why the export is full-frame.
    """
    import matplotlib.patches as mpatches
    for key, colour, label in (("roi", "#d4a13c", "metric"),
                               ("ct_rect", "#4ea3d8", "CT export"),
                               ("mri_rect", "#4bb573", "MRI export")):
        r = m.get(key)
        if not r or len(r) != 4 or r[2] < 1 or r[3] < 1:
            continue
        x, y, w, h = (float(v) for v in r)
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                        edgecolor=colour, lw=1.0, ls="--"))
        ax.text(x, y - 3, f"{label} {w:.0f}x{h:.0f} mm", color=colour, fontsize=7)


def _panel(fig, ct, mri, state, p, meta):
    """CT, MRI, overlay and difference in one figure."""
    m = meta_for(meta, p)
    specs = [("CT", ct, "gray"), ("MRI", mri, "gray"),
             (f"overlay {state['alpha']:.0%}", None, None),
             ("|CT - MRI|", np.abs(ct - mri), "inferno")]
    for k, (title, img, cmap) in enumerate(specs, 1):
        ax = fig.add_subplot(2, 2, k)
        ax.set_facecolor("black")
        if title.startswith("overlay"):
            ax.imshow(ct, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.imshow(mri, cmap="gray", vmin=0, vmax=1, alpha=state["alpha"],
                      interpolation="nearest")
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        _draw_roi(ax, m)
        ax.set_title(title, color="#8b98a5", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    bits = [f"{p['patient']} / {p['series']} / {p['name']}   "
            f"{state['i']+1} of {len(meta) or '?'}   {ct.shape[1]}x{ct.shape[0]} px (= mm)"]
    if m:
        line = (f"shift {m.get('reg_dx_mm',0):+.0f}, {m.get('reg_dy_mm',0):+.0f} mm "
                f"({'applied' if m.get('reg_applied') else 'NOT applied'})")
        mdx, mdy = m.get("manual_dx_mm", 0), m.get("manual_dy_mm", 0)
        if mdx or mdy:
            line += f"   manual {mdx:+.0f}, {mdy:+.0f} mm"
        if m.get("roi"):
            line += f"   crop {m['roi'][2]:.0f}x{m['roi'][3]:.0f}"
        if m.get("erased"):
            line += "   erased"
        bits.append(line)
    fig.suptitle("\n".join(bits), color="#dde5ec", fontsize=9, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))


def _match(a, shape):
    """Pad or trim to `shape` - CT and MRI can differ if per-modality crops were used."""
    if a.shape == tuple(shape):
        return a
    out = np.zeros(shape, dtype=a.dtype)
    h, w = min(a.shape[0], shape[0]), min(a.shape[1], shape[1])
    out[:h, :w] = a[:h, :w]
    return out


# ── viewer ────────────────────────────────────────────────────────────────────

def browse(pairs, meta, out_dir=None, db_path=None):
    import matplotlib.pyplot as plt

    state = {"i": 0, "mode": 4, "alpha": 0.5, "grid": False}
    MODES = ["CT", "MRI", "overlay", "difference", "all four"]

    fig = plt.figure(figsize=(9, 9.4))
    fig.canvas.manager.set_window_title("qc_app output viewer")
    fig.patch.set_facecolor("#0d1013")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")

    def draw():
        p = pairs[state["i"]]
        ct = np.load(p["ct"]).astype(np.float32)
        mri = _match(np.load(p["mri"]).astype(np.float32), ct.shape)
        mode = MODES[state["mode"]]

        # Panel view: everything at once, which is what makes an alignment
        # judgeable at a glance rather than by flicking between modes.
        fig.clf()
        if mode == "all four":
            _panel(fig, ct, mri, state, p, meta)
            fig.canvas.draw_idle()
            return
        ax = fig.add_subplot(111)
        ax.set_facecolor("black")
        if mode == "CT":
            ax.imshow(ct, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        elif mode == "MRI":
            ax.imshow(mri, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        elif mode == "overlay":
            ax.imshow(ct, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.imshow(mri, cmap="gray", vmin=0, vmax=1, alpha=state["alpha"],
                      interpolation="nearest")
        else:
            ax.imshow(np.abs(ct - mri), cmap="inferno", vmin=0, vmax=1,
                      interpolation="nearest")

        if state["grid"]:
            # 1 px = 1 mm, so a 20 px grid is a 20 mm ruler.
            for x in range(0, ct.shape[1], 20):
                ax.axvline(x, color="#4ea3d8", lw=0.3, alpha=0.4)
            for y in range(0, ct.shape[0], 20):
                ax.axhline(y, color="#4ea3d8", lw=0.3, alpha=0.4)

        m = meta_for(meta, p)
        _draw_roi(ax, m)
        bits = [f"{p['patient']} / {p['series']} / {p['name']}",
                f"{state['i']+1} of {len(pairs)}   {ct.shape[1]}x{ct.shape[0]} px (= mm)   [{mode}"
                + (f" {state['alpha']:.0%}]" if mode == "overlay" else "]")]
        if m:
            line = (f"shift {m.get('reg_dx_mm',0):+.0f}, {m.get('reg_dy_mm',0):+.0f} mm "
                    f"({'applied' if m.get('reg_applied') else 'NOT applied'})")
            mdx, mdy = m.get("manual_dx_mm", 0), m.get("manual_dy_mm", 0)
            if mdx or mdy:
                line += f"   manual {mdx:+.0f}, {mdy:+.0f} mm"
            if m.get("roi"):
                line += f"   crop {m['roi'][2]:.0f}x{m['roi'][3]:.0f}"
            if m.get("erased"):
                line += "   erased"
            if m.get("auto_erase"):
                line += "   auto-rails"
            if m.get("is_background"):
                line += "   background"
            bits.append(line)
            if m.get("orientation"):
                bits.append(f"{m['orientation']}  ·  {m.get('body_region','')}"
                            f"  ·  CT window {m.get('reg_note','')[:0]}")
        ax.set_title("\n".join(bits), color="#dde5ec", fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw_idle()

    def series_of(i):
        p = pairs[i]
        return (p["patient"], p["study"], p["series"])

    def jump_series(delta):
        cur = series_of(state["i"])
        rng = range(state["i"] + 1, len(pairs)) if delta > 0 else range(state["i"] - 1, -1, -1)
        for j in rng:
            if series_of(j) != cur:
                # land on the FIRST slice of that series when going forward
                if delta > 0:
                    state["i"] = j
                else:
                    tgt = series_of(j)
                    while j > 0 and series_of(j - 1) == tgt:
                        j -= 1
                    state["i"] = j
                return

    def on_key(e):
        if e.key in ("right", "n"):
            state["i"] = min(state["i"] + 1, len(pairs) - 1)
        elif e.key in ("left", "p"):
            state["i"] = max(state["i"] - 1, 0)
        elif e.key == "up":
            jump_series(-1)
        elif e.key == "down":
            jump_series(1)
        elif e.key in "12345":
            state["mode"] = int(e.key) - 1
        elif e.key == "o":
            state["alpha"] = {0.25: 0.5, 0.5: 0.75, 0.75: 0.25}.get(state["alpha"], 0.5)
            state["mode"] = 2
        elif e.key == "g":
            state["grid"] = not state["grid"]
        elif e.key == "s":
            p = pairs[state["i"]]
            out = f"{p['patient']}_{p['series']}_{p['name']}_{MODES[state['mode']]}.png"
            fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
            print(f"  saved {out}")
            return
        elif e.key == "r":
            # Re-read the manifest, so boxes drawn in the app since this window
            # opened appear without restarting it.
            meta.clear()
            meta.update(load_meta(out_dir, db_path))
            print("  reloaded crops/nudges from the manifest")
        elif e.key == "q":
            plt.close(fig)
            return
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    print("\n  arrows / n p = slice   up down = series   1-4 = view   o = opacity"
          "   g = grid   s = save   q = quit\n")
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv=None):
    from . import config as app_config

    default_out = app_config.AppConfig().output_dir
    ap = argparse.ArgumentParser(
        prog="python -m qc_app.view_output",
        description="View the .npy pairs written by the QC tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output_dir", default=default_out, help="Directory holding CT/ and MRI/")
    ap.add_argument("--patient", default=None, help="Only this patient, e.g. PA0_Ranjeet")
    ap.add_argument("--series", default=None,
                    help="Only this series, e.g. PA0_Ranjeet/ST0/SE0")
    ap.add_argument("--db", default=None,
                    help="Manifest to read crops/nudges from (default: <workspace>/qc.db)")
    ap.add_argument("--info", action="store_true", help="Print a summary and exit")
    ap.add_argument("--export", metavar="DIR", default=None,
                    help="Write side-by-side PNGs to DIR instead of opening a window")
    args = ap.parse_args(argv)

    pairs = find_pairs(args.output_dir, args.patient, args.series)
    if not pairs:
        print(f"  no CT/MRI .npy pairs found under {args.output_dir}"
              + (f" for {args.patient or args.series}" if (args.patient or args.series) else ""))
        return 1

    meta = load_meta(args.output_dir, args.db)
    if args.info:
        print_info(pairs, meta)
    elif args.export:
        export_pngs(pairs, args.export)
    else:
        browse(pairs, meta, args.output_dir, args.db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
