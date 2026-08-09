"""
server.py
─────────
FastAPI application: the JSON API and the static UI.

Series keys and pair ids contain slashes ("PA0_Ranjeet/ST0/SE0/0007"), so they
travel as query parameters rather than path segments. That keeps them readable
in the browser's network tab and avoids depending on how a proxy normalises
encoded slashes in a path.

Every route is read-mostly and cheap except the ones that explicitly queue
work. Registration itself never happens inside a request - it is submitted to
the worker and polled - which is what keeps the UI responsive while a series is
being processed.
"""

import json
import os
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as app_config
from . import db as db_mod
from . import outputs, render, registration_service as regsvc, scanner
from .worker import PRIORITY_PREFETCH, PRIORITY_USER, RegistrationWorker

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_app(cfg: app_config.AppConfig | None = None) -> FastAPI:
    cfg = cfg or app_config.AppConfig()
    cfg.ensure_dirs()

    manifest = db_mod.Manifest(cfg.db_path)

    # A series marked PROCESSING at startup was interrupted by the last
    # shutdown - no worker is touching it, and nothing will ever finish it.
    # Returning it to PENDING is what makes it eligible for prefetch again
    # instead of sitting in a state that only looks busy.
    stranded = manifest.execute(
        "UPDATE series SET reg_status='PENDING' WHERE reg_status='PROCESSING'").rowcount
    if stranded:
        print(f"  Requeued {stranded} series interrupted by the last shutdown")

    worker = RegistrationWorker(cfg, manifest)

    app = FastAPI(title="CT-MRI Registration QC", version="1.0")
    app.state.cfg = cfg
    app.state.db = manifest
    app.state.worker = worker
    app.state.scan_problems = manifest.get_setting("scan_problems", []) or []

    # ── helpers ──────────────────────────────────────────────────────────────
    def series_of(pair_row: dict) -> dict:
        row = manifest.query_one("SELECT * FROM series WHERE key=?",
                                 (pair_row["series_key"],))
        if row is None:
            raise HTTPException(404, f"series {pair_row['series_key']} not in manifest")
        return row

    def pair_by_id(pair_id: str) -> dict:
        row = manifest.query_one("SELECT * FROM pairs WHERE pair_id=?", (pair_id,))
        if row is None:
            raise HTTPException(404, f"no pair '{pair_id}'")
        return row

    def ensure_registered(series_row: dict, priority: int = PRIORITY_USER):
        """
        Cache for this series, queueing its registration if there is none.

        Returns None when the caller must wait; the UI polls /api/status and
        re-requests. This is the only place that decides a series needs work,
        so a cache deleted from disk behind the app's back self-heals.
        """
        cache = None
        if series_row.get("cache_path"):
            cache = worker.get_cache(series_row["key"], series_row["cache_path"])
        if cache is None and series_row["reg_status"] != "ERROR":
            worker.submit(series_row["key"], priority=priority)
        return cache

    def prefetch_from(seq: int) -> None:
        """Queue the series just ahead of the reviewer, at low priority."""
        rows = manifest.query(
            "SELECT DISTINCT series_key FROM pairs WHERE seq > ? "
            "ORDER BY seq LIMIT 400", (seq,))
        queued = 0
        for r in rows:
            if queued >= cfg.prefetch_depth:
                break
            s = manifest.query_one(
                "SELECT key, cache_path, reg_status FROM series WHERE key=?",
                (r["series_key"],))
            if s and s["reg_status"] == "PENDING":
                if worker.submit(s["key"], priority=PRIORITY_PREFETCH):
                    queued += 1

    # ── config & status ──────────────────────────────────────────────────────
    @app.get("/api/config")
    def get_config():
        return {
            "data_root":      cfg.data_root,
            "workspace":      cfg.workspace,
            "output_dir":     cfg.output_dir,
            "ct_root":        cfg.ct_root,
            "mri_root":       cfg.mri_root,
            "target_spacing": cfg.target_spacing,
            "reg_search_mm":  cfg.reg_search_mm,
            "n4_shrink":      cfg.n4_shrink,
            "prefetch_depth": cfg.prefetch_depth,
            "registration":   app_config.registration_settings(),
            "last_seq":       manifest.get_setting("last_seq", 0),
        }

    @app.get("/api/status")
    def get_status():
        return {"counts": manifest.counts(), "worker": worker.status(),
                "problems": app.state.scan_problems}

    class ScanRequest(BaseModel):
        data_root: str | None = None

    @app.post("/api/scan")
    def post_scan(req: ScanRequest | None = None):
        """
        Re-walk the dataset and merge what is found into the manifest.

        This is "Add to dataset": drop a new patient into CT/ and MRI/ under
        the same layout, press it, and the new series arrive as PENDING with
        their slice pairs enumerated. Existing rows keep their QC decisions -
        the merge is additive, never destructive.
        """
        if req and req.data_root:
            if not os.path.isdir(req.data_root):
                raise HTTPException(400, f"not a directory: {req.data_root}")
            cfg.data_root = req.data_root

        report = scanner.scan_dataset(cfg)
        added = manifest.sync_scan(report, regsvc.region_for_patient)
        app.state.scan_problems = report.problems
        manifest.set_setting("scan_problems", report.problems)
        return {
            "series_found":  len(report.pairs),
            "slice_pairs":   report.n_slice_pairs,
            "added_series":  added["added_series"],
            "added_pairs":   added["added_pairs"],
            "problems":      report.problems,
            "counts":        manifest.counts(),
        }

    # ── navigation ───────────────────────────────────────────────────────────
    @app.get("/api/pairs")
    def list_pairs(status: str = Query("all"), patient: str | None = None,
                   q: str | None = None, limit: int = 200, offset: int = 0,
                   around_seq: int | None = None):
        where, params = [], []
        # "nudged" and "erased" are not QC states - they are properties of the
        # edit, and there was previously no way to find them again once a
        # reviewer moved on. Exposed here as pseudo-filters rather than new
        # columns, so the status vocabulary stays PENDING/ACCEPTED/REJECTED/ERROR.
        if status == "nudged":
            where.append("(p.nudge_dx != 0 OR p.nudge_dy != 0)")
        elif status == "erased":
            where.append("p.erase_json IS NOT NULL")
        elif status and status.lower() != "all":
            where.append("p.qc_status = ?")
            params.append(status.upper())
        if patient:
            where.append("p.patient = ?")
            params.append(patient)
        if q:
            where.append("(p.patient LIKE ? OR p.ct_series LIKE ? OR "
                         "p.mri_series LIKE ? OR p.ct_file LIKE ?)")
            params.extend([f"%{q}%"] * 4)
        clause = ("WHERE " + " AND ".join(where)) if where else ""

        total = manifest.query_one(
            f"SELECT COUNT(*) AS n FROM pairs p {clause}", tuple(params))["n"]

        # `around_seq` asks for the window near a given pair. The offset must be
        # its RANK WITHIN THIS FILTERED SET, not its global seq: with a filter
        # applied, a reviewer sitting at global position 2281 has no meaningful
        # offset into a 109-row result, and using the global number asks for
        # rows past the end - which returns nothing at all while the count
        # still reads 109.
        if around_seq is not None:
            rank_clause = clause + (" AND " if clause else "WHERE ") + "p.seq < ?"
            rank = manifest.query_one(
                f"SELECT COUNT(*) AS n FROM pairs p {rank_clause}",
                tuple(params) + (int(around_seq),))["n"]
            offset = max(0, min(rank - 40, max(0, total - limit)))

        rows = manifest.query(
            f"""SELECT p.pair_id,p.seq,p.patient,p.study,p.ct_series,p.mri_series,
                       p.slice_index,p.ct_file,p.qc_status,p.is_background,
                       p.nudge_dx,p.nudge_dy,p.erase_json IS NOT NULL AS erased,
                       s.reg_status,s.reg_applied
                FROM pairs p JOIN series s ON s.key=p.series_key
                {clause} ORDER BY p.seq LIMIT ? OFFSET ?""",
            tuple(params) + (limit, offset))
        return {"total": total, "rows": rows, "limit": limit, "offset": offset}

    @app.get("/api/patients")
    def list_patients():
        return manifest.query(
            """SELECT patient,
                      COUNT(*) AS n,
                      SUM(qc_status='ACCEPTED') AS accepted,
                      SUM(qc_status='REJECTED') AS rejected,
                      SUM(qc_status='PENDING')  AS pending,
                      SUM(qc_status='ERROR')    AS errors,
                      MIN(seq) AS first_seq
               FROM pairs GROUP BY patient ORDER BY MIN(seq)""")

    @app.get("/api/pair")
    def get_pair(seq: int | None = None, pair_id: str | None = None,
                 remember: bool = True):
        if pair_id:
            row = pair_by_id(pair_id)
        else:
            total = manifest.query_one("SELECT COUNT(*) AS n FROM pairs")["n"]
            if total == 0:
                raise HTTPException(404, "manifest is empty - run a scan first")
            seq = max(0, min(int(seq or 0), total - 1))
            row = manifest.query_one("SELECT * FROM pairs WHERE seq=?", (seq,))
            if row is None:
                raise HTTPException(404, f"no pair at position {seq}")

        srow = series_of(row)
        cache = ensure_registered(srow)
        if remember:
            manifest.set_setting("last_seq", row["seq"])
        prefetch_from(row["seq"])

        total = manifest.query_one("SELECT COUNT(*) AS n FROM pairs")["n"]
        roi = json.loads(row["roi_json"]) if row.get("roi_json") else None
        shape = None
        if cache is not None:
            shape = [int(cache["ct"].shape[1]), int(cache["ct"].shape[2])]

        return {
            "pair": row,
            "total": total,
            "position": row["seq"] + 1,
            "ready": cache is not None,
            "nudge_dx": int(row.get("nudge_dx") or 0),
            "nudge_dy": int(row.get("nudge_dy") or 0),
            "roi": roi,
            "roi_mode": row.get("roi_mode") or "metric",
            "ct_rect": json.loads(row["ct_rect_json"]) if row.get("ct_rect_json") else None,
            "mri_rect": json.loads(row["mri_rect_json"]) if row.get("mri_rect_json") else None,
            "shape": shape,
            "series": {
                "key":          srow["key"],
                "reg_status":   srow["reg_status"],
                "orientation":  srow["orientation"],
                "series_desc":  srow["series_desc"],
                "body_region":  srow["body_region"],
                "n_pairs":      srow["n_pairs"],
                "n_ct_files":   srow["n_ct_files"],
                "n_mri_files":  srow["n_mri_files"],
                "ct_path":      srow["ct_path"],
                "mri_path":     srow["mri_path"],
                "reg_applied":  bool(srow["reg_applied"]),
                "reg_dx_mm":    srow["reg_dx_mm"],
                "reg_dy_mm":    srow["reg_dy_mm"],
                "reg_dx_px":    srow["reg_dx_px"],
                "reg_dy_px":    srow["reg_dy_px"],
                "reg_nmi_gain": srow["reg_nmi_gain"],
                "reg_spread_y_mm": srow["reg_spread_y_mm"],
                "reg_spread_x_mm": srow["reg_spread_x_mm"],
                "reg_n_probes": srow["reg_n_probes"],
                "reg_n_usable": srow["reg_n_usable"],
                "reg_hit_edge": srow["reg_hit_edge"],
                "reg_probe_slices": json.loads(srow["reg_probe_slices"])
                                    if srow.get("reg_probe_slices") else [],
                "reg_reason":   srow["reg_reason"],
                "reg_search_mm": srow["reg_search_mm"],
                "ct_win_min":   srow["ct_win_min"],
                "ct_win_max":   srow["ct_win_max"],
                "mri_p_low":    srow["mri_p_low"],
                "mri_p_high":   srow["mri_p_high"],
                "duration_s":   srow["duration_s"],
                "auto_erase":   bool(srow.get("auto_erase")),
                "error_message": srow["error_message"],
            },
        }

    # ── images ───────────────────────────────────────────────────────────────
    @app.get("/api/image")
    def get_image(pair_id: str, view: str = "fusion"):
        row = pair_by_id(pair_id)
        srow = series_of(row)
        cache = ensure_registered(srow)
        if cache is None:
            raise HTTPException(
                425, "series is not registered yet - it has been queued")

        i = row["slice_index"]
        if i >= cache["ct"].shape[0]:
            raise HTTPException(404, f"slice {i} is outside the cached stack")

        ct_a, mb_a, ma_a = cache["ct"][i], cache["mri_before"][i], cache["mri_after"][i]

        # The manual nudge rides on top of the measured shift, applied to the
        # cached result so adjusting it is instant. "before" is left alone: it
        # is the do-nothing baseline the metric scored against, and moving it
        # would destroy the only honest comparison the reviewer has.
        ndy, ndx = int(row.get("nudge_dy") or 0), int(row.get("nudge_dx") or 0)
        if ndy or ndx:
            ma_a = regsvc.apply_nudge(ma_a, ndy, ndx)

        # Automatic thin-structure removal, if this series has it on. Computed
        # from the CT and applied to both, exactly like a brush stroke.
        if srow.get("auto_erase"):
            auto = regsvc.thin_structure_mask(ct_a)
            if auto is not None:
                ct_a = regsvc.apply_erase(ct_a, auto)
                mb_a = regsvc.apply_erase(mb_a, auto)
                ma_a = regsvc.apply_erase(ma_a, auto)

        # Brush strokes are applied at render time, not baked into the cache.
        # That is what makes painting feel instant and undoable: the stored
        # pixels stay pristine, and the strokes are geometry that can be edited
        # or cleared without re-registering anything. Re-run folds them into the
        # measurement when the reviewer wants that.
        if row.get("erase_json"):
            try:
                mask = regsvc.rasterize_strokes(json.loads(row["erase_json"]), ct_a.shape)
            except (TypeError, ValueError):
                mask = None
            if mask is not None:
                ct_a = regsvc.apply_erase(ct_a, mask)
                mb_a = regsvc.apply_erase(mb_a, mask)
                ma_a = regsvc.apply_erase(ma_a, mask)

        try:
            png = render.render_view(view, ct_a, mb_a, ma_a)
        except ValueError as e:
            raise HTTPException(400, str(e))

        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    # ── erase brush ──────────────────────────────────────────────────────────
    class EraseRequest(BaseModel):
        pair_id: str
        strokes: list[dict] | None = None   # [{"r": px, "pts": [[x,y], ...]}, ...]
        copy_to_series: bool = False

    @app.post("/api/erase")
    def set_erase(req: EraseRequest):
        """
        Save the painted-out region for one slice.

        Strokes are stored per slice as geometry, in image pixels - which are
        millimetres, both stacks being at 1 mm. They take effect immediately in
        every rendered view and in anything exported afterwards; folding them
        into the registration measurement needs a re-run, because the shift is
        estimated once for the whole series.

        `copy_to_series` puts the same strokes on every slice of the series,
        which is the common case for a bed rail or head cradle: the patient
        does not move within a series, so the hardware sits in the same place
        on every slice.
        """
        row = pair_by_id(req.pair_id)
        strokes = req.strokes or None
        payload = json.dumps(strokes) if strokes else None
        now = time.time()

        if req.copy_to_series:
            manifest.execute(
                "UPDATE pairs SET erase_json=?, updated_at=? WHERE series_key=?",
                (payload, now, row["series_key"]))
            n = manifest.query_one(
                "SELECT COUNT(*) AS n FROM pairs WHERE series_key=?",
                (row["series_key"],))["n"]
        else:
            manifest.execute(
                "UPDATE pairs SET erase_json=?, updated_at=? WHERE pair_id=?",
                (payload, now, req.pair_id))
            n = 1

        return {"ok": True, "pair_id": req.pair_id, "slices_updated": n,
                "strokes": len(strokes or [])}

    # ── manual nudge & per-modality export rects ─────────────────────────────
    class NudgeRequest(BaseModel):
        pair_id: str
        dx: int = 0                 # px == mm, relative to the CT frame
        dy: int = 0
        target: str = "mri"         # "mri" or "ct"
        absolute: bool = False      # False adds to the current nudge
        apply_to_series: bool = False

    @app.post("/api/nudge")
    def nudge(req: NudgeRequest):
        """
        Adjust the alignment of ONE SLICE by hand, on top of what registration
        measured.

        PER SLICE, which is a deliberate departure from how the measured shift
        works. estimate_volume_translation produces one shift per stack
        precisely because independent per-slice shifts let the MRI shear
        through z (pipeline_config.py:140-153 records an 85 mm swing across one
        shoulder stack). A reviewer applying small corrections by eye is a
        different situation from an optimiser choosing freely per slice - but
        the risk is the same in kind, so the offsets are recorded per slice in
        metadata.csv and `apply_to_series` is there to make a single consistent
        offset the easy option.

        Only the relative offset is real, so exactly ONE number is stored:
        nudging the CT by (+dx, +dy) is recorded as nudging the MRI by
        (-dx, -dy). Offering both directions is a convenience; storing both
        would allow a pair to carry two offsets that contradict each other.

        The measured shift in reg_dx_mm/reg_dy_mm is never touched.
        """
        row = pair_by_id(req.pair_id)
        srow = series_of(row)

        sign = -1 if req.target == "ct" else 1
        dx, dy = sign * int(req.dx), sign * int(req.dy)
        if not req.absolute:
            dx += int(row.get("nudge_dx") or 0)
            dy += int(row.get("nudge_dy") or 0)

        limit = int(round(cfg.reg_search_mm * 2))
        if abs(dx) > limit or abs(dy) > limit:
            raise HTTPException(
                400, f"a manual offset beyond +/-{limit} mm is almost certainly "
                     f"a mistake; re-run with a wider search instead")

        now = time.time()
        if req.apply_to_series:
            manifest.execute(
                "UPDATE pairs SET nudge_dx=?, nudge_dy=?, updated_at=? WHERE series_key=?",
                (dx, dy, now, row["series_key"]))
            n = manifest.query_one(
                "SELECT COUNT(*) AS n FROM pairs WHERE series_key=?",
                (row["series_key"],))["n"]
        else:
            manifest.execute(
                "UPDATE pairs SET nudge_dx=?, nudge_dy=?, updated_at=? WHERE pair_id=?",
                (dx, dy, now, req.pair_id))
            n = 1

        return {"ok": True, "nudge_dx": dx, "nudge_dy": dy, "slices_updated": n,
                "total_dx_mm": (srow.get("reg_dx_mm") or 0) + dx,
                "total_dy_mm": (srow.get("reg_dy_mm") or 0) + dy}

    class ExportRectRequest(BaseModel):
        pair_id: str
        target: str                       # "ct" or "mri"
        rect: list[float] | None = None   # [x, y, w, h] in the common 1 mm frame
        apply_to_series: bool = False

    @app.post("/api/export_rect")
    def export_rect(req: ExportRectRequest):
        """
        Set the export rectangle for ONE modality, in the shared 1 mm frame.

        Two different rectangles are safe here because both are stated in the
        same frame: each says which physical millimetres of that shared
        coordinate system to keep. It does not move one image relative to the
        other. Saved slices may then differ in size between CT and MRI, which
        metadata.csv records per row.

        Export only - the metric still sees the full frame, or the shared ROI
        if one is set.
        """
        if req.target not in ("ct", "mri"):
            raise HTTPException(400, "target must be 'ct' or 'mri'")
        row = pair_by_id(req.pair_id)
        if req.rect is not None:
            if len(req.rect) != 4:
                raise HTTPException(400, "rect must be [x, y, w, h]")
            if req.rect[2] < 8 or req.rect[3] < 8:
                raise HTTPException(400, "export rectangle must be at least 8x8 px")

        col = "ct_rect_json" if req.target == "ct" else "mri_rect_json"
        payload = json.dumps(req.rect) if req.rect else None
        now = time.time()
        if req.apply_to_series:
            manifest.execute(f"UPDATE pairs SET {col}=?, updated_at=? WHERE series_key=?",
                             (payload, now, row["series_key"]))
        else:
            manifest.execute(f"UPDATE pairs SET {col}=?, updated_at=? WHERE pair_id=?",
                             (payload, now, req.pair_id))
        return {"ok": True, "target": req.target, "rect": req.rect}

    class AutoEraseRequest(BaseModel):
        series_key: str
        enabled: bool = True

    @app.post("/api/auto_erase")
    def set_auto_erase(req: AutoEraseRequest):
        """
        Turn automatic thin-structure removal on or off for a series.

        Removes table rails and the head cradle by SHAPE - anything thin that
        is not part of the main body. See registration_service.thin_structure_mask
        for why thickness and not size: on this dataset a size threshold would
        have had to choose between deleting a 21%-sized piece of anatomy and
        keeping the rails.

        Per series, since the hardware sits in the same place on every slice.
        Composes with hand-painted strokes rather than replacing them - the
        brush is still there for anything this misses.
        """
        srow = manifest.query_one("SELECT key FROM series WHERE key=?", (req.series_key,))
        if srow is None:
            raise HTTPException(404, f"no series '{req.series_key}'")
        manifest.execute(
            "UPDATE series SET auto_erase=?, updated_at=? WHERE key=?",
            (1 if req.enabled else 0, time.time(), req.series_key))
        # Pairs already written need re-exporting to pick this up.
        manifest.execute(
            "UPDATE pairs SET updated_at=? WHERE series_key=?",
            (time.time(), req.series_key))
        return {"ok": True, "series_key": req.series_key, "auto_erase": req.enabled}

    # ── QC decisions ─────────────────────────────────────────────────────────
    class DecisionRequest(BaseModel):
        pair_id: str
        note: str | None = None
        crop_export: bool = False
        save_preview: bool = True

    @app.post("/api/accept")
    def accept(req: DecisionRequest):
        row = pair_by_id(req.pair_id)
        srow = series_of(row)
        cache = ensure_registered(srow)
        if cache is None:
            raise HTTPException(425, "series is not registered yet - it has been queued")

        try:
            paths = outputs.write_pair(cfg, row, srow, cache,
                                       crop_export=req.crop_export,
                                       save_preview=req.save_preview)
        except Exception as exc:
            msg = regsvc.format_exception(exc)
            manifest.execute(
                "UPDATE pairs SET qc_status='ERROR', error_message=?, updated_at=? "
                "WHERE pair_id=?", (msg, time.time(), req.pair_id))
            raise HTTPException(500, f"could not write outputs: {msg}")

        now = time.time()
        manifest.execute(
            """UPDATE pairs SET qc_status='ACCEPTED', note=?, error_message=NULL,
               output_ct=?, output_mri=?, output_preview=?, reviewed_at=?,
               exported_at=?, updated_at=?
               WHERE pair_id=?""",
            (req.note, paths["ct"], paths["mri"], paths["preview"], now, now, now,
             req.pair_id))
        return {"ok": True, "pair_id": req.pair_id, "outputs": paths,
                "counts": manifest.counts()}

    @app.post("/api/reject")
    def reject(req: DecisionRequest):
        row = pair_by_id(req.pair_id)
        srow = series_of(row)
        # A pair accepted earlier has files on disk. Rejecting it has to take
        # them away, or the training set keeps a slice the reviewer just
        # rejected.
        outputs.remove_pair_outputs(cfg, row, srow)
        now = time.time()
        manifest.execute(
            """UPDATE pairs SET qc_status='REJECTED', note=?, error_message=NULL,
               output_ct=NULL, output_mri=NULL, output_preview=NULL,
               reviewed_at=?, updated_at=? WHERE pair_id=?""",
            (req.note, now, now, req.pair_id))
        return {"ok": True, "pair_id": req.pair_id, "counts": manifest.counts()}

    @app.post("/api/reset")
    def reset(req: DecisionRequest):
        row = pair_by_id(req.pair_id)
        outputs.remove_pair_outputs(cfg, row, series_of(row))
        now = time.time()
        manifest.execute(
            """UPDATE pairs SET qc_status='PENDING', note=NULL, error_message=NULL,
               output_ct=NULL, output_mri=NULL, output_preview=NULL,
               reviewed_at=NULL, updated_at=? WHERE pair_id=?""", (now, req.pair_id))
        return {"ok": True, "pair_id": req.pair_id, "counts": manifest.counts()}

    class BulkRequest(BaseModel):
        series_key: str
        decision: str = "ACCEPTED"
        crop_export: bool = False

    @app.post("/api/series/bulk")
    def bulk(req: BulkRequest):
        """
        Apply one decision to every slice of a series.

        Registration is per series, so a shift that is right or wrong is right
        or wrong for the whole stack. Once a reviewer has judged a few slices
        this saves them repeating it 18 times - and it stays a per-pair record,
        so any individual slice can still be overridden afterwards.
        """
        srow = manifest.query_one("SELECT * FROM series WHERE key=?", (req.series_key,))
        if srow is None:
            raise HTTPException(404, f"no series '{req.series_key}'")
        rows = manifest.query("SELECT * FROM pairs WHERE series_key=? ORDER BY slice_index",
                              (req.series_key,))
        decision = req.decision.upper()
        if decision not in ("ACCEPTED", "REJECTED", "PENDING"):
            raise HTTPException(400, f"unknown decision '{req.decision}'")

        done, failed = 0, []
        if decision == "ACCEPTED":
            cache = ensure_registered(srow)
            if cache is None:
                raise HTTPException(425, "series is not registered yet - it has been queued")
            now = time.time()
            for row in rows:
                try:
                    paths = outputs.write_pair(cfg, row, srow, cache,
                                               crop_export=req.crop_export)
                    manifest.execute(
                        """UPDATE pairs SET qc_status='ACCEPTED', error_message=NULL,
                           output_ct=?, output_mri=?, output_preview=?,
                           reviewed_at=?, exported_at=?, updated_at=? WHERE pair_id=?""",
                        (paths["ct"], paths["mri"], paths["preview"], now, now, now,
                         row["pair_id"]))
                    done += 1
                except Exception as exc:
                    failed.append({"pair_id": row["pair_id"],
                                   "error": regsvc.format_exception(exc)})
        else:
            now = time.time()
            for row in rows:
                outputs.remove_pair_outputs(cfg, row, srow)
                manifest.execute(
                    """UPDATE pairs SET qc_status=?, output_ct=NULL, output_mri=NULL,
                       output_preview=NULL, reviewed_at=?, updated_at=?
                       WHERE pair_id=?""",
                    (decision, now if decision != "PENDING" else None, now,
                     row["pair_id"]))
                done += 1

        return {"ok": True, "updated": done, "failed": failed,
                "counts": manifest.counts()}

    # ── re-run / ROI ─────────────────────────────────────────────────────────
    class RerunRequest(BaseModel):
        series_key: str
        roi: list[float] | None = None      # [x, y, w, h] in pixels of the 1 mm grid
        roi_mode: str = "metric"            # "metric" or "export"
        search_mm: float | None = None
        clear_roi: bool = False
        # False stores the ROI and stops there. Drawing or adjusting a crop is
        # not a request to re-measure - a reviewer may be tuning the rectangle
        # while the existing registration is perfectly good, and throwing that
        # away on every drag would be both slow and destructive.
        run: bool = True

    class RoiRequest(BaseModel):
        pair_id: str
        roi: list[float] | None = None
        roi_mode: str = "metric"
        clear_roi: bool = False
        apply_to_series: bool = False

    @app.post("/api/roi")
    def set_roi(req: RoiRequest):
        """
        Store the crop for ONE SLICE, without re-registering.

        Per slice, like the erase and the nudge. The metric ROI only bites on
        the five probe slices - the shift is estimated once per stack - but the
        export rectangles are genuinely per-slice, and adjusting one slice must
        not silently rewrite its twenty-three neighbours. `apply_to_series`
        does that deliberately when it is what you want.
        """
        row = pair_by_id(req.pair_id)
        roi = None if req.clear_roi else req.roi
        if roi is not None:
            if len(roi) != 4:
                raise HTTPException(400, "roi must be [x, y, w, h]")
            if roi[2] < 16 or roi[3] < 16:
                raise HTTPException(
                    400, "ROI must be at least 16x16 px - a smaller window has "
                         "too few samples for a stable joint histogram")

        payload = json.dumps(roi) if roi else None
        now = time.time()
        if req.apply_to_series:
            manifest.execute(
                "UPDATE pairs SET roi_json=?, roi_mode=?, updated_at=? WHERE series_key=?",
                (payload, req.roi_mode, now, row["series_key"]))
            n = manifest.query_one("SELECT COUNT(*) AS n FROM pairs WHERE series_key=?",
                                   (row["series_key"],))["n"]
        else:
            manifest.execute(
                "UPDATE pairs SET roi_json=?, roi_mode=?, updated_at=? WHERE pair_id=?",
                (payload, req.roi_mode, now, req.pair_id))
            n = 1
        return {"ok": True, "roi": roi, "slices_updated": n, "queued": False,
                "note": "stored - press Re-run to measure with it"}

    @app.post("/api/rerun")
    def rerun(req: RerunRequest):
        """
        Re-register a series, optionally restricting the metric to an ROI.

        The ROI is an index rectangle on the resampled grid. That is a physical
        rectangle here because the CT and the MRI already share that grid, so
        cropping both by the same rectangle cannot move one relative to the
        other - it only changes which pixels the NMI is computed over. Useful
        when a table, an arm or a coil dominates the frame and drags the match
        away from the anatomy.
        """
        srow = manifest.query_one("SELECT * FROM series WHERE key=?", (req.series_key,))
        if srow is None:
            raise HTTPException(404, f"no series '{req.series_key}'")

        roi = None if req.clear_roi else req.roi
        if roi is not None:
            if len(roi) != 4:
                raise HTTPException(400, "roi must be [x, y, w, h]")
            if roi[2] < 16 or roi[3] < 16:
                raise HTTPException(
                    400, "ROI must be at least 16x16 px - a smaller window has "
                         "too few samples for a stable joint histogram")

        # Clearing cache_path is what makes the re-run visible. While it is
        # NULL, /api/pair reports ready=false and the UI shows a spinner
        # instead of the previous run's images and metrics.
        manifest.execute(
            "UPDATE series SET reg_search_mm=?, reg_status='PENDING', "
            "cache_path=NULL, updated_at=? WHERE key=?",
            (req.search_mm or srow["reg_search_mm"] or cfg.reg_search_mm,
             time.time(), req.series_key))
        worker.invalidate(req.series_key)

        worker.submit(req.series_key, priority=PRIORITY_USER, roi=roi,
                      search_mm=req.search_mm, force=True)
        return {"ok": True, "queued": req.series_key, "roi": roi}

    class RerunRejectedRequest(BaseModel):
        search_mm: float = 90.0
        include_edge: bool = True      # best shift sat on the search boundary
        include_spread: bool = True    # probes disagreed by more than the limit
        include_gain: bool = False     # gain below threshold - see below
        dry_run: bool = False

    @app.post("/api/rerun_rejected")
    def rerun_rejected(req: RerunRejectedRequest):
        """
        Re-measure the series that ended with no shift applied, using a wider
        search.

        Not all rejections are the same, and the default selection reflects
        that:

          edge    the best shift landed on the boundary of the search square,
                  which is a wall rather than a peak - the true offset is
                  provably further out, so a wider search is exactly the fix.

          spread  the probes disagreed by more than REG_MAX_SPREAD_MM. Often
                  the same cause: when the real offset lies outside the window
                  each probe latches onto a different partial match. Worth
                  re-measuring.

          gain    the best shift was found but improved NMI by less than
                  REG_MIN_GAIN. That means the pair is ALREADY well aligned.
                  Widening the search here cannot improve matters and can make
                  them worse, by admitting a distant position that scores
                  marginally higher on noise. Off by default for that reason.

        Existing ROIs are preserved - a reviewer who focused the metric on the
        anatomy should not lose that by asking for a wider search.
        """
        rows = manifest.query(
            "SELECT key, reg_reason, reg_hit_edge FROM series "
            "WHERE reg_status='REGISTERED' AND reg_applied=0 ORDER BY key")

        selected = []
        for r in rows:
            reason = r["reg_reason"] or ""
            if r["reg_hit_edge"]:
                cat = "edge"
            elif "disagree" in reason:
                cat = "spread"
            elif "not worth moving" in reason:
                cat = "gain"
            else:
                cat = "other"
            if ((cat == "edge" and req.include_edge) or
                    (cat == "spread" and req.include_spread) or
                    (cat == "gain" and req.include_gain)):
                selected.append((r["key"], cat, None))

        if req.dry_run:
            return {"ok": True, "would_requeue": len(selected),
                    "series": [{"key": k, "category": c} for k, c, _ in selected]}

        for key, _cat, roi_json in selected:
            manifest.execute(
                "UPDATE series SET reg_status='PENDING', cache_path=NULL, "
                "reg_search_mm=?, error_message=NULL, updated_at=? WHERE key=?",
                (req.search_mm, time.time(), key))
            worker.invalidate(key)
            roi = json.loads(roi_json) if roi_json else None
            worker.submit(key, priority=PRIORITY_PREFETCH, roi=roi,
                          search_mm=req.search_mm, force=True)

        by_cat = {}
        for _k, c, _r in selected:
            by_cat[c] = by_cat.get(c, 0) + 1
        return {"ok": True, "requeued": len(selected), "search_mm": req.search_mm,
                "by_category": by_cat}

    @app.post("/api/retry_errors")
    def retry_errors():
        rows = manifest.query("SELECT key FROM series WHERE reg_status='ERROR'")
        for r in rows:
            manifest.execute(
                "UPDATE series SET reg_status='PENDING', error_message=NULL WHERE key=?",
                (r["key"],))
            worker.submit(r["key"], priority=PRIORITY_USER, force=True)
        return {"ok": True, "requeued": len(rows)}

    class ReexportRequest(BaseModel):
        only_stale: bool = True
        dry_run: bool = False

    @app.post("/api/reexport")
    def reexport(req: ReexportRequest):
        """
        Rewrite accepted pairs whose .npy files no longer match the manifest.

        Outputs are written once, at the moment of acceptance. Anything done
        afterwards - painting out a bed rail, nudging the alignment, or
        re-registering the series - updates the manifest but leaves the file on
        disk as it was. That is a silent divergence between what the reviewer
        sees and what the training set contains, so it needs an explicit way
        back into line.

        Staleness is `updated_at > exported_at`. Pairs are grouped by series so
        each cache is decoded once rather than once per slice.
        """
        rows = manifest.query(
            "SELECT * FROM pairs WHERE qc_status='ACCEPTED' "
            + ("AND (exported_at IS NULL OR updated_at > exported_at + 1) " if req.only_stale else "")
            + "ORDER BY series_key, slice_index")
        if req.dry_run:
            return {"ok": True, "would_rewrite": len(rows),
                    "series": sorted({r["series_key"] for r in rows})}

        written, failed, by_series = 0, [], {}
        for r in rows:
            by_series.setdefault(r["series_key"], []).append(r)

        now = time.time()
        for skey, items in by_series.items():
            srow = manifest.query_one("SELECT * FROM series WHERE key=?", (skey,))
            if srow is None:
                continue
            cache = ensure_registered(srow)
            if cache is None:
                failed.append({"series": skey, "error": "not registered yet - requeued"})
                continue
            for r in items:
                try:
                    paths = outputs.write_pair(cfg, r, srow, cache)
                    manifest.execute(
                        "UPDATE pairs SET output_ct=?, output_mri=?, output_preview=?, "
                        "exported_at=? WHERE pair_id=?",
                        (paths["ct"], paths["mri"], paths["preview"], now, r["pair_id"]))
                    written += 1
                except Exception as exc:
                    failed.append({"pair_id": r["pair_id"], "error": regsvc.format_exception(exc)})

        return {"ok": True, "rewritten": written, "failed": failed[:20],
                "n_failed": len(failed)}

    # ── export ───────────────────────────────────────────────────────────────
    @app.post("/api/export")
    def export_csv(include_rejected: bool = False):
        statuses = ("ACCEPTED", "REJECTED") if include_rejected else ("ACCEPTED",)
        return outputs.export_metadata_csv(cfg, manifest, statuses)

    # ── static UI ────────────────────────────────────────────────────────────
    @app.get("/")
    def index():
        """
        The page, with a version stamp on every asset it pulls in.

        Without this, a browser happily reuses a cached styles.css or app.js
        while the server serves a page that expects the new ones - which shows
        up as layout that is subtly or badly wrong for no visible reason, and
        costs a round of "have you hard-reloaded?" every time the UI changes.
        The stamp is each file's mtime, so the URL changes exactly when the
        file does and never otherwise.
        """
        path = os.path.join(STATIC_DIR, "index.html")
        html = open(path, encoding="utf-8").read()
        for asset in ("styles.css", "app.js"):
            try:
                stamp = int(os.path.getmtime(os.path.join(STATIC_DIR, asset)))
            except OSError:
                continue
            html = html.replace(f"/static/{asset}", f"/static/{asset}?v={stamp}")
        return Response(content=html, media_type="text/html",
                        headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.on_event("shutdown")
    def _shutdown():
        worker.stop()
        manifest.close()

    return app
