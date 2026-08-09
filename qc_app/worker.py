"""
worker.py
─────────
One background thread that registers series, so the UI never blocks.

WHY A THREAD AND NOT A PROCESS POOL
───────────────────────────────────
The expensive parts - N4, resampling, and the NMI sweep - are SimpleITK and
numpy calls that release the GIL, so a worker thread genuinely runs in parallel
with the request handlers. A process pool would add Windows spawn cost, a
second import of SimpleITK per worker, and pickling of whole volumes across a
pipe, to buy back very little. The instruction not to introduce multiprocessing
that conflicts with the existing pipeline points the same way.

ONE SERIES AT A TIME, ON PURPOSE
────────────────────────────────
A single worker keeps peak memory to one CT volume plus one MRI volume plus
their resampled copies. Nothing holds the dataset. Slices reach the browser
from the on-disk cache through a tiny LRU, never from a resident array of
everything.

PRIORITIES
──────────
    0  the reviewer is looking at this series right now, or asked for a re-run
    1  prefetch - the next few series ahead of the reviewer

A user action therefore overtakes any queued prefetch instead of waiting behind
it. Priority ties break on insertion order, so the queue stays deterministic.
"""

import heapq
import itertools
import json
import threading
import time
import traceback
from collections import OrderedDict

from . import outputs, registration_service as regsvc

PRIORITY_USER = 0
PRIORITY_PREFETCH = 1


class RegistrationWorker:
    def __init__(self, app_cfg, manifest, cache_slots: int = 3):
        self.cfg = app_cfg
        self.db = manifest

        self._heap = []
        self._queued = {}           # series_key -> priority already queued at
        self._counter = itertools.count()
        self._cv = threading.Condition()
        self._stop = threading.Event()

        self._current = None        # series_key being worked on
        self._progress = ""
        self._last_error = None

        # Decoded caches, most-recently-used last. Three is enough for the
        # series being reviewed plus the ones on either side of it.
        self._lru = OrderedDict()
        self._lru_lock = threading.Lock()
        self._lru_slots = cache_slots

        self._thread = threading.Thread(target=self._run, name="registration-worker",
                                        daemon=True)
        self._thread.start()

    # ── queue ────────────────────────────────────────────────────────────────
    def submit(self, series_key: str, priority: int = PRIORITY_USER,
               roi=None, search_mm=None, force: bool = False) -> bool:
        """
        Queue a series. Returns True if it was queued by this call.

        A series already queued at a weaker priority is promoted rather than
        queued twice, which is what stops a prefetch from making the reviewer
        wait for work they have since overtaken.
        """
        with self._cv:
            if series_key == self._current and not force:
                return False
            existing = self._queued.get(series_key)
            if existing is not None and existing <= priority and not force:
                return False
            self._queued[series_key] = priority
            heapq.heappush(self._heap,
                           (priority, next(self._counter),
                            {"key": series_key, "roi": roi, "search_mm": search_mm}))
            self._cv.notify()
            return True

    def status(self) -> dict:
        with self._cv:
            return {
                "current":    self._current,
                "progress":   self._progress,
                "queued":     len(self._heap),
                "queue":      [j[2]["key"] for j in sorted(self._heap)[:10]],
                "last_error": self._last_error,
                "alive":      self._thread.is_alive(),
            }

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    # ── the loop ─────────────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            job = None
            with self._cv:
                while not self._heap and not self._stop.is_set():
                    self._cv.wait(timeout=0.5)
                if self._stop.is_set():
                    return
                _, _, job = heapq.heappop(self._heap)
                self._queued.pop(job["key"], None)
                self._current = job["key"]
                self._progress = "starting"

            try:
                self._process(job)
            except Exception as exc:
                # Belt and braces: _process already records per-series failures.
                # Anything reaching here would otherwise kill the worker thread
                # and silently freeze the whole queue.
                self._last_error = f"{job['key']}: {regsvc.format_exception(exc)}"
                traceback.print_exc()
            finally:
                with self._cv:
                    self._current = None
                    self._progress = ""

    def _process(self, job: dict) -> None:
        key = job["key"]
        row = self.db.query_one("SELECT * FROM series WHERE key=?", (key,))
        if row is None:
            return

        # Crops live on the slices now, so the metric gets a dict keyed by
        # slice index. An explicit job roi (a fresh drag) still wins.
        roi = job.get("roi")
        if roi is None:
            roi = {}
            for r in self.db.query(
                    "SELECT slice_index, roi_json FROM pairs "
                    "WHERE series_key=? AND roi_json IS NOT NULL", (key,)):
                try:
                    v = json.loads(r["roi_json"])
                except (TypeError, ValueError):
                    continue
                if v:
                    roi[r["slice_index"]] = v
            roi = roi or None
        search_mm = job.get("search_mm") or row.get("reg_search_mm") or self.cfg.reg_search_mm

        self.db.execute(
            "UPDATE series SET reg_status='PROCESSING', error_message=NULL, updated_at=? "
            "WHERE key=?", (time.time(), key))

        def progress(msg):
            with self._cv:
                self._progress = msg

        # Per-slice brush strokes, keyed by slice index. Read fresh on every
        # run so a re-run always reflects what the reviewer has painted since.
        erase = {}
        for p in self.db.query(
                "SELECT slice_index, erase_json FROM pairs "
                "WHERE series_key=? AND erase_json IS NOT NULL", (key,)):
            try:
                strokes = json.loads(p["erase_json"])
            except (TypeError, ValueError):
                continue
            if strokes:
                erase[p["slice_index"]] = strokes

        try:
            result = regsvc.process_series(row, self.cfg, roi=roi,
                                           search_mm=search_mm, progress=progress,
                                           erase=erase or None)
        except Exception as exc:
            msg = regsvc.format_exception(exc)
            detail = traceback.format_exc()
            self.db.execute(
                "UPDATE series SET reg_status='ERROR', error_message=?, updated_at=? "
                "WHERE key=?", (f"{msg}\n\n{detail}", time.time(), key))
            # The slices of a series that could not be registered are errors
            # too, but only the ones nobody has ruled on - a decision a human
            # already made is never overwritten by machinery.
            self.db.execute(
                "UPDATE pairs SET qc_status='ERROR', error_message=?, updated_at=? "
                "WHERE series_key=? AND qc_status='PENDING'",
                (msg, time.time(), key))
            self._last_error = f"{key}: {msg}"
            return

        self._store_result(key, row, result)

    def _store_result(self, key: str, row: dict, result: dict) -> None:
        reg = result["reg"]
        now = time.time()
        self.db.execute(
            """UPDATE series SET
                 reg_status='REGISTERED', error_message=NULL,
                 reg_applied=?, reg_dx_px=?, reg_dy_px=?, reg_dx_mm=?, reg_dy_mm=?,
                 reg_nmi_gain=?, reg_spread_y_mm=?, reg_spread_x_mm=?,
                 reg_n_probes=?, reg_n_usable=?, reg_hit_edge=?, reg_probe_slices=?,
                 reg_reason=?, reg_search_mm=?,
                 mri_p_low=?, mri_p_high=?, ct_win_min=?, ct_win_max=?,
                 body_region=?, cache_path=?, n_pairs=?, duration_s=?, updated_at=?
               WHERE key=?""",
            (int(reg["applied"]), reg["dx"], reg["dy"], reg["dx_mm"], reg["dy_mm"],
             reg["mean_gain"], reg["spread_y_mm"], reg["spread_x_mm"],
             reg["n_probes"], reg["n_usable"], reg["hit_edge"],
             json.dumps(reg["probe_slices"]), reg["reason"], result["search_mm"],
             # The crop is NOT written back here any more - it belongs to the
             # slices, and the series row must not hold a second copy that can
             # disagree with them.
             result["mri_p_low"], result["mri_p_high"],
             result["ct_win_min"], result["ct_win_max"],
             result["body_region"], result["cache_path"], result["n_pairs"],
             result["duration_s"], now, key))

        # Background flags are a property of the pixels, so they are refreshed
        # on every registration, including a re-run with a different ROI.
        self.db.executemany(
            "UPDATE pairs SET is_background=?, updated_at=? WHERE pair_id=?",
            [(int(bg), now, f"{key}/{i:04d}")
             for i, bg in enumerate(result["is_background"])])

        # Slices previously marked ERROR by a failed run get another chance;
        # accepted and rejected decisions are left exactly as the reviewer set
        # them.
        self.db.execute(
            "UPDATE pairs SET qc_status='PENDING', error_message=NULL, updated_at=? "
            "WHERE series_key=? AND qc_status='ERROR'", (now, key))

        try:
            outputs.write_registration_record(self.cfg, row, result)
        except Exception:
            # A record that cannot be written must not invalidate a
            # registration that succeeded.
            traceback.print_exc()

        self._invalidate_cache(key)

    # ── slice cache ──────────────────────────────────────────────────────────
    def get_cache(self, series_key: str, cache_path: str):
        """Cached arrays for a series, decoded at most once per LRU slot."""
        with self._lru_lock:
            if series_key in self._lru:
                self._lru.move_to_end(series_key)
                return self._lru[series_key]

        data = regsvc.load_cache(cache_path)
        if data is None:
            return None

        with self._lru_lock:
            self._lru[series_key] = data
            self._lru.move_to_end(series_key)
            while len(self._lru) > self._lru_slots:
                self._lru.popitem(last=False)
        return data

    def invalidate(self, series_key: str) -> None:
        """
        Drop the decoded cache for a series.

        Called when a series is about to be re-registered. Without it, the
        endpoint that queued the re-run keeps serving the PREVIOUS result -
        which looks like a successful instant re-run and quietly shows a
        reviewer numbers from the settings they just changed away from.
        """
        self._invalidate_cache(series_key)

    def _invalidate_cache(self, series_key: str) -> None:
        with self._lru_lock:
            self._lru.pop(series_key, None)
