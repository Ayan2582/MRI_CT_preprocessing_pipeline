"""
db.py
─────
The persistent manifest. SQLite, one file under the workspace.

Two tables mirroring the two levels in scanner.py:

    series   one row per CT/MRI series pair - registration state lives here,
             because one shift is estimated per series, not per slice
    pairs    one row per slice pair - QC state lives here, because a human
             accepts or rejects one slice at a time

Everything the application needs to resume is in this file. Closing the app
mid-review and reopening it re-reads these tables and continues; nothing is
held only in memory, and no state is inferred by looking at which output files
happen to exist.

`seq` on `pairs` is a dense 0-based index over the whole dataset in review
order. It is what makes "next", "previous", "jump to 2037" and "3 of 2313"
single-row lookups instead of scans.
"""

import json
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    key             TEXT PRIMARY KEY,
    patient         TEXT NOT NULL,
    study           TEXT NOT NULL,
    ct_series       TEXT NOT NULL,
    mri_series      TEXT NOT NULL,
    ct_path         TEXT NOT NULL,
    mri_path        TEXT NOT NULL,
    orientation     TEXT,
    series_desc     TEXT,
    body_region     TEXT,
    n_ct_files      INTEGER,
    n_mri_files     INTEGER,
    n_pairs         INTEGER,
    reg_status      TEXT NOT NULL DEFAULT 'PENDING',
    reg_applied     INTEGER DEFAULT 0,
    reg_dx_px       INTEGER DEFAULT 0,
    reg_dy_px       INTEGER DEFAULT 0,
    reg_dx_mm       REAL DEFAULT 0.0,
    reg_dy_mm       REAL DEFAULT 0.0,
    reg_nmi_gain    REAL,
    reg_spread_y_mm REAL,
    reg_spread_x_mm REAL,
    reg_n_probes    INTEGER,
    reg_n_usable    INTEGER,
    reg_hit_edge    INTEGER,
    reg_probe_slices TEXT,
    reg_reason      TEXT,
    reg_search_mm   REAL,
    roi_json        TEXT,
    roi_mode        TEXT DEFAULT 'metric',
    mri_p_low       REAL,
    mri_p_high      REAL,
    ct_win_min      REAL,
    ct_win_max      REAL,
    cache_path      TEXT,
    error_message   TEXT,
    duration_s      REAL,
    created_at      REAL,
    updated_at      REAL
);

CREATE TABLE IF NOT EXISTS pairs (
    pair_id         TEXT PRIMARY KEY,
    seq             INTEGER NOT NULL,
    series_key      TEXT NOT NULL,
    patient         TEXT NOT NULL,
    study           TEXT NOT NULL,
    ct_series       TEXT NOT NULL,
    mri_series      TEXT NOT NULL,
    slice_index     INTEGER NOT NULL,
    ct_file         TEXT,
    mri_file        TEXT,
    qc_status       TEXT NOT NULL DEFAULT 'PENDING',
    is_background   INTEGER,
    output_ct       TEXT,
    output_mri      TEXT,
    output_preview  TEXT,
    note            TEXT,
    erase_json      TEXT,
    error_message   TEXT,
    reviewed_at     REAL,
    created_at      REAL,
    updated_at      REAL,
    FOREIGN KEY (series_key) REFERENCES series(key)
);

CREATE INDEX IF NOT EXISTS idx_pairs_seq        ON pairs(seq);
CREATE INDEX IF NOT EXISTS idx_pairs_status     ON pairs(qc_status);
CREATE INDEX IF NOT EXISTS idx_pairs_series     ON pairs(series_key);
CREATE INDEX IF NOT EXISTS idx_pairs_patient    ON pairs(patient);
CREATE INDEX IF NOT EXISTS idx_series_status    ON series(reg_status);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class Manifest:
    """
    Thin, explicit wrapper over the SQLite file.

    One connection shared across threads with a lock around writes. The
    alternative - a connection per thread - would mean the background worker
    and the request handlers seeing different WAL snapshots, and a reviewer
    could accept a pair whose registration the worker had already superseded.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """
        Add columns introduced after a manifest was first created.

        CREATE TABLE IF NOT EXISTS does nothing to a table that already exists,
        so a workspace built by an earlier version keeps its old columns and
        every query naming a new one fails. Adding them here means an existing
        review - decisions, registrations and all - survives an upgrade instead
        of having to be rebuilt.
        """
        wanted = {
            "pairs": {
                "erase_json": "TEXT",
                # Manual nudge, PER SLICE. Reviewers asked for slice-level
                # control after seeing pairs where one slice sits right and its
                # neighbour does not. See the note in server.nudge about what
                # per-slice offsets cost.
                "nudge_dx": "INTEGER DEFAULT 0",
                "nudge_dy": "INTEGER DEFAULT 0",
                # When the .npy files were last WRITTEN. Distinct from
                # reviewed_at, which is when a human decided. An edit after
                # acceptance (erase, nudge, or a re-registration of the series)
                # bumps updated_at and leaves the file behind - comparing the
                # two is what makes that detectable instead of silent.
                "exported_at": "REAL",
                # Crop, PER SLICE. It began as a series property; the metric
                # ROI still only bites on probe slices (the shift is estimated
                # once per stack), but the EXPORT rectangles are genuinely
                # per-slice, and a reviewer adjusting one slice should not be
                # silently changing the other twenty-three.
                "roi_json":      "TEXT",
                "roi_mode":      "TEXT DEFAULT 'metric'",
                "ct_rect_json":  "TEXT",
                "mri_rect_json": "TEXT",
            },
            "series": {
                # Opt-in automatic removal of thin non-body structures (table
                # rails, head cradle). Per series, because the hardware is the
                # same on every slice of one acquisition.
                "auto_erase": "INTEGER DEFAULT 0",
            },
        }
        # Columns an earlier schema created that nothing writes any more. Left
        # in place they read as all-zero data, which is worse than absent: the
        # nudge briefly lived on `series` before moving to `pairs`, and a
        # leftover series.nudge_dx of 0 looks exactly like "no nudge recorded".
        obsolete = {"series": ("nudge_dx", "nudge_dy", "roi_json", "roi_mode",
                               "ct_rect_json", "mri_rect_json")}

        # Carry existing series-level crops down onto their slices BEFORE the
        # series columns are dropped, so a review in progress keeps every crop
        # it has. Runs once: after the drop there is nothing left to copy.
        def _inherit_series_crops():
            have = {r["name"] for r in
                    self._conn.execute("PRAGMA table_info(series)").fetchall()}
            if "roi_json" not in have:
                return
            for col in ("roi_json", "roi_mode", "ct_rect_json", "mri_rect_json"):
                if col in have:
                    self._conn.execute(
                        f"UPDATE pairs SET {col} = ("
                        f"  SELECT s.{col} FROM series s WHERE s.key = pairs.series_key)"
                        f" WHERE {col} IS NULL AND EXISTS ("
                        f"  SELECT 1 FROM series s WHERE s.key = pairs.series_key"
                        f"    AND s.{col} IS NOT NULL)")

        with self._lock:
            for table, cols in wanted.items():
                have = {r["name"] for r in
                        self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
                for name, decl in cols.items():
                    if name not in have:
                        self._conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
            _inherit_series_crops()
            for table, names in obsolete.items():
                have = {r["name"] for r in
                        self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
                for name in names:
                    if name in have:
                        try:
                            self._conn.execute(f"ALTER TABLE {table} DROP COLUMN {name}")
                        except Exception:
                            pass   # older SQLite: harmless to leave it
            self._conn.commit()

    # ── low level ─────────────────────────────────────────────────────────────
    def query(self, sql: str, params=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def executemany(self, sql: str, seq):
        with self._lock:
            cur = self._conn.executemany(sql, seq)
            self._conn.commit()
            return cur

    def close(self):
        with self._lock:
            self._conn.close()

    # ── settings ──────────────────────────────────────────────────────────────
    def get_setting(self, key: str, default=None):
        row = self.query_one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value) -> None:
        self.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))

    # ── population ────────────────────────────────────────────────────────────
    def sync_scan(self, report, region_for) -> dict:
        """
        Bring the manifest in line with what is on disk.

        Additive and idempotent, which is what makes "Add to dataset" safe:
        rows already present keep their QC decisions and their registration,
        new ones arrive as PENDING. Nothing is ever deleted here - a series
        that vanished from disk keeps its history and is reported instead.

        `seq` is reassigned across the whole table afterwards, because a new
        patient inserted in the middle of the alphabet shifts review order.
        Positions are stored as pair_id everywhere else precisely so that
        renumbering cannot corrupt them.
        """
        now = time.time()
        added_series = added_pairs = 0

        for sp in report.pairs:
            existing = self.query_one("SELECT key FROM series WHERE key=?", (sp.key,))
            if not existing:
                self.execute(
                    """INSERT INTO series
                       (key,patient,study,ct_series,mri_series,ct_path,mri_path,
                        orientation,series_desc,body_region,n_ct_files,n_mri_files,
                        n_pairs,reg_status,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?, 'PENDING',?,?)""",
                    (sp.key, sp.patient, sp.study, sp.ct_series, sp.mri_series,
                     sp.ct_path, sp.mri_path, sp.orientation, sp.series_desc,
                     region_for(sp.patient), sp.n_ct_files, sp.n_mri_files,
                     sp.n_pairs, now, now))
                added_series += 1
            else:
                # Paths and counts can legitimately change if files were added.
                self.execute(
                    """UPDATE series SET ct_path=?,mri_path=?,orientation=?,
                       series_desc=?,body_region=?,n_ct_files=?,n_mri_files=?,
                       n_pairs=?,updated_at=? WHERE key=?""",
                    (sp.ct_path, sp.mri_path, sp.orientation, sp.series_desc,
                     region_for(sp.patient), sp.n_ct_files, sp.n_mri_files,
                     sp.n_pairs, now, sp.key))

            rows = []
            for i in range(sp.n_pairs):
                pair_id = f"{sp.key}/{i:04d}"
                rows.append((
                    pair_id, 0, sp.key, sp.patient, sp.study, sp.ct_series,
                    sp.mri_series, i,
                    sp.ct_files[i] if i < len(sp.ct_files) else None,
                    sp.mri_files[i] if i < len(sp.mri_files) else None,
                    now, now))
            before = self.query_one(
                "SELECT COUNT(*) AS n FROM pairs WHERE series_key=?", (sp.key,))["n"]
            self.executemany(
                """INSERT OR IGNORE INTO pairs
                   (pair_id,seq,series_key,patient,study,ct_series,mri_series,
                    slice_index,ct_file,mri_file,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", rows)

            # Existing rows keep their QC decisions, but their FILENAMES must
            # follow the scan. The pairing rule can change (it did, when slice
            # matching moved from stack position to image number), and a row
            # left advertising the old filenames would name one slice while the
            # pipeline processed another - the worst kind of wrong, because it
            # looks right.
            self.executemany(
                "UPDATE pairs SET ct_file=?, mri_file=?, updated_at=? WHERE pair_id=?",
                [(r[8], r[9], now, r[0]) for r in rows])
            after = self.query_one(
                "SELECT COUNT(*) AS n FROM pairs WHERE series_key=?", (sp.key,))["n"]
            added_pairs += after - before

        self.renumber()
        return {"added_series": added_series, "added_pairs": added_pairs}

    def renumber(self) -> None:
        """Assign dense review-order sequence numbers over the whole table."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT pair_id FROM pairs "
                "ORDER BY patient, study, ct_series, slice_index").fetchall()
            self._conn.executemany(
                "UPDATE pairs SET seq=? WHERE pair_id=?",
                [(i, r["pair_id"]) for i, r in enumerate(rows)])
            self._conn.commit()

    # ── counters ──────────────────────────────────────────────────────────────
    def counts(self) -> dict:
        rows = self.query("SELECT qc_status, COUNT(*) AS n FROM pairs GROUP BY qc_status")
        by = {r["qc_status"]: r["n"] for r in rows}
        total = sum(by.values())
        srows = self.query("SELECT reg_status, COUNT(*) AS n FROM series GROUP BY reg_status")
        sby = {r["reg_status"]: r["n"] for r in srows}
        return {
            "total":     total,
            "accepted":  by.get("ACCEPTED", 0),
            "rejected":  by.get("REJECTED", 0),
            "pending":   by.get("PENDING", 0),
            "errors":    by.get("ERROR", 0),
            "series": {
                "total":      sum(sby.values()),
                "pending":    sby.get("PENDING", 0),
                "processing": sby.get("PROCESSING", 0),
                "registered": sby.get("REGISTERED", 0),
                "error":      sby.get("ERROR", 0),
            },
        }
