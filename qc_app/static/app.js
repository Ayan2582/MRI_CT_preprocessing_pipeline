/*
 * app.js — the review client.
 *
 * Three ideas hold this together:
 *
 *  1. Each canvas owns its own view transform, fitted to its own viewport.
 *     The three panes are different sizes, so one shared absolute transform
 *     would centre the image in at most one of them.
 *
 *     "Link views" (default on) propagates a pan or zoom as a DELTA to the
 *     other canvases rather than copying the transform: they end up at the
 *     same magnification, showing the same region, each still centred in its
 *     own pane. Turn it off to move one image on its own.
 *
 *     Linking is on by default because comparing two images is only meaningful
 *     when both are showing the same region at the same magnification. Note
 *     that panning the MRI on its own does NOT change the registration — it
 *     moves pixels on screen only, and the metrics panel is unaffected.
 *
 *  2. The overlay blend is done client-side from the CT and MRI PNGs, so the
 *     opacity slider is instant. Fusion, checker and difference come from the
 *     server, where the arithmetic happens on the actual float arrays rather
 *     than on 8-bit screen pixels.
 *
 *  3. Nothing is cached across pairs except the browser's own image cache.
 *     Positions are addressed by seq, decisions by pair_id, and every mutation
 *     re-reads the counters from the response, so two tabs cannot drift apart.
 */

'use strict';

const $ = (id) => document.getElementById(id);

const canvases = {
  ct:     $('cv-ct'),
  mri:    $('cv-mri'),
  result: $('cv-result'),
};
const CANVAS_NAMES = Object.keys(canvases);

const state = {
  seq: 0,
  total: 0,
  pair: null,
  series: null,
  ready: false,
  view: 'overlay',
  alpha: 0.5,
  showBefore: false,
  cropMode: false,
  cropRect: null,          // live drag rectangle, in image pixels
  cropDrag: null,          // {handle, base, start} while resizing/moving a box
  eraseMode: false,
  nudgeMode: false,
  brushSize: 10,           // radius in image px == mm at 1 mm/px
  strokes: [],             // [{r, pts:[[x,y],…]}] for the current slice
  liveStroke: null,
  cursor: null,            // brush cursor position, in image px
  images: {},              // view name -> HTMLImageElement
  tfs: {                   // one view transform per canvas
    ct:     { scale: 1, tx: 0, ty: 0 },
    mri:    { scale: 1, tx: 0, ty: 0 },
    result: { scale: 1, tx: 0, ty: 0 },
  },
  linked: true,
  fitted: false,
  filter: 'all',
  query: '',
  busy: false,
  imgVersion: 0,     // bumped on every edit, to defeat the browser image cache
};

/* ── tiny helpers ─────────────────────────────────────────────────────────── */

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const post = (path, body) => api(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {}),
});

let toastTimer = null;
function toast(msg, kind) {
  const el = $('toast');
  el.textContent = msg;
  el.className = 'toast ' + (kind || '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), 3600);
}

const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : Number(v).toFixed(d);

/* ── view transforms ──────────────────────────────────────────────────────── */

function sizeCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  return dpr;
}

function viewport(name) {
  const cv = canvases[name];
  const dpr = sizeCanvas(cv);
  return { dpr, w: cv.width / dpr, h: cv.height / dpr };
}

function imageSize() {
  const img = state.images.ct;
  return img && img.complete && img.naturalWidth
    ? { w: img.naturalWidth, h: img.naturalHeight } : null;
}

/* Fit one canvas to its OWN viewport. The panes differ in size, so each needs
   its own scale and centring — a single shared transform leaves the image
   off-centre or clipped in every pane but the one it was computed from. */
const lastSize = { ct: null, mri: null, result: null };

/* A pane changed size - a toolbar appeared, the window moved, the layout
   reflowed. Keep the current magnification and re-anchor so the image point
   that was at the centre stays at the centre. Re-fitting instead would throw
   away a zoom the reviewer deliberately set, which is what made toggling a
   mode look like the picture "shrank by itself". */
function reanchor(name) {
  const prev = lastSize[name];
  const vp = viewport(name);
  lastSize[name] = { w: vp.w, h: vp.h };
  if (!prev || !imageSize()) return;
  if (Math.abs(prev.w - vp.w) < 0.5 && Math.abs(prev.h - vp.h) < 0.5) return;
  const tf = state.tfs[name];
  const cx = (prev.w / 2 - tf.tx) / tf.scale;
  const cy = (prev.h / 2 - tf.ty) / tf.scale;
  tf.tx = vp.w / 2 - cx * tf.scale;
  tf.ty = vp.h / 2 - cy * tf.scale;
}

function fitOne(name) {
  const dim = imageSize();
  if (!dim) return;
  const vp = viewport(name);
  lastSize[name] = { w: vp.w, h: vp.h };
  const scale = Math.min(vp.w / dim.w, vp.h / dim.h);
  state.tfs[name] = {
    scale,
    tx: (vp.w - dim.w * scale) / 2,
    ty: (vp.h - dim.h * scale) / 2,
  };
}

function fitView() {
  CANVAS_NAMES.forEach(fitOne);
  state.fitted = true;
  drawAll();
}

function panBy(name, dx, dy) {
  const targets = state.linked ? CANVAS_NAMES : [name];
  for (const n of targets) {
    state.tfs[n].tx += dx;
    state.tfs[n].ty += dy;
  }
}

/* Zoom by factor k. The canvas under the cursor keeps the point beneath the
   cursor fixed; linked canvases zoom about their own centre, so they stay
   centred in their own pane instead of inheriting a position that only made
   sense in the pane the wheel event came from. */
function zoomBy(name, k, clientX, clientY) {
  const targets = state.linked ? CANVAS_NAMES : [name];
  for (const n of targets) {
    const tf = state.tfs[n];
    const vp = viewport(n);
    let ax, ay;
    if (n === name && clientX !== undefined) {
      const r = canvases[n].getBoundingClientRect();
      ax = clientX - r.left;
      ay = clientY - r.top;
    } else {
      ax = vp.w / 2;
      ay = vp.h / 2;
    }
    const ix = (ax - tf.tx) / tf.scale;
    const iy = (ay - tf.ty) / tf.scale;
    tf.scale = Math.max(0.05, Math.min(40, tf.scale * k));
    tf.tx = ax - ix * tf.scale;
    tf.ty = ay - iy * tf.scale;
  }
}

/* Canvas point -> image pixel, for that canvas's own transform. */
function toImage(name, clientX, clientY) {
  const r = canvases[name].getBoundingClientRect();
  const tf = state.tfs[name];
  return {
    x: (clientX - r.left - tf.tx) / tf.scale,
    y: (clientY - r.top - tf.ty) / tf.scale,
  };
}

/* ── drawing ──────────────────────────────────────────────────────────────── */

function paintBase(name) {
  const cv = canvases[name];
  const vp = viewport(name);
  const ctx = cv.getContext('2d');
  ctx.setTransform(vp.dpr, 0, 0, vp.dpr, 0, 0);
  ctx.clearRect(0, 0, vp.w, vp.h);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, vp.w, vp.h);
  return ctx;
}

function paintWith(ctx, img, tf, alpha) {
  if (!img || !img.complete || !img.naturalWidth) return;
  ctx.save();
  ctx.translate(tf.tx, tf.ty);
  ctx.scale(tf.scale, tf.scale);
  ctx.imageSmoothingEnabled = tf.scale < 2;
  if (alpha !== undefined) ctx.globalAlpha = alpha;
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

function paintImage(name, ctx, img, alpha) {
  if (!img || !img.complete || !img.naturalWidth) return;
  const tf = state.tfs[name];
  ctx.save();
  ctx.translate(tf.tx, tf.ty);
  ctx.scale(tf.scale, tf.scale);
  // Nearest-neighbour above 2x: at that magnification a reviewer is looking at
  // individual pixels, and smoothing invents edges that are not there.
  ctx.imageSmoothingEnabled = tf.scale < 2;
  if (alpha !== undefined) ctx.globalAlpha = alpha;
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

function drawPane(name, img) {
  // Deliberately NO stroke overlay here. These panes show the erase RESULT -
  // the server bakes it into every rendered view, so what you see is the CT
  // and the MRI with those pixels actually zeroed. Painting the red marker on
  // top would hide the one thing the panes exist to confirm: whether the bed
  // line is gone. The marker stays on the result view, where the painting
  // happens and where feedback is needed.
  paintImage(name, paintBase(name), img);
}

function drawResult() {
  const ctx = paintBase('result');

  if (state.view === 'overlay') {
    paintImage('result', ctx, state.images.ct);
    paintImage('result', ctx,
      state.showBefore ? state.images.mri_before : state.images.mri, state.alpha);
  } else {
    const key = state.showBefore && state.view !== 'checker'
      ? state.view + '_before' : state.view;
    paintImage('result', ctx, state.images[key] || state.images[state.view]);
  }

  drawStrokes(ctx, 'result');
  drawRoi(ctx);
}

/* Painted regions, drawn exactly as the server rasterises them: round caps and
   round joins at radius r. What you see is what gets zeroed. */
function drawStrokes(ctx, name = 'result') {
  const all = state.liveStroke ? state.strokes.concat([state.liveStroke]) : state.strokes;
  if (!all.length) return;

  const tf = state.tfs[name];
  ctx.save();
  ctx.translate(tf.tx, tf.ty);
  ctx.scale(tf.scale, tf.scale);
  ctx.strokeStyle = 'rgba(217,96,90,.45)';
  ctx.fillStyle = 'rgba(217,96,90,.45)';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const s of all) {
    if (!s.pts.length) continue;
    if (s.pts.length > 1) {
      ctx.beginPath();
      ctx.lineWidth = s.r * 2;
      ctx.moveTo(s.pts[0][0], s.pts[0][1]);
      for (let i = 1; i < s.pts.length; i++) ctx.lineTo(s.pts[i][0], s.pts[i][1]);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(s.pts[0][0], s.pts[0][1], s.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

/* The brush outline, on its own overlay canvas so moving the pointer does not
   force a redraw of the image underneath. */
function drawBrushCursor() {
  const cv = $('cv-brush');
  const vp = viewport('result');
  const dpr = window.devicePixelRatio || 1;
  const r = canvases.result.getBoundingClientRect();
  if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) {
    cv.width = Math.round(r.width * dpr);
    cv.height = Math.round(r.height * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cv.width / dpr, cv.height / dpr);
  if (!state.eraseMode || !state.cursor) return;

  const tf = state.tfs.result;
  ctx.beginPath();
  ctx.arc(tf.tx + state.cursor.x * tf.scale,
          tf.ty + state.cursor.y * tf.scale,
          state.brushSize * tf.scale, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,.9)';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.strokeStyle = 'rgba(0,0,0,.6)';
  ctx.lineWidth = 1;
  ctx.arc(tf.tx + state.cursor.x * tf.scale,
          tf.ty + state.cursor.y * tf.scale,
          state.brushSize * tf.scale + 1, 0, Math.PI * 2);
  ctx.stroke();
}

const HANDLES = ['nw', 'n', 'ne', 'w', 'e', 'sw', 's', 'se'];

function handlePoints(roi) {
  const [x, y, w, h] = roi;
  return {
    nw: [x, y], n: [x + w / 2, y], ne: [x + w, y],
    w: [x, y + h / 2], e: [x + w, y + h / 2],
    sw: [x, y + h], s: [x + w / 2, y + h], se: [x + w, y + h],
  };
}

/* Which handle is under the pointer, in SCREEN space - so the grab area stays
   the same physical size however far the view is zoomed. */
function hitHandle(roi, clientX, clientY) {
  if (!roi) return null;
  const tf = state.tfs.result;
  const r = canvases.result.getBoundingClientRect();
  const pts = handlePoints(roi);
  for (const k of HANDLES) {
    const sx = r.left + tf.tx + pts[k][0] * tf.scale;
    const sy = r.top + tf.ty + pts[k][1] * tf.scale;
    if (Math.hypot(clientX - sx, clientY - sy) <= 9) return k;
  }
  return null;
}

function insideRoi(roi, p) {
  return roi && p.x >= roi[0] && p.x <= roi[0] + roi[2]
             && p.y >= roi[1] && p.y <= roi[1] + roi[3];
}

/* Drag one edge or corner. Only the named sides move, so extending a box is
   one gesture rather than redrawing it. */
function resizeRoi(base, handle, p) {
  let [x0, y0] = [base[0], base[1]];
  let [x1, y1] = [base[0] + base[2], base[1] + base[3]];
  if (handle.includes('w')) x0 = p.x;
  if (handle.includes('e')) x1 = p.x;
  if (handle.includes('n')) y0 = p.y;
  if (handle.includes('s')) y1 = p.y;
  return [Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)];
}

function clampRoi(roi) {
  const dim = imageSize();
  if (!dim) return roi;
  const x = Math.max(0, Math.min(roi[0], dim.w - 1));
  const y = Math.max(0, Math.min(roi[1], dim.h - 1));
  return [x, y, Math.min(roi[2], dim.w - x), Math.min(roi[3], dim.h - y)];
}

function drawRoi(ctx) {
  const roi = state.cropRect ||
    (state.pair && state.pair.roi ? state.pair.roi : null);
  if (!roi) return;

  const tf = state.tfs.result;
  ctx.save();
  ctx.translate(tf.tx, tf.ty);
  ctx.scale(tf.scale, tf.scale);

  // Everything outside the ROI is dimmed, so the region the metric actually
  // sees is unmistakable rather than being a thin line you have to hunt for.
  //
  // Only once the rectangle HAS an area. On mousedown it is [x, y, 0, 0], and
  // an even-odd fill whose inner rectangle encloses nothing dims the entire
  // frame - which reads as the image vanishing the moment you click.
  const dim = imageSize();
  if (dim && roi[2] >= 1 && roi[3] >= 1) {
    ctx.fillStyle = 'rgba(6,9,12,.55)';
    ctx.beginPath();
    ctx.rect(0, 0, dim.w, dim.h);
    ctx.rect(roi[0], roi[1], roi[2], roi[3]);
    ctx.fill('evenodd');
  }

  ctx.strokeStyle = state.cropRect ? '#4ea3d8' : '#d4a13c';
  ctx.lineWidth = Math.max(0.5, 1.5 / tf.scale);
  ctx.setLineDash(state.cropRect ? [] : [6 / tf.scale, 4 / tf.scale]);
  ctx.strokeRect(roi[0], roi[1], roi[2], roi[3]);
  ctx.restore();

  // Grab handles, in crop mode only, drawn at a fixed screen size.
  if (state.cropMode) {
    const pts = handlePoints(roi);
    ctx.save();
    ctx.fillStyle = '#4ea3d8';
    ctx.strokeStyle = '#0d1013';
    ctx.lineWidth = 1;
    for (const k of HANDLES) {
      const sx = tf.tx + pts[k][0] * tf.scale;
      const sy = tf.ty + pts[k][1] * tf.scale;
      ctx.beginPath();
      ctx.rect(sx - 3.5, sy - 3.5, 7, 7);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  // Size label in screen pixels, so it stays readable at any zoom.
  ctx.save();
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillStyle = '#dde5ec';
  ctx.fillText(`${Math.round(roi[2])} x ${Math.round(roi[3])} mm`,
               tf.tx + roi[0] * tf.scale, tf.ty + roi[1] * tf.scale - 5);
  ctx.restore();
}

/* The crop pane: the current view, framed to the ROI and scaled to fill.
   It answers "what does the crop actually contain" without making you zoom
   the main view, and it is where an edge you have just dragged shows its
   effect immediately. */
function drawCropPane() {
  const cv = $('cv-crop');
  if (!cv) return;
  const dpr = sizeCanvas(cv);
  const vw = cv.width / dpr, vh = cv.height / dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, vw, vh);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, vw, vh);

  const roi = state.cropRect || (state.pair && state.pair.roi) || null;
  const cap = $('cap-crop');
  if (!roi || roi[2] < 1 || roi[3] < 1) {
    if (cap) cap.textContent = 'no crop';
    ctx.fillStyle = '#5c6874';
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('no crop on this slice', vw / 2, vh / 2);
    ctx.textAlign = 'start';
    return;
  }
  if (cap) cap.textContent = `${Math.round(roi[2])} x ${Math.round(roi[3])} mm`;

  const sc = Math.min(vw / roi[2], vh / roi[3]);
  const tf = {
    scale: sc,
    tx: (vw - roi[2] * sc) / 2 - roi[0] * sc,
    ty: (vh - roi[3] * sc) / 2 - roi[1] * sc,
  };

  // Clip to the rectangle before painting. Framing alone is not enough: the
  // transform positions the box, but the whole slice is still drawn through
  // it, so anatomy outside the crop spills into the pane. Clipping is what
  // makes this pane show the crop RESULT - exactly the pixels the rectangle
  // encloses, and nothing else.
  ctx.save();
  ctx.beginPath();
  ctx.rect(tf.tx + roi[0] * tf.scale, tf.ty + roi[1] * tf.scale,
           roi[2] * tf.scale, roi[3] * tf.scale);
  ctx.clip();

  if (state.view === 'overlay') {
    paintWith(ctx, state.images.ct, tf);
    paintWith(ctx, state.showBefore ? state.images.mri_before : state.images.mri,
              tf, state.alpha);
  } else {
    const key = state.showBefore && state.view !== 'checker'
      ? state.view + '_before' : state.view;
    paintWith(ctx, state.images[key] || state.images[state.view], tf);
  }
  ctx.restore();

  // A thin outline so the crop's own edge is visible against a dark pane.
  ctx.strokeStyle = 'rgba(212,161,60,.55)';
  ctx.lineWidth = 1;
  ctx.strokeRect(tf.tx + roi[0] * tf.scale + 0.5, tf.ty + roi[1] * tf.scale + 0.5,
                 roi[2] * tf.scale - 1, roi[3] * tf.scale - 1);
}

function drawAll() {
  if (!state.fitted) { fitView(); return; }
  drawPane('ct', state.images.ct);
  drawPane('mri', state.showBefore ? state.images.mri_before : state.images.mri);
  drawResult();
  drawCropPane();
}

/* ── image loading ────────────────────────────────────────────────────────── */

function loadImage(pairId, view) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    // The URL is otherwise identical after a nudge or an erase, and browsers
    // will happily reuse the in-memory image for it even under no-store. The
    // version token is what forces a real refetch.
    img.src = `/api/image?pair_id=${encodeURIComponent(pairId)}&view=${view}`
            + `&v=${state.imgVersion}`;
  });
}

async function loadImagesFor(pair) {
  const wanted = new Set(['ct', 'mri', 'mri_before']);
  if (state.view !== 'overlay') {
    wanted.add(state.view);
    if (state.showBefore && state.view !== 'checker') wanted.add(state.view + '_before');
  }
  const names = [...wanted];
  const imgs = await Promise.all(names.map((v) => loadImage(pair.pair_id, v)));
  state.images = {};
  names.forEach((n, i) => { if (imgs[i]) state.images[n] = imgs[i]; });
}

/* ── loading a pair ───────────────────────────────────────────────────────── */

let pollTimer = null;

async function loadPair(seq, pairId) {
  let data;
  try {
    const q = pairId ? `pair_id=${encodeURIComponent(pairId)}` : `seq=${seq}`;
    data = await api(`/api/pair?${q}`);
  } catch (e) {
    toast(e.message, 'bad');
    return;
  }

  // What we are leaving. Captured before state.pair is replaced, because it
  // is the box that seeds the slice we are about to open.
  const leaving = state.pair
    ? { seriesKey: state.pair.series_key,
        roi: state.pair.roi ? state.pair.roi.slice() : null,
        roi_mode: state.pair.roi_mode || 'metric' }
    : null;

  state.pair = data.pair;
  // The server sends the PARSED crop and nudge at the top level of the
  // response; data.pair is the raw DB row and carries only *_json strings.
  // Copying them onto state.pair is what makes one object the single source
  // the renderers read - reading half from each is how the crop went missing.
  state.pair.roi      = data.roi ?? null;
  state.pair.roi_mode = data.roi_mode || 'metric';
  carryCropForward(leaving);
  state.pair.ct_rect  = data.ct_rect ?? null;
  state.pair.mri_rect = data.mri_rect ?? null;
  state.pair.nudge_dx = data.nudge_dx ?? 0;
  state.pair.nudge_dy = data.nudge_dy ?? 0;
  state.series = data.series;
  state.seq = data.pair.seq;
  state.total = data.total;
  state.ready = data.ready;
  state.cropRect = null;
  state.fitted = false;
  state.liveStroke = null;
  // Strokes belong to the slice, so they are reloaded with it.
  try {
    state.strokes = data.pair.erase_json ? JSON.parse(data.pair.erase_json) : [];
  } catch (_) {
    state.strokes = [];
  }

  renderIdentity();
  renderMetrics();
  renderProgress();
  renderNudge();
  renderCropRects();
  $('auto-erase').checked = !!state.series.auto_erase;

  $('not-ready').classList.toggle('hidden', data.ready);
  if (!data.ready) {
    $('not-ready-text').textContent = state.series.reg_status === 'ERROR'
      ? 'This series failed to register — see the error below.'
      : 'Registering this series…';
    state.images = {};
    CANVAS_NAMES.forEach(paintBase);
    schedulePoll();
  } else {
    await loadImagesFor(data.pair);
    fitView();
  }
  refreshList();
}

function schedulePoll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (state.pair && !state.ready) await loadPair(state.seq, state.pair.pair_id);
  }, 1500);
}

/* ── rendering the chrome ─────────────────────────────────────────────────── */

function renderIdentity() {
  const p = state.pair, s = state.series;
  $('id-patient').textContent = p.patient;
  $('id-series').textContent = p.ct_series === p.mri_series
    ? p.ct_series : `${p.ct_series} ↔ ${p.mri_series}`;
  $('id-image').textContent = p.ct_file || `#${p.slice_index}`;

  $('id-orientation').textContent =
    `${s.orientation || 'unknown'} · slice ${p.slice_index + 1} of ${s.n_pairs}`;
  $('id-region').textContent = s.body_region || '—';
  $('id-bg').classList.toggle('hidden', !p.is_background);

  const st = $('id-status');
  st.textContent = p.qc_status.toLowerCase();
  st.className = 'chip is-' + p.qc_status.toLowerCase();

  $('cap-ct').textContent = p.ct_file || '';
  $('cap-mri').textContent = p.mri_file || '';
}

function renderProgress() {
  $('pos').textContent = state.seq + 1;
  $('total').textContent = state.total;
  $('progress-fill').style.width =
    state.total ? `${((state.seq + 1) / state.total) * 100}%` : '0';
}

function renderMetrics() {
  const s = state.series;
  const applied = $('m-applied');
  applied.textContent = s.reg_applied ? 'yes' : 'no';
  applied.className = 'mv ' + (s.reg_applied ? 'good' : 'bad');

  $('m-shift').textContent = s.reg_applied || s.reg_dx_mm || s.reg_dy_mm
    ? `${fmt(s.reg_dx_mm, 0)} , ${fmt(s.reg_dy_mm, 0)} mm` : '—';
  $('m-gain').textContent = s.reg_nmi_gain === null || s.reg_nmi_gain === undefined
    ? '—' : (s.reg_nmi_gain >= 0 ? '+' : '') + fmt(s.reg_nmi_gain, 4);
  $('m-spread').textContent = (s.reg_spread_y_mm === null && s.reg_spread_x_mm === null)
    ? '—' : `${fmt(s.reg_spread_y_mm, 0)} / ${fmt(s.reg_spread_x_mm, 0)} mm`;
  $('m-probes').textContent = s.reg_n_probes
    ? `${s.reg_n_usable} of ${s.reg_n_probes}` +
      (s.reg_hit_edge ? ` (${s.reg_hit_edge} at edge)` : '') : '—';
  $('m-search').textContent = s.reg_search_mm ? `±${fmt(s.reg_search_mm, 0)} mm` : '—';
  const proi = state.pair && state.pair.roi;
  $('m-roi').textContent = proi
    ? `${Math.round(proi[2])} x ${Math.round(proi[3])} mm @ ${Math.round(proi[0])},${Math.round(proi[1])}`
    : 'full frame';
  $('m-window').textContent = (s.ct_win_min === null || s.ct_win_min === undefined)
    ? '—' : `${s.ct_win_min} … ${s.ct_win_max} HU`;

  $('m-reason').textContent = s.reg_reason ||
    (s.reg_status === 'PENDING' ? 'Not registered yet.' : '—');

  const err = $('m-error');
  const msg = s.error_message || state.pair.error_message;
  err.textContent = msg || '';
  err.classList.toggle('hidden', !msg);
}

/* ── the pair list ────────────────────────────────────────────────────────── */

let listTimer = null;
function refreshList() {
  clearTimeout(listTimer);
  listTimer = setTimeout(async () => {
    const params = new URLSearchParams({ status: state.filter, limit: '300' });
    if (state.query) params.set('q', state.query);
    // Window the list around the reviewer rather than always starting at 0,
    // so 2313 pairs do not mean scrolling to find where you are. The server
    // converts this into an offset within the FILTERED set - computing it here
    // from the global seq is what made a filtered list come back empty.
    params.set('around_seq', String(state.seq));

    let data;
    try { data = await api('/api/pairs?' + params); } catch (_) { return; }

    $('list-count').textContent = data.total;
    const ul = $('pair-list');
    ul.innerHTML = '';
    for (const r of data.rows) {
      const li = document.createElement('li');
      li.className = r.pair_id === (state.pair && state.pair.pair_id) ? 'on' : '';
      const marks =
        (r.nudge_dx || r.nudge_dy
          ? `<span class="pl-mark" title="nudged ${r.nudge_dx}, ${r.nudge_dy} mm">N</span>` : '') +
        (r.erased ? '<span class="pl-mark pl-mark-e" title="has erase strokes">E</span>' : '');
      li.innerHTML =
        `<span class="pl-dot s-${r.qc_status}"></span>` +
        `<span class="pl-name">${r.patient} · ${r.ct_series} · ${r.ct_file || r.slice_index}</span>` +
        marks +
        `<span class="pl-seq">${r.seq + 1}</span>`;
      li.onclick = () => loadPair(r.seq, r.pair_id);
      ul.appendChild(li);
    }
    const on = ul.querySelector('.on');
    if (on) on.scrollIntoView({ block: 'nearest' });
  }, 120);
}

async function refreshStatus() {
  let data;
  try { data = await api('/api/status'); } catch (_) { return; }

  const c = data.counts;
  $('c-accepted').textContent = c.accepted;
  $('c-rejected').textContent = c.rejected;
  $('c-pending').textContent = c.pending;
  $('c-errors').textContent = c.errors;

  const w = data.worker;
  const dot = $('worker-dot');
  dot.className = 'dot' + (w.current ? ' busy' : (w.last_error ? ' err' : ''));
  $('worker-state').textContent = w.current
    ? `registering ${w.current.split('/')[0]}…`
    : (w.queued ? `${w.queued} queued` : 'idle');
  $('worker-detail').textContent = w.current
    ? (w.progress || '—')
    : `${c.series.registered}/${c.series.total} series registered`;

  renderProblems(data.problems || []);

  // The series we are looking at may have just finished.
  if (state.pair && !state.ready && w.current !== (state.series && state.series.key)) {
    loadPair(state.seq, state.pair.pair_id);
  }
}

let problemsRendered = -1;
function renderProblems(problems) {
  if (problems.length === problemsRendered) return;
  problemsRendered = problems.length;
  const box = $('problems');
  box.innerHTML = '';
  for (const p of problems.slice(0, 60)) {
    const d = document.createElement('div');
    d.className = 'p-' + p.level;
    d.textContent = `${p.where}: ${p.message}`;
    box.appendChild(d);
  }
}

/* ── actions ──────────────────────────────────────────────────────────────── */

async function decide(kind) {
  if (!state.pair || state.busy) return;
  state.busy = true;
  try {
    await post(`/api/${kind}`, {
      pair_id: state.pair.pair_id,
      crop_export: $('crop-export').checked,
    });
    state.pair.qc_status =
      kind === 'accept' ? 'ACCEPTED' : kind === 'reject' ? 'REJECTED' : 'PENDING';
    renderIdentity();
    refreshStatus();
    refreshList();
    if (kind !== 'reset') next();
  } catch (e) {
    toast(e.status === 425
      ? 'Still registering this series — try again in a moment.'
      : e.message, 'bad');
  } finally {
    state.busy = false;
  }
}

function next() { if (state.seq + 1 < state.total) loadPair(state.seq + 1); }
function prev() { if (state.seq > 0) loadPair(state.seq - 1); }

/* Save a crop region without touching the registration. The Re-run button
   lights up so it is obvious the stored region has not been measured with. */
async function storeRoi(roi, clearRoi, toSeries) {
  if (!state.pair) return;
  try {
    const r = await post('/api/roi', {
      pair_id: state.pair.pair_id,
      roi: roi || null,
      clear_roi: !!clearRoi,
      roi_mode: $('crop-export').checked ? 'export' : 'metric',
      apply_to_series: !!toSeries,
    });
    state.pair.roi = roi || null;
    state.cropRect = null;
    drawAll();
    if (toSeries) toast(`Crop applied to ${r.slices_updated} slices`, 'ok');
    state.roiDirty = true;
    $('btn-rerun').classList.add('on');
    renderMetrics();
    drawResult();
    toast(roi ? 'Crop saved — press Re-run to register with it'
              : 'Crop cleared — press Re-run to register without it', 'ok');
  } catch (e) {
    toast(e.message, 'bad');
  }
}

async function rerun(roi, clearRoi) {
  if (!state.series) return;
  try {
    // No roi here: crops live on the slices, and the worker collects every
    // slice's own crop when it runs. Passing one would override all of them.
    await post('/api/rerun', {
      series_key: state.series.key,
      search_mm: null,
    });
    toast(roi ? 'Re-running on the selected region…'
              : 'Re-running this series…', 'ok');
    state.ready = false;
    state.roiDirty = false;
    $('btn-rerun').classList.remove('on');
    state.cropRect = null;
    $('not-ready').classList.remove('hidden');
    $('not-ready-text').textContent = 'Registering this series…';
    schedulePoll();
  } catch (e) {
    toast(e.message, 'bad');
  }
}

async function bulk(decision) {
  if (!state.series) return;
  try {
    const res = await post('/api/series/bulk', {
      series_key: state.series.key,
      decision,
      crop_export: $('crop-export').checked,
    });
    toast(`${decision.toLowerCase()} — ${res.updated} slice${res.updated === 1 ? '' : 's'}` +
          (res.failed.length ? `, ${res.failed.length} failed` : ''),
          res.failed.length ? 'bad' : 'ok');
    refreshStatus();
    loadPair(state.seq, state.pair.pair_id);
  } catch (e) {
    toast(e.status === 425
      ? 'Still registering this series — try again in a moment.'
      : e.message, 'bad');
  }
}

/* ── interaction ──────────────────────────────────────────────────────────── */

function setView(view) {
  state.view = view;
  document.querySelectorAll('#view-tabs .tab').forEach((t) =>
    t.classList.toggle('tab-on', t.dataset.view === view));
  $('alpha-wrap').classList.toggle('hidden', view !== 'overlay');
  if (state.pair && state.ready) loadImagesFor(state.pair).then(drawAll);
}

function setCropMode(on) {
  state.cropMode = on;
  if (on) { setEraseMode(false); setNudgeMode(false); }
  $('btn-crop').classList.toggle('on', on);
  $('crop-target-bar').classList.toggle('hidden', !on);
  canvases.result.classList.toggle('cropping', on);
  if (!on) state.cropRect = null;
  drawAll();
}

function setEraseMode(on) {
  state.eraseMode = on;
  if (on) { setCropMode(false); setNudgeMode(false); }
  $('btn-erase').classList.toggle('on', on);
  $('brush-bar').classList.toggle('hidden', !on);
  canvases.result.classList.toggle('erasing', on);
  if (!on) state.cursor = null;
  drawBrushCursor();
  // No re-fit here. Showing the bar resizes the pane, and the ResizeObserver
  // re-anchors the view for it - keeping whatever zoom is set.
  drawAll();
}

/* Give a freshly-opened slice the box from the slice before it, in the same
   series, when it has none of its own. Anatomy moves slowly down a stack, so
   the previous rectangle is nearly always the right starting point - and it
   arrives as a real, editable box rather than something you have to redraw
   before you can nudge one edge.

   Only ever fills a gap: a slice that already has a crop is left alone. */
function sameRect(a, b) {
  return a && b && a.length === 4 && b.length === 4 &&
         a.every((v, i) => Math.abs(v - b[i]) < 0.5);
}

/* Carry the crop from the slice you just left onto the one you just opened.

   Deterministic on purpose. Two earlier versions were conditional - first
   "only where a slice has no box" (which fired on 10 of 2313 slices, because
   almost every slice already carried an inherited one), then "only if you
   edited it" (which does nothing when you simply page forward). Both were
   invisible in normal use. The rule is now exactly what it sounds like: step
   from slice N to slice N+1 in the same series and N+1 gets N's rectangle,
   ready to adjust.

   Two limits keep it safe: it never crosses a series boundary, since a box
   measured against one acquisition means nothing in the next; and it is off
   entirely when the `carry forward` box is unticked. */
function carryCropForward(leaving) {
  const p = state.pair;
  if (!p || !leaving || !leaving.roi) return;
  if (leaving.seriesKey !== p.series_key) return;

  const on = $('crop-carry') && $('crop-carry').checked;
  if (!on || sameRect(p.roi, leaving.roi)) return;

  p.roi = leaving.roi.slice();
  p.roi_mode = leaving.roi_mode || 'metric';
  // Persist, or it would vanish the moment you moved on again.
  post('/api/roi', { pair_id: p.pair_id, roi: p.roi, roi_mode: p.roi_mode })
    .catch(() => {});
}

/* ── manual nudge ─────────────────────────────────────────────────────────── */

function setNudgeMode(on) {
  state.nudgeMode = on;
  if (on) { setCropMode(false); setEraseMode(false); }
  $('btn-nudge').classList.toggle('on', on);
  $('nudge-bar').classList.toggle('hidden', !on);
  drawAll();
}

function renderNudge() {
  const s = state.series;
  if (!s || !state.pair) return;
  // Per slice, so the numbers live on the pair, not the series.
  const dx = state.pair.nudge_dx || 0, dy = state.pair.nudge_dy || 0;
  $('nudge-val').textContent = `manual ${dx >= 0 ? '+' : ''}${dx}, ${dy >= 0 ? '+' : ''}${dy} mm`;
  const tdx = (s.reg_dx_mm || 0) + dx, tdy = (s.reg_dy_mm || 0) + dy;
  $('nudge-total').textContent = `total ${tdx >= 0 ? '+' : ''}${tdx}, ${tdy >= 0 ? '+' : ''}${tdy} mm`;
  $('btn-nudge').classList.toggle('on', state.nudgeMode || !!(dx || dy));
}

async function nudge(dx, dy, absolute, toSeries) {
  if (!state.pair) return;
  const step = Math.max(1, parseInt($('nudge-step').value, 10) || 1);
  try {
    const r = await post('/api/nudge', {
      pair_id: state.pair.pair_id,
      dx: dx * (absolute ? 1 : step),
      dy: dy * (absolute ? 1 : step),
      target: $('nudge-target').value,
      absolute: !!absolute,
      apply_to_series: !!toSeries,
    });
    state.pair.nudge_dx = r.nudge_dx;
    state.pair.nudge_dy = r.nudge_dy;
    state.imgVersion++;
    if (toSeries) toast(`Offset applied to ${r.slices_updated} slices`, 'ok');
    renderNudge();
    // Only the registered MRI moves, so a refetch of that one view is enough.
    if (state.pair && state.ready) loadImagesFor(state.pair).then(drawAll);
  } catch (e) {
    toast(e.message, 'bad');
  }
}

/* ── per-modality export rectangles ───────────────────────────────────────── */

function renderCropRects() {
  const p = state.pair;
  if (!p) return;
  const f = (r) => r ? `${Math.round(r[2])}x${Math.round(r[3])}` : 'full';
  $('crop-rects').textContent =
    `metric ${f(p.roi)} · CT ${f(p.ct_rect)} · MRI ${f(p.mri_rect)}`;
}

async function setExportRect(target, rect, toSeries) {
  if (!state.pair) return;
  try {
    await post('/api/export_rect', {
      pair_id: state.pair.pair_id, target, rect, apply_to_series: !!toSeries });
    state.pair[target === 'ct' ? 'ct_rect' : 'mri_rect'] = rect;
    renderCropRects();
    drawResult();
    toast(rect ? `${target.toUpperCase()} export rectangle set — affects saved files only`
               : `${target.toUpperCase()} export rectangle cleared`, 'ok');
  } catch (e) {
    toast(e.message, 'bad');
  }
}

function setBrushSize(v) {
  state.brushSize = Math.max(2, Math.min(60, Math.round(v)));
  $('brush-size').value = state.brushSize;
  $('brush-size-val').textContent = `${state.brushSize} mm`;
  drawBrushCursor();
}

async function saveStrokes(copyToSeries) {
  if (!state.pair) return;
  try {
    const res = await post('/api/erase', {
      pair_id: state.pair.pair_id,
      strokes: state.strokes,
      copy_to_series: !!copyToSeries,
    });
    state.pair.erase_json = state.strokes.length ? JSON.stringify(state.strokes) : null;
    if (copyToSeries) {
      toast(`Erased region copied to ${res.slices_updated} slices — ` +
            `Re-run to fold it into the registration`, 'ok');
    }
  } catch (e) {
    toast(e.message, 'bad');
  }
}

function refreshErasedImages() {
  state.imgVersion++;
  // The server bakes the strokes into the rendered PNGs, so re-fetch once a
  // stroke is finished. Until then the local overlay is what you see.
  if (state.pair && state.ready) loadImagesFor(state.pair).then(drawAll);
}

function setLinked(on) {
  state.linked = on;
  $('link-views').checked = on;
  if (on) fitView();   // re-fit so the three panes agree again from a known state
}

function wireViewer(name) {
  const cv = canvases[name];
  let dragging = false, last = null, cropStart = null;

  cv.addEventListener('mousedown', (e) => {
    if (state.eraseMode && name === 'result' && e.button === 0) {
      const p = toImage(name, e.clientX, e.clientY);
      state.liveStroke = { r: state.brushSize, pts: [[p.x, p.y]] };
      drawResult();
    } else if (state.cropMode && name === 'result') {
      const cur = state.cropRect || (state.pair && state.pair.roi) || null;
      const p = toImage(name, e.clientX, e.clientY);
      const h = hitHandle(cur, e.clientX, e.clientY);
      if (h) {
        // Grab an edge or corner of the existing box and stretch it.
        state.cropDrag = { handle: h, base: cur.slice(), start: p };
        state.cropRect = cur.slice();
      } else if (insideRoi(cur, p)) {
        state.cropDrag = { handle: 'move', base: cur.slice(), start: p };
        state.cropRect = cur.slice();
      } else {
        cropStart = p;
        state.cropRect = [p.x, p.y, 0, 0];
      }
      drawResult();
    } else {
      dragging = true;
      last = { x: e.clientX, y: e.clientY };
    }
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => {
    if (state.eraseMode && name === 'result') {
      const p = toImage(name, e.clientX, e.clientY);
      state.cursor = p;
      if (state.liveStroke) {
        const last = state.liveStroke.pts[state.liveStroke.pts.length - 1];
        // Drop sub-pixel moves: they bloat the stored stroke without changing
        // a single rasterised pixel.
        if (Math.hypot(p.x - last[0], p.y - last[1]) >= 0.75) {
          state.liveStroke.pts.push([p.x, p.y]);
          drawResult();
        }
      }
      drawBrushCursor();
      if (state.liveStroke) return;
    }
    if (state.cropDrag && name === 'result') {
      const p = toImage(name, e.clientX, e.clientY);
      const d = state.cropDrag;
      state.cropRect = d.handle === 'move'
        ? [d.base[0] + (p.x - d.start.x), d.base[1] + (p.y - d.start.y), d.base[2], d.base[3]]
        : resizeRoi(d.base, d.handle, p);
      drawResult();
      drawCropPane();
    } else if (cropStart) {
      const p = toImage(name, e.clientX, e.clientY);
      state.cropRect = [
        Math.min(cropStart.x, p.x), Math.min(cropStart.y, p.y),
        Math.abs(p.x - cropStart.x), Math.abs(p.y - cropStart.y),
      ];
      drawResult();
      drawCropPane();
    } else if (dragging) {
      panBy(name, e.clientX - last.x, e.clientY - last.y);
      last = { x: e.clientX, y: e.clientY };
      if (state.linked) drawAll(); else drawOne(name);
    }
  });

  window.addEventListener('mouseup', () => {
    if (state.cropDrag) {
      const r = clampRoi(state.cropRect);
      state.cropDrag = null;
      if (r[2] >= 16 && r[3] >= 16) {
        state.cropRect = null;             // the stored value takes over
        const target = $('crop-target').value;
        const toSeries = $('crop-to-series').checked;
        if (target === 'ct' || target === 'mri') setExportRect(target, r, toSeries);
        else storeRoi(r, false, toSeries);
      } else {
        toast('That region is under 16 x 16 mm.', 'bad');
        state.cropRect = null;
        drawAll();
      }
      return;
    }
    if (state.liveStroke) {
      state.strokes.push(state.liveStroke);
      state.liveStroke = null;
      drawResult();
      saveStrokes(false).then(refreshErasedImages);
      return;
    }
    if (cropStart) {
      cropStart = null;
      const r = state.cropRect;
      if (r && r[2] >= 16 && r[3] >= 16) {
        const dim = imageSize();
        if (dim) {
          // Clamp to the frame: a rectangle dragged past the edge is a
          // rectangle the reviewer meant to end at the edge.
          const x = Math.max(0, Math.min(r[0], dim.w - 1));
          const y = Math.max(0, Math.min(r[1], dim.h - 1));
          const w = Math.min(r[2], dim.w - x);
          const h = Math.min(r[3], dim.h - y);
          const target = $('crop-target').value;
          const toSeries = $('crop-to-series').checked;
          if (target === 'ct' || target === 'mri') {
            // Per-modality: export framing only, nothing measured.
            setExportRect(target, [x, y, w, h], toSeries);
            state.cropRect = null;
          } else {
            // Shared: stored only. Adjusting a crop is not a request to
            // re-measure - the existing registration may be perfectly good,
            // and Re-run is the explicit way to say otherwise.
            state.cropRect = [x, y, w, h];
            storeRoi([x, y, w, h], false, toSeries);
          }
          setCropMode(false);
        }
      } else if (r) {
        toast('That region is under 16 x 16 mm — too few pixels for a stable histogram.', 'bad');
        state.cropRect = null;
        drawResult();
      }
    }
    dragging = false;
  });

  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    // Scroll always zooms, in every mode - erasing at 1:1 on a 250 px frame is
    // guesswork, so the brush must not take the gesture you need most. Brush
    // size moves to Alt/Shift+scroll, matching how image editors bind it, and
    // is also on [ / ] and the slider.
    if (state.eraseMode && name === 'result' && (e.altKey || e.shiftKey)) {
      setBrushSize(state.brushSize * (e.deltaY < 0 ? 1.15 : 1 / 1.15));
      return;
    }
    zoomBy(name, e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX, e.clientY);
    if (state.linked) drawAll(); else drawOne(name);
    drawBrushCursor();
  }, { passive: false });

  cv.addEventListener('mouseleave', () => {
    if (name === 'result') { state.cursor = null; drawBrushCursor(); }
  });

  // Left button paints while erasing, so panning falls to the middle or right
  // button. Suppressing the context menu is what makes a right-drag usable.
  cv.addEventListener('contextmenu', (e) => {
    if (state.eraseMode) e.preventDefault();
  });

  cv.addEventListener('dblclick', () => {
    if (state.linked) { fitView(); } else { fitOne(name); drawOne(name); }
  });
}

function drawOne(name) {
  if (name === 'result') { drawResult(); return; }
  drawPane(name, name === 'ct'
    ? state.images.ct
    : (state.showBefore ? state.images.mri_before : state.images.mri));
}

/* ── wiring ───────────────────────────────────────────────────────────────── */

function init() {
  CANVAS_NAMES.forEach(wireViewer);

  document.querySelectorAll('#view-tabs .tab').forEach((t) => {
    t.onclick = () => setView(t.dataset.view);
  });

  $('alpha').oninput = (e) => {
    state.alpha = e.target.value / 100;
    $('alpha-val').textContent = `${e.target.value}%`;
    drawResult();
    drawCropPane();
  };

  $('show-before').onchange = (e) => {
    state.showBefore = e.target.checked;
    if (state.pair && state.ready) loadImagesFor(state.pair).then(drawAll);
  };

  $('link-views').onchange = (e) => setLinked(e.target.checked);

  $('btn-fit').onclick = fitView;
  $('btn-accept').onclick = () => decide('accept');
  $('btn-reject').onclick = () => decide('reject');
  $('btn-next').onclick = next;
  $('btn-prev').onclick = prev;
  $('btn-crop').onclick = () => setCropMode(!state.cropMode);
  $('btn-erase').onclick = () => setEraseMode(!state.eraseMode);
  $('btn-nudge').onclick = () => setNudgeMode(!state.nudgeMode);

  // data-dx/data-dy are SCREEN direction (right/down positive). They are
  // negated into registration_idea's convention, where sample_window reads
  // out[y][x] = mri[y+dy][x+dx] - so a positive dx samples further right in
  // the source and the image appears to move LEFT. Verified: dx=+1 shifts a
  // test pixel from column 10 to column 9. Storage stays in the pipeline's
  // convention so a manual offset adds cleanly to reg_dx_mm/reg_dy_mm.
  document.querySelectorAll('#nudge-bar .pad-grid button').forEach((b) => {
    b.onclick = () => nudge(-Number(b.dataset.dx), -Number(b.dataset.dy));
  });
  $('nudge-reset').onclick = () => nudge(0, 0, true);
  // Copy THIS slice's offset to the whole series - the easy way back to one
  // consistent shift after fixing a slice by eye. Posted directly rather than
  // through nudge(), which negates screen direction into pipeline convention
  // and would flip an already-stored value.
  $('nudge-all').onclick = async () => {
    if (!state.pair) return;
    try {
      const r = await post('/api/nudge', {
        pair_id: state.pair.pair_id,
        dx: state.pair.nudge_dx || 0,
        dy: state.pair.nudge_dy || 0,
        target: 'mri', absolute: true, apply_to_series: true,
      });
      toast(`Offset ${r.nudge_dx}, ${r.nudge_dy} mm applied to ${r.slices_updated} slices`, 'ok');
    } catch (e) { toast(e.message, 'bad'); }
  };
  $('crop-clear-ct').onclick = () => setExportRect('ct', null, $('crop-to-series').checked);
  $('crop-clear-mri').onclick = () => setExportRect('mri', null, $('crop-to-series').checked);
  $('crop-clear-roi').onclick = () => storeRoi(null, true, $('crop-to-series').checked);

  $('brush-size').oninput = (e) => setBrushSize(Number(e.target.value));
  $('brush-undo').onclick = () => {
    if (!state.strokes.length) return;
    state.strokes.pop();
    drawResult();
    saveStrokes(false).then(refreshErasedImages);
  };
  $('brush-clear').onclick = () => {
    state.strokes = [];
    drawResult();
    saveStrokes(false).then(refreshErasedImages);
  };
  $('brush-all').onclick = () => saveStrokes(true);
  $('auto-erase').onchange = async (e) => {
    if (!state.series) return;
    try {
      await post('/api/auto_erase', { series_key: state.series.key, enabled: e.target.checked });
      state.series.auto_erase = e.target.checked;
      state.imgVersion++;
      toast(e.target.checked
        ? 'Rails removed on every slice of this series — Re-export edited to update saved files'
        : 'Automatic rail removal off', 'ok');
      if (state.pair && state.ready) loadImagesFor(state.pair).then(drawAll);
    } catch (err) { toast(err.message, 'bad'); }
  };
  $('btn-rerun').onclick = () => rerun();

  $('btn-series').onclick = (e) => {
    e.stopPropagation();
    $('series-menu').classList.toggle('hidden');
  };
  document.addEventListener('click', () => $('series-menu').classList.add('hidden'));
  $('series-menu').onclick = (e) => e.stopPropagation();
  $('series-menu').querySelectorAll('button[data-decision]').forEach((b) => {
    b.onclick = () => { bulk(b.dataset.decision); $('series-menu').classList.add('hidden'); };
  });
  $('btn-clear-roi').onclick = () => {
    storeRoi(null, true, true);
    $('series-menu').classList.add('hidden');
  };

  $('filter').onchange = (e) => { state.filter = e.target.value; refreshList(); };
  $('search').oninput = (e) => { state.query = e.target.value.trim(); refreshList(); };
  $('jump-go').onclick = () => {
    const n = parseInt($('jump').value, 10);
    if (!Number.isNaN(n)) loadPair(Math.max(0, Math.min(n - 1, state.total - 1)));
  };
  $('jump').onkeydown = (e) => { if (e.key === 'Enter') $('jump-go').click(); };

  $('btn-rescan').onclick = async () => {
    toast('Scanning dataset…');
    try {
      const r = await post('/api/scan');
      toast(`${r.series_found} series, ${r.slice_pairs} pairs — ${r.added_pairs} new`, 'ok');
      refreshStatus();
      refreshList();
    } catch (e) { toast(e.message, 'bad'); }
  };

  $('btn-export').onclick = async () => {
    try {
      const r = await post('/api/export');
      toast(`metadata.csv written — ${r.rows} accepted pairs`, 'ok');
    } catch (e) { toast(e.message, 'bad'); }
  };

  $('btn-reexport').onclick = async () => {
    try {
      const plan = await post('/api/reexport', { dry_run: true });
      if (!plan.would_rewrite) { toast('Every accepted file is up to date', 'ok'); return; }
      if (!confirm(`${plan.would_rewrite} accepted pairs were edited after they were ` +
                   `accepted, so their .npy files are out of date.

Rewrite them now?`)) return;
      toast('Rewriting…');
      const r = await post('/api/reexport', {});
      toast(`${r.rewritten} files rewritten` + (r.n_failed ? `, ${r.n_failed} failed` : ''),
            r.n_failed ? 'bad' : 'ok');
    } catch (e) { toast(e.message, 'bad'); }
  };

  $('btn-rerun-rejected').onclick = async () => {
    const mm = Number($('wide-search-mm').value) || 90;
    try {
      // Ask first, so the reviewer sees what will be touched before it starts.
      const plan = await post('/api/rerun_rejected', { search_mm: mm, dry_run: true });
      if (!plan.would_requeue) { toast('No rejected series need a wider search', 'ok'); return; }
      if (!confirm(`Re-measure ${plan.would_requeue} rejected series at ±${mm} mm?

` +
                   `Series already aligned (gain below threshold) are left alone.
` +
                   `Existing crops are kept. This runs in the background.`)) return;
      const r = await post('/api/rerun_rejected', { search_mm: mm });
      toast(`${r.requeued} series requeued at ±${mm} mm ` +
            `(${Object.entries(r.by_category).map(([k, v]) => `${v} ${k}`).join(', ')})`, 'ok');
      refreshStatus();
    } catch (e) { toast(e.message, 'bad'); }
  };

  $('btn-retry').onclick = async () => {
    try {
      const r = await post('/api/retry_errors');
      toast(r.requeued ? `${r.requeued} series requeued` : 'No failed series', 'ok');
      refreshStatus();
    } catch (e) { toast(e.message, 'bad'); }
  };

  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    switch (e.key.toLowerCase()) {
      case 'a': decide('accept'); break;
      case 'r': decide('reject'); break;
      case 'c': setCropMode(!state.cropMode); break;
      case 'e': setEraseMode(!state.eraseMode); break;
      case 'n': setNudgeMode(!state.nudgeMode); break;
      case '[': setBrushSize(state.brushSize - 2); break;
      case ']': setBrushSize(state.brushSize + 2); break;
      case 'f': fitView(); break;
      case 'b': $('show-before').click(); break;
      case 'l': setLinked(!state.linked); break;
      case 'arrowright':
        if (e.shiftKey) { e.preventDefault(); nudge(-1, 0); break; }
        e.preventDefault(); next(); break;
      case 'arrowleft':
        if (e.shiftKey) { e.preventDefault(); nudge(1, 0); break; }
        prev(); break;
      case 'arrowup':   if (e.shiftKey) { e.preventDefault(); nudge(0, 1); } break;
      case 'arrowdown': if (e.shiftKey) { e.preventDefault(); nudge(0, -1); } break;
      case ' ': e.preventDefault(); next(); break;
      case 'escape': setCropMode(false); setEraseMode(false); setNudgeMode(false); break;
      case '1': setView('overlay'); break;
      case '2': setView('fusion'); break;
      case '3': setView('checker'); break;
      case '4': setView('difference'); break;
    }
  });

  // Every layout change that resizes a pane - a toolbar opening, the window
  // resizing, the sidebar reflowing - arrives here. Anchoring on the observed
  // size is what makes it reliable; trying to predict which UI actions change
  // the layout is what produced the stale, shrunken draws.
  if (window.ResizeObserver) {
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) reanchor(e.target.dataset.pane);
      drawAll();
      drawBrushCursor();
    });
    CANVAS_NAMES.forEach((n) => {
      canvases[n].dataset.pane = n;
      ro.observe(canvases[n]);
    });
  } else {
    let resizeTimer = null;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(fitView, 100);
    });
  }

  api('/api/config')
    .then((c) => loadPair(c.last_seq || 0))
    .catch((e) => toast(e.message, 'bad'));

  refreshStatus();
  setInterval(refreshStatus, 1500);
}

init();
