"use strict";
// downwind map -- satellite canvas (ghost-fleet projection core) under a light,
// airy skin. Two color codes, same direction (darker = worse air):
//   predicted field  = warm ramp (pale yellow -> deep red), pixelated raster
//   measured station = cool ramp (pale teal -> deep indigo), solid dots
const api = (p) => fetch(p).then(r => r.json());
const cv = document.getElementById("map"), cx = cv.getContext("2d");

const FRAME = { lon0: 1.5, lon1: 22.5, lat0: 34.5, lat1: 53.5 };
let cam = { cx: (FRAME.lon0 + FRAME.lon1) / 2, cy: (FRAME.lat0 + FRAME.lat1) / 2, scale: 1 };
let DPR = 1, W = 0, H = 0;
const S = { stations: [], field: null, pol: "pm25", tIdx: null, playing: false,
            sel: null, status: "starting" };
const VMAX = { pm25: 50, no2: 80 };   // ramp ceiling per pollutant (ug/m3)

// ---- projection (Web Mercator, tiles align pixel-perfect) ----
const MER = 180 / Math.PI;
const mercY = lat => MER * Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
const imercY = y => (2 * Math.atan(Math.exp(y / MER)) - Math.PI / 2) * 180 / Math.PI;
function proj(lon, lat) {
  const k = cam.scale;
  return { x: W / 2 + (lon - cam.cx) * k, y: H / 2 - (mercY(lat) - mercY(cam.cy)) * k };
}
function unproj(x, y) {
  const k = cam.scale;
  return { lon: cam.cx + (x - W / 2) / k, lat: imercY(mercY(cam.cy) - (y - H / 2) / k) };
}
function resize() {
  DPR = Math.min(2, window.devicePixelRatio || 1);
  W = cv.clientWidth || window.innerWidth;
  H = cv.clientHeight || window.innerHeight;
  if (!W || !H) { requestAnimationFrame(resize); return; }
  cv.width = W * DPR; cv.height = H * DPR;
  const sx = W / (FRAME.lon1 - FRAME.lon0);
  const sy = H / (mercY(FRAME.lat1) - mercY(FRAME.lat0));
  if (!resize.done) { cam.scale = Math.min(sx, sy) * 0.96; resize.done = true; }
  draw();
}

// ---- color ramps ----
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const a = stops[i - 1], b = stops[i], k = (t - a[0]) / (b[0] - a[0]);
      return a[1].map((v, j) => Math.round(v + (b[1][j] - v) * k));
    }
  }
  return stops[stops.length - 1][1];
}
const PRED = [[0, [254, 249, 195]], [0.3, [251, 191, 36]], [0.55, [249, 115, 22]],
              [0.8, [220, 38, 38]], [1, [127, 29, 29]]];
const MEAS = [[0, [204, 251, 241]], [0.3, [45, 212, 191]], [0.55, [8, 145, 178]],
              [0.8, [30, 64, 175]], [1, [30, 27, 75]]];

// ---- satellite tiles (Esri World Imagery, free) under a bright airy wash ----
const TILES = new Map();
function getTile(z, x, y) {
  const key = z + "/" + x + "/" + y;
  let t = TILES.get(key);
  if (t) return t;
  if (TILES.size > 400) TILES.delete(TILES.keys().next().value);
  t = { img: new Image(), ok: false };
  t.img.onload = () => { t.ok = true; draw(); };
  t.img.src = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`;
  TILES.set(key, t);
  return t;
}
function drawTiles() {
  const k = cam.scale, myc = mercY(cam.cy);
  let z = Math.max(3, Math.min(12, Math.round(Math.log2(k * 360 / 256))));
  let n = 1 << z;
  const lonW = W / k, myH = H / k;
  while (z > 3 && (lonW / (360 / n)) * (myH / (360 / n)) > 200) { z--; n = 1 << z; }
  const tx0 = Math.floor((cam.cx - lonW / 2 + 180) / 360 * n);
  const tx1 = Math.floor((cam.cx + lonW / 2 + 180) / 360 * n);
  const ty0 = Math.max(0, Math.floor((1 - (myc + myH / 2) / 180) / 2 * n));
  const ty1 = Math.min(n - 1, Math.floor((1 - (myc - myH / 2) / 180) / 2 * n));
  let drew = 0;
  for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) {
    const t = getTile(z, ((tx % n) + n) % n, ty);
    if (!t.ok) continue;
    const lonA = tx / n * 360 - 180, lonB = (tx + 1) / n * 360 - 180;
    const myA = 180 * (1 - 2 * ty / n), myB = 180 * (1 - 2 * (ty + 1) / n);
    const x0 = W / 2 + (lonA - cam.cx) * k, x1 = W / 2 + (lonB - cam.cx) * k;
    const y0 = H / 2 - (myA - myc) * k, y1 = H / 2 - (myB - myc) * k;
    cx.drawImage(t.img, x0, y0, x1 - x0 + 0.6, y1 - y0 + 0.6);
    drew++;
  }
  // light haze wash: keeps satellite texture but lifts it to the airy palette
  if (drew) { cx.fillStyle = "rgba(240,249,255,.42)"; cx.fillRect(0, 0, W, H); }
  return drew > 0;
}

// ---- the predicted field: crisp pixel raster, one cell per grid point ----
let rasterBuf = null;
function drawField() {
  const f = S.field;
  if (!f || !f.points || S.tIdx == null) return;
  const vals = f[S.pol][S.tIdx];
  if (!vals) return;
  const SC = 4;
  const gw = Math.max(1, Math.ceil(W / SC)), gh = Math.max(1, Math.ceil(H / SC));
  const acc = new Float32Array(gw * gh), wsum = new Float32Array(gw * gh);
  // grid spacing in raster cells decides the splat radius: cells overlap just
  // enough to read as a continuous pixelated field, not confetti
  const stepPx = Math.max(2, 0.15 * cam.scale / SC);
  const rad = Math.max(2, stepPx * 0.85);
  const vmax = VMAX[S.pol];
  for (let i = 0; i < f.points.length; i++) {
    const v = vals[i];
    if (v == null) continue;
    const p = f.points[i], q = proj(p.lon, p.lat);
    const gx = q.x / SC, gy = q.y / SC;
    if (gx < -rad || gx > gw + rad || gy < -rad || gy > gh + rad) continue;
    const x0 = Math.max(0, (gx - rad) | 0), x1 = Math.min(gw - 1, (gx + rad) | 0);
    const y0 = Math.max(0, (gy - rad) | 0), y1 = Math.min(gh - 1, (gy + rad) | 0);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const dx = x - gx, dy = y - gy, d2 = dx * dx + dy * dy;
      if (d2 > rad * rad) continue;
      const w = Math.exp(-d2 / (rad * rad * 0.45));
      acc[y * gw + x] += v * w; wsum[y * gw + x] += w;
    }
  }
  if (!rasterBuf) rasterBuf = document.createElement("canvas");
  rasterBuf.width = gw; rasterBuf.height = gh;
  const rcx = rasterBuf.getContext("2d"), img = rcx.createImageData(gw, gh), D = img.data;
  for (let i = 0; i < acc.length; i++) {
    if (wsum[i] < 0.05) { D[i * 4 + 3] = 0; continue; }
    const v = acc[i] / wsum[i], t = v / vmax;
    const c = ramp(PRED, t);
    D[i * 4] = c[0]; D[i * 4 + 1] = c[1]; D[i * 4 + 2] = c[2];
    D[i * 4 + 3] = Math.round(120 + 110 * Math.min(1, t));
  }
  rcx.putImageData(img, 0, 0);
  cx.imageSmoothingEnabled = false;              // pixelated on purpose
  cx.drawImage(rasterBuf, 0, 0, gw, gh, 0, 0, gw * SC, gh * SC);
  cx.imageSmoothingEnabled = true;
}

// ---- measured stations: cool-ramp dots with white halo ----
function drawStations() {
  const vmax = VMAX[S.pol];
  for (const s of S.stations) {
    const q = proj(s.lon, s.lat);
    if (q.x < -20 || q.x > W + 20 || q.y < -20 || q.y > H + 20) continue;
    const v = s[S.pol];
    const has = v != null;
    cx.beginPath(); cx.arc(q.x, q.y, has ? 5 : 3, 0, 7);
    cx.fillStyle = "rgba(255,255,255,.95)"; cx.fill();
    cx.beginPath(); cx.arc(q.x, q.y, has ? 3.6 : 1.8, 0, 7);
    cx.fillStyle = has ? `rgb(${ramp(MEAS, v / vmax).join(",")})` : "#94a3b8";
    cx.fill();
    if (s === S.sel) {
      cx.strokeStyle = "#0ea5e9"; cx.lineWidth = 2;
      cx.beginPath(); cx.arc(q.x, q.y, 9, 0, 7); cx.stroke();
    }
  }
}

let clickMark = null;
function drawClickMark() {
  if (!clickMark) return;
  const q = proj(clickMark.lon, clickMark.lat);
  cx.strokeStyle = "#0ea5e9"; cx.lineWidth = 2;
  cx.beginPath(); cx.arc(q.x, q.y, 10, 0, 7); cx.stroke();
  cx.beginPath(); cx.moveTo(q.x - 14, q.y); cx.lineTo(q.x - 5, q.y);
  cx.moveTo(q.x + 5, q.y); cx.lineTo(q.x + 14, q.y);
  cx.moveTo(q.x, q.y - 14); cx.lineTo(q.x, q.y - 5);
  cx.moveTo(q.x, q.y + 5); cx.lineTo(q.x, q.y + 14); cx.stroke();
}

function draw() {
  cx.save(); cx.scale(DPR, DPR);
  cx.fillStyle = "#dbeafe"; cx.fillRect(0, 0, W, H);
  drawTiles();
  drawField();
  drawStations();
  drawClickMark();
  cx.restore();
}

// ---- player ----
const slider = document.getElementById("tslider"), playBtn = document.getElementById("play");
function setFrame(i, fromSlider) {
  const f = S.field; if (!f || !f.times) return;
  S.tIdx = Math.max(0, Math.min(f.times.length - 1, i));
  if (!fromSlider) slider.value = S.tIdx;
  const iso = f.times[S.tIdx], isNow = S.tIdx === f.now_idx;
  const rel = S.tIdx - f.now_idx;
  document.getElementById("tlabel").innerHTML = isNow ? "<b>now</b>" :
    `<b>${iso.slice(11, 16)}Z</b> (${rel > 0 ? "+" : ""}${rel}h)`;
  draw();
}
let playTimer = null;
playBtn.onclick = () => {
  S.playing = !S.playing;
  playBtn.innerHTML = S.playing ? "&#10074;&#10074;" : "&#9654;";
  clearInterval(playTimer);
  if (S.playing) playTimer = setInterval(() => {
    const f = S.field; if (!f) return;
    setFrame(S.tIdx + 1 > f.times.length - 1 ? 0 : S.tIdx + 1);
  }, 450);
};
slider.oninput = () => setFrame(+slider.value, true);

// ---- pollutant toggle + legend ----
document.querySelectorAll(".toggle button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".toggle button").forEach(x => x.classList.remove("on"));
  b.classList.add("on");
  S.pol = b.getAttribute("data-p");
  document.getElementById("leg-title").textContent =
    (S.pol === "pm25" ? "PM2.5" : "NO2") + ", ug/m3";
  document.getElementById("leg-max-p").textContent = VMAX[S.pol] + "+";
  document.getElementById("leg-max-m").textContent = VMAX[S.pol] + "+";
  renderRail(); draw();
});

// ---- attention rail ----
function renderRail() {
  const el = document.getElementById("spots");
  const hs = (S.hotspots && S.hotspots[S.pol]) || [];
  el.innerHTML = hs.map(h =>
    `<div class="spot" data-lat="${h.lat}" data-lon="${h.lon}">
       <span class="v" style="color:rgb(${ramp(PRED, h.value / VMAX[S.pol]).join(",")})">${h.value.toFixed(0)}</span>
       <span style="font-size:11px;color:#64748b">ug/m3 predicted</span>
       <div class="w">${h.lat.toFixed(2)}N ${h.lon.toFixed(2)}E &middot; ${h.dist_km.toFixed(0)} km from any sensor</div>
     </div>`).join("") ||
    `<div class="spot" style="cursor:default"><div class="w">hotspots surface once the field is computed&hellip;</div></div>`;
  el.querySelectorAll(".spot[data-lat]").forEach(c => c.onclick = () => {
    const lat = +c.getAttribute("data-lat"), lon = +c.getAttribute("data-lon");
    panTo(lon, lat); inspect(lat, lon);
  });
}

let panAnim = null;
function panTo(lon, lat) {
  const x0 = cam.cx, y0 = cam.cy, t0 = performance.now(), dur = 420;
  cancelAnimationFrame(panAnim);
  (function step(t) {
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    cam.cx = x0 + (lon - x0) * e; cam.cy = y0 + (lat - y0) * e; draw();
    if (k < 1) panAnim = requestAnimationFrame(step);
  })(t0);
}

// ---- click-anywhere inspection ----
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function inspect(lat, lon) {
  clickMark = { lat, lon }; draw();
  const el = document.getElementById("card");
  el.innerHTML = `<div class="x">&times;</div><h3>reading the air at ${lat.toFixed(3)}N ${lon.toFixed(3)}E&hellip;</h3>`;
  el.classList.add("show");
  el.querySelector(".x").onclick = closeCard;
  api(`api/point?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`).then(d => {
    if (d.error) { el.innerHTML = `<div class="x">&times;</div><h3>${esc(d.error)}</h3>`; el.querySelector(".x").onclick = closeCard; return; }
    const big = ["pm25", "no2"].map(p => {
      const c = ramp(PRED, d[p].value / VMAX[p]);
      return `<div><div class="v" style="color:rgb(${c.join(",")})">${d[p].value.toFixed(0)}</div>
        <div class="u">${p === "pm25" ? "PM2.5" : "NO2"} ug/m3<br>CAMS prior ${d[p].cams_prior.toFixed(0)}</div></div>`;
    }).join("");
    const n0 = d.nearest && d.nearest[0];
    el.innerHTML = `<div class="x">&times;</div>
      <h3>${lat.toFixed(3)}N ${lon.toFixed(3)}E ${d.status === "model" ? "&middot; model estimate" : "&middot; CAMS prior"}</h3>
      <div class="big">${big}</div>
      <ul>${(d.reasons || []).map(r => `<li>${esc(r)}</li>`).join("")}</ul>
      ${n0 ? `<div class="near">nearest sensor <b>${esc(n0.eoi)}</b> (${n0.dist_km} km): ` +
        `PM2.5 ${n0.pm25 ?? "&ndash;"}, NO2 ${n0.no2 ?? "&ndash;"}${n0.obs_ts ? " at " + esc(n0.obs_ts) + "Z" : ""}</div>` : ""}
      <div style="margin-top:10px"><a href="${d.links.eea}" target="_blank">EEA index</a>
      <a href="${d.links.cams}" target="_blank">Copernicus CAMS</a></div>`;
    el.querySelector(".x").onclick = closeCard;
  }).catch(() => {});
}
function closeCard() {
  document.getElementById("card").classList.remove("show");
  clickMark = null; S.sel = null; draw();
}

function nearestStation(px, py, tol) {
  let best = null, bd = tol;
  for (const s of S.stations) {
    const q = proj(s.lon, s.lat);
    const d = Math.hypot(q.x - px, q.y - py);
    if (d < bd) { bd = d; best = s; }
  }
  return best;
}

// ---- interaction: drag pan, wheel zoom, click inspect ----
let drag = null;
const dragDist = e => Math.hypot(e.clientX - drag.x, e.clientY - drag.y);
cv.addEventListener("pointerdown", e => {
  drag = { x: e.clientX, y: e.clientY, cx: cam.cx, my: mercY(cam.cy), moved: 0 };
});
cv.addEventListener("pointermove", e => {
  if (!drag) {
    cv.style.cursor = nearestStation(e.clientX, e.clientY, 12) ? "pointer" : "grab";
    return;
  }
  drag.moved = Math.max(drag.moved, dragDist(e));
  cam.cx = drag.cx - (e.clientX - drag.x) / cam.scale;
  cam.cy = imercY(drag.my + (e.clientY - drag.y) / cam.scale);
  draw();
});
cv.addEventListener("pointerup", e => {
  if (drag && drag.moved < 5) {
    const st = nearestStation(e.clientX, e.clientY, 12);
    if (st) { S.sel = st; inspect(st.lat, st.lon); }
    else { const g = unproj(e.clientX, e.clientY); inspect(g.lat, g.lon); }
  }
  drag = null;
});
cv.addEventListener("wheel", e => {
  e.preventDefault();
  const before = unproj(e.clientX, e.clientY);
  cam.scale *= Math.exp(-e.deltaY * 0.0016);
  const after = unproj(e.clientX, e.clientY);
  cam.cx += before.lon - after.lon;
  cam.cy = Math.max(-80, Math.min(80, cam.cy + before.lat - after.lat));
  draw();
}, { passive: false });

// ---- state ingest ----
function ingest(d) {
  S.stations = d.stations || S.stations;
  S.field = d.field && d.field.times ? d.field : S.field;
  S.hotspots = d.hotspots || S.hotspots;
  S.status = d.status;
  document.getElementById("s-stations").textContent = S.stations.length;
  const chip = document.getElementById("chip-status");
  if (d.status === "model") {
    const v = d.models && d.models.pm25 ? " v" + d.models.pm25.version : "";
    chip.textContent = "model live" + v; chip.className = "chip live"; chip.id = "chip-status";
    chip.classList.add("live");
  } else if (d.status === "cams-prior") {
    chip.textContent = "CAMS prior (model training)"; chip.classList.add("prior");
  } else chip.textContent = d.status;
  if (S.field && S.field.times) {
    slider.max = S.field.times.length - 1;
    if (S.tIdx == null) setFrame(S.field.now_idx);
  }
  renderRail(); draw();
}

async function tick() {
  try { ingest(await api("api/state")); } catch (e) { /* transient */ }
}

function clockTick() {
  const d = new Date(), p = n => String(n).padStart(2, "0");
  document.getElementById("s-clock").textContent =
    p(d.getUTCHours()) + ":" + p(d.getUTCMinutes()) + "Z";
}
setInterval(clockTick, 30000); clockTick();

window.addEventListener("resize", resize);
resize();
if (window.__INIT__) ingest(window.__INIT__);
tick();
setInterval(tick, 120000);
