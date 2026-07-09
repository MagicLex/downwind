"""downwind -- custom Hopsworks app (FastAPI, light canvas map).

Thin client of the FTI system:
- the FIELD: a grid of points over the covered countries, current-hour CAMS +
  weather fetched from open-meteo (batched), run through the registered
  gap-filling models -> a predicted ground PM2.5/NO2 surface denser than the
  sensor net. Refreshed hourly, cached in memory.
- the TRUTH: EEA stations with their latest validated readings from the label FG.
- click anywhere: same features, same models, one point -- with the CAMS prior,
  the nearest real stations and plain-word context. No skew: the exact feature
  columns the model trained on (model.joblib carries them).

Models load from the registry when training has registered them; until then the
app serves the raw CAMS prior and says so. No heavy reads per request: the store
is read in batch on a slow loop, predictions run on cached features.
"""

import asyncio
import glob
import json
import math
import os
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


def _find_root():
    cand = Path(__file__).resolve().parents[1]
    for p in [cand] + [Path(g) for g in sorted(glob.glob("/hopsfs/Users/*/downwind"))]:
        if (p / "downwind_features.py").exists():
            return p
    raise RuntimeError("repo root not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT))
STATIC = ROOT / "app" / "static"

from collect import eea, openmeteo  # noqa: E402

COUNTRIES = {"AL", "BA", "BE", "LU", "MT"}
POLLUTANTS = ("pm25", "no2")
CAMS_PRIOR = {"no2": "cams_no2", "pm25": "cams_pm25"}
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
AQ_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
GRID_STEP = 0.15          # deg; ~CAMS native resolution, keeps the grid honest
MAX_BOX_PTS = 420         # coarsen a big box instead of exploding the call count
HOTSPOT_KM = 25.0         # "unmonitored" = farther than this from any station

S = {"stations": [], "field": {}, "field_ts": 0, "models": {}, "labels_ts": 0,
     "status": "starting"}


# ---- open-meteo current conditions, batched ------------------------------------
def _om_key():
    return os.environ.get("OPENMETEO_API_KEY")


def _om_url(api, lats, lons, hourly):
    url = (f"{api}?latitude={','.join(f'{v:.4f}' for v in lats)}"
           f"&longitude={','.join(f'{v:.4f}' for v in lons)}"
           f"&hourly={','.join(hourly)}&past_days=1&forecast_days=1"
           f"&wind_speed_unit=kmh&timezone=UTC")
    key = _om_key()
    if key:  # Standard plan covers forecast + air-quality (archive is the excluded one)
        url = url.replace("https://", "https://customer-") + f"&apikey={key}"
    return url


def _fetch_json(url, tries=4):
    for attempt in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=60).read())
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt * 2)


WINDOW_H = 12  # hours each side of now in the animation window


def hourly_conditions(points):
    """points: [(lat, lon), ...] -> (times, frames) where times is the +/-WINDOW_H
    hour axis around now (UTC ISO strings) and frames[t] is a DataFrame with one
    row per point (lat, lon, altitude + weather + cams). Batched 100 locs/call;
    the hourly= series costs the same call count as current= but buys the timeline."""
    from concurrent.futures import ThreadPoolExecutor

    def fetch_chunk(chunk):
        lats, lons = [p[0] for p in chunk], [p[1] for p in chunk]
        wx = _fetch_json(_om_url(FORECAST_API, lats, lons, list(openmeteo.WEATHER)))
        aq = _fetch_json(_om_url(AQ_API, lats, lons, list(openmeteo.CAMS)))
        return (chunk, wx if isinstance(wx, list) else [wx],
                aq if isinstance(aq, list) else [aq])

    chunks = [points[i:i + 100] for i in range(0, len(points), 100)]
    per_point = []
    times = None
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_chunk, chunks))
    for chunk, wx, aq in results:
        for (lat, lon), w, a in zip(chunk, wx, aq):
            h_w, h_a = w.get("hourly") or {}, a.get("hourly") or {}
            if times is None and h_w.get("time"):
                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
                axis = h_w["time"]
                c = axis.index(now) if now in axis else len(axis) // 2
                lo, hi = max(0, c - WINDOW_H), min(len(axis), c + WINDOW_H + 1)
                times = (axis[lo:hi], lo, hi)
            rec = {"lat": lat, "lon": lon, "altitude": float(w.get("elevation") or 0.0)}
            for src, dst in openmeteo.WEATHER.items():
                rec[dst] = h_w.get(src)
            for src, dst in openmeteo.CAMS.items():
                rec[dst] = h_a.get(src)
            per_point.append(rec)
    if times is None:
        return [], []
    axis, lo, hi = times
    frames = []
    for k in range(lo, hi):
        rows = []
        for rec in per_point:
            r = {"lat": rec["lat"], "lon": rec["lon"], "altitude": rec["altitude"]}
            for dst in list(openmeteo.WEATHER.values()) + list(openmeteo.CAMS.values()):
                seq = rec[dst]
                r[dst] = seq[k] if isinstance(seq, list) and k < len(seq) else None
            rows.append(r)
        frames.append(pd.DataFrame(rows))
    return axis, frames


# ---- prediction (mirrors train.py feature assembly exactly) ---------------------
def predict(df, pol, t=None):
    """df: rows with lat, lon, altitude + weather + cams. Returns (pred, prior).
    Missing one-hot station columns fill 0 except rural background = the neutral
    'no local source' station the field semantics want."""
    prior = df[CAMS_PRIOR[pol]].astype(float).to_numpy()
    m = S["models"].get(pol)
    if not m:
        return prior, prior
    t = t or datetime.now(timezone.utc)
    X = df.copy()
    X["hour_sin"] = math.sin(2 * math.pi * t.hour / 24.0)
    X["hour_cos"] = math.cos(2 * math.pi * t.hour / 24.0)
    doy = t.timetuple().tm_yday
    X["doy_sin"] = math.sin(2 * math.pi * doy / 365.25)
    X["doy_cos"] = math.cos(2 * math.pi * doy / 365.25)
    X["is_weekend"] = 1.0 if t.weekday() >= 5 else 0.0
    X = X.reindex(columns=m["features"], fill_value=0)
    for neutral in ("station_area_rural", "station_type_background"):
        if neutral in X.columns:
            X[neutral] = 1
    pred = m["model"].predict(X)
    return np.maximum(pred, 0.0), prior


# ---- the field -------------------------------------------------------------------
EU_BBOX = (-10.0, 35.0, 30.0, 60.0)   # lon0, lat0, lon1, lat1 -- the whole theatre
EU_STEP = 0.35                          # deg; ~8k points, one call batch of ~160/refresh


def grid_points(_stations):
    """One continuous grid over Europe. The model's whole point is predicting
    where no sensor exists; CAMS + weather cover the continent, so the field
    does too. Station density only affects trust (dist_km), not coverage."""
    x0, y0, x1, y1 = EU_BBOX
    lats = np.arange(y0, y1 + 1e-9, EU_STEP)
    lons = np.arange(x0, x1 + 1e-9, EU_STEP)
    return [(round(float(la), 4), round(float(lo), 4)) for la in lats for lo in lons]


def _dist_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.0
    dx = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def nearest_stations(lat, lon, n=3):
    scored = sorted(S["stations"], key=lambda s: _dist_km(lat, lon, s["lat"], s["lon"]))
    return [{**s, "dist_km": round(_dist_km(lat, lon, s["lat"], s["lon"]), 1)}
            for s in scored[:n]]


def refresh_field():
    pts = grid_points(S["stations"])
    if not pts:
        return
    times, frames = hourly_conditions(pts)
    if not frames:
        return
    base = frames[0][["lat", "lon"]].reset_index(drop=True)
    plat, plon = base["lat"].to_numpy(), base["lon"].to_numpy()
    slat = np.array([s["lat"] for s in S["stations"]])
    slon = np.array([s["lon"] for s in S["stations"]])
    # nearest-station distance, vectorized (equirectangular is fine at this scale)
    dy = (plat[:, None] - slat[None, :]) * 111.0
    dx = (plon[:, None] - slon[None, :]) * 111.0 * np.cos(np.radians(plat))[:, None]
    dmin = np.sqrt(dx ** 2 + dy ** 2).min(axis=1) if len(slat) else np.full(len(plat), 9e9)
    points = [{"lat": round(float(la), 4), "lon": round(float(lo), 4),
               "dist_km": round(float(d), 1)}
              for la, lo, d in zip(plat, plon, dmin)]
    values = {p: [] for p in POLLUTANTS}
    for iso, df in zip(times, frames):
        t = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
        ok = df["cams_pm25"].notna() & df["cams_no2"].notna()
        for pol in POLLUTANTS:
            pred, _ = predict(df.where(ok), pol, t)
            values[pol].append([None if (v is None or not np.isfinite(v)) else round(float(v), 1)
                                for v in pred])
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
    S["field"] = {"times": times, "step": EU_STEP,
                  "now_idx": times.index(now_iso) if now_iso in times else len(times) // 2,
                  "points": points,
                  "pm25": values["pm25"], "no2": values["no2"]}
    S["field_ts"] = time.time()
    S["status"] = "model" if S["models"] else "cams-prior"
    print(f"field: {len(points)} points x {len(times)} frames, status={S['status']}")


def hotspots(pol, n=6):
    f = S["field"]
    if not f:
        return []
    vals = f[pol][f["now_idx"]]
    rows = [{**p, "value": v} for p, v in zip(f["points"], vals)
            if v is not None and p["dist_km"] > HOTSPOT_KM]
    return sorted(rows, key=lambda r: -r["value"])[:n]


# ---- slow loops -------------------------------------------------------------------
def load_stations():
    meta = eea.stations_for_pollutants(["NO2", "PM2.5"])
    meta = meta[meta["station_eoi"].str[:2].isin(COUNTRIES)]
    S["stations"] = [
        {"eoi": r["station_eoi"], "lat": round(float(r["lat"]), 4),
         "lon": round(float(r["lon"]), 4),
         "area": str(r.get("station_area") or ""), "type": str(r.get("station_type") or ""),
         "pm25": None, "no2": None, "obs_ts": None}
        for _, r in meta.iterrows()]
    print(f"{len(S['stations'])} stations loaded")


def refresh_labels():
    """Latest validated reading per station from the label FG (batch, bounded)."""
    import hopsworks
    fs = hopsworks.login().get_feature_store()
    fg = fs.get_feature_group("station_measurement", version=1)
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=21)).to_pydatetime()
    # .filter() = event-time pushdown; read(start_time=) is COMMIT-time travel
    # and returns 0 rows / FlightServerError here (scar 2026-07-09)
    df = fg.select(["station_eoi", "pollutant", "value", "start_time"]) \
           .filter(fg.start_time >= start).read(dataframe_type="pandas")
    if df.empty:
        return
    df = df.sort_values("start_time").groupby(["station_eoi", "pollutant"]).last()
    by = {s["eoi"]: s for s in S["stations"]}
    for (eoi, pol), r in df.iterrows():
        s = by.get(eoi)
        if s is not None and pol in POLLUTANTS:
            s[pol] = round(float(r["value"]), 1)
            s["obs_ts"] = str(r["start_time"])[:16]
    S["labels_ts"] = time.time()
    print(f"labels: latest readings for {df.index.get_level_values(0).nunique()} stations")


def load_models():
    import hopsworks
    import joblib
    mr = hopsworks.login().get_model_registry()
    for pol in POLLUTANTS:
        try:
            models = mr.get_models(f"air_quality_{pol}")
            if not models:
                continue
            best = max(models, key=lambda m: m.version)
            have = S["models"].get(pol, {}).get("version")
            if have == best.version:
                continue  # newer registered versions hot-swap in
            d = best.download()
            S["models"][pol] = {**joblib.load(f"{d}/model.joblib"),
                                "version": best.version,
                                "metrics": best.training_metrics or {}}
            print(f"model air_quality_{pol} v{best.version} loaded")
        except Exception as exc:
            print(f"model {pol}: not available yet ({str(exc)[:80]})")


async def _loop(fn, every, name):
    while True:
        delay = every
        try:
            await asyncio.to_thread(fn)
        except Exception as exc:
            print(f"{name} loop error: {str(exc)[:120]}")
            delay = min(every, 300)  # a failed pass retries soon, not next period
        await asyncio.sleep(delay)


# ---- app ---------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=2048)


@app.get("/", response_class=HTMLResponse)
def index():
    html = (STATIC / "index.html").read_text()
    pol = "pm25"
    hs = hotspots(pol)
    rows = "".join(
        f"<li>{h['value']:.0f} ug/m3 at {h['lat']:.2f}N {h['lon']:.2f}E, "
        f"{h['dist_km']:.0f} km from the nearest sensor</li>"
        for h in hs)
    init = {"stations": S["stations"], "field": S["field"],
            "status": S["status"], "field_age_s": int(time.time() - S["field_ts"]) if S["field_ts"] else None,
            "models": {p: {"version": m.get("version"), "metrics": m.get("metrics")}
                       for p, m in S["models"].items()}}
    html = html.replace("/*__INIT__*/null", json.dumps(init))
    html = html.replace("<!--__HOTSPOTS__-->", rows)
    return html


@app.get("/static/{name}")
def static(name: str):
    p = STATIC / Path(name).name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


@app.get("/api/state")
def state():
    return {"stations": S["stations"], "field": S["field"], "status": S["status"],
            "field_age_s": int(time.time() - S["field_ts"]) if S["field_ts"] else None,
            "hotspots": {p: hotspots(p) for p in POLLUTANTS},
            "models": {p: {"version": m.get("version"), "metrics": m.get("metrics")}
                       for p, m in S["models"].items()}}


@app.get("/api/point")
def point(lat: float, lon: float):
    if not (34.0 <= lat <= 55.0 and -2.0 <= lon <= 24.0):
        return JSONResponse({"error": "out of theatre"}, status_code=400)
    times, frames = hourly_conditions([(lat, lon)])
    if not frames:
        return JSONResponse({"error": "no data here"}, status_code=502)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
    df = frames[times.index(now_iso) if now_iso in times else len(times) // 2]
    if df.empty or df["cams_pm25"].isna().all():
        return JSONResponse({"error": "no CAMS data here"}, status_code=502)
    near = nearest_stations(lat, lon)
    out = {"lat": lat, "lon": lon, "status": S["status"], "nearest": near,
           "weather": {k: (None if pd.isna(v) else round(float(v), 1))
                       for k, v in df.iloc[0][list(openmeteo.WEATHER.values())].items()}}
    for pol in POLLUTANTS:
        pred, prior = predict(df, pol)
        out[pol] = {"value": round(float(pred[0]), 1), "cams_prior": round(float(prior[0]), 1)}
    # plain words: what the reader needs to trust or distrust the number
    reasons = []
    if S["models"]:
        for pol in POLLUTANTS:
            dv = out[pol]["value"] - out[pol]["cams_prior"]
            if abs(dv) >= 1:
                reasons.append(f"model adjusts CAMS {pol} {'up' if dv > 0 else 'down'} "
                               f"by {abs(dv):.0f} ug/m3 from local weather + context")
    else:
        reasons.append("raw CAMS prior (model not registered yet)")
    if near:
        n0 = near[0]
        reasons.append(f"nearest sensor {n0['eoi']} is {n0['dist_km']:.0f} km away")
        if n0["dist_km"] > HOTSPOT_KM:
            reasons.append("this point is effectively unmonitored")
    out["reasons"] = reasons
    out["links"] = {
        "eea": "https://airindex.eea.europa.eu/",
        "cams": "https://atmosphere.copernicus.eu/european-air-quality-index",
    }
    return out


@app.get("/health")
def health():
    return {"ok": True, "stations": len(S["stations"]), "field": len(S["field"]),
            "status": S["status"]}


async def _lifespan(_):
    def boot():
        try:
            if "OPENMETEO_API_KEY" not in os.environ:
                import hopsworks
                hopsworks.login()  # secrets api needs an active connection
                os.environ["OPENMETEO_API_KEY"] = \
                    hopsworks.get_secrets_api().get_secret("OPENMETEO_API_KEY").value
                print("open-meteo key loaded from secrets")
        except Exception as exc:
            print(f"open-meteo key unavailable, keyless mode: {str(exc)[:80]}")
        load_stations()
        for step in (load_models, refresh_field):
            try:  # degraded boot beats no boot; the loops retry
                step()
            except Exception as exc:
                print(f"boot {step.__name__}: {str(exc)[:100]}")
    await asyncio.to_thread(boot)
    tasks = [asyncio.create_task(_loop(refresh_field, 3600, "field")),
             asyncio.create_task(_loop(load_models, 600, "models")),
             asyncio.create_task(_loop(refresh_labels, 21600, "labels"))]
    yield
    for t in tasks:
        t.cancel()


# the proxy forwards the FULL mount path into the container (ghost-fleet scar):
# mount the app at APP_BASE_URL_PATH so prefixed requests route; browser links
# stay relative so they resolve under the public mount either way.
BASE = os.environ.get("APP_BASE_URL_PATH", "").rstrip("/")
asgi = FastAPI(lifespan=asynccontextmanager(_lifespan))
asgi.mount(BASE or "/", app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi, host="0.0.0.0", port=int(os.environ.get("APP_PORT", 8000)))
