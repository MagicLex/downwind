"""open-meteo archive client -- keyless. Two coarse signals per station-hour:

  weather (ERA5 archive)          the column-to-ground modulator
  CAMS air quality (reanalysis)   the ~10km modelled prior the model refines

One call per endpoint returns the full hourly series for a location, so a station
costs two calls over the whole window. Rate-limit aware (bounded backoff on 429).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pandas as pd

WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"
CAMS_API = "https://air-quality-api.open-meteo.com/v1/air-quality"

# open-meteo variable -> our feature name.
# boundary_layer_height is intentionally absent: open-meteo's archive returns it null
# for recent dates, which would be train/serve skew. CAMS already models the trapping.
WEATHER = {
    "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir",
    "temperature_2m": "temp", "relative_humidity_2m": "humidity",
    "precipitation": "precip", "surface_pressure": "pressure",
}
CAMS = {
    "pm2_5": "cams_pm25", "nitrogen_dioxide": "cams_no2", "ozone": "cams_o3",
    "sulphur_dioxide": "cams_so2", "carbon_monoxide": "cams_co",
    "dust": "cams_dust", "aerosol_optical_depth": "cams_aod",
}


def _get_json(url: str, tries: int = 5) -> dict:
    for attempt in range(tries):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=90).read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:   # rate limited, back off
                time.sleep(2 ** attempt * 5)
                continue
            raise
    raise RuntimeError("unreachable")


def _series(api: str, lat: float, lon: float, start: str, end: str, rename: dict) -> pd.DataFrame:
    params = (f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
              f"&hourly={','.join(rename)}")
    d = _get_json(api + params)
    h = d.get("hourly")
    if not h or not h.get("time"):
        return pd.DataFrame()
    out = pd.DataFrame({"valid_time": pd.to_datetime(h["time"], utc=True)})
    for src, dst in rename.items():
        # force float64: whole-number responses infer as bigint and clash with the
        # double FG schema (playbook dtype gotcha).
        out[dst] = pd.to_numeric(pd.Series(h.get(src)), errors="coerce").astype("float64")
    return out


def station_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Merged weather + CAMS hourly series for one station location."""
    w = _series(WEATHER_API, lat, lon, start, end, WEATHER)
    c = _series(CAMS_API, lat, lon, start, end, CAMS)
    if w.empty or c.empty:
        return pd.DataFrame()
    return w.merge(c, on="valid_time", how="inner")
