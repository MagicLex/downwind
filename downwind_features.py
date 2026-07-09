"""downwind shared feature extractor -- the ONE skew-free module.

Given a point (lat, lon, time) and the source values already looked up for that
point (satellite column, weather, static land context), assemble the exact feature
vector the model trains on and serves on. The training-pair builder, the batch map
builder and the KServe predictor all import `assemble` so there is one and only one
definition of a feature. Never duplicate this logic.

The source lookups (which granule, which weather cell, which context row) live in the
pipelines and the predictor and pass their results in as plain dicts. This module owns
the contract and the pure transforms, not the I/O.
"""

from __future__ import annotations

import math
from datetime import datetime

# --- the feature contract -------------------------------------------------------
# Order is not load-bearing (the model reads by name), but this is the full set.
SATELLITE = ["no2_column", "aerosol_index"]
WEATHER = ["wind_speed", "wind_dir_sin", "wind_dir_cos", "blh", "temp",
           "humidity", "precip", "pressure"]
CONTEXT = ["landcover_class", "road_density", "population", "elevation",
           "dist_point_source", "cams_pm25", "cams_no2"]
TEMPORAL = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_weekend"]
GEO = ["lat", "lon"]

FEATURE_NAMES = GEO + SATELLITE + WEATHER + CONTEXT + TEMPORAL


# --- pure transforms ------------------------------------------------------------
def temporal_features(ts: datetime) -> dict:
    """Cyclic time-of-day and time-of-year plus a weekend flag."""
    hour = ts.hour + ts.minute / 60.0
    doy = ts.timetuple().tm_yday
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "is_weekend": 1.0 if ts.weekday() >= 5 else 0.0,
    }


def wind_components(speed: float, direction_deg: float) -> dict:
    """Split a wind vector so the model sees direction without a discontinuity at 360."""
    rad = math.radians(direction_deg)
    return {
        "wind_speed": speed,
        "wind_dir_sin": math.sin(rad),
        "wind_dir_cos": math.cos(rad),
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance, used for distance-to-nearest-station and point sources."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def assemble(lat: float, lon: float, ts: datetime,
             satellite: dict, weather: dict, context: dict) -> dict:
    """Compose the model feature vector for one (lat, lon, ts).

    satellite: {no2_column, aerosol_index}                  from tropomi_column
    weather:   {wind_speed, wind_dir, blh, temp, humidity, precip, pressure}  from weather_cell
    context:   {landcover_class, road_density, population, elevation,
                dist_point_source, cams_pm25, cams_no2}      from cell_context
    """
    feat = {"lat": lat, "lon": lon}
    feat.update({k: satellite.get(k) for k in SATELLITE})
    feat.update(wind_components(weather["wind_speed"], weather["wind_dir"]))
    for k in ("blh", "temp", "humidity", "precip", "pressure"):
        feat[k] = weather.get(k)
    feat.update({k: context.get(k) for k in CONTEXT})
    feat.update(temporal_features(ts))
    return feat
