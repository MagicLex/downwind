"""EEA air quality download client -- keyless, the ground-truth label source.

Two pieces the pipeline needs:
  1. hourly measurements per sampling point   (the download API, parquet)
  2. station coordinates + context            (the AQViewer metadata CSV)

The measurement parquet carries a `Samplingpoint` like `LU/SPO-LU0104A_00008_100`.
The station EoI code (`LU0104A`) is the token after `SPO-`, up to the first `_`,
and joins to the metadata `Air Quality Station EoI Code`. Coordinates are per
station (its sampling points share the location).
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile

import pandas as pd

API = "https://eeadmz1-downloads-api-appservice.azurewebsites.net"
METADATA_URL = ("https://discomap.eea.europa.eu/App/AQViewer/download"
                "?fqn=Airquality_Dissem.b2g.measurements&f=csv")

# EEA pollutant notation -> we resolve the vocabulary URI at runtime from /Pollutant
POLLUTANTS = {"no2": "NO2", "pm25": "PM2.5"}


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"accept": "*/*"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def _post(path: str, body: dict, timeout: int = 120) -> str:
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "accept": "*/*"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")


def pollutant_uris(notations: list[str]) -> dict[str, str]:
    """Resolve {'NO2': 'http://.../8', ...} from the live vocabulary."""
    pol = json.loads(_get(API + "/Pollutant", timeout=60))
    by_notation = {p["notation"]: p["id"] for p in pol}
    return {n: by_notation[n] for n in notations}


def countries() -> list[str]:
    return [c["countryCode"] for c in json.loads(_get(API + "/Country", timeout=60))]


def parquet_urls(country: str, pollutant_uri: str, dataset: int = 2) -> list[str]:
    """URLs of hourly measurement parquets. source='Api' and aggregationType='hour'
    are both required. Dataset ids: 1 = UTD (current year), 2 = Verified (2013->recent),
    3 = Historical (2004-2012)."""
    body = {"countries": [country], "cities": [], "pollutants": [pollutant_uri],
            "dataset": dataset, "source": "Api", "aggregationType": "hour", "compress": False}
    resp = _post("/ParquetFile/urls", body)
    return [u.strip() for u in resp.splitlines() if u.strip().startswith("http")]


def station_metadata() -> pd.DataFrame:
    """One row per station EoI: lat, lon, altitude, area, type."""
    raw = _get(METADATA_URL, timeout=300)
    z = zipfile.ZipFile(io.BytesIO(raw))
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
    keep = {
        "Air Quality Station EoI Code": "station_eoi",
        "Longitude": "lon", "Latitude": "lat", "Altitude": "altitude",
        "Air Quality Station Area": "station_area",
        "Air Quality Station Type": "station_type",
    }
    df = df[list(keep)].rename(columns=keep)
    for c in ("lon", "lat", "altitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["station_eoi", "lat", "lon"])
    return df.groupby("station_eoi", as_index=False).first()


def station_eoi_from_samplingpoint(sp: str) -> str | None:
    """`LU/SPO-LU0104A_00008_100` -> `LU0104A`. Format-agnostic on the EoI internals."""
    tail = sp.split("/")[-1]
    if not tail.startswith("SPO-"):
        return None
    return tail[4:].split("_")[0] or None


def read_measurements(url: str) -> pd.DataFrame:
    """Read one measurement parquet, keep valid hourly rows with a numeric value."""
    df = pd.read_parquet(url)
    df = df[df["Validity"] >= 1].copy()
    df["value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["station_eoi"] = df["Samplingpoint"].map(station_eoi_from_samplingpoint)
    return df[["Samplingpoint", "station_eoi", "Start", "End", "value", "Validity"]]
