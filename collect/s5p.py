"""Sentinel-5P TROPOMI client -- CDSE OData search (public) + S3 download (keyed).

Two daily L2 products sampled at station locations:

  L2__NO2___  tropospheric NO2 column (mol/m2)   the raw satellite signal
  L2__AER_AI  UV aerosol index 354/388           a PM2.5 proxy

Search is the public OData catalogue (no auth). Download is the eodata S3 bucket
and needs the CDSE S3 key pair (free account; keys expire and need rotation --
current pair expires 2026-07-31). OFFL granules only: better calibration than
NRTI, available ~5 days behind real time, which matches the EEA label lag.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"

# product type -> (hdf5 variable under /PRODUCT, minimum qa_value)
PRODUCTS = {
    "L2__NO2___": ("nitrogendioxide_tropospheric_column", 0.75),
    "L2__AER_AI": ("aerosol_index_354_388", 0.50),
}
MAX_PIXEL_KM = 8.0  # nearest-pixel cutoff; TROPOMI pixel is ~5.5 km


def search_granules(product: str, bbox: tuple, day: str) -> list[dict]:
    """OFFL granules of `product` intersecting bbox (lon0, lat0, lon1, lat1) on `day`."""
    lon0, lat0, lon1, lat1 = bbox
    poly = (f"POLYGON(({lon0} {lat0},{lon1} {lat0},{lon1} {lat1},"
            f"{lon0} {lat1},{lon0} {lat0}))")
    flt = (f"Collection/Name eq 'SENTINEL-5P' and contains(Name,'{product}') "
           f"and contains(Name,'OFFL') "
           f"and OData.CSC.Intersects(area=geography'SRID=4326;{poly}') "
           f"and ContentDate/Start ge {day}T00:00:00.000Z "
           f"and ContentDate/Start lt {day}T23:59:59.999Z")
    url = f"{ODATA}?$filter={urllib.parse.quote(flt)}&$top=50"
    d = json.loads(urllib.request.urlopen(url, timeout=60).read())
    return [{"name": v["Name"], "s3_path": v["S3Path"]} for v in d.get("value", [])]


def _s3():
    import boto3
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["CDSE_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["CDSE_S3_SECRET_KEY"])


def download_granule(s3_path: str, dest_dir: str) -> str:
    """S3Path is '/eodata/Sentinel-5P/.../<name>' -- a directory holding one .nc file."""
    prefix = s3_path.removeprefix("/eodata/")
    s3 = _s3()
    keys = [o["Key"] for o in
            s3.list_objects_v2(Bucket="eodata", Prefix=prefix).get("Contents", [])
            if o["Key"].endswith(".nc")]
    if not keys:
        raise FileNotFoundError(f"no .nc under {s3_path}")
    local = os.path.join(dest_dir, os.path.basename(keys[0]))
    s3.download_file("eodata", keys[0], local)
    return local


def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def sample_granule(nc_path: str, product: str, stations: pd.DataFrame) -> pd.DataFrame:
    """Nearest qualifying pixel per station -> rows (station_eoi, value).

    stations: df with station_eoi, lat, lon. Stations outside the swath or with no
    pixel passing the qa threshold within MAX_PIXEL_KM are absent from the result.
    """
    var, qa_min = PRODUCTS[product]
    with h5py.File(nc_path, "r") as f:
        g = f["PRODUCT"]
        lat = g["latitude"][0].ravel()
        lon = g["longitude"][0].ravel()
        qa = g["qa_value"][0].ravel()
        ds = g[var]
        val = ds[0].ravel().astype("float64")
        fill = ds.attrs.get("_FillValue")
        if fill is not None:
            val = np.where(val == np.asarray(fill, dtype="float64"), np.nan, val)
        # qa_value is stored scaled (uint8 * 0.01) in some processor versions
        sf = g["qa_value"].attrs.get("scale_factor")
        if sf is not None and qa.dtype.kind in "iu":
            qa = qa * np.asarray(sf, dtype="float64")

    ok = np.isfinite(val) & (qa >= qa_min)
    if not ok.any():
        return pd.DataFrame()
    tree = cKDTree(_unit_vectors(lat[ok], lon[ok]))
    # chord length on the unit sphere for the km cutoff
    max_chord = 2.0 * np.sin(MAX_PIXEL_KM / 6371.0 / 2.0)
    dist, idx = tree.query(_unit_vectors(
        stations["lat"].to_numpy(), stations["lon"].to_numpy()), k=1)
    hit = dist <= max_chord
    if not hit.any():
        return pd.DataFrame()
    return pd.DataFrame({
        "station_eoi": stations["station_eoi"].to_numpy()[hit],
        "value": val[ok][idx[hit]],
    })
