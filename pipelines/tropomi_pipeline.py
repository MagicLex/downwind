"""F1 -- Sentinel-5P satellite feature pipeline.

TROPOMI NO2 tropospheric column + UV aerosol index, sampled daily at EEA station
locations -> FG `tropomi_column` (station_eoi + obs_date). The raw satellite signal
that CAMS only assimilates; the v1.5 SOTA leg on top of the CAMS-based baseline.

Granules are searched per station-cluster bounding box (public OData), downloaded
once each from the eodata S3 bucket (CDSE key pair, read from Hopsworks secrets or
env), sampled with a nearest-qualifying-pixel lookup, and deleted. Disk use is one
granule at a time (~450 MB), memory is one day's rows.

Config via argv (`--key value`) or env (KEY); argv wins. Keys:
  countries    csv of country codes, or ALL           (default ALL)
  start        first date, YYYY-MM-DD                  (default 2026-01-01)
  end          last date, YYYY-MM-DD                   (default today - 6d, OFFL lag)
  resume       1 = skip dates already in the FG        (default 1)
  dry          1 = one day, print, no FG write         (default 0)
"""

import glob
import os
import re
import sys
import tempfile
from datetime import date, timedelta

import pandas as pd

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _root in [_here] + sorted(glob.glob("/hopsfs/Users/*/downwind")):
    if os.path.exists(os.path.join(_root, "downwind_features.py")):
        sys.path.insert(0, _root)
        break
from collect import eea, s5p  # noqa: E402


def config():
    argv = {}
    a = sys.argv[1:]
    i = 0
    while i < len(a):
        if a[i].startswith("--") and i + 1 < len(a):
            argv[a[i][2:].replace("-", "_")] = a[i + 1]
            i += 2
        else:
            i += 1

    def get(k, d):
        return argv.get(k, os.environ.get(k.upper(), d))

    return {
        "countries": get("countries", "ALL"),
        "start": get("start", "2026-01-01"),
        "end": get("end", str(date.today() - timedelta(days=6))),
        "resume": get("resume", "1") == "1",
        "dry": get("dry", "0") == "1",
    }


def get_fg(fs):
    from hsfs.feature import Feature
    feats = [
        Feature("station_eoi", "string", description="EEA station EoI code"),
        Feature("obs_date", "timestamp", description="Overpass day (event time)"),
        Feature("lat", "double", description="Station latitude"),
        Feature("lon", "double", description="Station longitude"),
        Feature("no2_column", "double",
                description="TROPOMI tropospheric NO2 column, mol/m2, qa>=0.75, "
                            "nearest pixel within 8 km, daily mean over overpasses"),
        Feature("aerosol_index", "double",
                description="TROPOMI UV aerosol index 354/388 (PM2.5 proxy), "
                            "qa>=0.5, nearest pixel within 8 km, daily mean"),
    ]
    return fs.get_or_create_feature_group(
        name="tropomi_column", version=1,
        description="Daily Sentinel-5P TROPOMI NO2 column + aerosol index sampled at "
                    "EEA station locations. The raw satellite signal the model "
                    "downscales; CAMS assimilates this, F1 carries it raw.",
        primary_key=["station_eoi"], event_time="obs_date",
        online_enabled=False, statistics_config=False, features=feats)


def load_secrets():
    """CDSE S3 keys: env wins (local runs), Hopsworks secrets otherwise (jobs)."""
    if "CDSE_S3_ACCESS_KEY" in os.environ:
        return
    import hopsworks
    api = hopsworks.get_secrets_api()
    os.environ["CDSE_S3_ACCESS_KEY"] = api.get_secret("CDSE_S3_ACCESS_KEY").value
    os.environ["CDSE_S3_SECRET_KEY"] = api.get_secret("CDSE_S3_SECRET_KEY").value


def station_boxes(stations: pd.DataFrame, cell: float = 5.0, margin: float = 0.5):
    """Coarse bounding boxes around the station clusters (BE/LU, AL/BA, MT...):
    snap to `cell`-deg tiles, one box per occupied tile. Keeps the OData polygon
    small instead of one Europe-wide box that matches every orbit."""
    boxes = {}
    for _, r in stations.iterrows():
        key = (int(r["lon"] // cell), int(r["lat"] // cell))
        b = boxes.get(key)
        if b is None:
            boxes[key] = [r["lon"], r["lat"], r["lon"], r["lat"]]
        else:
            b[0], b[1] = min(b[0], r["lon"]), min(b[1], r["lat"])
            b[2], b[3] = max(b[2], r["lon"]), max(b[3], r["lat"])
    return [(b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin)
            for b in boxes.values()]


def done_dates(fg) -> set:
    try:
        s = fg.select(["obs_date"]).read(dataframe_type="pandas")
        return set(pd.to_datetime(s["obs_date"]).dt.date)
    except Exception:
        return set()


def sample_day(day: str, boxes, stations, workdir) -> pd.DataFrame:
    """All products, all boxes, granules deduped by name -> one row per station
    that had at least one qualifying overpass, values averaged over overpasses."""
    per_product = {}
    for product in s5p.PRODUCTS:
        granules = {}
        for box in boxes:
            for g in s5p.search_granules(product, box, day):
                # daytime passes only: over Europe S5P crosses ~09-14 UTC; night
                # granules fail qa everywhere (no UV) and waste a 600 MB download
                m = re.search(r"(\d{8}T\d{2})", g["name"])
                if m and 8 <= int(m.group(1)[-2:]) <= 14:
                    granules[g["name"]] = g
        vals = []
        for g in granules.values():
            try:
                nc = s5p.download_granule(g["s3_path"], workdir)
            except Exception as exc:
                print(f"  ! {g['name']}: download failed: {exc}")
                continue
            try:
                df = s5p.sample_granule(nc, product, stations)
            finally:
                os.remove(nc)
            if not df.empty:
                vals.append(df)
        if vals:
            per_product[product] = (pd.concat(vals)
                                    .groupby("station_eoi")["value"].mean())
    if not per_product:
        return pd.DataFrame()
    out = stations[["station_eoi", "lat", "lon"]].copy()
    out["obs_date"] = pd.Timestamp(day, tz="UTC")
    out["no2_column"] = out["station_eoi"].map(per_product.get("L2__NO2___", {}))
    out["aerosol_index"] = out["station_eoi"].map(per_product.get("L2__AER_AI", {}))
    out = out.dropna(subset=["no2_column", "aerosol_index"], how="all")
    return out[["station_eoi", "obs_date", "lat", "lon", "no2_column", "aerosol_index"]]


def main():
    cfg = config()
    print("config:", cfg)
    stations = eea.stations_for_pollutants(["NO2", "PM2.5"])
    if cfg["countries"] != "ALL":
        codes = {c.strip() for c in cfg["countries"].split(",")}
        stations = stations[stations["station_eoi"].str[:2].isin(codes)]
    stations = stations.sort_values("station_eoi").reset_index(drop=True)
    boxes = station_boxes(stations)
    print(f"{len(stations)} stations in {len(boxes)} search boxes")

    fg = done = None
    if not cfg["dry"]:
        import hopsworks
        hopsworks.login()
        load_secrets()
        fg = get_fg(hopsworks.login().get_feature_store())
        done = done_dates(fg) if cfg["resume"] else set()
        if done:
            print(f"resume: {len(done)} dates already present")

    days = pd.date_range(cfg["start"], cfg["end"], freq="D").date
    days = [d for d in days if not (done and d in done)]
    if cfg["dry"]:
        days = days[:1]
    print(f"{len(days)} days to sample")

    total, buf = 0, []
    with tempfile.TemporaryDirectory() as workdir:
        for d in days:
            df = sample_day(str(d), boxes, stations, workdir)
            print(f"  {d}: {len(df)} station rows")
            if df.empty:
                continue
            if cfg["dry"]:
                print(df.head(6).to_string())
                continue
            buf.append(df)
            total += len(df)
            if sum(len(b) for b in buf) >= 50_000:
                fg.insert(pd.concat(buf, ignore_index=True), wait=True)
                buf = []
    if buf:
        fg.insert(pd.concat(buf, ignore_index=True), wait=True)
    print(f"DONE: {total} rows written to tropomi_column")


if __name__ == "__main__":
    main()
