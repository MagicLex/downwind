"""F3 -- weather + CAMS feature pipeline.

open-meteo ERA5 weather + CAMS air quality, hourly, per station -> FG
`station_features`. The coarse modulator (wind, boundary layer, precipitation) and
the ~10km modelled prior (CAMS pm2.5/no2/...) that the model refines with local
context into a station reading. Keyed by (station_eoi, valid_time); F5 joins it to
the station labels.

Independent of the F2 backfill: the station set + coords come from EEA metadata,
not from the label FG, so this runs in parallel.

Config via argv (`--key value`) or env (KEY); argv wins. Keys:
  countries    csv of country codes, or ALL          (default ALL)
  start        first date, YYYY-MM-DD                 (default 2019-01-01)
  end          last date, YYYY-MM-DD                  (default today - 7d, archive lag)
  limit-stations  cap number of stations, 0 = all     (default 0)
  resume       1 = skip stations already in the FG     (default 1)
  dry          1 = assemble and print, no FG write     (default 0)
"""

import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _root in [_here] + sorted(glob.glob("/hopsfs/Users/*/downwind")):
    if os.path.exists(os.path.join(_root, "downwind_features.py")):
        sys.path.insert(0, _root)
        break
from collect import eea, openmeteo  # noqa: E402

FEATURES = list(openmeteo.WEATHER.values()) + list(openmeteo.CAMS.values())


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
        "start": get("start", "2019-01-01"),
        "end": get("end", str(date.today() - timedelta(days=7))),
        "limit_stations": int(get("limit_stations", "0")),
        "sample": int(get("sample", "0")),
        "batch_rows": int(get("batch_rows", "1000000")),
        "parallel": int(get("parallel", "16")),
        "resume": get("resume", "1") == "1",
        "dry": get("dry", "0") == "1",
    }


def get_fg(fs):
    from hsfs.feature import Feature
    feats = [
        Feature("station_eoi", "string", description="EEA station EoI code"),
        Feature("valid_time", "timestamp", description="Hour the values are valid (event time)"),
        Feature("lat", "double", description="Station latitude"),
        Feature("lon", "double", description="Station longitude"),
        Feature("wind_speed", "double", description="10m wind speed, m/s"),
        Feature("wind_dir", "double", description="10m wind direction, deg"),
        Feature("temp", "double", description="2m temperature, C"),
        Feature("humidity", "double", description="2m relative humidity, %"),
        Feature("precip", "double", description="Precipitation, mm"),
        Feature("pressure", "double", description="Surface pressure, hPa"),
        Feature("cams_pm25", "double", description="CAMS modelled PM2.5, ug/m3"),
        Feature("cams_no2", "double", description="CAMS modelled NO2, ug/m3"),
        Feature("cams_o3", "double", description="CAMS modelled O3, ug/m3"),
        Feature("cams_so2", "double", description="CAMS modelled SO2, ug/m3"),
        Feature("cams_co", "double", description="CAMS modelled CO, ug/m3"),
        Feature("cams_dust", "double", description="CAMS dust, ug/m3"),
        Feature("cams_aod", "double", description="CAMS aerosol optical depth"),
    ]
    return fs.get_or_create_feature_group(
        name="station_features", version=1,
        description="Hourly open-meteo ERA5 weather + CAMS air quality per EEA station. "
                    "The coarse prior + meteorology the downwind model refines to a reading.",
        primary_key=["station_eoi"], event_time="valid_time",
        online_enabled=False, statistics_config=False, features=feats)


def already_done(fg) -> set:
    try:
        s = fg.select(["station_eoi"]).read(dataframe_type="pandas")
        return set(s["station_eoi"].unique())
    except Exception:
        return set()


def main():
    cfg = config()
    print("config:", cfg)
    stations = eea.stations_for_pollutants(["NO2", "PM2.5"])
    if cfg["countries"] != "ALL":
        codes = {c.strip() for c in cfg["countries"].split(",")}
        stations = stations[stations["station_eoi"].str[:2].isin(codes)]
    if cfg["sample"] and cfg["sample"] < len(stations):
        # fixed-seed spread across all countries (a first-model subset under the
        # open-meteo daily call cap), not the first-N by alphabet.
        stations = stations.sample(n=cfg["sample"], random_state=42)
    stations = stations.sort_values("station_eoi").reset_index(drop=True)
    print(f"{len(stations)} NO2/PM2.5 stations")

    fg = done = None
    if not cfg["dry"]:
        import hopsworks
        fg = get_fg(hopsworks.login().get_feature_store())
        done = already_done(fg) if cfg["resume"] else set()
        if done:
            print(f"resume: {len(done)} stations already present")

    cols = ["station_eoi", "valid_time", "lat", "lon"] + FEATURES

    def fetch_one(row):
        """One station -> its hourly features df (or None). Runs in a worker thread;
        the slow part is the two open-meteo HTTP calls, so fetching is parallel."""
        eoi = row["station_eoi"]
        try:
            df = openmeteo.station_hourly(row["lat"], row["lon"], cfg["start"], cfg["end"])
        except Exception as exc:
            print(f"  ! {eoi} open-meteo failed: {exc}")
            return None
        if df.empty:
            return None
        df.insert(0, "station_eoi", eoi)
        df["lat"], df["lon"] = row["lat"], row["lon"]
        df = df.dropna(subset=["cams_no2", "cams_pm25"])  # need the prior
        return None if df.empty else df[cols]

    todo = [row for _, row in stations.iterrows() if not (done and row["station_eoi"] in done)]
    if cfg["limit_stations"]:
        todo = todo[:cfg["limit_stations"]]
    print(f"{len(todo)} stations to fetch ({cfg['parallel']} in parallel)")

    n, total = 0, 0
    buf, buf_rows = [], 0

    def flush():
        # one insert per ~1M rows, not one Delta commit per station (commit-cost scar)
        nonlocal buf, buf_rows, total
        if not buf:
            return
        big = pd.concat(buf, ignore_index=True)
        buf, buf_rows = [], 0
        total += len(big)
        fg.insert(big, wait=True)
        print(f"  wrote {len(big)} rows ({n} stations done, running {total})")

    with ThreadPoolExecutor(max_workers=cfg["parallel"]) as ex:
        for df in ex.map(fetch_one, todo):  # inserts stay serial in the main thread
            if df is None:
                continue
            n += 1
            if cfg["dry"]:
                if n == 1:
                    print(df[["station_eoi", "valid_time", "wind_speed", "pressure",
                              "cams_no2", "cams_pm25"]].head(4).to_string())
                continue
            buf.append(df)
            buf_rows += len(df)
            if buf_rows >= cfg["batch_rows"]:
                flush()
    if not cfg["dry"]:
        flush()
    print(f"DONE: {n} stations, {total} rows written to station_features")


if __name__ == "__main__":
    main()
