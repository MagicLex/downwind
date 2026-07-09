"""F2 -- ground-station label pipeline.

EEA hourly PM2.5 / NO2 -> join station coordinates + context -> FG `station_measurement`.
Offline FG (training-label data, no online serving). The sparse ground truth every
gap-filled estimate is validated against.

Inserts per parquet file so memory stays bounded on big countries (DE/FR/IT carry
millions of rows over the full window).

Config via argv (`--key value`) or env (`KEY`); argv wins. Keys:
  countries   csv of country codes, or ALL   (default ALL European)
  pollutants  csv of no2,pm25                 (default no2,pm25)
  min-year    drop readings before this year  (default 2019, the Sentinel-5P era)
  datasets    EEA dataset ids csv             (default 2,1 = Verified + UTD)
  limit-urls  cap parquet files per country/pollutant/dataset, 0 = all  (default 0)
  dry         1 = assemble and print, no FG write               (default 0)
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `collect`
from collect import eea  # noqa: E402


def config():
    """argv `--key value` overrides env KEY overrides default."""
    argv = {}
    a = sys.argv[1:]
    for i in range(0, len(a) - 1, 2):
        if a[i].startswith("--"):
            argv[a[i][2:].replace("-", "_")] = a[i + 1]

    def get(key, default):
        return argv.get(key, os.environ.get(key.upper(), default))

    return {
        "countries": get("countries", "ALL"),
        "pollutants": [p.strip() for p in get("pollutants", "no2,pm25").split(",") if p.strip()],
        "min_year": int(get("min_year", "2019")),
        "datasets": [int(d) for d in get("datasets", "2,1").split(",") if d.strip()],
        "limit_urls": int(get("limit_urls", "0")),
        "dry": get("dry", "0") == "1",
    }


def assemble_file(url: str, pol: str, country: str, min_year: int, meta: pd.DataFrame) -> pd.DataFrame:
    """One parquet -> label rows: valid, recent, placed on a station, deduped."""
    m = eea.read_measurements(url)
    if m.empty:
        return m
    m["pollutant"] = pol
    m["start_time"] = pd.to_datetime(m["Start"], utc=True)
    m["end_time"] = pd.to_datetime(m["End"], utc=True)
    m = m[m["start_time"].dt.year >= min_year]
    if m.empty:
        return m
    m = m.merge(meta, on="station_eoi", how="inner")  # inner: drop what we cannot place
    if m.empty:
        return m
    m["country"] = country
    m["validity"] = m["Validity"].astype("int64")
    out = m[["Samplingpoint", "station_eoi", "pollutant", "value", "lat", "lon",
             "altitude", "station_area", "station_type", "country",
             "start_time", "end_time", "validity"]].rename(
        columns={"Samplingpoint": "sampling_point"})
    return out.drop_duplicates(subset=["sampling_point", "start_time"])


def get_fg(fs):
    from hsfs.feature import Feature
    return fs.get_or_create_feature_group(
        name="station_measurement", version=1,
        description="EEA hourly ground-station PM2.5/NO2 with station coordinates and "
                    "context. The sparse ground-truth label for the downwind gap-filler.",
        primary_key=["sampling_point"], event_time="start_time",
        online_enabled=False, statistics_config=False,
        features=[
            Feature("sampling_point", "string", description="EEA sampling point id"),
            Feature("station_eoi", "string", description="EEA station EoI code"),
            Feature("pollutant", "string", description="no2 or pm25"),
            Feature("value", "double", description="Concentration, ug/m3"),
            Feature("lat", "double", description="Station latitude"),
            Feature("lon", "double", description="Station longitude"),
            Feature("altitude", "double", description="Station altitude, m"),
            Feature("station_area", "string", description="urban / suburban / rural"),
            Feature("station_type", "string", description="background / traffic / industrial"),
            Feature("country", "string", description="EEA country code"),
            Feature("start_time", "timestamp", description="Measurement start (event time)"),
            Feature("end_time", "timestamp", description="Measurement end"),
            Feature("validity", "bigint", description="EEA validity flag (>=1 valid)"),
        ],
    )


def main():
    cfg = config()
    print("config:", cfg)
    pol_uris = eea.pollutant_uris([eea.POLLUTANTS[p] for p in cfg["pollutants"]])
    print("downloading station metadata ...")
    meta = eea.station_metadata()
    print(f"  {len(meta)} stations with coordinates")

    country_list = eea.countries() if cfg["countries"] == "ALL" \
        else [c.strip() for c in cfg["countries"].split(",")]
    print(f"countries: {len(country_list)} -> {country_list}")

    fg = None
    if not cfg["dry"]:
        import hopsworks
        fg = get_fg(hopsworks.login().get_feature_store())

    total = 0
    for i, country in enumerate(country_list, 1):
        c_rows = 0
        for pol in cfg["pollutants"]:
            uri = pol_uris[eea.POLLUTANTS[pol]]
            for dataset in cfg["datasets"]:
                try:
                    urls = eea.parquet_urls(country, uri, dataset=dataset)
                except Exception as exc:
                    print(f"  ! {country}/{pol}/d{dataset} url list failed: {exc}")
                    continue
                if cfg["limit_urls"]:
                    urls = urls[:cfg["limit_urls"]]
                for url in urls:
                    try:
                        df = assemble_file(url, pol, country, cfg["min_year"], meta)
                    except Exception as exc:  # a single bad file never kills the run
                        print(f"  ! {country}/{pol}/d{dataset} skip {url.split('/')[-1]}: {exc}")
                        continue
                    if df.empty:
                        continue
                    c_rows += len(df)
                    if cfg["dry"]:
                        if total == 0:
                            print(df.head(4).to_string())
                    else:
                        fg.insert(df, wait=True)
                    total += len(df)
        print(f"[{i}/{len(country_list)}] {country}: {c_rows} rows")
    print(f"DONE: {total} station-measurement rows across {len(country_list)} countries")


if __name__ == "__main__":
    main()
