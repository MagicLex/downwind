# downwind: ground-level air quality where no sensor reaches

Spatial ML system. The prediction is a **ground-level pollutant concentration** (PM2.5,
NO2) at any point in Europe, at the places the ground-station network does not cover.
Satellite gives a coarse column from space, the ground stations give sparse truth, weather
and land context modulate what actually reaches the air people breathe. The model learns
the map between them at the stations, then fills the gaps between the dots. The deal-maker
is the feature store: satellite (daily, coarse grid), weather (hourly), static land context
and sparse station labels fused point-in-time, no leakage, no train/serve skew.

**Honesty rule (untested / ghost lineage):** the label is the sparse EEA ground-station
network, so the headline is the **error reduction over spatial interpolation** at
**held-out stations** (leave-stations-out CV), never an absolute R2 that a station the
model already saw would inflate. The map is an **estimate with an uncertainty band**, not a
measurement. Far from any station or under unusual meteorology it reads *low confidence*,
the applicability-domain rule from the-untested. The output is a *modelled estimate for
screening and sensor-siting, never a regulatory measurement*.

**Scope v1:** continental Europe on a working grid (roughly 0.1 degree, or H3 res 5),
two pollutants PM2.5 and NO2 (the two with the strongest satellite and health signal).
Bounded grid + two pollutants keeps ingest and training sane, same discipline as the
ghost bboxes.

---

## AI-system card

| Field | Value |
|---|---|
| **Prediction problem** | Regression (spatial gap-filling / downscaling): ground concentration of PM2.5 and NO2 at a (lat, lon, time) |
| **Entity / key** | Grid cell (H3 res 5 or 0.1deg lat/lon) x hour, per pollutant. Validation entity = EEA station id |
| **KPI** | Trust an estimate where no monitor exists, and rank where the next monitor should go. Watch-value: coverage of population living far from any station, served with a calibrated estimate |
| **ML proxy metric** | RMSE / MAE / R2 per pollutant at **held-out stations** (leave-stations-out + spatial-block CV), and **error reduction over interpolation** (IDW / kriging from the other stations) and over the raw satellite column. Uncertainty calibration reported |
| **System type** | Batch map builder (renders the continuous map) plus an on-demand point endpoint, over an offline training loop |
| **Consumption** | Custom web app: a continuous European pollution map denser than the sensor net, station truth-dots on top, click-anywhere local estimate with uncertainty and plain-word reasons, an attention rail of worst unmonitored hotspots |
| **Monitoring** | Every new station reading scores the map at that cell (held-out), rolling live error on the app; feature drift and score distribution tracked |

---

## Shared extractor (no-skew rule)

`downwind_features.py` is the ONE module that, given a `(lat, lon, time)`, assembles the
feature vector: the satellite column at that cell and day, weather at that point and hour
(wind speed and direction, boundary-layer height, temperature, humidity, precipitation,
pressure), the static land context (land-cover class, road density, population density,
elevation, distance to nearest industrial point source), and temporal encodings (hour,
day-of-week, season, cyclic). Used by the training-pair builder AND the serving predictor
AND the batch map builder. Never duplicated. Same discipline as `ghost_features.py` and
`skywatch_features.py`.

---

## Data sources (all open, Euro-native, to be verified pod-reachable in the data phase)

1. **EEA ground stations** (the label): European Environment Agency air quality download
   service / European Air Quality Portal, hourly validated and up-to-date PM2.5, PM10, NO2,
   O3 from thousands of stations. OpenAQ v3 (free key) aggregates the same and is the
   easier API if the EEA service is slow from the pod. This is the sparse ground truth.
2. **Sentinel-5P TROPOMI** (the satellite signal): Copernicus, tropospheric NO2 column and
   the UV aerosol index (a PM2.5 proxy), daily, ~5.5 km. Via the Copernicus Data Space
   Ecosystem (STAC / S3) or Sentinel Hub. Coarse from space, the model downscales it.
3. **Copernicus CAMS** (context and a baseline): atmospheric composition analysis with
   modelled ground PM2.5 and NO2 at ~10 km. Used as a feature and as a baseline the local
   model should refine. Free via the Atmosphere Data Store.
4. **Weather** (the modulator): open-meteo (free, pod-proven in ghost-fleet) or ERA5, hourly
   at stations and grid cells: wind speed and direction (transport), boundary-layer height
   (trapping), temperature, humidity, precipitation (washout), pressure. Meteorology is what
   turns a column into a ground concentration.
5. **Static land context** (the covariates): CORINE Land Cover (Copernicus), OpenStreetMap
   road density (a traffic proxy, aerial only, no Mapillary), population density (GHSL /
   Eurostat), Copernicus DEM elevation, E-PRTR industrial point sources. These are what let
   a coarse column resolve to a local concentration.

---

## Pipelines (ordered, feature -> training -> inference)

### F1. Satellite ingestion feature pipeline  `[staged to v1.5]`
- Staged after a working CAMS-based model (MVPS: ship the prediction service first, then add
  the raster leg). Job `downwind-tropomi` (daily): pull Sentinel-5P NO2 column + aerosol
  index at station locations, write FG **`tropomi_column`** v1 (offline; station_eoi + date).
  Sampled per station, same shape as F3, so it slots into the FV without reshaping. Needs the
  free Copernicus Data Space token for granule download (the only credential in the system).
- In v1 the coarse satellite signal is carried by CAMS (F3), which assimilates Sentinel-5P.
  F1 adds the *raw* column as the SOTA enhancement, the same staged discipline as
  the-untested (FP-GBM bar, then the GNN).
- Skill: **hops-features** -> hops-data-sources, hops-fg.

### F2. Ground-station label pipeline  `[blocked-by: nothing]`
- Job `downwind-stations` (hourly / daily): pull EEA (or OpenAQ) station measurements for
  Europe, write FG **`station_measurement`** v1 (offline; station_id, lat, lon, pollutant,
  value, ts). `event_time` = measurement time. The weak sparse ground truth.
- Skill: **hops-features** -> hops-data-sources, hops-fg.

### F3. Weather + CAMS feature pipeline  `[blocked-by: nothing]`
- Job `downwind-weather` (hourly): open-meteo ERA5 weather + CAMS air quality, hourly per
  station, write FG **`station_features`** v1 (offline; station_eoi + valid_time). Weather
  (wind speed and direction, temp, humidity, precip, pressure) plus the CAMS ~10 km modelled
  prior (pm2.5, no2, o3, so2, co, dust, aerosol optical depth). CAMS is the v1 coarse signal
  (it already assimilates Sentinel-5P); the model refines it to a station reading with local
  context. Station set and coordinates come from the EEA metadata, not the F2 label FG, so
  this runs in parallel with the F2 backfill.
- Skill: **hops-features** -> hops-data-sources, hops-fg.

### F4. Static context pipeline  `[blocked-by: nothing]`
- Job `downwind-context` (one-time build, seasonal refresh): CORINE land cover, OSM road
  density, population, elevation, distance-to-point-source per cell, write FG
  **`cell_context`** v1 (offline; key cell). `event_time` = build.
- Skill: **hops-features** -> hops-data-sources, hops-fg.

### F5 / FV. Feature view (the join is the pairing)  `[blocked-by: F2, F3]`
- No separate pairs FG. The feature store does the fusion: FV **`air_quality_fv`** v1 uses
  `station_measurement` (label) as the spine, joined to `station_features` on `station_eoi`
  with a point-in-time match on time (the weather + CAMS valid at the reading's hour). This
  is the feature-store showpiece, one join, no materialised copy, no skew. Static context
  (lat, lon, altitude, station area/type) rides along on the label FG.
- Tree models first, so minimal MDTs (log-target for the skewed concentration, cyclic-encode
  hour and day-of-year). **Leakage rule: split BY station and by spatial block, never random
  rows**, so validation measures true gap-filling, the sky group-by-flight and untested
  scaffold-split discipline. Lowercase every feature name in any exclusion set.
- Skill: **hops-fv**, **hops-transformations**.

### T. Training pipeline  `[blocked-by: FV]`
- EDA first: profile the concentration distribution, check leakage (a covariate that
  trivially encodes the station id or its region), confirm the held-out-station protocol.
  Skill: **hops-eda** / hops-eda-checklist.
- Baselines first: (a) spatial interpolation (IDW / kriging) from the other stations, (b)
  the raw satellite column, (c) the CAMS modelled ground value. Then a gradient-boosting
  regressor (HistGradientBoosting / XGBoost / LightGBM) per pollutant on the full feature
  set. **Leave-stations-out grouped CV plus spatial-block CV.** Loss matches the headline
  metric (the btc lesson). Metrics: RMSE / MAE / R2 per pollutant, **error reduction over
  interpolation and over the raw column**, and uncertainty calibration (coverage vs error).
- Register models **`air_quality_pm25`** and **`air_quality_no2`** (or one multi-output)
  with eval images: the **error-vs-distance-to-nearest-station** money-shot plot, spatial
  error map, predicted-vs-observed scatter, feature importance, and a model card carrying
  the estimate-not-measurement and out-of-domain caveats loud. Autoresearch recipe if
  round one is close.
- Skill: **hops-train**.

### I1. Batch map builder  `[blocked-by: T]`
- Job `downwind-map` (scheduled, after each satellite + weather refresh): predict every
  grid cell across Europe for the latest inputs, write FG **`grid_prediction`** v1 (cell,
  pollutant, value, uncertainty, ts). This is the render source for the app. Logs inputs
  and predictions for monitoring.
- Skill: **hops-batch-inference**.

### I2. On-demand point endpoint  `[blocked-by: T]`
- KServe deployment **`airscorer`** (clone `pandas-inference-pipeline` base): given a
  `(lat, lon)` and optional time, compute features **on-demand** (ODT: pull the latest
  satellite cell + live weather + the precomputed static context from the online FV), fuse,
  return concentration + uncertainty + the top plain-word reasons ("downwind of the A20 at
  6 m/s, boundary layer 250 m traps it, satellite NO2 column in the top decile"). The
  precomputed-context + on-demand-satellite-and-weather fusion is the feature-store
  showpiece. Logs inputs + predictions.
- Skill: **hops-online-inference**, **hops-transformations** (ODTs), **hops-environments**.

### M. Monitoring / self-scoring  `[blocked-by: I1]`
- Job `downwind-score` (hourly / daily): as new station readings arrive, join the predicted
  grid value at that station cell (a held-out validation station) with the actual reading,
  write FG **`prediction_scored`** v1 (rolling live error). This FG is the live accuracy
  line in the app, the sky self-scoring loop and the untested honest-meter, same spirit.
  Feature drift + score distribution + drift alert.
- Skill: **hops-monitoring**.

### A. App  `[blocked-by: I1, I2]`
- Custom web app **`airlive`** (FastAPI + map, server-rendered so content is in the initial
  payload, no SPA). Backend reads `grid_prediction` for the continuous map and calls
  `airscorer` for click-anywhere point queries, backfills the latest window on start so it
  is warm immediately. Sober earth-observation skin (Copernicus / defense feel, not
  gaming): a continuous PM2.5 / NO2 field denser than the sensor net, EEA stations as truth
  dots on top, click anywhere for a local estimate with its uncertainty and plain-word
  reasons and the nearest real station for cross-check, an attention rail of the worst
  **unmonitored** hotspots (high predicted pollution far from any sensor, where a monitor
  should go). Money-shot: hide the stations and the map still shows the plume, reveal them
  and they agree. External links to the EEA portal and Copernicus for every point.
- Skill: **hops-app**.

---

## Dependency graph

```
F1 tropomi ──┐
F2 stations ─┤                                            ┌─► I1 downwind-map ─┐
F3 weather ──┼─► F5 pairs ─► FV ─► T train ─► registry ───┤                    ├─► A airlive
F4 context ──┘                                            └─► I2 airscorer ────┘
                                                              I1 ─► M downwind-score (live scoreboard) ─► A
```

## v1 vs later
- **v1**: F1-F5, T, I1, I2, M, A on the European grid, PM2.5 and NO2.
- **v2**: O3 and PM10; higher-resolution downscaling under cities (100 m with building and
  street context); a proper geospatial model (a small CNN or graph over cells) past the
  GBM plateau, the same staged discipline as the-untested GNN; a time-forward claim (next
  24 h concentration, the honest forecasting leg btc never earned); CAMS as a learned
  residual target rather than a feature.

## Honesty and ethics
The output is a modelled estimate for screening, exposure awareness, and sensor-siting, not
a regulatory measurement. Every point carries its uncertainty, out-of-domain reads unknown
rather than a false precision, and every point links to the raw EEA and Copernicus
source-of-truth services. The headline is the error reduction over interpolation at
held-out stations, never an absolute score that a seen station would inflate.
