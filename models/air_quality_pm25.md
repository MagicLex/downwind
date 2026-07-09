# air_quality_pm25

Ground-level PM2.5 gap-filler: given the CAMS modelled prior, current weather and station
context at a point, predict what a ground station would measure there.

## v1 (2026-07-09)

| | |
|---|---|
| registry | `air_quality_pm25` v1, Hopsworks model registry |
| architecture | sklearn HistGradientBoostingRegressor (400 iter, lr 0.06, 63 leaves, L2 1.0, early stopping) |
| features | lat, lon, altitude + 6 weather (ERA5) + 7 CAMS species + 5 temporal encodings + station area/type one-hots |
| training data | 3.3M station-hours, 80 stations, 5 countries (AL/BA/BE/LU/MT), 2019 to 2026-07, through `air_quality_fv` v1 |
| validation | leave-stations-out GroupKFold (5 folds, grouped by station): every score comes from stations the fold never saw |

### Evaluation

| metric | model | raw CAMS prior |
|---|---|---|
| RMSE | **11.67 ug/m3** | 14.76 ug/m3 |
| MAE | **4.89 ug/m3** | 5.51 ug/m3 |
| r2 | **0.61** | 0.38 |

**20.9% RMSE reduction over the CAMS prior at held-out stations.** Pred-vs-obs and
error-vs-distance-to-nearest-sensor plots are attached to the registry entry.

### Intended use and limits

- Screening, exposure awareness, sensor-siting. Not a regulatory measurement.
- Trained on 5 countries; predictions elsewhere in Europe lean on the CAMS prior plus
  weather and carry no local station context. The app draws the sensor-anchored frontier
  for exactly this reason.
- Arbitrary points are predicted as a rural background station (neutral one-hots): urban
  street-canyon peaks will be underestimated.
- Labels are EEA validated data, which lag real time by days; the model refines a prior,
  it does not nowcast sudden events (fires, fireworks) beyond what CAMS carries.
