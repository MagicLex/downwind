# air_quality_no2

Ground-level NO2 gap-filler, same architecture and validation as
[air_quality_pm25](air_quality_pm25.md).

## v1 (2026-07-09) -- withdrawn

The first run trained on 4.4M station-hours across 110 stations and produced a held-out
RMSE of 1313 ug/m3 against a CAMS-prior RMSE of 1199 ug/m3, both physically absurd (ambient
NO2 tops out around 200). A handful of stations carry validated-flagged readings in the
thousands of ug/m3; squared error let them dominate model and baseline alike, and both r2
went negative. The model was worse than the prior it was meant to refine, so v1 was deleted
from the registry rather than served.

**Root cause: label hygiene, not architecture.** The pipeline now applies physical label
bounds before training (NO2 kept in (0, 500) ug/m3, PM2.5 in (0, 800)).

## v2 -- retraining

Trains on the bounded labels over the full rebuilt station set; registered with the same
leave-stations-out protocol once it beats the CAMS prior at held-out stations. Until then
the app serves the raw CAMS prior for NO2 and says so.
