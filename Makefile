# downwind -- FTI on Hopsworks: ground-level air quality where no sensor reaches
# Feature (stations + weather/CAMS + satellite FGs) -> Training (GBM gap-filler, leave-stations-out) -> Inference (embedded-model map app)
FEAT_ENV = python-feature-pipeline
TRAIN_ENV = pandas-training-pipeline
S5P_ENV = downwind-s5p-env

stations-job:        ## deploy + run the EEA station label pull
	hops job deploy downwind-stations pipelines/stations_pipeline.py --env $(FEAT_ENV) --overwrite --run

weather-job:         ## deploy + run the open-meteo weather + CAMS pull (per grid cell)
	hops job deploy downwind-weather pipelines/weather_pipeline.py --env $(FEAT_ENV) --overwrite --run

tropomi-job:         ## deploy + run Sentinel-5P column ingestion (needs CDSE keys in secrets)
	hops job deploy downwind-tropomi pipelines/tropomi_pipeline.py --env $(S5P_ENV) --overwrite --run

train-job:           ## deploy + run the air_quality retrain (registers to the model registry)
	hops job deploy downwind-train pipelines/train.py --env $(TRAIN_ENV) --overwrite --run

app:                 ## deploy the map app
	python3 app/deploy_app.py

# NOTE: `hops job deploy --overwrite` resets job resources to 1 core / 2 GB.
# weather + tropomi want 8 GB, train wants 16 GB / 4 cores -- re-apply via
# job.config["resourceConfig"] after a redeploy.

smoke-stations:      ## run the EEA station pull from the terminal pod
	python3 pipelines/stations_pipeline.py
smoke-tropomi:       ## dry-run one Sentinel-5P day from the terminal pod
	python3 pipelines/tropomi_pipeline.py --dry 1

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/  --/'
.PHONY: stations-job weather-job tropomi-job train-job app smoke-stations smoke-tropomi help
