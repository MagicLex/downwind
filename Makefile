# downwind -- FTI on Hopsworks: ground-level air quality where no sensor reaches
# Feature (satellite + stations + weather + context FGs) -> Training (GBM gap-filler, leave-stations-out) -> Inference (grid map + KServe + earth-obs app)
FEAT_ENV = python-feature-pipeline
TRAIN_ENV = pandas-training-pipeline

envs:                ## clone the collector / serve envs and pin deps
	python3 tools/build_envs.py

tropomi-job:         ## deploy + schedule Sentinel-5P column ingestion (daily)
	hops job deploy downwind-tropomi pipelines/tropomi_pipeline.py --env $(FEAT_ENV) --overwrite
	python3 tools/schedule.py downwind-tropomi "0 30 5 ? * *" --run

stations-job:        ## deploy + schedule EEA station label pull (hourly)
	hops job deploy downwind-stations pipelines/stations_pipeline.py --env $(FEAT_ENV) --overwrite
	python3 tools/schedule.py downwind-stations "0 15 0/1 ? * *" --run

weather-job:         ## deploy + schedule open-meteo weather pull (hourly)
	hops job deploy downwind-weather pipelines/weather_pipeline.py --env $(FEAT_ENV) --overwrite
	python3 tools/schedule.py downwind-weather "0 25 0/1 ? * *" --run

context-job:         ## deploy + run the static land context build (one-time, seasonal refresh)
	hops job deploy downwind-context pipelines/context_pipeline.py --env $(FEAT_ENV) --overwrite --run

pairs-job:           ## deploy + schedule the station x hour training-sample builder
	hops job deploy downwind-pairs pipelines/pairs_pipeline.py --env $(FEAT_ENV) --overwrite
	python3 tools/schedule.py downwind-pairs "0 40 0/1 ? * *" --run

train-job:           ## deploy + schedule the air_quality retrain (daily; every run registered, serve best by held-out error)
	hops job deploy downwind-train pipelines/train.py --env $(TRAIN_ENV) --overwrite
	python3 tools/schedule.py downwind-train "0 40 2 ? * *"

map-job:             ## deploy + schedule the continuous grid prediction (after satellite + weather refresh)
	hops job deploy downwind-map pipelines/map_pipeline.py --env $(TRAIN_ENV) --overwrite
	python3 tools/schedule.py downwind-map "0 0 6 ? * *" --run

score-job:           ## deploy + schedule the held-out station self-scoring (hourly)
	hops job deploy downwind-score pipelines/score_pipeline.py --env $(FEAT_ENV) --overwrite
	python3 tools/schedule.py downwind-score "0 50 0/1 ? * *" --run

serve:               ## deploy the airscorer KServe endpoint (after train)
	python3 serving/deploy_serving.py

app:                 ## deploy the airlive earth-observation app
	python3 app/deploy_app.py

smoke-stations:      ## run the EEA station pull from the terminal pod
	python3 pipelines/stations_pipeline.py
smoke-tropomi:       ## run a small Sentinel-5P pull from the terminal pod
	python3 pipelines/tropomi_pipeline.py

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/  --/'
.PHONY: envs tropomi-job stations-job weather-job context-job pairs-job train-job map-job score-job serve app smoke-stations smoke-tropomi help
