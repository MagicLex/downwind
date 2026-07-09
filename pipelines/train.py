"""T -- feature view + training.

Creates `air_quality_fv` (station_measurement label joined point-in-time to
station_features on station_eoi), then trains one gradient-boosting regressor per
pollutant that predicts the ground reading from the CAMS prior + weather + local
station context. Honest metric: error reduction over the raw CAMS prior, measured
with LEAVE-STATIONS-OUT CV (a station in the test fold is never in train, so the
number reflects true gap-filling, not memorising a sensor).

  --peek   build/inspect the FV and print the joined training data, no training
  --pollutants no2,pm25   which models to train (default both)
"""

import glob
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _root in [_here] + sorted(glob.glob("/hopsfs/Users/*/downwind")):
    if os.path.exists(os.path.join(_root, "downwind_features.py")):
        sys.path.insert(0, _root)
        break
import downwind_features as dwf  # noqa: E402

WEATHER = ["wind_speed", "wind_dir", "temp", "humidity", "precip", "pressure"]
CAMS = ["cams_pm25", "cams_no2", "cams_o3", "cams_so2", "cams_co", "cams_dust", "cams_aod"]
CONTEXT = ["lat", "lon", "altitude"]
CAT = ["station_area", "station_type"]
CAMS_PRIOR = {"no2": "cams_no2", "pm25": "cams_pm25"}  # the baseline each model must beat


def argval(flag, default):
    a = sys.argv
    return a[a.index(flag) + 1] if flag in a and a.index(flag) + 1 < len(a) else default


def get_fv(fs):
    label = fs.get_feature_group("station_measurement", version=1)
    feat = fs.get_feature_group("station_features", version=1)
    query = label.select(
        ["value", "pollutant", "station_eoi"] + CONTEXT + CAT
    ).join(feat.select(WEATHER + CAMS), on=["station_eoi"], join_type="inner")
    return fs.get_or_create_feature_view(
        name="air_quality_fv", version=1,
        description="Ground PM2.5/NO2 (label) joined point-in-time to CAMS + weather "
                    "per station. Read per pollutant; group CV by station_eoi.",
        query=query, labels=["value"],
        training_helper_columns=["station_eoi", "pollutant"],
    )


def add_temporal(X, times):
    t = pd.to_datetime(times, utc=True)
    X = X.copy()
    X["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24.0)
    X["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365.25)
    X["is_weekend"] = (t.dt.weekday >= 5).astype(float)
    return X


def metrics(y, p):
    err = np.asarray(p) - np.asarray(y)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - np.sum(err ** 2) / ss) if ss > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_pollutant(pol, X, y, times, groups, out_dir):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    feats = CONTEXT + WEATHER + CAMS + ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_weekend"]
    Xt = add_temporal(X[CONTEXT + WEATHER + CAMS], times)
    Xt = pd.concat([Xt, pd.get_dummies(X[CAT].astype(str), prefix=CAT)], axis=1)
    feat_cols = [c for c in Xt.columns]
    prior = X[CAMS_PRIOR[pol]].to_numpy()

    # leave-stations-out out-of-fold predictions
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    for tr, te in gkf.split(Xt, y, groups):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          max_leaf_nodes=63, l2_regularization=1.0,
                                          early_stopping=True, random_state=42)
        m.fit(Xt.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(Xt.iloc[te])

    model_m = metrics(y, oof)
    cams_m = metrics(y, prior)
    reduction = {k: round(1 - model_m[k] / cams_m[k], 4) for k in ("rmse", "mae")}
    print(f"[{pol}] n={len(y)} stations={groups.nunique()}")
    print(f"  model (leave-stations-out): {model_m}")
    print(f"  CAMS prior baseline:        {cams_m}")
    print(f"  error reduction over CAMS:  {reduction}")

    # final fit on all data
    final = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          max_leaf_nodes=63, l2_regularization=1.0,
                                          early_stopping=True, random_state=42)
    final.fit(Xt, y)
    plots = eval_plots(pol, y.to_numpy(), oof, prior, X, groups, out_dir)
    return final, feat_cols, {"model": model_m, "cams_baseline": cams_m,
                              "reduction_over_cams": reduction, "n": len(y),
                              "stations": int(groups.nunique())}, plots


def eval_plots(pol, y, oof, prior, X, groups, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []

    # predicted vs observed (model vs CAMS)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    hi = np.nanpercentile(y, 99)
    for a, p, t in ((ax[0], oof, "downwind (leave-stations-out)"), (ax[1], prior, "raw CAMS prior")):
        a.hexbin(y, p, gridsize=45, cmap="YlGnBu", mincnt=1, extent=(0, hi, 0, hi))
        a.plot([0, hi], [0, hi], "k--", lw=1)
        a.set_xlabel(f"observed {pol} (ug/m3)"); a.set_ylabel(f"predicted {pol}")
        a.set_title(t); a.set_xlim(0, hi); a.set_ylim(0, hi)
    fig.tight_layout(); pvo = f"{out_dir}/pred_vs_obs_{pol}.png"; fig.savefig(pvo, dpi=110); plt.close(fig)
    paths.append(pvo)

    # money-shot: error vs distance to nearest training station (does gap-filling hold far from sensors?)
    coords = X[["lat", "lon"]].to_numpy()
    st = pd.DataFrame({"g": groups.to_numpy(), "lat": coords[:, 0], "lon": coords[:, 1]}) \
        .groupby("g").first()
    abs_err = np.abs(oof - y)
    d = np.full(len(y), np.nan)
    stll = st[["lat", "lon"]].to_numpy()
    gi = {g: i for i, g in enumerate(st.index)}
    for i, g in enumerate(groups.to_numpy()):
        others = np.delete(stll, gi[g], axis=0)
        dd = np.sqrt(((others - coords[i]) ** 2).sum(1)) * 111.0  # deg->km approx
        d[i] = dd.min() if len(dd) else np.nan
    bins = pd.cut(d, [0, 10, 25, 50, 100, 1e6], labels=["<10", "10-25", "25-50", "50-100", ">100"])
    fig, ax = plt.subplots(figsize=(7, 5))
    dfb = pd.DataFrame({"bin": bins, "model": abs_err, "cams": np.abs(prior - y)}).groupby("bin", observed=True).mean()
    dfb.plot(kind="bar", ax=ax, color=["#10b981", "#9ca3af"])
    ax.set_ylabel(f"mean abs error {pol} (ug/m3)"); ax.set_xlabel("km to nearest other station")
    ax.set_title(f"{pol}: error vs distance to nearest sensor")
    fig.tight_layout(); msp = f"{out_dir}/error_vs_distance_{pol}.png"; fig.savefig(msp, dpi=110); plt.close(fig)
    paths.append(msp)
    return paths


def main():
    import hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    fv = get_fv(fs)
    print("feature view ready:", fv.name, "v", fv.version)

    print("reading training data (in-memory) ...")
    # The offline read via AFS is intermittently flaky (Errno 255 HDFS read on some
    # attempts, clean on others). Bounded retries catch a good read.
    for attempt in range(6):
        try:
            X, y = fv.training_data(training_helper_columns=True, event_time=True)
            break
        except Exception as exc:
            if attempt == 5:
                raise
            print(f"  read attempt {attempt + 1} failed ({str(exc).splitlines()[0][:70]}), retry")
            time.sleep(15)
    print("training rows:", len(X), "| columns:", list(X.columns))

    if "--peek" in sys.argv:
        print(X.head(5).to_string())
        print("pollutant counts:", X["pollutant"].value_counts().to_dict())
        return

    # event-time helpers come back PREFIXED (<fg>_<col>) from the FV read
    tcol = (next((c for c in X.columns if c.endswith("start_time")), None)
            or next((c for c in X.columns if c.endswith("valid_time")), None))
    times = X[tcol]
    pollutants = argval("--pollutants", "no2,pm25").split(",")
    out_dir = tempfile.mkdtemp()
    mr = project.get_model_registry()

    for pol in pollutants:
        mask = (X["pollutant"] == pol).to_numpy()
        if mask.sum() < 500:
            print(f"[{pol}] only {mask.sum()} rows, skip")
            continue
        Xp, yp = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)["value"]
        tp = pd.to_datetime(times[mask].reset_index(drop=True), utc=True)
        groups = Xp["station_eoi"]
        model, feat_cols, evals, plots = train_pollutant(pol, Xp, yp, tp, groups, out_dir)

        import joblib
        mdir = tempfile.mkdtemp()
        joblib.dump({"model": model, "features": feat_cols}, f"{mdir}/model.joblib")
        with open(f"{mdir}/metrics.json", "w") as f:
            json.dump(evals, f, indent=2)
        for p in plots:
            os.rename(p, f"{mdir}/{os.path.basename(p)}")
        name = f"air_quality_{pol}"
        reg = mr.python.create_model(
            name=name,
            metrics={"rmse": evals["model"]["rmse"], "mae": evals["model"]["mae"],
                     "r2": evals["model"]["r2"],
                     "rmse_reduction_over_cams": evals["reduction_over_cams"]["rmse"]},
            description=f"downwind {pol} gap-filler: GBM on CAMS+weather+context, "
                        f"leave-stations-out. Beats raw CAMS by "
                        f"{evals['reduction_over_cams']['rmse']:.0%} RMSE.",
            feature_view=fv,
        )
        reg.save(mdir)
        print(f"registered {name} v{reg.version}")


if __name__ == "__main__":
    main()
