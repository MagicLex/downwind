"""Deploy downwind as a custom (FastAPI) Hopsworks app.

Runs on downwind-app-env (python-app-pipeline clone + scikit-learn==1.8.0):
the embedded gap-filler model is a sklearn 1.8.0 pickle, and the unpickle
dies on any other sklearn (`No module named '_loss'`). Every version in the
entrypoint install is pinned for the same reason -- an unpinned install at
pod start is a time bomb that goes off on someone else's release day.
server.py runs from the FUSE repo path so a git pull + restart redeploys.
Redeploy uses the full recovery sequence from ghost-fleet (stop, purge
lingering k8s deployment, drain, stop zombie executions, settle) --
app.stop() returns before the execution dies. NOTE: environment and
entrypoint are fixed at create; changing them = delete app + recreate.
"""
import subprocess
import time
from pathlib import Path

import hopsworks

APP_NAME = "downwind"
ENV_NAME = "downwind-app-env"  # python-app-pipeline clone + scikit-learn==1.8.0 (model pickle needs it)

_here = Path(__file__).resolve()
rel = str(_here).split("/hopsfs/", 1)[1]
APP_PATH = str(Path(rel).parent / "server.py")
ENTRYPOINT = ('bash -lc "python -m uv pip install --system --no-cache '
              "'fastapi==0.139.0' 'uvicorn==0.49.0' 'scikit-learn==1.8.0' 'joblib==1.5.3' && "
              f'exec python /hopsfs/{rel.rsplit("/", 1)[0]}/server.py"')


def _pods():
    out = subprocess.run(["kubectl", "get", "pods"], capture_output=True, text=True).stdout
    return [l.split()[0] for l in out.splitlines() if APP_NAME in l]


def _purge_k8s():
    out = subprocess.run(["kubectl", "get", "deployment"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if APP_NAME in line:
            name = line.split()[0]
            subprocess.run(["kubectl", "delete", "deployment", name], capture_output=True)
            print(f"purged k8s deployment {name}", flush=True)
    for _ in range(60):
        if not _pods():
            return
        time.sleep(5)
    raise RuntimeError("app pods refused to drain")


def _stop_zombies(project):
    job = project.get_job_api().get_job(APP_NAME)
    if job is None:
        return
    for ex in job.get_executions() or []:
        if ex.final_status in ("UNDEFINED", None):
            try:
                ex.stop()
                print(f"stopped zombie execution {ex.id}", flush=True)
            except Exception:
                pass


def main():
    project = hopsworks.login()
    apps = project.get_app_api()
    print(f"app_path={APP_PATH}", flush=True)
    app = apps.get_app(APP_NAME)
    if app is None:
        app = apps.create_app(
            name=APP_NAME, app_path=APP_PATH, app_kind="CUSTOM",
            entrypoint_command=ENTRYPOINT, app_port=8000,
            environment=ENV_NAME, memory=4096, cores=1.0,
            description="downwind -- ground-level air quality where no sensor "
                        "exists: CAMS prior refined by weather + context, "
                        "measured stations as truth dots, 24h animated field")
    else:
        try:
            app.stop()
        except Exception:
            pass
        _purge_k8s()
        _stop_zombies(project)
        time.sleep(45)
    app.run(await_serving=True)
    print(f"URL: {app.app_url}", flush=True)


if __name__ == "__main__":
    main()
