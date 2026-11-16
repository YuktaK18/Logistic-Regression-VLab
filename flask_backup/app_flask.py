"""Logistic Regression VLab (Flask BACKUP version, used only if Streamlit/pandas cannot run)."""
import os
import sys
import threading
import webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flask import Flask, Response, render_template, request

from lab import ml_core as core

app = Flask(__name__)

NAV = [
    ("theory", "/", "📘", "Aim & Theory"),
    ("procedure", "/procedure", "🧭", "Procedure"),
    ("dataset", "/dataset", "🗂️", "Dataset Explorer"),
    ("experiment", "/experiment", "🧪", "Run Experiment"),
    ("observations", "/observations", "📒", "Observations"),
]


@app.context_processor
def inject():
    return {"nav": NAV, "datasets": core.DATASETS, "models": core.MODELS}


@app.route("/")
def theory():
    return render_template("theory.html", page="theory", title="Aim & Theory",
                           sigmoid_img=core.sigmoid_plot())


@app.route("/procedure")
def procedure():
    return render_template("procedure.html", page="procedure", title="Experiment Procedure")


@app.route("/dataset")
def dataset():
    key = request.args.get("ds", "iris")
    if key not in core.DATASETS:
        key = "iris"
    return render_template("dataset.html", page="dataset", title="Dataset Explorer",
                           info=core.dataset_info(key), selected=key)


@app.route("/experiment", methods=["GET", "POST"])
def experiment():
    defaults = {"dataset": request.args.get("ds", "iris"), "model": "lr_l2", "test_pct": 25, "C": 1.0,
                "max_iter": 200, "solver": "lbfgs", "scale": True, "seed": 42}
    if defaults["dataset"] not in core.DATASETS:
        defaults["dataset"] = "iris"
    res, error, params, obs_no = None, None, defaults, None
    if request.method == "POST":
        try:
            params = core.parse_params(request.form)
            res = core.run_experiment(params)
            obs_no, obs_new = core.obs_add(res)
        except ValueError as exc:
            error = str(exc)
            params = {**defaults, **{k: request.form.get(k, v) for k, v in defaults.items() if k != "scale"},
                      "scale": request.form.get("scale") == "on"}
        except Exception as exc:  # noqa: BLE001
            error = f"Unexpected error while running the experiment: {exc}"
    return render_template("experiment.html", page="experiment", title="Run Experiment",
                           res=res, error=error, p=params, obs_no=obs_no)


@app.route("/download", methods=["POST"])
def download():
    try:
        res = core.run_experiment(core.parse_params(request.form))
    except ValueError as exc:
        return Response(str(exc), status=400)
    name = f"logreg_predictions_{res['params']['dataset']}.csv"
    return Response(core.predictions_csv(res), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


@app.route("/observations", methods=["GET", "POST"])
def observations():
    if request.method == "POST" and request.form.get("action") == "clear":
        core.obs_clear()
    rows = core.obs_load()
    return render_template("observations.html", page="observations", title="Observations", rows=rows,
                           headers=core.OBS_HEADERS, table=core.obs_table_rows(rows),
                           insights=core.obs_insights(rows), chart=core.obs_chart(rows) if rows else None)


@app.route("/observations.csv")
def observations_csv():
    return Response(core.obs_csv(core.obs_load()), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=observations.csv"})


@app.errorhandler(404)
def not_found(_):
    return render_template("theory.html", page="theory", title="Page not found",
                           sigmoid_img=core.sigmoid_plot()), 404


if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  Logistic Regression VLab (backup) running at {url}  (Ctrl+C to stop)\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
