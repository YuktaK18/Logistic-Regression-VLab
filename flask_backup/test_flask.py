"""Run with:  python flask_backup/test_flask.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["VLAB_OBS_PATH"] = "/tmp/obs_flask_test.json" if os.name != "nt" else os.path.join(os.environ.get("TEMP", "."), "obs_flask_test.json")
from flask_backup.app_flask import app
from lab import ml_core as core


def check(ds):
    p = core.parse_params({"dataset": ds, "model": "lr_l2", "test_pct": "25", "C": "1", "max_iter": "200",
                           "solver": "lbfgs", "scale": "on", "seed": "42"})
    r = core.run_experiment(p)
    m = r["metrics"]
    assert 0.5 < m["accuracy"] <= 1 and 0.5 < m["auc"] <= 1, (ds, m)
    assert len(r["preds"]) == r["n_test"] and all(r["images"].values())
    assert "id,actual" or core.predictions_csv(r).count("\n") == r["n_test"] + 1
    print(f"[PASS] {ds:14s} acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f} wrong={r['n_wrong']}")


for d in core.DATASETS:
    check(d)

c = app.test_client()
for url in ["/", "/procedure", "/dataset", "/dataset?ds=wine", "/dataset?ds=bad", "/experiment", "/observations", "/nope"]:
    code = c.get(url).status_code
    print(f"[GET ] {url:22s} -> {code}")
    assert code in (200, 404)
form = {"dataset": "breast_cancer", "model": "lr_l1", "test_pct": "30", "C": "0.5", "max_iter": "100", "solver": "newton-cg", "scale": "on", "seed": "1"}
r = c.post("/experiment", data=form); assert r.status_code == 200 and b"Conclusion" in r.data
r = c.post("/download", data=form); assert r.status_code == 200 and r.mimetype == "text/csv"
print("[POST] experiment + CSV download OK")
for bad in ({"dataset": "x"}, {"dataset": "iris", "C": "-3"}, {"dataset": "iris", "C": "abc"}, {"dataset": "iris", "max_iter": "0"}):
    r = c.post("/experiment", data=bad); assert b"alert err" in r.data, bad
print("[POST] invalid inputs handled gracefully")
# low iteration / no scaling case still works
f2 = {**form, "max_iter": "5", "scale": ""}; r = c.post("/experiment", data=f2); assert r.status_code == 200
r = c.get("/observations"); assert b"Observation Table" in r.data and b"What the results show" in r.data
r = c.get("/observations.csv"); assert r.mimetype == "text/csv"
r = c.post("/observations", data={"action": "clear"}); assert b"No observations yet" in r.data
print("[PASS] observations page, CSV, clear")
print("ALL TESTS PASSED")
