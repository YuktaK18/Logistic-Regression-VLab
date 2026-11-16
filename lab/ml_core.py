"""
Core machine-learning logic for the Logistic Regression Virtual Lab.
Uses only NumPy + scikit-learn + matplotlib (no pandas).
Everything returned to the UI is computed from a real trained model.
"""
import base64
import csv
import inspect
import io
import json
import os
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no GUI needed - draw to memory
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn import datasets  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, auc, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler, label_binarize  # noqa: E402

PALETTE = ["#4f46e5", "#f59e0b", "#10b981", "#ef4444", "#06b6d4", "#8b5cf6"]

DATASETS = {
    "iris": {
        "label": "Iris Flowers (3 classes)",
        "loader": datasets.load_iris,
        "about": "Classic dataset by R.A. Fisher. Measurements of 150 iris flowers from 3 species. "
                 "Goal: identify the species from petal and sepal size.",
        "kind": "Multiclass (3 classes)",
    },
    "breast_cancer": {
        "label": "Breast Cancer Wisconsin (2 classes)",
        "loader": datasets.load_breast_cancer,
        "about": "569 tumour samples; 30 numeric features computed from digitised cell-nucleus images. "
                 "Goal: classify a tumour as malignant or benign.",
        "kind": "Binary (2 classes)",
    },
    "wine": {
        "label": "Wine Recognition (3 classes)",
        "loader": datasets.load_wine,
        "about": "178 wines from 3 cultivars in Italy, described by 13 chemical measurements. "
                 "Goal: identify the cultivar from its chemical analysis.",
        "kind": "Multiclass (3 classes)",
    },
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1e1b4b")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def load_data(key):
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{key}'.")
    d = DATASETS[key]["loader"]()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=int)
    return X, y, [str(f) for f in d.feature_names], [str(t) for t in d.target_names]


def sigmoid_plot():
    z = np.linspace(-8, 8, 400)
    s = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(z, s, color=PALETTE[0], lw=2.5)
    ax.axhline(0.5, color="#9ca3af", ls="--", lw=1)
    ax.axvline(0, color="#9ca3af", ls="--", lw=1)
    ax.fill_between(z, s, 0.5, where=(z > 0), color=PALETTE[2], alpha=0.12)
    ax.fill_between(z, s, 0.5, where=(z < 0), color=PALETTE[3], alpha=0.12)
    ax.text(3, 0.2, "predict class 1\n(p > 0.5)", color="#047857", fontsize=9)
    ax.text(-7.5, 0.8, "predict class 0\n(p < 0.5)", color="#b91c1c", fontsize=9)
    _style(ax, "Sigmoid function  σ(z) = 1 / (1 + e⁻ᶻ)", "z = w·x + b", "σ(z) = probability")
    return fig_to_b64(fig)


# ----------------------------------------------------------------------------
# dataset page
# ----------------------------------------------------------------------------
def dataset_info(key):
    X, y, fnames, tnames = load_data(key)
    counts = np.bincount(y, minlength=len(tnames))
    stats = [
        {"name": n, "min": X[:, i].min(), "max": X[:, i].max(),
         "mean": X[:, i].mean(), "std": X[:, i].std()}
        for i, n in enumerate(fnames)
    ]
    preview = [
        {"vals": [float(v) for v in X[i]], "target": tnames[y[i]]}
        for i in range(10)
    ]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar(tnames, counts, color=PALETTE[: len(tnames)])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=9)
    _style(ax, "Class distribution", "Class", "Samples")
    dist_img = fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for k, name in enumerate(tnames):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=18, alpha=0.75, color=PALETTE[k % len(PALETTE)], label=name)
    _style(ax, "First two features", fnames[0][:28], fnames[1][:28])
    ax.legend(fontsize=8)
    scatter_img = fig_to_b64(fig)

    cfg = DATASETS[key]
    return {
        "key": key, "label": cfg["label"], "about": cfg["about"], "kind": cfg["kind"],
        "n_samples": X.shape[0], "n_features": X.shape[1], "n_classes": len(tnames),
        "features": fnames, "classes": list(zip(tnames, counts.tolist())),
        "stats": stats, "preview": preview, "dist_img": dist_img, "scatter_img": scatter_img,
    }


# ----------------------------------------------------------------------------
# parameter validation
# ----------------------------------------------------------------------------
MODELS = {
    "lr_l2": {"label": "Logistic Regression – L2 (Ridge) [default]", "lr": True, "has_C": True,
              "about": "Standard Logistic Regression. L2 shrinks all weights smoothly towards zero."},
    "lr_l1": {"label": "Logistic Regression – L1 (Lasso)", "lr": True, "has_C": True,
              "about": "L1 penalty can push some weights exactly to zero, acting as automatic feature selection."},
    "lr_en": {"label": "Logistic Regression – Elastic Net", "lr": True, "has_C": True,
              "about": "Mix of L1 and L2 penalties (50/50)."},
    "lr_none": {"label": "Logistic Regression – No regularisation", "lr": True, "has_C": False,
                "about": "Plain Logistic Regression with no penalty (C is ignored). May overfit."},
    "knn": {"label": "K-Nearest Neighbours (k=5) – comparison", "lr": False, "has_C": False,
            "about": "Baseline model for comparison. Classifies by majority vote of the 5 nearest samples."},
    "tree": {"label": "Decision Tree (depth 4) – comparison", "lr": False, "has_C": False,
             "about": "Baseline model for comparison. Learns if-else rules; can capture non-linear patterns."},
    "svm": {"label": "Support Vector Machine (RBF) – comparison", "lr": False, "has_C": True,
            "about": "Baseline model for comparison. Uses the same C idea (margin vs. error trade-off)."},
}

# scikit-learn >= 1.8 replaced `penalty` by `l1_ratio`; support both old and new versions
_NEW_LR_API = inspect.signature(LogisticRegression.__init__).parameters["penalty"].default == "deprecated" \
    if "penalty" in inspect.signature(LogisticRegression.__init__).parameters else True


def build_model(key, C, max_iter, solver, seed):
    if key.startswith("lr_"):
        kind = key[3:]
        kw = dict(C=C, max_iter=max_iter, solver=solver, random_state=seed)
        if kind in ("l1", "en"):
            kw["solver"] = "saga"          # the solver that supports L1 / elastic-net
        if _NEW_LR_API:
            kw["l1_ratio"] = {"l2": 0.0, "l1": 1.0, "en": 0.5, "none": 0.0}[kind]
            if kind == "none":
                kw["C"] = np.inf
        else:
            kw["penalty"] = {"l2": "l2", "l1": "l1", "en": "elasticnet", "none": None}[kind]
            if kind == "en":
                kw["l1_ratio"] = 0.5
        return LogisticRegression(**kw)
    if key == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if key == "tree":
        return DecisionTreeClassifier(max_depth=4, random_state=seed)
    if key == "svm":
        return SVC(C=C, probability=True, random_state=seed)
    raise ValueError("Unknown model.")


def parse_params(form):
    def num(name, default, cast, lo, hi, label):
        raw = form.get(name, "")
        raw = default if raw in ("", None) else raw
        try:
            val = cast(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number.")
        if not (lo <= val <= hi):
            raise ValueError(f"{label} must be between {lo} and {hi}.")
        return val

    ds = form.get("dataset", "iris")
    if ds not in DATASETS:
        raise ValueError("Please choose a valid dataset.")
    model = form.get("model", "lr_l2")
    if model not in MODELS:
        raise ValueError("Please choose a valid model.")
    solver = form.get("solver", "lbfgs")
    if solver not in ("lbfgs", "newton-cg"):
        raise ValueError("Unsupported solver.")
    return {
        "dataset": ds, "model": model,
        "test_pct": num("test_pct", 25, int, 10, 50, "Test size (%)"),
        "C": num("C", 1.0, float, 0.0001, 10000, "C"),
        "max_iter": num("max_iter", 200, int, 5, 10000, "Max iterations"),
        "solver": solver,
        "scale": form.get("scale") in ("on", "1", "true", True),
        "seed": num("seed", 42, int, 0, 99999, "Random seed"),
    }


# ----------------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------------
def _fit(model_key, C, max_iter, solver, seed, Xtr, ytr):
    model = build_model(model_key, C, max_iter, solver, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr)
    return model


def run_experiment(p):
    X, y, fnames, tnames = load_data(p["dataset"])
    n_classes = len(tnames)
    idx = np.arange(len(y))
    mkey = p.get("model", "lr_l2")
    minfo = MODELS[mkey]

    # 1) split (stratified keeps class proportions equal in train and test)
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, idx, test_size=p["test_pct"] / 100, random_state=p["seed"], stratify=y)

    # 2) preprocessing - scaler is fitted on TRAIN data only (no data leakage)
    if p["scale"]:
        scaler = StandardScaler().fit(Xtr)
        Xtr_m, Xte_m = scaler.transform(Xtr), scaler.transform(Xte)
    else:
        Xtr_m, Xte_m = Xtr, Xte

    # 3) training
    model = _fit(mkey, p["C"], p["max_iter"], p["solver"], p["seed"], Xtr_m, ytr)
    if isinstance(model, LogisticRegression):
        n_iter = int(np.max(model.n_iter_))
        converged = n_iter < p["max_iter"]
        solver_used = model.solver
    else:
        n_iter, converged, solver_used = None, True, "-"

    # 4) testing
    ypred = model.predict(Xte_m)
    proba = model.predict_proba(Xte_m)
    ypred_tr = model.predict(Xtr_m)

    # 5) evaluation
    avg = "binary" if n_classes == 2 else "weighted"
    metrics = {
        "accuracy": accuracy_score(yte, ypred),
        "precision": precision_score(yte, ypred, average=avg, zero_division=0),
        "recall": recall_score(yte, ypred, average=avg, zero_division=0),
        "f1": f1_score(yte, ypred, average=avg, zero_division=0),
        "train_accuracy": accuracy_score(ytr, ypred_tr),
    }
    if n_classes == 2:
        metrics["auc"] = roc_auc_score(yte, proba[:, 1])
        avg_note = f"Binary: positive class = “{tnames[1]}”"
    else:
        metrics["auc"] = roc_auc_score(yte, proba, multi_class="ovr", average="macro")
        avg_note = "Multiclass: weighted average over classes"
    cm = confusion_matrix(yte, ypred, labels=list(range(n_classes)))

    rep = classification_report(yte, ypred, target_names=tnames, output_dict=True, zero_division=0)
    report_rows = []
    for name in tnames + ["macro avg", "weighted avg"]:
        r = rep[name]
        report_rows.append({"name": name, "precision": r["precision"], "recall": r["recall"],
                            "f1": r["f1-score"], "support": int(r["support"])})

    # ---- plots -------------------------------------------------------------
    images = {}

    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels([t[:12] for t in tnames], rotation=30, ha="right")
    ax.set_yticklabels([t[:12] for t in tnames])
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1e1b4b")
    ax.set_title("Confusion matrix", fontsize=11, fontweight="bold", color="#1e1b4b")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax, fraction=0.046)
    images["cm"] = fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(yte, proba[:, 1])
        ax.plot(fpr, tpr, color=PALETTE[0], lw=2.5, label=f"AUC = {auc(fpr, tpr):.3f}")
    else:
        Y = label_binarize(yte, classes=list(range(n_classes)))
        for k in range(n_classes):
            fpr, tpr, _ = roc_curve(Y[:, k], proba[:, k])
            ax.plot(fpr, tpr, lw=2, color=PALETTE[k], label=f"{tnames[k][:14]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Random guess")
    ax.legend(loc="lower right", fontsize=8)
    _style(ax, "ROC curve" + (" (one-vs-rest)" if n_classes > 2 else ""),
           "False Positive Rate", "True Positive Rate")
    images["roc"] = fig_to_b64(fig)

    # learned coefficients (only for linear models)
    n_zero = None
    coef = model.coef_ if hasattr(model, "coef_") else None
    if coef is not None:
        n_zero = int(np.sum(np.abs(coef) < 1e-8))
        if n_classes == 2:
            order = np.argsort(np.abs(coef[0]))[::-1][:15][::-1]
            fig, ax = plt.subplots(figsize=(5.4, max(3, 0.32 * len(order) + 1)))
            ax.barh([fnames[i][:26] for i in order], coef[0][order],
                    color=[PALETTE[2] if v > 0 else PALETTE[3] for v in coef[0][order]])
            _style(ax, f"Top coefficients (+ pushes towards “{tnames[1]}”)", "Coefficient value", "")
        else:
            fig, ax = plt.subplots(figsize=(5.8, 3.6))
            lim = max(np.abs(coef).max(), 1e-9)
            im = ax.imshow(coef, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_yticks(range(n_classes)); ax.set_yticklabels([t[:14] for t in tnames])
            ax.set_xticks(range(len(fnames)))
            ax.set_xticklabels([f[:14] for f in fnames], rotation=70, ha="right", fontsize=8)
            ax.set_title("Learned coefficients (class × feature)", fontsize=11, fontweight="bold", color="#1e1b4b")
            fig.colorbar(im, ax=ax, fraction=0.03)
        images["coef"] = fig_to_b64(fig)
    else:
        images["coef"] = None

    # effect of C (same split, same preprocessing) - only when the model has C
    best_C, best_C_acc = None, None
    if minfo["has_C"]:
        Cs = np.logspace(-3, 3, 13)
        tr_acc, te_acc = [], []
        for c in Cs:
            m = _fit(mkey, c, p["max_iter"], p["solver"], p["seed"], Xtr_m, ytr)
            tr_acc.append(accuracy_score(ytr, m.predict(Xtr_m)))
            te_acc.append(accuracy_score(yte, m.predict(Xte_m)))
        best_i = int(np.argmax(te_acc))
        best_C, best_C_acc = float(Cs[best_i]), float(te_acc[best_i])
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.semilogx(Cs, tr_acc, "o-", color=PALETTE[0], label="Train accuracy")
        ax.semilogx(Cs, te_acc, "s-", color=PALETTE[1], label="Test accuracy")
        ax.axvline(p["C"], color=PALETTE[3], ls="--", lw=1.2, label=f"Your C = {p['C']:g}")
        ax.legend(fontsize=8, loc="lower right")
        _style(ax, "Effect of regularisation strength C", "C (log scale)", "Accuracy")
        images["sweep"] = fig_to_b64(fig)
    else:
        images["sweep"] = None

    # 2-feature decision regions (illustration)
    if coef is not None:
        top2 = np.argsort(np.abs(coef).sum(axis=0))[::-1][:2]
    else:
        corr = [abs(np.corrcoef(Xtr_m[:, j], ytr)[0, 1]) if Xtr_m[:, j].std() > 0 else 0 for j in range(X.shape[1])]
        top2 = np.argsort(corr)[::-1][:2]
    m2 = _fit(mkey, p["C"], max(p["max_iter"], 200), p["solver"], p["seed"], Xtr_m[:, top2], ytr)
    pad = 0.5 * (Xtr_m[:, top2].std(axis=0))
    mn, mx = Xtr_m[:, top2].min(axis=0) - pad, Xtr_m[:, top2].max(axis=0) + pad
    gx, gy = np.meshgrid(np.linspace(mn[0], mx[0], 250), np.linspace(mn[1], mx[1], 250))
    zz = m2.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.contourf(gx, gy, zz, levels=np.arange(-0.5, n_classes, 1), alpha=0.18, colors=PALETTE[:n_classes])
    for k in range(n_classes):
        mk = yte == k
        ax.scatter(Xte_m[mk, top2[0]], Xte_m[mk, top2[1]], s=22, color=PALETTE[k],
                   edgecolor="white", linewidth=0.5, label=tnames[k][:14])
    suffix = " (scaled)" if p["scale"] else ""
    _style(ax, "Decision regions (2 most influential features)",
           fnames[top2[0]][:26] + suffix, fnames[top2[1]][:26] + suffix)
    ax.legend(fontsize=8)
    images["boundary"] = fig_to_b64(fig)

    # model comparison on the same split (model selection)
    comp = []
    for k, info in MODELS.items():
        mm = model if k == mkey else _fit(k, p["C"], p["max_iter"], p["solver"], p["seed"], Xtr_m, ytr)
        comp.append((info["label"].split(" [")[0].replace(" – comparison", ""),
                     accuracy_score(yte, mm.predict(Xte_m)), k == mkey))
    comp.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh([c[0][:40] for c in comp], [c[1] for c in comp],
            color=[PALETTE[0] if c[2] else "#c7d2fe" for c in comp])
    for i, c in enumerate(comp):
        ax.text(c[1] + 0.003, i, f"{c[1]*100:.1f}%", va="center", fontsize=8)
    ax.set_xlim(max(0, min(c[1] for c in comp) - 0.1), 1.08)
    _style(ax, "Model comparison – test accuracy (selected model in dark blue)", "Accuracy", "")
    images["compare"] = fig_to_b64(fig)
    best_model = max(comp, key=lambda t: t[1])

    # ---- predictions table ---------------------------------------------------
    conf = proba.max(axis=1)
    preds = []
    for r in range(len(yte)):
        preds.append({
            "sample": int(ite[r]), "actual": tnames[yte[r]], "predicted": tnames[ypred[r]],
            "ok": bool(yte[r] == ypred[r]), "confidence": float(conf[r]),
            "probs": [float(v) for v in proba[r]], "features": [float(v) for v in Xte[r]],
        })
    n_wrong = int((yte != ypred).sum())

    res = {
        "params": p, "dataset": DATASETS[p["dataset"]]["label"], "classes": tnames,
        "model_label": minfo["label"].split(" [")[0], "is_lr": minfo["lr"], "has_C": minfo["has_C"],
        "solver_used": solver_used, "n_zero": n_zero, "n_coef": None if coef is None else int(coef.size),
        "best_model": best_model[0], "best_model_acc": float(best_model[1]),
        "feature_names": fnames, "n_classes": n_classes,
        "n_train": len(ytr), "n_test": len(yte), "n_features": X.shape[1],
        "n_iter": n_iter, "converged": converged, "metrics": metrics, "avg_note": avg_note,
        "report": report_rows, "cm": cm.tolist(), "images": images, "preds": preds,
        "n_wrong": n_wrong, "best_C": best_C, "best_C_acc": best_C_acc,
    }
    res["conclusion"] = make_conclusion(res, cm, tnames)
    return res


def _grade(acc):
    if acc >= 0.97: return "excellent"
    if acc >= 0.93: return "very good"
    if acc >= 0.85: return "good"
    if acc >= 0.70: return "moderate"
    return "poor"


def _article(word):
    return "an" if word[0] in "aeiou" else "a"


def make_conclusion(res, cm, tnames):
    """Builds the experiment conclusion purely from the measured results."""
    p, m = res["params"], res["metrics"]
    out = []
    kind = "binary" if res["n_classes"] == 2 else f"{res['n_classes']}-class"
    cfg = f"C = {p['C']:g}, " if res["has_C"] else ""
    if res["is_lr"]:
        cfg += f"solver = {res['solver_used']}, max_iter = {p['max_iter']}, "
    out.append(
        f"A {kind} classification experiment was run with the model “{res['model_label']}” ({cfg.rstrip(', ')}) "
        f"on the {res['dataset']} dataset: {res['n_train']} samples for training and {res['n_test']} unseen samples "
        f"for testing ({p['test_pct']}% test split, {'with' if p['scale'] else 'without'} feature standardisation).")
    out.append(
        f"The model achieved a test accuracy of {m['accuracy']*100:.2f}% — {_article(_grade(m['accuracy']))} "
        f"{_grade(m['accuracy'])} result — with precision {m['precision']:.3f}, recall {m['recall']:.3f}, "
        f"F1-score {m['f1']:.3f} and ROC-AUC {m['auc']:.3f}. It misclassified {res['n_wrong']} of {res['n_test']} test samples.")
    gap = m["train_accuracy"] - m["accuracy"]
    if gap > 0.05:
        out.append(f"Training accuracy ({m['train_accuracy']*100:.2f}%) is noticeably higher than test accuracy, "
                   "which hints at overfitting; try stronger regularisation (smaller C) or a simpler model.")
    elif m["train_accuracy"] < 0.80 and m["accuracy"] < 0.80:
        out.append("Both training and test accuracy are low, which hints at underfitting; try a larger C, "
                   "more iterations or enabling feature standardisation.")
    else:
        out.append(f"Training accuracy ({m['train_accuracy']*100:.2f}%) and test accuracy are close, "
                   "so the model generalises well and does not overfit noticeably.")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    if off.sum() > 0:
        i, j = np.unravel_index(np.argmax(off), off.shape)
        out.append(f"The most frequent confusion was actual “{tnames[i]}” predicted as “{tnames[j]}” "
                   f"({int(off[i, j])} time(s)).")
    else:
        out.append("The confusion matrix is perfectly diagonal: every test sample was classified correctly.")
    if res["n_zero"] is not None and p["model"] in ("lr_l1", "lr_en"):
        out.append(f"The {'L1' if p['model'] == 'lr_l1' else 'elastic-net'} penalty set {res['n_zero']} of "
                   f"{res['n_coef']} coefficients exactly to zero, i.e. it removed those features from the model.")
    if not res["converged"]:
        out.append(f"Warning: the solver used all {p['max_iter']} iterations and may not have converged. "
                   "Increase max iterations or enable standardisation.")
    if res["best_C"] is not None and abs(res["best_C"] - p["C"]) / p["C"] > 0.01 \
            and res["best_C_acc"] > m["accuracy"] + 1e-9:
        out.append(f"In the C-sweep, C ≈ {res['best_C']:g} gave a higher test accuracy "
                   f"({res['best_C_acc']*100:.2f}%) than your chosen value.")
    if res["best_model_acc"] > m["accuracy"] + 1e-9:
        out.append(f"In the model comparison, {res['best_model']} scored highest ({res['best_model_acc']*100:.2f}%) "
                   "on this split.")
    else:
        out.append("In the model comparison, the selected model was among the best on this split "
                   f"(best accuracy {res['best_model_acc']*100:.2f}%).")
    return out


def predictions_csv(res):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sample_index", "actual", "predicted", "correct", "confidence"]
               + [f"prob_{c}" for c in res["classes"]] + res["feature_names"])
    for r in res["preds"]:
        w.writerow([r["sample"], r["actual"], r["predicted"], "yes" if r["ok"] else "no",
                    f"{r['confidence']:.4f}"] + [f"{v:.4f}" for v in r["probs"]]
                   + [f"{v:.4f}" for v in r["features"]])
    return buf.getvalue()


# ----------------------------------------------------------------------------
# OBSERVATIONS - every experiment run is logged automatically
# ----------------------------------------------------------------------------
OBS_PATH = os.environ.get("VLAB_OBS_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "observations_log.json")
_OBS_KEYS = ("dataset", "model", "test_pct", "C", "max_iter", "solver", "scale", "seed")
_PARAM_LABEL = {"model": "the model", "scale": "feature standardisation", "C": "C",
                "max_iter": "max iterations", "test_pct": "test size (%)", "solver": "the solver",
                "seed": "the random seed"}


def obs_load():
    try:
        with open(OBS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def obs_save(rows):
    try:
        with open(OBS_PATH, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
        return True
    except OSError:
        return False


def obs_clear():
    return obs_save([])


def obs_add(res):
    """Record one finished experiment. Returns (observation number, is_new)."""
    p, m = res["params"], res["metrics"]
    row = {
        "dataset": p["dataset"], "dataset_label": DATASETS[p["dataset"]]["label"],
        "model": p["model"], "model_label": res["model_label"],
        "test_pct": p["test_pct"], "C": p["C"] if res["has_C"] else None,
        "max_iter": p["max_iter"] if res["is_lr"] else None,
        "solver": res["solver_used"] if res["is_lr"] else None,
        "scale": bool(p["scale"]), "seed": p["seed"],
        "train_acc": m["train_accuracy"], "accuracy": m["accuracy"], "precision": m["precision"],
        "recall": m["recall"], "f1": m["f1"], "auc": m["auc"], "n_iter": res["n_iter"],
        "converged": res["converged"], "n_wrong": res["n_wrong"], "n_test": res["n_test"],
        "time": time.strftime("%d-%b %H:%M"),
    }
    rows = obs_load()
    key = tuple(row[k] for k in _OBS_KEYS)
    for i, r in enumerate(rows):          # same settings again -> update instead of duplicating
        if tuple(r.get(k) for k in _OBS_KEYS) == key:
            rows[i] = row
            obs_save(rows)
            return i + 1, False
    rows.append(row)
    obs_save(rows)
    return len(rows), True


def obs_table_rows(rows):
    out = []
    for i, r in enumerate(rows, 1):
        out.append([i, r["dataset_label"].split(" (")[0], r["model_label"].replace("Logistic Regression", "LogReg"),
                    f"{r['test_pct']}%", "-" if r["C"] is None else f"{r['C']:g}",
                    "-" if r["max_iter"] is None else r["max_iter"], "ON" if r["scale"] else "OFF",
                    f"{r['train_acc']*100:.2f}%", f"{r['accuracy']*100:.2f}%", f"{r['precision']:.3f}",
                    f"{r['recall']:.3f}", f"{r['f1']:.3f}", f"{r['auc']:.3f}",
                    "-" if r["n_iter"] is None else r["n_iter"]])
    return out


OBS_HEADERS = ["Sr.", "Dataset", "Model", "Test size", "C", "Max iter", "Scaling", "Train acc",
               "Test acc", "Precision", "Recall", "F1", "AUC", "Iterations"]


def obs_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(OBS_HEADERS)
    w.writerows(obs_table_rows(rows))
    return buf.getvalue()


def obs_chart(rows):
    rows = rows[-25:]
    start = len(obs_load()) - len(rows) + 1
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(rows) + 3), 3.8))
    ax.bar(x - 0.2, [r["train_acc"] for r in rows], 0.4, color="#c7d2fe", label="Train accuracy")
    ax.bar(x + 0.2, [r["accuracy"] for r in rows], 0.4, color=PALETTE[0], label="Test accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{start + i}" for i in range(len(rows))])
    lo = min(min(r["train_acc"], r["accuracy"]) for r in rows)
    ax.set_ylim(max(0, lo - 0.08), 1.02)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    _style(ax, "Accuracy of every recorded run", "Observation number", "Accuracy")
    return fig_to_b64(fig)


def _fmt(param, row):
    v = row[param]
    if param == "scale":
        return "ON" if v else "OFF"
    if param == "model":
        return row["model_label"]
    if param == "C":
        return "-" if v is None else f"{v:g}"
    return str(v)


def obs_insights(rows):
    """Plain-English findings computed from the recorded runs (nothing is assumed)."""
    out = []
    if not rows:
        return out
    best = max(rows, key=lambda r: (r["accuracy"], r["f1"]))
    worst = min(rows, key=lambda r: (r["accuracy"], r["f1"]))
    bi, wi = rows.index(best) + 1, rows.index(worst) + 1
    out.append(f"{len(rows)} run(s) recorded. The best test accuracy was {best['accuracy']*100:.2f}% "
               f"(observation #{bi}: {best['model_label']} on {best['dataset_label'].split(' (')[0]}, "
               f"scaling {'ON' if best['scale'] else 'OFF'}).")
    if len(rows) > 1:
        out.append(f"The lowest test accuracy was {worst['accuracy']*100:.2f}% (observation #{wi}: "
                   f"{worst['model_label']} on {worst['dataset_label'].split(' (')[0]}, "
                   f"scaling {'ON' if worst['scale'] else 'OFF'}).")

    # controlled comparisons: runs that differ in exactly ONE setting
    lines = []
    for param in ("scale", "C", "max_iter", "test_pct", "model", "solver", "seed"):
        buckets = {}
        for r in rows:
            if r[param] is None and param != "model":
                continue
            skip = {param} | ({"C", "max_iter", "solver"} if param == "model" else set())
            other = tuple(r[k] for k in _OBS_KEYS if k not in skip)
            buckets.setdefault(other, []).append(r)
        for grp in buckets.values():
            vals = {}
            for r in grp:
                k = str(r[param])
                # for model comparisons keep each model's best run (fair comparison)
                if param != "model" or k not in vals or r["accuracy"] > vals[k]["accuracy"]:
                    vals[k] = r
            if len(vals) < 2:
                continue
            grp = sorted(vals.values(), key=lambda r: (r[param] is None, r[param] if param != "model" else r["model_label"]))
            accs = [r["accuracy"] for r in grp]
            first = grp[0]
            where = f"On {first['dataset_label'].split(' (')[0]}" + ("" if param == "model" else f" with {first['model_label']}")
            seq = " → ".join(_fmt(param, r) for r in grp)
            if param == "model":
                seq += "; best run of each model"
            accseq = " → ".join(f"{a*100:.2f}%" for a in accs)
            if max(accs) - min(accs) < 0.0005:
                verdict = f"{_PARAM_LABEL[param]} made no difference to accuracy here."
            else:
                tops = [r for r, a in zip(grp, accs) if a >= max(accs) - 1e-9]
                if len(tops) > 1:
                    verdict = (f"best accuracy shared by {_PARAM_LABEL[param]} = "
                               + " and ".join(_fmt(param, r) for r in tops) + ".")
                else:
                    verdict = f"best accuracy with {_PARAM_LABEL[param]} = {_fmt(param, tops[0])}."
            lines.append(f"{where}, changing {_PARAM_LABEL[param]} ({seq}) gave test accuracy {accseq}; {verdict}")
    out.extend(lines[:8])
    if not lines and len(rows) > 1:
        out.append("Change only ONE setting between runs (e.g. only C, or only scaling) to see its effect here.")

    over = [i + 1 for i, r in enumerate(rows) if r["train_acc"] - r["accuracy"] > 0.05]
    if over:
        out.append("Possible overfitting (train accuracy more than 5 points above test accuracy) in observation(s): "
                   + ", ".join(f"#{i}" for i in over) + ".")
    nonc = [i + 1 for i, r in enumerate(rows) if not r["converged"]]
    if nonc:
        out.append("The solver hit the iteration limit (not converged) in observation(s): "
                   + ", ".join(f"#{i}" for i in nonc) + " — increase max iterations or enable scaling.")
    return out
