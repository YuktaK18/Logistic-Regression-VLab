"""
Logistic Regression VLab  (Streamlit app)
Run with:   python -m streamlit run app.py
All ML code lives in lab/ml_core.py (NumPy + scikit-learn + matplotlib).
Tables/graphs are drawn as HTML/PNG so this app never calls st.dataframe / st.pyplot.
"""
import html

import streamlit as st

from lab import ml_core as core

st.set_page_config(page_title="Logistic Regression VLab", page_icon="🧪",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container{padding-top:3.6rem;max-width:1250px}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e1b4b,#312e81)}
section[data-testid="stSidebar"] *{color:#e0e7ff !important}
section[data-testid="stSidebar"] label p{font-size:16px}
.hero{background:linear-gradient(120deg,#4f46e5,#7c3aed);color:#fff;border-radius:14px;padding:20px 28px;margin-bottom:18px;box-shadow:0 10px 28px rgba(79,70,229,.25)}
.hero p{margin:0;font-size:12px;letter-spacing:.12em;text-transform:uppercase;opacity:.85;color:#fff}
.hero h1{margin:2px 0 0;font-size:28px;color:#fff;padding:0}
.brand{font-size:19px;font-weight:700;color:#fff;margin-bottom:2px}
.pill{display:inline-block;padding:3px 12px;border-radius:20px;background:#e0e7ff;color:#4f46e5;font-size:13px;font-weight:600;margin:2px}
.pill.red{background:#fee2e2;color:#b91c1c}
.steps{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 18px}
.step{flex:1 1 170px;background:#fff;border:1px solid #e5e7eb;border-top:4px solid #4f46e5;border-radius:12px;padding:10px 14px;color:#1e1b4b}
.step b{color:#4f46e5}.step p{margin:3px 0 0;font-size:13px;color:#4b5563}
.explain{background:#f5f3ff;border-radius:10px;padding:10px 14px;font-size:14px;color:#4c1d95;margin:4px 0 12px}
.tbl{overflow:auto;max-height:430px;border:1px solid #e5e7eb;border-radius:10px;background:#fff}
.tbl table{border-collapse:collapse;width:100%;font-size:14px;color:#1e1b4b}
.tbl th{background:#eef2ff;color:#4f46e5;text-align:left;padding:8px 12px;position:sticky;top:0}
.tbl td{padding:7px 12px;border-bottom:1px solid #eef0f4;white-space:nowrap}
.tbl .num{text-align:right;font-variant-numeric:tabular-nums}
.tbl tr.wrong td{background:#fef2f2}
img.plot{max-width:100%;height:auto;border-radius:10px;border:1px solid #e5e7eb;background:#fff;display:block;margin:auto}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#eef2ff,#fff);border:1px solid #c7d2fe;border-radius:12px;padding:10px 14px}
div[data-testid="stMetricValue"]{color:#4f46e5}
</style>
""", unsafe_allow_html=True)

PAGES = ["📘 Aim & Theory", "🧭 Procedure", "🗂️ Dataset Explorer", "🧪 Run Experiment", "📒 Observations"]


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def hero(title):
    st.markdown(f'<div class="hero"><p>Logistic Regression VLab</p><h1>{title}</h1></div>',
                unsafe_allow_html=True)


def explain(text):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def show_img(b64, max_width=None):
    style = f' style="max-width:{max_width}"' if max_width else ""
    st.markdown(f'<img class="plot"{style} src="data:image/png;base64,{b64}">', unsafe_allow_html=True)


def table(headers, rows, num_cols=(), row_classes=None):
    """rows = list of lists of plain values (escaped automatically)."""
    h = "".join(f'<th class="{"num" if i in num_cols else ""}">{html.escape(str(x))}</th>'
                for i, x in enumerate(headers))
    body = ""
    for r_i, r in enumerate(rows):
        cls = row_classes[r_i] if row_classes else ""
        tds = "".join(f'<td class="{"num" if i in num_cols else ""}">{html.escape(str(v))}</td>'
                      for i, v in enumerate(r))
        body += f'<tr class="{cls}">{tds}</tr>'
    st.markdown(f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>',
                unsafe_allow_html=True)


def pills(items, cls=""):
    st.markdown("".join(f'<span class="pill {cls}">{html.escape(str(i))}</span>' for i in items),
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_sigmoid():
    return core.sigmoid_plot()


@st.cache_data(show_spinner=False)
def cached_dataset_info(key):
    return core.dataset_info(key)


# ----------------------------------------------------------------------------
# Page 1 - Theory
# ----------------------------------------------------------------------------
def page_theory():
    hero("Aim & Theory")
    with st.container(border=True):
        st.subheader("🎯 Aim")
        st.write("To implement **Logistic Regression** for classification, train it on a real dataset and evaluate "
                 "its performance using accuracy, precision, recall, F1-score, confusion matrix and the ROC curve.")
        st.markdown("**Objectives**")
        st.markdown("- Understand how a linear model + sigmoid function produces class probabilities.\n"
                    "- Split data into training and testing sets and apply feature scaling.\n"
                    "- Study the effect of the regularisation parameter **C**, the penalty type (L1 / L2 / Elastic Net) "
                    "and the number of iterations.\n"
                    "- Compare Logistic Regression with other models and interpret evaluation metrics and graphs "
                    "for binary and multiclass problems.")
    with st.container(border=True):
        st.subheader("📖 Introduction")
        st.write("Despite its name, Logistic Regression is a **classification** algorithm. It predicts the "
                 "*probability* that a sample belongs to a class. If that probability is above a threshold "
                 "(normally 0.5) the sample is assigned to that class. It is a **supervised** learning method — it "
                 "learns from labelled examples — and is one of the most widely used baseline models in machine learning.")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("🧮 Mathematical Explanation")
            st.markdown("**1. Linear score** (w = weights, b = bias)")
            st.latex(r"z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = \mathbf{w}\cdot\mathbf{x} + b")
            st.markdown("**2. Sigmoid (logistic) function** – squashes any number into (0, 1)")
            st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
            st.markdown("**3. Probability and decision** – predict class 1 if P ≥ 0.5")
            st.latex(r"P(y=1\mid \mathbf{x}) = \sigma(\mathbf{w}\cdot\mathbf{x} + b)")
            st.markdown("**4. Log-odds (logit)** – the model is linear in the log-odds")
            st.latex(r"\ln\!\left(\frac{p}{1-p}\right) = \mathbf{w}\cdot\mathbf{x} + b")
            st.markdown("**5. Loss function** – log-loss with L2 regularisation (C is the *inverse* strength)")
            st.latex(r"J(\mathbf{w}) = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\ln p_i + (1-y_i)\ln(1-p_i)\right] + \frac{1}{2C}\lVert \mathbf{w}\rVert^2")
            st.markdown("**6. Multiclass (softmax)** – one score per class, probabilities sum to 1")
            st.latex(r"P(y=k\mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j} e^{z_j}}")
            st.markdown("**7. Regularisation types** – L2 shrinks weights, L1 can make weights exactly zero "
                        "(feature selection), Elastic Net mixes both.")
    with right:
        with st.container(border=True):
            st.subheader("📈 The Sigmoid Curve")
            show_img(cached_sigmoid())
            explain("When z = 0 the probability is exactly 0.5. Large positive z → probability near 1; large "
                    "negative z → near 0. This turns a straight-line score into a probability.")
        with st.container(border=True):
            st.subheader("⚙️ Working Principle")
            st.markdown("1. Start with zero/random weights.\n2. Compute z and the probability p for every training sample.\n"
                        "3. Measure the error with log-loss.\n4. Update the weights to reduce the loss (optimiser such as L-BFGS).\n"
                        "5. Repeat until the loss stops improving (convergence) or max iterations is reached.\n"
                        "6. Use the final weights to predict probabilities for new data.")

    with st.container(border=True):
        st.subheader("📝 Algorithm (step-by-step)")
        st.markdown("1. **Load** the dataset (features X and labels y).\n"
                    "2. **Split** into training and testing sets (stratified).\n"
                    "3. **Standardise** features: x' = (x − mean) / std, using *training* statistics only.\n"
                    "4. **Select** the model and set C, solver and max_iter.\n"
                    "5. **Train** (fit) the model on the training data by minimising log-loss.\n"
                    "6. **Predict** labels and probabilities for the test set.\n"
                    "7. **Evaluate** with accuracy, precision, recall, F1, confusion matrix and ROC-AUC.\n"
                    "8. **Tune / compare** C, penalty and models; write the conclusion.")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.subheader("👍 Advantages")
        st.markdown("- Simple, fast, easy to implement\n- Gives probabilities, not just labels\n- Interpretable coefficients\n"
                    "- Works well on linearly separable data\n- Regularisation reduces overfitting")
    with c2.container(border=True):
        st.subheader("👎 Limitations")
        st.markdown("- Assumes a linear decision boundary\n- Sensitive to outliers and unscaled features\n"
                    "- Struggles with complex non-linear patterns\n- Needs reasonably independent features")
    with c3.container(border=True):
        st.subheader("🌍 Applications")
        st.markdown("- Disease diagnosis (benign / malignant)\n- Spam e-mail detection\n- Credit-risk and fraud detection\n"
                    "- Customer churn prediction\n- Click-through-rate prediction")


# ----------------------------------------------------------------------------
# Page 2 - Procedure
# ----------------------------------------------------------------------------
def page_procedure():
    hero("Experiment Procedure")
    st.markdown("""
<div class="steps">
<div class="step"><b>1 · Dataset</b><p>Choose Iris, Breast Cancer or Wine and inspect it.</p></div>
<div class="step"><b>2 · Preprocessing</b><p>Stratified train/test split + feature standardisation.</p></div>
<div class="step"><b>3 · Training</b><p>Fit the selected model by minimising the loss.</p></div>
<div class="step"><b>4 · Testing</b><p>Predict labels &amp; probabilities on unseen data.</p></div>
<div class="step"><b>5 · Evaluation</b><p>Metrics, confusion matrix, ROC and conclusion.</p></div>
</div>""", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("🧭 Procedure")
        st.markdown(
            "1. Open **Dataset Explorer**, select a dataset and study its rows, columns, features, target and class balance.\n"
            "2. Open **Run Experiment** and select the dataset.\n"
            "3. Select a **model**: a Logistic Regression variant (L2, L1, Elastic Net, none) or a comparison model (KNN, Decision Tree, SVM).\n"
            "4. Set the **test size** (e.g. 25%). A larger test set gives a more reliable estimate but leaves less data for training.\n"
            "5. Set **C** (default 1). Small C = strong regularisation (simpler model). Large C = weak regularisation.\n"
            "6. Set **max iterations** (default 200), the solver and keep **standardisation** ticked.\n"
            "7. Click **Run Experiment**. A real scikit-learn model is trained and evaluated.\n"
            "8. Study the **Results**, **Graphs**, **Predictions** and **Conclusion** tabs.\n"
            "9. Repeat with different settings (change **one** setting at a time). Every run is **recorded automatically** on the **Observations** page, together with findings generated from your actual results.")
    a, b = st.columns(2)
    with a.container(border=True):
        st.subheader("🔍 What each stage means")
        st.markdown("**Dataset** – rows are samples, columns are features, the target is what we predict.")
        st.markdown("**Preprocessing** – features have different scales; standardising puts them on the same scale so the "
                    "optimiser converges faster and C treats every feature fairly.")
        st.markdown("**Training** – the model searches for weights that minimise the loss on the training set.")
        st.markdown("**Testing** – predictions on data the model never saw show how it behaves in the real world.")
        st.markdown("**Evaluation** – metrics quantify the quality; plots show *where* the model makes mistakes.")
    with b.container(border=True):
        st.subheader("📏 Metrics at a glance")
        st.latex(r"\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}")
        st.latex(r"\text{Precision} = \frac{TP}{TP+FP}\qquad \text{Recall} = \frac{TP}{TP+FN}")
        st.latex(r"F_1 = \frac{2\cdot \text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}}")
        st.write("**ROC curve** plots True-Positive-Rate vs False-Positive-Rate at every threshold. "
                 "**AUC** = area under it (1.0 perfect, 0.5 random).")


# ----------------------------------------------------------------------------
# Page 3 - Dataset
# ----------------------------------------------------------------------------
def _use_in_experiment():
    st.session_state["exp_dataset"] = st.session_state["ds_page"]
    st.session_state["nav"] = PAGES[3]


def page_dataset():
    hero("Dataset Explorer")
    with st.container(border=True):
        st.subheader("🗂️ Select Dataset")
        c1, c2 = st.columns([3, 2])
        key = c1.selectbox("Dataset", list(core.DATASETS), key="ds_page",
                           format_func=lambda k: core.DATASETS[k]["label"])
        c2.write("")
        c2.write("")
        c2.button("Use this dataset in the experiment →", on_click=_use_in_experiment)
    info = cached_dataset_info(key)
    with st.container(border=True):
        st.subheader("ℹ️ " + info["label"])
        st.write(info["about"])
        m = st.columns(4)
        m[0].metric("Rows (samples)", info["n_samples"])
        m[1].metric("Columns (features)", info["n_features"])
        m[2].metric("Target classes", info["n_classes"])
        m[3].metric("Problem type", info["kind"].split(" (")[0])
        st.markdown("**Target classes**")
        pills([f"{i} = {n} ({c})" for i, (n, c) in enumerate(info["classes"])])
        st.markdown("**Features**")
        pills(info["features"])
    a, b = st.columns(2)
    with a.container(border=True):
        st.subheader("📊 Class balance")
        show_img(info["dist_img"])
    with b.container(border=True):
        st.subheader("🔵 Feature scatter")
        show_img(info["scatter_img"])
    with st.container(border=True):
        st.subheader("👀 Dataset preview (first 10 rows)")
        rows = [[i] + [f"{v:.3f}" for v in r["vals"]] + [r["target"]] for i, r in enumerate(info["preview"])]
        table(["#"] + info["features"] + ["target"], rows, num_cols=set(range(1, info["n_features"] + 1)))
    with st.container(border=True):
        st.subheader("📐 Feature statistics")
        rows = [[s["name"], f"{s['min']:.3f}", f"{s['max']:.3f}", f"{s['mean']:.3f}", f"{s['std']:.3f}"]
                for s in info["stats"]]
        table(["Feature", "Min", "Max", "Mean", "Std"], rows, num_cols={1, 2, 3, 4})
        explain("Notice how different the feature ranges can be — this is why <b>standardisation</b> is important.")


# ----------------------------------------------------------------------------
# Page 4 - Experiment
# ----------------------------------------------------------------------------
def page_experiment():
    hero("Run Experiment")
    st.session_state.setdefault("exp_dataset", "iris")
    st.session_state.setdefault("exp_model", "lr_l2")

    with st.container(border=True):
        st.subheader("⚙️ Experiment Configuration")
        with st.form("exp_form"):
            c1, c2, c3 = st.columns(3)
            dataset = c1.selectbox("Dataset", list(core.DATASETS), key="exp_dataset",
                                   format_func=lambda k: core.DATASETS[k]["label"])
            model = c2.selectbox("Model", list(core.MODELS), key="exp_model",
                                 format_func=lambda k: core.MODELS[k]["label"])
            test_pct = c3.slider("Test size (%)", 10, 50, 25, 5, help="The rest of the data is used for training")
            c4, c5, c6, c7 = st.columns(4)
            C = c4.number_input("C (inverse regularisation)", min_value=0.0001, max_value=10000.0, value=1.0,
                                step=0.1, format="%g", help="Small = simpler model, large = fits training data closely")
            max_iter = c5.number_input("Max iterations", min_value=5, max_value=10000, value=200, step=50)
            solver = c6.selectbox("Solver (L2 / none)", ["lbfgs", "newton-cg"],
                                  help="L1 and Elastic Net automatically use the 'saga' solver")
            seed = c7.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1,
                                   help="Same seed = same split every time")
            scale = st.checkbox("Standardise features (StandardScaler)", value=True)
            run = st.form_submit_button("▶ Run Experiment", type="primary")
        with st.expander("ℹ️ About the available models"):
            for k, v in core.MODELS.items():
                st.markdown(f"**{v['label']}** — {v['about']}")

    if run:
        try:
            params = core.parse_params({"dataset": dataset, "model": model, "test_pct": test_pct, "C": C,
                                        "max_iter": int(max_iter), "solver": solver, "scale": scale,
                                        "seed": int(seed)})
            with st.spinner("Training the model and generating graphs…"):
                st.session_state["res"] = core.run_experiment(params)
            n, is_new = core.obs_add(st.session_state["res"])
            st.session_state["obs_msg"] = (f"📒 Recorded as observation #{n}." if is_new else
                                           f"📒 Same settings as observation #{n} — its entry was refreshed.")
        except ValueError as exc:
            st.error(f"⚠️ {exc}")
            st.session_state.pop("res", None)
            st.session_state.pop("obs_msg", None)
        except Exception as exc:  # noqa: BLE001
            st.error(f"⚠️ Unexpected error while running the experiment: {exc}")
            st.session_state.pop("res", None)

    res = st.session_state.get("res")
    if not res:
        st.info("👋 Choose your settings above and press **Run Experiment**. A real model will be trained and all "
                "results, graphs and the conclusion will appear here.")
        return
    show_results(res)


def show_results(res):
    p, m = res["params"], res["metrics"]
    if st.session_state.get("obs_msg"):
        st.success(st.session_state["obs_msg"] + " See the **Observations** page.")
    iters = (f"<br>Used {res['n_iter']} iteration(s) {'✓ converged' if res['converged'] else '⚠ not converged'}"
             if res["n_iter"] else "")
    cfg = f", C={p['C']:g}" if res["has_C"] else ""
    st.markdown(f"""
<div class="steps">
<div class="step"><b>1 · Dataset</b><p>{html.escape(res['dataset'])}<br>{res['n_train'] + res['n_test']} samples, {res['n_features']} features, {res['n_classes']} classes</p></div>
<div class="step"><b>2 · Preprocessing</b><p>Stratified split: {res['n_train']} train / {res['n_test']} test.<br>{'StandardScaler fitted on train data' if p['scale'] else 'No scaling applied'}</p></div>
<div class="step"><b>3 · Training</b><p>{html.escape(res['model_label'])}{cfg}.{iters}</p></div>
<div class="step"><b>4 · Testing</b><p>Predicted {res['n_test']} unseen samples.<br>{res['n_test'] - res['n_wrong']} correct, {res['n_wrong']} wrong</p></div>
<div class="step"><b>5 · Evaluation</b><p>Accuracy {m['accuracy'] * 100:.2f}%<br>AUC {m['auc']:.3f}</p></div>
</div>""", unsafe_allow_html=True)
    if not res["converged"]:
        st.warning("The solver hit the iteration limit and may not have fully converged. "
                   "Increase **max iterations** or enable standardisation.")

    t1, t2, t3, t4 = st.tabs(["📊 Results", "📈 Graphs & Visualization", "🔮 Predictions", "✅ Conclusion"])

    with t1:
        explain(f"Computed on the <b>test set</b> (data the model has never seen). {html.escape(res['avg_note'])}.")
        c = st.columns(6)
        c[0].metric("Test Accuracy", f"{m['accuracy'] * 100:.2f}%")
        c[1].metric("Precision", f"{m['precision']:.3f}")
        c[2].metric("Recall", f"{m['recall']:.3f}")
        c[3].metric("F1-score", f"{m['f1']:.3f}")
        c[4].metric("ROC-AUC", f"{m['auc']:.3f}")
        c[5].metric("Train Accuracy", f"{m['train_accuracy'] * 100:.2f}%")
        st.markdown("#### 📋 Classification Report")
        table(["Class", "Precision", "Recall", "F1-score", "Support"],
              [[r["name"], f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}", r["support"]]
               for r in res["report"]], num_cols={1, 2, 3, 4})
        st.markdown("#### 🟦 Confusion matrix (numbers)")
        table(["Actual \\ Predicted"] + res["classes"],
              [[res["classes"][i]] + row for i, row in enumerate(res["cm"])], num_cols=set(range(1, res["n_classes"] + 1)))

    with t2:
        a, b = st.columns(2)
        with a.container(border=True):
            st.markdown("#### 🟦 Confusion Matrix")
            explain("Rows = actual class, columns = predicted class. The diagonal shows correct predictions.")
            show_img(res["images"]["cm"])
        with b.container(border=True):
            st.markdown("#### 📉 ROC Curve & AUC")
            explain("A curve hugging the top-left corner means good class separation. The dashed line is random guessing.")
            show_img(res["images"]["roc"])
        a, b = st.columns(2)
        if res["images"]["coef"]:
            with a.container(border=True):
                st.markdown("#### 🔬 Learned Coefficients")
                extra = f" <b>{res['n_zero']}</b> coefficient(s) are exactly zero." if res["n_zero"] else ""
                explain("Weights show how strongly each feature pushes the prediction." + extra)
                show_img(res["images"]["coef"])
        if res["images"]["sweep"]:
            with b.container(border=True):
                st.markdown("#### 🎚️ Effect of C")
                explain(f"Best test accuracy in this sweep: <b>{res['best_C_acc'] * 100:.2f}%</b> at C ≈ <b>{res['best_C']:g}</b>.")
                show_img(res["images"]["sweep"])
        a, b = st.columns(2)
        with a.container(border=True):
            st.markdown("#### 🗺️ Decision Regions")
            explain("A model trained on only the 2 most influential features, so the boundary can be drawn in 2-D.")
            show_img(res["images"]["boundary"])
        with b.container(border=True):
            st.markdown("#### 🏁 Model Comparison")
            explain("All models trained on the same split — this helps you justify your model choice.")
            show_img(res["images"]["compare"])

    with t3:
        explain('Each row is one unseen sample. <span class="pill red">Red rows</span> are wrong predictions. '
                f'Showing the first 40 of {res["n_test"]} rows — the CSV has all of them (with feature values).')
        rows, cls = [], []
        for r in res["preds"][:40]:
            rows.append([r["sample"], r["actual"], r["predicted"], "✅" if r["ok"] else "❌",
                         f"{r['confidence'] * 100:.1f}%"] + [f"{v:.3f}" for v in r["probs"]])
            cls.append("" if r["ok"] else "wrong")
        table(["Sample #", "Actual", "Predicted", "Result", "Confidence"] + [f"P({c[:10]})" for c in res["classes"]],
              rows, num_cols=set(range(4, 5 + res["n_classes"])), row_classes=cls)
        st.write("")
        st.download_button("⬇ Download predictions (CSV)", core.predictions_csv(res),
                           file_name=f"predictions_{p['dataset']}_{p['model']}.csv", mime="text/csv")

    with t4:
        st.markdown("#### ✅ Conclusion (auto-generated from your results)")
        for line in res["conclusion"]:
            st.markdown(f"- {line}")


# ----------------------------------------------------------------------------
# Page 5 - Observations (actual results of the runs you performed)
# ----------------------------------------------------------------------------
def page_observations():
    hero("Observations")
    explain("Every time you press <b>Run Experiment</b>, the settings and results are recorded here automatically. "
            "These are your <b>actual observations</b> — nothing is typed by hand. The log is saved in "
            "<code>observations_log.json</code>, so it is still here after you restart the lab.")
    rows = core.obs_load()
    if not rows:
        st.info("No observations yet. Go to **Run Experiment**, run the model with a few different settings "
                "(e.g. change only C, or switch scaling off) and come back.")
        return

    best = max(rows, key=lambda r: (r["accuracy"], r["f1"]))
    m = st.columns(4)
    m[0].metric("Experiments run", len(rows))
    m[1].metric("Datasets used", len({r["dataset"] for r in rows}))
    m[2].metric("Models tried", len({r["model"] for r in rows}))
    m[3].metric("Best test accuracy", f"{best['accuracy'] * 100:.2f}%")

    with st.container(border=True):
        st.subheader("📒 Observation Table")
        table(core.OBS_HEADERS, core.obs_table_rows(rows), num_cols=set(range(3, 14)))
    with st.container(border=True):
        st.subheader("📊 Accuracy of every run")
        show_img(core.obs_chart(rows))
    with st.container(border=True):
        st.subheader("🔎 What the results show")
        for line in core.obs_insights(rows):
            st.markdown(f"- {line}")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("⬇ Download observations (CSV)", core.obs_csv(rows), file_name="observations.csv",
                       mime="text/csv")
    if c2.button("🗑️ Clear all observations"):
        core.obs_clear()
        st.rerun()


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">🧪 Logistic Regression VLab</div>', unsafe_allow_html=True)
    st.caption("Computer Engineering · VES")
    st.write("")
    page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    st.write("")
    st.markdown("---")
    st.markdown("**Logistic Regression VLab**  \nVirtual Lab on localhost  \nPython · scikit-learn · Streamlit")

{PAGES[0]: page_theory, PAGES[1]: page_procedure, PAGES[2]: page_dataset,
 PAGES[3]: page_experiment, PAGES[4]: page_observations}[page]()
