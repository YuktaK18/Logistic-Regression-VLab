# Logistic Regression VLab

A Virtual Lab running on localhost. **Python + Streamlit + scikit-learn + NumPy + matplotlib.**
All tables and graphs are rendered as HTML/PNG, so the app code never calls `st.dataframe` / `pandas`.

## Main parts of the lab
| Requirement | Where |
|---|---|
| **Theory** | *Aim & Theory* and *Procedure* pages |
| **Dataset selection** | *Dataset Explorer* page and the *Dataset* box in *Run Experiment* (Iris, Breast Cancer, Wine) |
| **Model selection** | *Model* box: Logistic Regression L2 / L1 / Elastic Net / none, plus KNN, Decision Tree, SVM for comparison |
| **Result section** | *Results* tab (accuracy, precision, recall, F1, AUC, classification report), *Predictions* tab (+ CSV), *Conclusion* tab |
| **Graph & visualization section** | *Graphs & Visualization* tab: confusion matrix, ROC, coefficients, effect of C, decision regions, model comparison |
| **Observations** | *Observations* page: every run is logged automatically (table, chart, findings, CSV download) |

## Requirements
Windows 10/11 · Python 3.10–3.13 (64-bit, from python.org, "Add python.exe to PATH") · internet for the first install.

## Check Python first
```powershell
python --version
python -c "import sys, struct; print(sys.executable); print(struct.calcsize('P')*8, 'bit')"
python -m pip --version
```

## Run – PowerShell
```powershell
cd "C:\Users\<you>\Documents\Logistic_Regression_VLab"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_env.py
python -m streamlit run app.py
```
Browser opens at **http://localhost:8501**. Stop with `Ctrl + C`.
(If Streamlit asks for an e-mail, press Enter. If scripts are blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.)

Or double-click **RUN_VLAB.bat** – it does all of this, and if pandas is blocked it starts the backup version automatically.

## pandas DLL blocked ("Application Control policy has blocked this file")
Streamlit itself imports pandas, so it can only run if `python -c "import pandas"` works. Try in order:
1. See what is blocked: `python -c "import pandas"`.
2. Use **Python from python.org (64-bit, 3.11 or 3.12)**, not the Microsoft Store Python; keep the project and venv in a normal user folder, not Downloads/Desktop/OneDrive.
3. `pip install --force-reinstall --no-cache-dir numpy pandas`; install *Microsoft Visual C++ Redistributable 2015-2022 (x64)* if the error says "DLL load failed".
4. If an **Application Control policy** blocks it: on Windows 11 this is often *Smart App Control* (Windows Security → App & browser control); on a college-managed laptop it is an IT policy – ask IT to allow the Python folder.
5. Run the lab on a college lab PC or another laptop and take screenshots there.
6. **Backup (no pandas, no Streamlit):** `python flask_backup\app_flask.py` → http://127.0.0.1:5000 (same pages, models, results, graphs, observations).

## Project structure
```
Logistic_Regression_VLab/
├─ app.py                  Streamlit app (UI + navigation)   ← main project
├─ lab/ml_core.py          ML code: datasets, models, preprocessing, metrics, plots, conclusion, observation log
├─ observations_log.json   created automatically when you run experiments
├─ .streamlit/config.toml  Theme
├─ check_env.py            Checks libraries
├─ RUN_VLAB.bat            One-click start (Windows)
├─ requirements.txt
├─ tests/test_streamlit.py Headless test of every page, dataset and model
└─ flask_backup/           Backup version without Streamlit/pandas
```
Self-test: `python tests\test_streamlit.py`
