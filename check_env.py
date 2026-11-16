"""Checks that every library loads. Exit code: 0 = all OK, 2 = only Streamlit/pandas broken, 1 = core broken."""
import sys
from importlib.metadata import PackageNotFoundError, version

CORE = ["numpy", "scipy", "sklearn", "matplotlib", "flask"]
STREAMLIT_STACK = ["pandas", "streamlit"]   # streamlit imports pandas internally


def check(name):
    try:
        __import__(name)
        try:
            ver = version("scikit-learn" if name == "sklearn" else name)
        except PackageNotFoundError:
            ver = ""
        print(f"  [OK]   {name:<11} {ver}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {name:<11} {type(exc).__name__}: {str(exc)[:160]}")
        return False


core_ok = all([check(n) for n in CORE])
st_ok = all([check(n) for n in STREAMLIT_STACK])

if core_ok and st_ok:
    print("\n  Environment OK - Streamlit lab will start.")
    sys.exit(0)
if core_ok and not st_ok:
    print("\n  Streamlit cannot start because pandas/streamlit failed to load (see README -> 'pandas DLL blocked').")
    print("  NumPy + scikit-learn work, so the BACKUP version can still run:  python flask_backup\\app_flask.py")
    sys.exit(2)
print("\n  A core library failed to load. See README.md -> Troubleshooting.")
sys.exit(1)
