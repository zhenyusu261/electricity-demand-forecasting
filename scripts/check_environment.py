from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "tensorflow",
    "keras-tuner",
]


def main():
    print(f"Python: {sys.version.split()[0]}")
    python_ok = sys.version_info[:2] in {(3, 10), (3, 11)}
    print(f"Python version supported: {python_ok}")

    for package in PACKAGES:
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: NOT INSTALLED")

    data_path = Path(
        os.environ.get("CHANGZHOU_DATA", PROJECT_ROOT / "data" / "changzhou.csv")
    )
    print(f"Data path: {data_path}")
    print(f"Data exists: {data_path.exists()}")

    x13_path = os.environ.get("X13AS_PATH", "")
    print(f"X13AS_PATH: {x13_path or 'NOT SET'}")
    print(f"X13AS exists: {bool(x13_path) and Path(x13_path).exists()}")


if __name__ == "__main__":
    main()
