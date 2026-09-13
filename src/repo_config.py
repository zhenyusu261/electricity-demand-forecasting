from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(
    os.environ.get("CHANGZHOU_DATA", PROJECT_ROOT / "data" / "changzhou.csv")
)
X13_PATH = os.environ.get("X13AS_PATH", "x13as")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
MODEL_DIR = OUTPUT_DIR / "models"
TUNER_DIR = OUTPUT_DIR / "tuners"
