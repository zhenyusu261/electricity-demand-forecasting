from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    PROJECT_ROOT / "src" / "CNN-ATT.py",
    PROJECT_ROOT / "src" / "X13-CNN-ATT.py",
    PROJECT_ROOT / "src" / "HP-CNN-ATT.py",
    PROJECT_ROOT / "src" / "X13-HP-CNN-ATT.py",
    PROJECT_ROOT / "src" / "X13-HP-CNN-B-ATT.py",
    PROJECT_ROOT / "src" / "X13-HP-CNN-C-ATT.py",
    PROJECT_ROOT / "src" / "X13-HP-CNN-D-ATT.py",
    PROJECT_ROOT / "src" / "X13-HP-CNN-E-ATT.py",
    PROJECT_ROOT / "src" / "X13-HP-BiLSTM-ATT.py",
    PROJECT_ROOT / "src" / "X13-HP-LSTM-ATT.py",
    PROJECT_ROOT / "src" / "X13-HP-RNN-ATT.py",
    PROJECT_ROOT / "src" / "X13-HP-CNN-BiLSTM.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n=== Running {script.name} ===", flush=True)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
