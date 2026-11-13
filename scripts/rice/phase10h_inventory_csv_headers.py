from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "rice" / "raw"

for path in sorted(RAW_DIR.rglob("*.csv")):
    print("\n" + "=" * 75)
    print("FILE:", path.relative_to(ROOT))

    try:
        df = pd.read_csv(path, nrows=3)
        print("COLUMNS:", df.columns.tolist())
        print("SAMPLE:")
        print(df.to_string(index=False))
    except Exception as e:
        print("READ ERROR:", e)
