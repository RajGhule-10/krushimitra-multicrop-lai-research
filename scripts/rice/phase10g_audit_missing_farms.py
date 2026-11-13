from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "rice" / "raw"
PROCESSED_DIR = ROOT / "data" / "rice" / "processed"

MISSING = {
    1: [
        "AR020",
        "AR021",
        "AR022",
        "AR025",
        "AR026",
        "AR028",
        "AR030",
        "AR031",
        "AR041",
    ],
    2: [
        "AR001",
        "AR002",
        "AR009",
        "AR030",
        "AR031",
        "AR032",
        "AR038",
        "AR052",
        "AR061",
        "AR065",
        "AR066",
        "AR067",
        "AR069",
        "AR072",
    ],
    3: [],
}


def normalize_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()
    match = re.match(r"^(AR)0*(\d+)$", value)

    if match:
        return f"AR{int(match.group(2)):03d}"

    return value


def main():

    print("=" * 70)
    print("PHASE 10G: MISSING FARM / CHLOROPHYLL AUDIT")
    print("=" * 70)

    raw_files = list(RAW_DIR.rglob("*.csv"))

    print("\nRaw CSV files found:", len(raw_files))

    for path in raw_files:
        try:
            df = pd.read_csv(path, nrows=5)
            cols = [str(c).strip() for c in df.columns]

            farm_cols = [
                c for c in cols if c.lower() in ("farm_id", "farm id", "parcel id")
            ]

            chl_cols = [c for c in cols if c.lower() in ("chl", "chlorophyll")]

            if farm_cols or chl_cols:
                print(
                    f"\n{path.name}\n"
                    f"  Farm columns: {farm_cols}\n"
                    f"  Chl columns: {chl_cols}"
                )

        except Exception as exc:
            print(f"Could not inspect {path.name}: {exc}")

    for campaign, missing_ids in MISSING.items():
        if not missing_ids:
            continue

        print("\n" + "-" * 70)
        print(f"CAMPAIGN {campaign}")
        print("-" * 70)

        candidates = []

        for path in raw_files:
            try:
                df = pd.read_csv(path)

                farm_col = next(
                    (
                        c
                        for c in df.columns
                        if str(c).strip().lower() in ("farm_id", "farm id", "parcel id")
                    ),
                    None,
                )

                chl_col = next(
                    (
                        c
                        for c in df.columns
                        if str(c).strip().lower() in ("chl", "chlorophyll")
                    ),
                    None,
                )

                if farm_col and chl_col:
                    candidates.append((path, df, farm_col, chl_col))

            except Exception:
                continue

        if not candidates:
            print("No raw CSV containing both Farm_ID and Chl found.")
            continue

        for path, df, farm_col, chl_col in candidates:
            df["_farm_norm"] = df[farm_col].apply(normalize_id)

            print("\nCandidate file:", path.name)

            for farm in missing_ids:
                rows = df[df["_farm_norm"] == farm]

                if rows.empty:
                    print(f"{farm}: NO CHL RECORD")
                else:
                    values = pd.to_numeric(rows[chl_col], errors="coerce")

                    print(
                        f"{farm}: {len(rows)} row(s), "
                        f"valid Chl={values.notna().sum()}, "
                        f"missing Chl={values.isna().sum()}"
                    )

    print("\n" + "=" * 70)
    print("PHASE 10G COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
