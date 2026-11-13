from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SEARCH_DIRS = [
    Path.home() / "Downloads" / "rice_biophysical_project_complete",
    ROOT / "data" / "rice",
]

ZIP_FILES = [
    Path.home()
    / "Downloads"
    / "rice_biophysical_project_complete"
    / "satellite_imagery.zip",
    Path.home()
    / "Downloads"
    / "rice_biophysical_project_complete"
    / "Parcels Geodatabase_updated.zip",
]

KEYWORDS = ["dualex", "chlorophyll", "chl", "campaign"]


def inspect_csv(path):
    try:
        df = pd.read_csv(path, nrows=3)
        cols = df.columns.tolist()

        relevant = [
            c
            for c in cols
            if any(k in str(c).lower() for k in ["farm", "chl", "chlor"])
        ]

        print("CSV:", path)
        print("Columns:", cols)
        print("Relevant columns:", relevant)

    except Exception as e:
        print("CSV read error:", path, e)


def main():

    print("=" * 75)
    print("PHASE 10I: FIND ORIGINAL DUALEX DATA")
    print("=" * 75)

    seen = set()

    for folder in SEARCH_DIRS:
        print("\nSEARCH DIRECTORY:", folder)

        if not folder.exists():
            print("Directory not found.")
            continue

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            name = path.name.lower()

            if not any(k in name for k in KEYWORDS):
                continue

            print("\nMATCH:", path)

            if path.suffix.lower() == ".csv":
                inspect_csv(path)

            seen.add(str(path))

    for archive in ZIP_FILES:
        print("\nZIP ARCHIVE:", archive)

        if not archive.exists():
            print("Archive not found.")
            continue

        try:
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if not any(k in name.lower() for k in KEYWORDS):
                        continue

                    print("ZIP MATCH:", name)

                    if name.lower().endswith(".csv"):
                        try:
                            with z.open(name) as f:
                                df = pd.read_csv(f, nrows=3)

                            print("Columns:", df.columns.tolist())

                        except Exception as e:
                            print("Could not read CSV:", e)

        except Exception as e:
            print("ZIP error:", e)

    print("\n" + "=" * 75)
    print("PHASE 10I COMPLETED")
    print("Matching filesystem files:", len(seen))
    print("=" * 75)


if __name__ == "__main__":
    main()
