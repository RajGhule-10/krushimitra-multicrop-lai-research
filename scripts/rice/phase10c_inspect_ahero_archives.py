from pathlib import Path
from zipfile import ZipFile, BadZipFile

ROOT = Path.home() / "Downloads" / "rice_biophysical_project_complete"

ARCHIVES = {
    "SATELLITE IMAGERY": ROOT / "satellite_imagery.zip",
    "PARCEL GEODATABASE": ROOT / "Parcels Geodatabase_updated.zip",
}

KEY_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".gpkg",
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".kml",
    ".geojson",
    ".csv",
}


def inspect_archive(label, archive_path):

    print("\n" + "=" * 75)
    print(label)
    print("=" * 75)

    if not archive_path.exists():
        print(f"ERROR: Archive not found: {archive_path}")
        return

    print(f"Archive: {archive_path}")
    print(f"Size: {archive_path.stat().st_size / (1024**2):.2f} MB")

    try:
        with ZipFile(archive_path, "r") as z:
            files = [item for item in z.infolist() if not item.is_dir()]

            print(f"Total files: {len(files)}")

            print("\nRelevant files:")

            for item in files:
                path = Path(item.filename)

                if path.suffix.lower() in KEY_EXTENSIONS:
                    print(f"{item.filename} ({item.file_size / (1024**2):.2f} MB)")

            print("\nArchive integrity test:")
            bad_file = z.testzip()

            if bad_file:
                print(f"FAILED: {bad_file}")
            else:
                print("PASS: No corrupted archive members detected")

    except BadZipFile:
        print("ERROR: Invalid ZIP archive")


def main():
    for label, path in ARCHIVES.items():
        inspect_archive(label, path)

    print("\nARCHIVE INSPECTION COMPLETED")


if __name__ == "__main__":
    main()
