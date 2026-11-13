import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NOTEBOOK = (
    Path.home()
    / "Downloads"
    / "rice_biophysical_project_complete"
    / "final-revised-thesis-code.ipynb"
)

OUTPUT = (
    ROOT / "outputs" / "rice" / "reports" / "phase10" / "original_notebook_audit.txt"
)

KEYWORDS = [
    "Farm_ID",
    "farm_id",
    "Ahero",
    "merge(",
    "dropna",
    "drop_duplicates",
    "groupby",
    "LAI",
    "zonal",
    "geometry",
    "raster",
    "mask",
    "nan",
    "isna",
    "notna",
    "inner",
    "left",
    "concat",
    "filter",
    "super_resolved",
]


def main():

    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"Original notebook not found: {NOTEBOOK}")

    with open(NOTEBOOK, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    lines = []

    for index, cell in enumerate(notebook["cells"]):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if cell_type != "code":
            continue

        source_lines = source.splitlines()

        matched = [
            i
            for i, line in enumerate(source_lines)
            if any(keyword.lower() in line.lower() for keyword in KEYWORDS)
        ]

        if not matched:
            continue

        lines.append("\n" + "=" * 80)
        lines.append(f"CELL {index}")
        lines.append("=" * 80)

        included = set()

        for match in matched:
            for position in range(max(0, match - 3), min(len(source_lines), match + 5)):
                included.add(position)

        previous = -2

        for position in sorted(included):
            if position > previous + 1:
                lines.append("...")

            lines.append(f"{position + 1:04d}: {source_lines[position]}")

            previous = position

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    print("NOTEBOOK AUDIT COMPLETED")
    print("Total cells:", len(notebook["cells"]))
    print("Relevant code sections:", len(lines))
    print("Saved to:", OUTPUT)

    print("\nFirst 250 lines:\n")

    for line in lines[:250]:
        print(line)


if __name__ == "__main__":
    main()
