from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

file_path = ROOT / "data/cassava/raw/Ground-LAI-Clean.csv"

df = pd.read_csv(file_path)

# Normalize plot IDs
df["Plot_ID"] = (
    df["Plot"]
    .astype(str)
    .str.strip()
    .str.upper()
    .str.replace(r"([A-Z]+)0+(\d+)", r"\1\2", regex=True)
)

df["MAP"] = pd.to_numeric(df["MAP"], errors="coerce")

print("\nCASSAVA LAI MEASUREMENT-TYPE CHECK")
print("=" * 55)

print("\n1. Measurement types:")
print(df["Plot_type"].value_counts(dropna=False))

print("\n2. Measurement types by MAP:")
print(pd.crosstab(df["MAP"], df["Plot_type"], dropna=False))

# Check whether a plot-MAP group contains multiple types
type_counts = df.groupby(["Plot_ID", "MAP"])["Plot_type"].nunique(dropna=False)

mixed_groups = type_counts[type_counts > 1]

print("\n3. Plot-MAP groups containing multiple measurement types:")
print("Count:", len(mixed_groups))

if len(mixed_groups):
    print(mixed_groups.to_string())
else:
    print("None")

# Check whether plot-MAP-type combinations are repeated
group_counts = df.groupby(["Plot_ID", "MAP", "Plot_type"]).size()

print("\n4. Number of LAI samples per plot-MAP-type:")
print(group_counts.describe())

print("\n5. Missing measurement types:")
print(df["Plot_type"].isna().sum())

print("\nCheck completed.")
