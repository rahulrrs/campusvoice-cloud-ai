import os
import sys
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to import clean_text, but don't fail if src isn't available
try:
    from src.utils.helpers import clean_text
except Exception:
    clean_text = None

# ✅ Use cleaned dataset
FILE_PATH = r"data\dataset_clean.csv"

print("Loading dataset...")

if FILE_PATH.endswith((".xlsx", ".xls")):
    df = pd.read_excel(FILE_PATH)
else:
    df = pd.read_csv(FILE_PATH, low_memory=False)

print("✅ Dataset Loaded Successfully\n")

print("Columns:", df.columns.tolist())
print("Total Rows:", len(df))
print("\nFirst 3 rows:")
print(df.head(3))

# ✅ Updated required columns
required = {"text", "label", "label_id", "priority_id_fixed"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")
print("\n✅ Required columns found")

print("\nMissing values:")
print(df.isnull().sum())

# Clean text (optional)
df["text"] = df["text"].astype(str)
if clean_text is not None:
    df["text"] = df["text"].map(clean_text)
else:
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

empty_text_count = (df["text"].str.len() == 0).sum()
print(f"\nEmpty text rows: {empty_text_count}")

duplicates = df.duplicated(subset=["text"]).sum()
print(f"Duplicate text rows: {duplicates}")

# Validate label_id int
try:
    df["label_id"] = pd.to_numeric(df["label_id"], errors="raise").astype(int)
    print("\n✅ label_id is integer")
except Exception as e:
    print("\n❌ label_id is NOT integer:", e)

# ✅ Validate priority_id_fixed int and values
try:
    df["priority_id_fixed"] = pd.to_numeric(df["priority_id_fixed"], errors="raise").astype(int)
    bad = (~df["priority_id_fixed"].isin([0, 1, 2])).sum()
    if bad == 0:
        print("✅ priority_id_fixed is valid (0/1/2)")
    else:
        print(f"⚠️ priority_id_fixed has invalid values (not 0/1/2): {bad}")
except Exception as e:
    print("\n❌ priority_id_fixed is NOT integer:", e)

print("\nLabel Distribution (label):")
print(df["label"].value_counts())

print("\nPriority Distribution (priority_id_fixed):")
print(df["priority_id_fixed"].value_counts().sort_index())

# Text length stats
df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))
print("\nText Length Stats:")
print(df["text_length"].describe())

print("\nLongest Text Sample (first 500 chars):")
print(df.sort_values("text_length", ascending=False)["text"].iloc[0][:500])

print("\n===============================")
print("GPU CHECK")
print("===============================")
if torch.cuda.is_available():
    print("✅ GPU Available:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU NOT Available")

print("\nSanity Check Completed Successfully 🚀")