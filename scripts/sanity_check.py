import pandas as pd
import torch

# ===============================
# 🔹 CHANGE FILE NAME HERE
# ===============================
FILE_PATH = r"data\dataset.csv"
# Example:
# FILE_PATH = r"D:\Dev\programming\dataset\finalDataset0.2.xlsx"

print("Loading dataset...")

# Load file
if FILE_PATH.endswith(".xlsx") or FILE_PATH.endswith(".xls"):
    df = pd.read_excel(FILE_PATH)
else:
    df = pd.read_csv(FILE_PATH)

print("✅ Dataset Loaded Successfully\n")

# ===============================
# BASIC INFO
# ===============================
print("Columns:", df.columns.tolist())
print("Total Rows:", len(df))
print("\nFirst 3 rows:")
print(df.head(3))

# ===============================
# REQUIRED COLUMNS CHECK
# ===============================
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("❌ Dataset must contain 'text' and 'label' columns")
else:
    print("\n✅ Required columns found")

# ===============================
# MISSING VALUES CHECK
# ===============================
print("\nMissing values:")
print(df.isnull().sum())

# ===============================
# REMOVE EMPTY TEXT
# ===============================
df["text"] = df["text"].astype(str).str.strip()
empty_text_count = (df["text"].str.len() == 0).sum()
print(f"\nEmpty text rows: {empty_text_count}")

# ===============================
# DUPLICATE CHECK
# ===============================
duplicates = df.duplicated(subset=["text"]).sum()
print(f"Duplicate text rows: {duplicates}")

# ===============================
# LABEL CHECK
# ===============================
try:
    df["label"] = df["label"].astype(int)
    print("\n✅ Labels are integers")
except:
    print("\n❌ Labels are NOT integers")

print("\nLabel Distribution:")
print(df["label"].value_counts())

# ===============================
# TEXT LENGTH CHECK
# ===============================
df["text_length"] = df["text"].apply(lambda x: len(x.split()))

print("\nText Length Stats:")
print(df["text_length"].describe())

print("\nLongest Text Sample:")
print(df.sort_values("text_length", ascending=False)["text"].iloc[0][:500])

# ===============================
# GPU CHECK
# ===============================
print("\n===============================")
print("GPU CHECK")
print("===============================")

if torch.cuda.is_available():
    print("✅ GPU Available:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU NOT Available")

print("\nSanity Check Completed Successfully 🚀")
