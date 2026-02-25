import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch
from src.utils.helpers import clean_text

FILE_PATH = r"data\dataset.csv"

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

required = {"text", "label", "label_id"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")
print("\n✅ Required columns found")

print("\nMissing values:")
print(df.isnull().sum())

df["text"] = df["text"].map(clean_text)
empty_text_count = (df["text"].str.len() == 0).sum()
print(f"\nEmpty text rows: {empty_text_count}")

duplicates = df.duplicated(subset=["text"]).sum()
print(f"Duplicate text rows: {duplicates}")

try:
    df["label_id"] = df["label_id"].astype(int)
    print("\n✅ label_id is integer")
except Exception as e:
    print("\n❌ label_id is NOT integer:", e)

print("\nLabel Distribution (label):")
print(df["label"].value_counts())

df["text_length"] = df["text"].apply(lambda x: len(x.split()))
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