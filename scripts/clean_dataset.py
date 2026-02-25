import os
import json
import pandas as pd
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.utils.helpers import clean_text

IN_PATH = r"data\dataset.csv"
OUT_PATH = r"data\dataset_clean.csv"
MAP_PATH = r"outputs\edu_classifier\label_mapping.json"

os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)

print("📥 Loading:", IN_PATH)
df = pd.read_csv(IN_PATH, low_memory=False)

# Clean
df["text"] = df["text"].map(clean_text)

# Remove empty
before = len(df)
df = df[df["text"].str.len() > 0].copy()
print("🧹 Removed empty:", before - len(df))

# Ensure label_id is int
df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
before = len(df)
df = df[df["label_id"].notna()].copy()
df["label_id"] = df["label_id"].astype(int)
print("🧹 Removed invalid label_id:", before - len(df))

# Conflicting duplicates check (same text appears with multiple label_id)
conflicts = df.groupby("text")["label_id"].nunique()
conflict_texts = conflicts[conflicts > 1].index.tolist()
print("⚠️ Conflicting duplicate texts:", len(conflict_texts))

# If you want to DROP conflicts completely, uncomment:
# df = df[~df["text"].isin(conflict_texts)].copy()

# Drop duplicates
before = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print("🧹 Dropped duplicates:", before - len(df))

# Create mapping from label_id to label
id_to_label = (
    df.groupby("label_id")["label"]
      .agg(lambda x: x.value_counts().index[0])
      .to_dict()
)
label_to_id = {v: k for k, v in id_to_label.items()}

# Save cleaned dataset
df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print("💾 Saved cleaned dataset:", OUT_PATH, "rows:", len(df))

# Save mapping
with open(MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {"id_to_label": {str(k): v for k, v in id_to_label.items()},
         "label_to_id": label_to_id},
        f, ensure_ascii=False, indent=2
    )
print("💾 Saved label mapping:", MAP_PATH)