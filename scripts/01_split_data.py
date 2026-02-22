import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path("data/dataset.csv")
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

print("Loading...")
df = pd.read_csv(DATA_PATH)

# Clean broken dash + trim
df["text"] = df["text"].astype(str).str.replace("â€”", "—", regex=False).str.strip()

print("Total rows:", len(df))
print("Unique labels:", df["label"].nunique())

# 1) Remove exact duplicates (same text + same label)
# This reduces repetition and makes splitting safer.
df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
print("After dropping duplicate (text,label):", len(df))

# 2) Split by unique text so the same text cannot be in multiple splits
unique_texts = df["text"].unique().tolist()

train_texts, temp_texts = train_test_split(
    unique_texts, test_size=0.2, random_state=42
)

val_texts, test_texts = train_test_split(
    temp_texts, test_size=0.5, random_state=42
)

train_df = df[df["text"].isin(train_texts)].copy()
val_df   = df[df["text"].isin(val_texts)].copy()
test_df  = df[df["text"].isin(test_texts)].copy()

# 3) Safety check: NO overlap
train_set = set(train_df["text"])
val_set   = set(val_df["text"])
test_set  = set(test_df["text"])

assert train_set.isdisjoint(val_set), "Leakage: train and val share texts!"
assert train_set.isdisjoint(test_set), "Leakage: train and test share texts!"
assert val_set.isdisjoint(test_set),   "Leakage: val and test share texts!"

# 4) Save splits
train_df.to_csv(OUT_DIR / "train.csv", index=False)
val_df.to_csv(OUT_DIR / "val.csv", index=False)
test_df.to_csv(OUT_DIR / "test.csv", index=False)

print("\n✅ Split sizes (no text leakage):")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

print("\nLabel distribution (train):")
print(train_df["label"].value_counts().sort_index())

print("\nLabel distribution (val):")
print(val_df["label"].value_counts().sort_index())

print("\nLabel distribution (test):")
print(test_df["label"].value_counts().sort_index())
