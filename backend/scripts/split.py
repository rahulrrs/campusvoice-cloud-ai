import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FILE_PATH = Path(os.getenv("DATASET_SPLIT_SOURCE", str(DATA_DIR / "dataset_clean.csv")))
OUT_TRAIN = DATA_DIR / "train.csv"
OUT_VAL = DATA_DIR / "val.csv"
OUT_TEST = DATA_DIR / "test.csv"
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.80"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.10"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.10"))

if min(TRAIN_RATIO, VAL_RATIO, TEST_RATIO) <= 0:
    raise ValueError("TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must all be > 0")
if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1.0e-9:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

df = pd.read_csv(FILE_PATH, low_memory=False)
print("Original rows:", len(df))

PRIO_COL = "priority_id_fixed"
LABEL_COL = "label_id"

need = {"text", "label", LABEL_COL, PRIO_COL}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {FILE_PATH}: {missing}")

df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int)
df[PRIO_COL] = pd.to_numeric(df[PRIO_COL], errors="raise").astype(int)

bad_prio = df[~df[PRIO_COL].isin([0, 1, 2])]
if len(bad_prio):
    raise ValueError(f"Invalid {PRIO_COL} values: {bad_prio[PRIO_COL].unique().tolist()}")

combo = df[LABEL_COL].astype(str) + "__" + df[PRIO_COL].astype(str)
min_combo_count = combo.value_counts().min()

if min_combo_count >= 4:
    print("All label+priority combos have >=4 rows; using combo stratification")
    strat_col = combo
else:
    rare = combo.value_counts()[combo.value_counts() < 4]
    print(f"{len(rare)} label+priority combos have <4 rows; stratifying on label only")
    strat_col = df[LABEL_COL].astype(str)

df["_strat"] = strat_col.values

train_df, temp_df = train_test_split(
    df,
    test_size=(1.0 - TRAIN_RATIO),
    stratify=df["_strat"],
    random_state=42,
)

temp_strat = temp_df["_strat"]
temp_combo_min = temp_strat.value_counts().min()

if temp_combo_min >= 2:
    val_strat = temp_strat
else:
    print("Temp split has singleton strat keys; using label-only for val/test split")
    val_strat = temp_df[LABEL_COL].astype(str)

temp_df = temp_df.copy()
temp_df["_strat"] = val_strat.values

temp_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

val_df, test_df = train_test_split(
    temp_df,
    test_size=temp_test_ratio,
    stratify=temp_df["_strat"],
    random_state=42,
)

for split_df in (train_df, val_df, test_df):
    split_df.drop(columns=["_strat"], inplace=True, errors="ignore")

os.makedirs(OUT_TRAIN.parent, exist_ok=True)
train_df.to_csv(OUT_TRAIN, index=False)
val_df.to_csv(OUT_VAL, index=False)
test_df.to_csv(OUT_TEST, index=False)

print("\nSplit completed")
print(
    f"Ratios used -> train: {TRAIN_RATIO:.2f}, val: {VAL_RATIO:.2f}, test: {TEST_RATIO:.2f}"
)
print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")

print("\nTrain label distribution:")
print(train_df["label"].value_counts().to_string())

for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n{split_name} priority distribution ({PRIO_COL}):")
    print(split_df[PRIO_COL].value_counts().sort_index().to_string())
    print(f"\n{split_name} category x priority:")
    print(pd.crosstab(split_df["label"], split_df[PRIO_COL]).to_string())

print("\nExamination rows")
for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    exam = split_df[split_df["label"] == "Examination"]
    high = (exam[PRIO_COL] == 2).sum()
    print(f"{split_name}: {len(exam)} rows | High-priority: {high}")
