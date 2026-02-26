import os
import pandas as pd
from sklearn.model_selection import train_test_split

FILE_PATH = r"data\dataset_clean.csv"
OUT_TRAIN = r"data\train.csv"
OUT_VAL   = r"data\val.csv"
OUT_TEST  = r"data\test.csv"

df = pd.read_csv(FILE_PATH, low_memory=False)
print("Original rows:", len(df))

PRIO_COL  = "priority_id_fixed"
LABEL_COL = "label_id"

# ── Safety checks ─────────────────────────────────────────────────────────────
need = {"text", "label", LABEL_COL, PRIO_COL}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing columns in {FILE_PATH}: {missing}")

df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int)
df[PRIO_COL]  = pd.to_numeric(df[PRIO_COL],  errors="raise").astype(int)

bad_prio = df[~df[PRIO_COL].isin([0, 1, 2])]
if len(bad_prio):
    raise ValueError(f"❌ Invalid {PRIO_COL} values: {bad_prio[PRIO_COL].unique().tolist()}")

# ── Stratification strategy ───────────────────────────────────────────────────
# Using label+priority combos causes failures when High-priority rows are so
# rare per class that sklearn can't guarantee ≥1 example in every split.
# Fix: stratify on LABEL only; priority distribution will follow naturally
# because High-priority rows are spread across all label classes.
#
# Only fall back to label+priority if every combo has ≥ 4 rows.

combo = df[LABEL_COL].astype(str) + "__" + df[PRIO_COL].astype(str)
min_combo_count = combo.value_counts().min()

if min_combo_count >= 4:
    print(f"✅ All label+priority combos have ≥4 rows — using combo stratification")
    strat_col = combo
else:
    rare = combo.value_counts()[combo.value_counts() < 4]
    print(f"⚠️  {len(rare)} label+priority combos have <4 rows → stratifying on LABEL only")
    strat_col = df[LABEL_COL].astype(str)

df["_strat"] = strat_col.values

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["_strat"],
    random_state=42,
)

# Re-evaluate for temp split (smaller N)
temp_strat = temp_df["_strat"]
temp_combo_min = temp_strat.value_counts().min()

if temp_combo_min >= 2:
    val_strat = temp_strat
else:
    print("⚠️  Temp split has singleton strat keys → using label-only for val/test split")
    val_strat = temp_df[LABEL_COL].astype(str)

temp_df = temp_df.copy()
temp_df["_strat"] = val_strat.values

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["_strat"],
    random_state=42,
)

# Drop helper column
for d in (train_df, val_df, test_df):
    d.drop(columns=["_strat"], inplace=True, errors="ignore")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_TRAIN) or ".", exist_ok=True)
train_df.to_csv(OUT_TRAIN, index=False)
val_df.to_csv(OUT_VAL,   index=False)
test_df.to_csv(OUT_TEST,  index=False)

print("\n✅ Split completed")
print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")

print("\nTrain label distribution:")
print(train_df["label"].value_counts().to_string())

for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"\n{split_name} priority distribution ({PRIO_COL}):")
    print(split_df[PRIO_COL].value_counts().sort_index().to_string())

# ── Sanity: check Examination split ──────────────────────────────────────────
print("\n── Examination rows ──")
for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    exam = split_df[split_df["label"] == "Examination"]
    high = (exam[PRIO_COL] == 2).sum()
    print(f"  {split_name}: {len(exam)} rows | High-priority: {high}")