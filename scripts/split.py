import pandas as pd
from sklearn.model_selection import train_test_split

# ========= LOAD DATA =========
FILE_PATH = r"data\dataset_clean.csv"   # ✅ FIXED
df = pd.read_csv(FILE_PATH)

print("Original rows:", len(df))

# ========= TRAIN vs TEMP =========
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],   # IMPORTANT
    random_state=42
)

# ========= VAL vs TEST =========
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# ========= SAVE FILES =========
train_df.to_csv(r"data\train.csv", index=False)
val_df.to_csv(r"data\val.csv", index=False)
test_df.to_csv(r"data\test.csv", index=False)

# ========= PRINT STATS =========
print("\n✅ Split completed")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain label distribution:")
print(train_df["label"].value_counts())