import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
default_data_path = PROJECT_ROOT / "data" / "dataset_clean.csv"
DATA_PATH = Path(os.getenv("CORRECTION_SOURCE_DATASET", str(default_data_path)))
default_review_path = PROJECT_ROOT / "outputs" / "edu_classifier_multitask" / "review" / "review_candidates.csv"
REVIEW_PATH = Path(os.getenv("REVIEW_CANDIDATES_PATH", str(default_review_path)))
default_out_path = PROJECT_ROOT / "data" / "dataset_corrected.csv"
OUT_PATH = Path(os.getenv("CORRECTED_DATASET_PATH", str(default_out_path)))

PRIORITY_TO_ID = {"Low": 0, "Medium": 1, "High": 2}


def clean_optional_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def main() -> None:
    data_df = pd.read_csv(DATA_PATH, low_memory=False)
    review_df = pd.read_csv(REVIEW_PATH, low_memory=False)

    data_df = data_df.copy()
    data_df["review_status"] = ""
    data_df["review_notes"] = ""
    data_df["review_has_notes"] = 0
    data_df["review_was_corrected"] = 0

    required_review_cols = {
        "row_index",
        "review_status",
        "corrected_label",
        "corrected_priority",
    }
    missing = required_review_cols - set(review_df.columns)
    if missing:
        raise ValueError(f"Missing columns in {REVIEW_PATH}: {sorted(missing)}")

    reviewed = review_df[
        review_df.apply(
            lambda row: any(
                clean_optional_text(row.get(column))
                for column in ("review_status", "corrected_label", "corrected_priority", "review_notes")
            ),
            axis=1,
        )
    ].copy()
    approved = reviewed[
        reviewed["review_status"].astype(str).str.strip().str.lower().eq("approved")
    ].copy()

    allowed_labels = set(data_df["label"].astype(str).str.strip().unique().tolist())

    reviewed_rows = 0
    applied = 0
    for row in reviewed.to_dict("records"):
        idx = int(row["row_index"])
        if idx < 0 or idx >= len(data_df):
            continue

        review_status = clean_optional_text(row.get("review_status"))
        review_notes = clean_optional_text(row.get("review_notes"))
        corrected_label = clean_optional_text(row.get("corrected_label"))
        corrected_priority = clean_optional_text(row.get("corrected_priority"))
        changed = False

        if corrected_label and review_status.lower() == "approved":
            if corrected_label not in allowed_labels:
                raise ValueError(
                    f"Invalid corrected label '{corrected_label}' at row_index={idx}. "
                    f"Expected one of existing dataset labels: {sorted(allowed_labels)}"
                )
            data_df.at[idx, "label"] = corrected_label
            changed = True
        if corrected_priority and review_status.lower() == "approved":
            if corrected_priority not in PRIORITY_TO_ID:
                raise ValueError(f"Invalid corrected priority '{corrected_priority}' at row_index={idx}")
            data_df.at[idx, "priority"] = corrected_priority
            data_df.at[idx, "priority_id_fixed"] = PRIORITY_TO_ID[corrected_priority]
            changed = True

        data_df.at[idx, "review_status"] = review_status
        data_df.at[idx, "review_notes"] = review_notes
        data_df.at[idx, "review_has_notes"] = int(bool(review_notes))
        data_df.at[idx, "review_was_corrected"] = int(changed)
        reviewed_rows += 1
        if changed:
            applied += 1

    label_names = sorted(data_df["label"].astype(str).str.strip().unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    data_df["label"] = data_df["label"].astype(str).str.strip()
    data_df["label_id"] = data_df["label"].map(label_to_id).astype(int)

    data_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(
        json.dumps(
            {
                "reviewed_rows": int(reviewed_rows),
                "approved_corrections": int(len(approved)),
                "applied_rows": int(applied),
                "output_path": str(OUT_PATH),
                "num_labels": int(len(label_to_id)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
