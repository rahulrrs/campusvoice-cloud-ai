import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "outputs" / "edu_classifier_multitask"
ANALYSIS_DIR = MODEL_DIR / "analysis"
OUT_DIR = MODEL_DIR / "review"

MISCLASSIFIED_PATH = Path(
    os.getenv(
        "MISCLASSIFIED_PREDICTIONS_PATH",
        str(ANALYSIS_DIR / "misclassified_test_predictions.csv"),
    )
)
OUT_PATH = Path(
    os.getenv(
        "CONFUSION_FOCUS_OUT_PATH",
        str(OUT_DIR / "confusion_focus_review.csv"),
    )
)

FOCUS_PAIRS = {
    ("Discipline", "Faculty"),
    ("Faculty", "Discipline"),
    ("Infrastructure", "Lost & Found"),
    ("Lost & Found", "Infrastructure"),
    ("Safety & Security", "Infrastructure"),
    ("Infrastructure", "Safety & Security"),
    ("Examination", "Certificate & Records"),
    ("Certificate & Records", "Examination"),
    ("Library", "Safety & Security"),
    ("Safety & Security", "Library"),
}


def main() -> None:
    df = pd.read_csv(MISCLASSIFIED_PATH, low_memory=False)
    required = {
        "text",
        "true_label",
        "pred_label",
        "true_priority",
        "pred_priority",
        "error_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {MISCLASSIFIED_PATH}: {sorted(missing)}")

    filtered = df[
        df.apply(
            lambda row: (str(row["true_label"]).strip(), str(row["pred_label"]).strip()) in FOCUS_PAIRS,
            axis=1,
        )
    ].copy()

    filtered["focus_pair"] = filtered.apply(
        lambda row: f'{str(row["true_label"]).strip()} -> {str(row["pred_label"]).strip()}',
        axis=1,
    )
    filtered["suggested_action"] = filtered.apply(
        lambda row: (
            "review_label_and_priority"
            if str(row["error_type"]).strip() == "label_and_priority"
            else "review_label"
        ),
        axis=1,
    )
    filtered["review_status"] = ""
    filtered["review_notes"] = ""
    filtered["corrected_label"] = ""
    filtered["corrected_priority"] = ""

    priority_distance = (
        filtered["pred_priority_id"].astype(int) - filtered["true_priority_id"].astype(int)
    ).abs() if {"pred_priority_id", "true_priority_id"}.issubset(filtered.columns) else 0
    filtered["priority_distance"] = priority_distance

    sort_columns = ["focus_pair", "error_type"]
    ascending = [True, True]
    if "priority_distance" in filtered.columns:
        sort_columns.append("priority_distance")
        ascending.append(False)
    filtered = filtered.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(
        json.dumps(
            {
                "source_path": str(MISCLASSIFIED_PATH),
                "rows_exported": int(len(filtered)),
                "output_path": str(OUT_PATH),
                "focus_pairs": sorted(f"{left} -> {right}" for left, right in FOCUS_PAIRS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
