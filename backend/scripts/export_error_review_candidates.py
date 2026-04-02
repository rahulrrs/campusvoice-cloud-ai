import hashlib
import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "outputs" / "edu_classifier_multitask"
ANALYSIS_DIR = MODEL_DIR / "analysis"
REVIEW_DIR = MODEL_DIR / "review"

default_data_path = DATA_DIR / "dataset_corrected.csv"
if not default_data_path.exists():
    default_data_path = DATA_DIR / "dataset_clean.csv"

DATASET_PATH = Path(os.getenv("ERROR_REVIEW_DATASET_PATH", str(default_data_path)))
ERRORS_PATH = Path(
    os.getenv(
        "MISCLASSIFIED_PREDICTIONS_PATH",
        str(ANALYSIS_DIR / "misclassified_test_predictions.csv"),
    )
)
OUT_PATH = Path(
    os.getenv(
        "ERROR_REVIEW_OUT_PATH",
        str(REVIEW_DIR / "error_review_candidates.csv"),
    )
)
ERROR_TYPES = {
    item.strip()
    for item in os.getenv(
        "ERROR_REVIEW_TYPES",
        "label_and_priority,label_only,priority_only",
    ).split(",")
    if item.strip()
}


def text_hash(value: str) -> str:
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def main() -> None:
    dataset_df = pd.read_csv(DATASET_PATH, low_memory=False)
    errors_df = pd.read_csv(ERRORS_PATH, low_memory=False)

    required_dataset_cols = {"text", "label", "priority"}
    missing_dataset = required_dataset_cols - set(dataset_df.columns)
    if missing_dataset:
        raise ValueError(f"Missing columns in dataset {DATASET_PATH}: {sorted(missing_dataset)}")

    required_error_cols = {
        "text",
        "true_label",
        "pred_label",
        "true_priority",
        "pred_priority",
        "error_type",
    }
    missing_error_cols = required_error_cols - set(errors_df.columns)
    if missing_error_cols:
        raise ValueError(f"Missing columns in misclassified file {ERRORS_PATH}: {sorted(missing_error_cols)}")

    if ERROR_TYPES:
        errors_df = errors_df[errors_df["error_type"].astype(str).isin(ERROR_TYPES)].copy()

    dataset_hashes = dataset_df["text"].astype(str).map(text_hash)
    hash_to_indices: dict[str, list[int]] = {}
    for idx, row_hash in enumerate(dataset_hashes.tolist()):
        hash_to_indices.setdefault(row_hash, []).append(idx)

    review_rows: list[dict] = []
    unmatched = 0
    ambiguous = 0

    for error_row in errors_df.to_dict("records"):
        text = clean_text(error_row.get("text"))
        row_hash = text_hash(text)
        matched_indices = hash_to_indices.get(row_hash, [])

        if len(matched_indices) == 1:
            row_index = int(matched_indices[0])
            match_status = "matched"
        elif len(matched_indices) == 0:
            row_index = int(error_row.get("row_index", -1)) if str(error_row.get("row_index", "")).strip() else -1
            match_status = "unmatched"
            unmatched += 1
        else:
            row_index = int(matched_indices[0])
            match_status = "ambiguous"
            ambiguous += 1

        dataset_label = clean_text(dataset_df.at[row_index, "label"]) if row_index >= 0 and row_index < len(dataset_df) else ""
        dataset_priority = clean_text(dataset_df.at[row_index, "priority"]) if row_index >= 0 and row_index < len(dataset_df) else ""

        true_label = clean_text(error_row.get("true_label"))
        pred_label = clean_text(error_row.get("pred_label"))
        true_priority = clean_text(error_row.get("true_priority"))
        pred_priority = clean_text(error_row.get("pred_priority"))
        error_type = clean_text(error_row.get("error_type"))

        review_reason_parts = [f"eval_{error_type}"]
        if true_label and pred_label and true_label != pred_label:
            review_reason_parts.append("label_mismatch")
        if true_priority and pred_priority and true_priority != pred_priority:
            review_reason_parts.append("priority_mismatch")
        if match_status != "matched":
            review_reason_parts.append(f"dataset_{match_status}")

        review_rows.append(
            {
                "source_dataset": str(DATASET_PATH),
                "source_errors_file": str(ERRORS_PATH),
                "row_index": row_index,
                "text": text,
                "text_hash": row_hash,
                "current_label": dataset_label or true_label,
                "predicted_label": pred_label,
                "current_priority": dataset_priority or true_priority,
                "predicted_priority": pred_priority,
                "review_reason": "|".join(review_reason_parts),
                "match_status": match_status,
                "dataset_label": dataset_label,
                "dataset_priority": dataset_priority,
                "eval_true_label": true_label,
                "eval_true_priority": true_priority,
                "suggested_label": pred_label if pred_label and pred_label != (dataset_label or true_label) else "",
                "suggested_priority": (
                    pred_priority if pred_priority and pred_priority != (dataset_priority or true_priority) else ""
                ),
                "review_status": "",
                "corrected_text": "",
                "corrected_label": "",
                "corrected_priority": "",
                "review_notes": "",
            }
        )

    out_df = pd.DataFrame(review_rows)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset_path": str(DATASET_PATH),
                "errors_path": str(ERRORS_PATH),
                "rows_exported": int(len(out_df)),
                "unmatched_rows": int(unmatched),
                "ambiguous_rows": int(ambiguous),
                "output_path": str(OUT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
