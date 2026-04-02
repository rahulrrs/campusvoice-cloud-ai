import hashlib
import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_corrected.csv"
REVIEW_PATH = PROJECT_ROOT / "outputs" / "edu_classifier_multitask" / "review" / "confusion_focus_review.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "edu_classifier_multitask" / "review" / "confusion_focus_apply_candidates.csv"


def text_hash(value: str) -> str:
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()


def main() -> None:
    dataset_path = Path(os.getenv("CONFUSION_DATASET_PATH", str(DATASET_PATH)))
    review_path = Path(os.getenv("CONFUSION_REVIEW_PATH", str(REVIEW_PATH)))
    out_path = Path(os.getenv("CONFUSION_APPLY_OUT_PATH", str(OUT_PATH)))

    data_df = pd.read_csv(dataset_path, low_memory=False)
    review_df = pd.read_csv(review_path, low_memory=False)

    required_dataset = {"text", "label", "priority"}
    required_review = {"text", "review_status", "corrected_label", "corrected_priority", "review_notes"}
    missing_dataset = required_dataset - set(data_df.columns)
    missing_review = required_review - set(review_df.columns)
    if missing_dataset:
        raise ValueError(f"Missing columns in dataset: {sorted(missing_dataset)}")
    if missing_review:
        raise ValueError(f"Missing columns in confusion review file: {sorted(missing_review)}")

    data_hashes = data_df["text"].astype(str).map(text_hash)
    hash_to_index: dict[str, int] = {}
    for idx, row_hash in enumerate(data_hashes.tolist()):
        hash_to_index.setdefault(row_hash, idx)

    rows = []
    unmatched = 0
    for review_row in review_df.to_dict("records"):
        text = str(review_row.get("text", "")).strip()
        row_hash = text_hash(text)
        row_index = hash_to_index.get(row_hash, -1)
        if row_index < 0:
            unmatched += 1
            continue

        rows.append(
            {
                "source_dataset": str(dataset_path),
                "row_index": int(row_index),
                "text": text,
                "text_hash": row_hash,
                "current_label": str(data_df.at[row_index, "label"]).strip(),
                "current_priority": str(data_df.at[row_index, "priority"]).strip(),
                "review_status": str(review_row.get("review_status", "")).strip(),
                "corrected_text": "",
                "corrected_label": str(review_row.get("corrected_label", "")).strip(),
                "corrected_priority": str(review_row.get("corrected_priority", "")).strip(),
                "review_notes": str(review_row.get("review_notes", "")).strip(),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "review_path": str(review_path),
                "rows_written": int(len(out_df)),
                "unmatched_rows": int(unmatched),
                "output_path": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
