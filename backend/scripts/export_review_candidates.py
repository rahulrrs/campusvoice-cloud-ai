import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_model_only import predict_texts


DEFAULT_REVIEW_DATA_PATH = PROJECT_ROOT / "data" / "dataset_corrected.csv"
if not DEFAULT_REVIEW_DATA_PATH.exists():
    DEFAULT_REVIEW_DATA_PATH = PROJECT_ROOT / "data" / "dataset_clean.csv"

DATA_PATH = Path(
    os.getenv("REVIEW_DATASET_SOURCE", str(DEFAULT_REVIEW_DATA_PATH))
)
OUT_DIR = PROJECT_ROOT / "outputs" / "edu_classifier_multitask" / "review"
OUT_PATH = OUT_DIR / "review_candidates.csv"

PRIORITY_HOTSPOTS = {
    ("Low", "Medium"),
    ("High", "Medium"),
    ("Medium", "High"),
}
INFRASTRUCTURE_LABEL = "Infrastructure"


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = {"text", "label", "priority"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {DATA_PATH}: {sorted(missing)}")

    texts = df["text"].astype(str).tolist()
    predictions = predict_texts(texts)

    review_rows: list[dict] = []
    for idx, (row, pred) in enumerate(zip(df.to_dict("records"), predictions)):
        true_label = str(row["label"]).strip()
        true_priority = str(row["priority"]).strip()
        pred_label = str(pred["label"]).strip()
        pred_priority = str(pred["priority"]).strip()
        label_conf = float(pred["label_confidence"])
        priority_conf = float(pred["priority_confidence"])

        label_mismatch = true_label != pred_label and label_conf >= 0.80
        priority_hotspot = (true_priority, pred_priority) in PRIORITY_HOTSPOTS and priority_conf >= 0.70
        infra_priority_mismatch = (
            true_label == INFRASTRUCTURE_LABEL and true_priority != pred_priority and priority_conf >= 0.60
        )

        if not (label_mismatch or priority_hotspot or infra_priority_mismatch):
            continue

        review_reason: list[str] = []
        if label_mismatch:
            review_reason.append("high_conf_label_mismatch")
        if priority_hotspot:
            review_reason.append("priority_hotspot")
        if infra_priority_mismatch:
            review_reason.append("infrastructure_priority_mismatch")

        review_rows.append(
            {
                "row_index": idx,
                "text": row["text"],
                "current_label": true_label,
                "predicted_label": pred_label,
                "label_confidence": round(label_conf, 4),
                "current_priority": true_priority,
                "predicted_priority": pred_priority,
                "priority_confidence": round(priority_conf, 4),
                "review_reason": "|".join(review_reason),
                "review_status": "",
                "corrected_label": "",
                "corrected_priority": "",
                "review_notes": "",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(review_rows)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    summary = {
        "dataset_rows": int(len(df)),
        "review_rows": int(len(out_df)),
        "output_path": str(OUT_PATH),
        "reason_counts": out_df["review_reason"].value_counts().to_dict() if len(out_df) else {},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
