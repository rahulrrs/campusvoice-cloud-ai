import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_model_only import predict_texts


DATA_PATH = PROJECT_ROOT / "data" / "test.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "fairness_eval.json"


def detect_language_proxy(text: str) -> str:
    normalized = str(text or "").lower()
    if any(token in normalized for token in ("kripya", "dhanyavad", "nahi")):
        return "hi"
    if any(token in normalized for token in ("nanri", "ungal", "vendum")):
        return "ta"
    if any(token in normalized for token in ("dayachesi", "meeru", "ledu")):
        return "te"
    return "en"


def build_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["anonymity_group"] = (
        out["is_anonymous"].map(lambda value: "anonymous" if bool(value) else "identified")
        if "is_anonymous" in out.columns
        else "identified"
    )
    out["language_group"] = (
        out["source_language"].fillna("unknown").astype(str)
        if "source_language" in out.columns
        else out["text"].astype(str).map(detect_language_proxy)
    )
    out["category_group"] = out["label"].astype(str)
    if "user_group" in out.columns:
        out["user_group_eval"] = out["user_group"].fillna("unknown").astype(str)
    elif "user_id" in out.columns:
        counts = out["user_id"].astype(str).value_counts()
        out["user_group_eval"] = out["user_id"].astype(str).map(
            lambda value: "repeat_submitter" if counts.get(value, 0) >= 3 else "standard_submitter"
        )
    else:
        out["user_group_eval"] = "unknown"
    return out


def group_metrics(df: pd.DataFrame, group_col: str, true_col: str, pred_col: str) -> list[dict]:
    results: list[dict] = []
    for group_value, group_df in df.groupby(group_col):
        if len(group_df) < 3:
            continue
        results.append(
            {
                "group": str(group_value),
                "count": int(len(group_df)),
                "accuracy": round(accuracy_score(group_df[true_col], group_df[pred_col]), 4),
                "f1_macro": round(
                    f1_score(group_df[true_col], group_df[pred_col], average="macro", zero_division=0),
                    4,
                ),
            }
        )
    return sorted(results, key=lambda item: item["count"], reverse=True)


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = {"text", "label", "priority"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {DATA_PATH}: {sorted(missing)}")

    # Fairness evaluation should measure the raw model, not post-hoc demo rules.
    predictions = predict_texts(df["text"].astype(str).tolist(), apply_rules=False)
    df = df.copy()
    df["pred_label"] = [row["label"] for row in predictions]
    df["pred_priority"] = [row["priority"] for row in predictions]
    df = build_group_columns(df)

    report = {
        "overall": {
            "label_accuracy": round(accuracy_score(df["label"], df["pred_label"]), 4),
            "label_f1_macro": round(f1_score(df["label"], df["pred_label"], average="macro", zero_division=0), 4),
            "priority_accuracy": round(accuracy_score(df["priority"], df["pred_priority"]), 4),
            "priority_f1_macro": round(
                f1_score(df["priority"], df["pred_priority"], average="macro", zero_division=0),
                4,
            ),
        },
        "groups": {
            "anonymity": {
                "label": group_metrics(df, "anonymity_group", "label", "pred_label"),
                "priority": group_metrics(df, "anonymity_group", "priority", "pred_priority"),
            },
            "language": {
                "label": group_metrics(df, "language_group", "label", "pred_label"),
                "priority": group_metrics(df, "language_group", "priority", "pred_priority"),
            },
            "category": {
                "label": group_metrics(df, "category_group", "label", "pred_label"),
                "priority": group_metrics(df, "category_group", "priority", "pred_priority"),
            },
            "user_group": {
                "label": group_metrics(df, "user_group_eval", "label", "pred_label"),
                "priority": group_metrics(df, "user_group_eval", "priority", "pred_priority"),
            },
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
