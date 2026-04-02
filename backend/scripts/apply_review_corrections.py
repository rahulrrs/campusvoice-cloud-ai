import json
import os
import hashlib
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
default_data_path = PROJECT_ROOT / "data" / "dataset_corrected.csv"
if not default_data_path.exists():
    default_data_path = PROJECT_ROOT / "data" / "dataset_clean.csv"
DATA_PATH = Path(os.getenv("CORRECTION_SOURCE_DATASET", str(default_data_path)))
default_review_path = PROJECT_ROOT / "outputs" / "edu_classifier_multitask" / "review" / "review_candidates.csv"
REVIEW_PATH = Path(os.getenv("REVIEW_CANDIDATES_PATH", str(default_review_path)))
default_out_path = PROJECT_ROOT / "data" / "dataset_corrected.csv"
OUT_PATH = Path(os.getenv("CORRECTED_DATASET_PATH", str(default_out_path)))
SKIP_UNMATCHED_REVIEW_ROWS = os.getenv("SKIP_UNMATCHED_REVIEW_ROWS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

PRIORITY_TO_ID = {"Low": 0, "Medium": 1, "High": 2}


def clean_optional_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def text_hash(value: str) -> str:
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()


def load_review_file(path: Path) -> pd.DataFrame:
    if path.exists():
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        try:
            return pd.read_csv(path, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
            return pd.read_excel(path)

    xlsx_fallback = path.with_suffix(path.suffix + ".xlsx")
    if xlsx_fallback.exists():
        return pd.read_excel(xlsx_fallback)

    excel_fallback = path.with_suffix(".xlsx")
    if excel_fallback.exists():
        return pd.read_excel(excel_fallback)

    raise FileNotFoundError(f"Review candidates file not found: {path}")


def build_appended_row(
    data_df: pd.DataFrame,
    row: dict,
    review_status: str,
    review_notes: str,
    corrected_text: str,
    corrected_label: str,
    corrected_priority: str,
    is_approved: bool,
) -> dict:
    base = {column: "" for column in data_df.columns}
    if "review_has_notes" in base:
        base["review_has_notes"] = 0
    if "review_was_corrected" in base:
        base["review_was_corrected"] = 0

    final_text = corrected_text or clean_optional_text(row.get("text"))
    final_label = corrected_label or clean_optional_text(row.get("current_label"))
    final_priority = corrected_priority or clean_optional_text(row.get("current_priority"))

    base["text"] = final_text
    if "label" in base:
        base["label"] = final_label
    if "priority" in base:
        base["priority"] = final_priority
    if "priority_id_fixed" in base and final_priority in PRIORITY_TO_ID:
        base["priority_id_fixed"] = PRIORITY_TO_ID[final_priority]

    if "review_status" in base:
        base["review_status"] = review_status or ("approved" if is_approved else "")
    if "review_notes" in base:
        base["review_notes"] = review_notes
    if "review_has_notes" in base:
        base["review_has_notes"] = int(bool(review_notes))
    if "review_was_corrected" in base:
        base["review_was_corrected"] = int(
            bool(corrected_text or corrected_label or corrected_priority)
        )

    return base


def main() -> None:
    data_df = pd.read_csv(DATA_PATH, low_memory=False)
    review_df = load_review_file(REVIEW_PATH)

    data_df = data_df.copy()
    if "review_status" not in data_df.columns:
        data_df["review_status"] = ""
    else:
        data_df["review_status"] = data_df["review_status"].fillna("").astype(str)
    if "review_notes" not in data_df.columns:
        data_df["review_notes"] = ""
    else:
        data_df["review_notes"] = data_df["review_notes"].fillna("").astype(str)
    if "review_has_notes" not in data_df.columns:
        data_df["review_has_notes"] = 0
    else:
        data_df["review_has_notes"] = pd.to_numeric(data_df["review_has_notes"], errors="coerce").fillna(0).astype(int)
    if "review_was_corrected" not in data_df.columns:
        data_df["review_was_corrected"] = 0
    else:
        data_df["review_was_corrected"] = pd.to_numeric(
            data_df["review_was_corrected"], errors="coerce"
        ).fillna(0).astype(int)

    required_review_cols = {
        "row_index",
        "review_status",
        "corrected_text",
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
        reviewed.apply(
            lambda row: (
                clean_optional_text(row.get("review_status")).lower() == "approved"
                or any(
                    clean_optional_text(row.get(column))
                    for column in ("corrected_text", "corrected_label", "corrected_priority")
                )
            ),
            axis=1,
        )
    ].copy()

    allowed_labels = set(data_df["label"].astype(str).str.strip().unique().tolist())
    data_text_hashes = data_df["text"].astype(str).map(text_hash)
    hash_to_indices: dict[str, list[int]] = {}
    for row_idx, row_hash in enumerate(data_text_hashes.tolist()):
        hash_to_indices.setdefault(row_hash, []).append(row_idx)

    reviewed_rows = 0
    applied = 0
    skipped_unmatched = 0
    appended_unmatched = 0
    for row in reviewed.to_dict("records"):
        idx = int(row["row_index"])
        review_status = clean_optional_text(row.get("review_status"))
        review_notes = clean_optional_text(row.get("review_notes"))
        corrected_text = clean_optional_text(row.get("corrected_text"))
        corrected_label = clean_optional_text(row.get("corrected_label"))
        corrected_priority = clean_optional_text(row.get("corrected_priority"))
        is_approved = (
            review_status.lower() == "approved"
            or bool(corrected_text or corrected_label or corrected_priority)
        )

        expected_text = clean_optional_text(row.get("text"))
        expected_text_hash = clean_optional_text(row.get("text_hash"))
        row_matches = idx >= 0 and idx < len(data_df)
        if row_matches and expected_text:
            row_matches = str(data_df.at[idx, "text"]).strip() == expected_text
        if row_matches and expected_text_hash:
            row_matches = text_hash(data_df.at[idx, "text"]) == expected_text_hash

        if not row_matches:
            candidate_indices: list[int] = []
            if expected_text_hash:
                candidate_indices = hash_to_indices.get(expected_text_hash, [])
            elif expected_text:
                candidate_indices = data_df.index[data_df["text"].astype(str).str.strip().eq(expected_text)].tolist()

            if len(candidate_indices) == 1:
                idx = int(candidate_indices[0])
            elif len(candidate_indices) == 0:
                if SKIP_UNMATCHED_REVIEW_ROWS:
                    appended_row = build_appended_row(
                        data_df=data_df,
                        row=row,
                        review_status=review_status,
                        review_notes=review_notes,
                        corrected_text=corrected_text,
                        corrected_label=corrected_label,
                        corrected_priority=corrected_priority,
                        is_approved=is_approved,
                    )
                    data_df = pd.concat([data_df, pd.DataFrame([appended_row])], ignore_index=True)
                    new_idx = len(data_df) - 1
                    new_hash = text_hash(data_df.at[new_idx, "text"])
                    hash_to_indices.setdefault(new_hash, []).append(new_idx)
                    print(
                        f"Appended unmatched reviewed sample from row_index={row.get('row_index')} "
                        f"to dataset bottom at row_index={new_idx}"
                    )
                    appended_unmatched += 1
                    reviewed_rows += 1
                    applied += 1
                    continue
                raise ValueError(
                    f"Could not locate reviewed sample from row_index={row.get('row_index')} "
                    "in the source dataset. Re-export review candidates from the current dataset."
                )
            else:
                if SKIP_UNMATCHED_REVIEW_ROWS:
                    print(
                        f"Skipping ambiguous reviewed sample from row_index={row.get('row_index')} "
                        f"(matched_rows={len(candidate_indices)})"
                    )
                    skipped_unmatched += 1
                    continue
                raise ValueError(
                    f"Multiple dataset rows match reviewed sample from row_index={row.get('row_index')}. "
                    "Use a dataset with unique complaint texts or re-export review candidates."
                )

        changed = False

        if corrected_text and is_approved:
            data_df.at[idx, "text"] = corrected_text
            changed = True
        if corrected_label and is_approved:
            if corrected_label not in allowed_labels:
                raise ValueError(
                    f"Invalid corrected label '{corrected_label}' at row_index={idx}. "
                    f"Expected one of existing dataset labels: {sorted(allowed_labels)}"
                )
            data_df.at[idx, "label"] = corrected_label
            changed = True
        if corrected_priority and is_approved:
            if corrected_priority not in PRIORITY_TO_ID:
                raise ValueError(f"Invalid corrected priority '{corrected_priority}' at row_index={idx}")
            data_df.at[idx, "priority"] = corrected_priority
            data_df.at[idx, "priority_id_fixed"] = PRIORITY_TO_ID[corrected_priority]
            changed = True

        data_df.at[idx, "review_status"] = review_status or ("approved" if is_approved else "")
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
                "skipped_unmatched_rows": int(skipped_unmatched),
                "appended_unmatched_rows": int(appended_unmatched),
                "output_path": str(OUT_PATH),
                "num_labels": int(len(label_to_id)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
