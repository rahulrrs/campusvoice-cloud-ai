import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# ================== CONFIG ==================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "edu_classifier_multitask"
INPUT_CANDIDATES = [
    DATA_DIR / "dataset.csv",
    DATA_DIR / "dataset.xlsx",
    DATA_DIR / "dataset.xls",
]
OUT_PATH = DATA_DIR / "dataset_clean.csv"
OUT_DIR = OUTPUTS_DIR
MIN_WORDS = 3

DEFAULT_PRIORITY_ID = 1  # 0=Low, 1=Medium, 2=High

# Label-noise fix: reclassify course-review texts under "Examination" -> "Academic"
FIX_EXAM_LABEL_NOISE = False

# Synthetic augmentation for underrepresented High-priority exam complaints
AUGMENT_EXAM_HIGH = False
# ===========================================

_URGENCY_RE = re.compile(
    r"\b(today|tomorrow|tonight|in\s*\d+\s*(hours|hrs)|within\s*\d+\s*(hours|hrs)|next\s*day)\b",
    re.I,
)
_EXAM_COMPLAINT_RE = re.compile(
    r"\b(hall\s*ticket|admit\s*card|timetable|time\s*table|seating|venue|result|revaluation|"
    r"registration|deadline|last\s*date|not\s*released|schedule|roll\s*no|seat\s*number|"
    r"exam\s*centre|exam\s*date)\b",
    re.I,
)
_COURSE_REVIEW_RE = re.compile(
    r"\b(prof|professor|lecture|course|assignment|midterm|mark|grading|\bTA\b|quiz|"
    r"textbook|readings?|semester|instructor|courseload|syllabus|coursework|"
    r"professor's?|lectures?|assignments?)\b",
    re.I,
)
_EXAM_BLOCKER_RE = re.compile(
    r"\b(exam|examination|timetable|time\s*table|schedule|hall\s*ticket|admit\s*card|result|"
    r"revaluation|registration|enroll|deadline|starts?\s*(tomorrow|today))\b",
    re.I,
)
_WATER_OUTAGE_RE = re.compile(
    r"\b(no\s*water|water\s*(is\s*)?not\s*available|water\s*supply\s*.{0,20}\bdown\b|"
    r"water\s*problem|water\s*outage|water\s*cut)\b",
    re.I,
)
_DURATION_RE = re.compile(r"\b(\d+)\s*(day|days|d)\b", re.I)

_PRIORITY_MAP = {
    "low": 0,
    "l": 0,
    "0": 0,
    "medium": 1,
    "med": 1,
    "m": 1,
    "1": 1,
    "high": 2,
    "h": 2,
    "2": 2,
}

_COLUMN_ALIASES = {
    "text": "text",
    "complaint_text": "text",
    "description": "text",
    "content": "text",
    "label": "label",
    "category": "label",
    "class": "label",
    "priority": "priority",
    "priority_fixed": "priority_fixed",
    "source": "source",
    "source_type": "source_type",
    "source_ty": "source_type",
    "text_len": "text_len",
    "text_length": "text_len",
    "word_count": "word_count",
    "word_cou": "word_count",
    "label_id": "label_id",
}


def _is_exam_label_noise(text: str) -> bool:
    if _EXAM_COMPLAINT_RE.search(text):
        return False
    return bool(_COURSE_REVIEW_RE.search(text))


def _is_exam_urgent(text: str) -> bool:
    t = (text or "").lower()
    if any(k in t for k in ["scholarship", "discount", "fee waiver", "fees", "financial aid"]):
        return False
    return bool(_EXAM_BLOCKER_RE.search(t) and _URGENCY_RE.search(t))


def _is_hostel_water_outage(text: str) -> bool:
    t = (text or "").lower()
    if not _WATER_OUTAGE_RE.search(t):
        return False
    m = _DURATION_RE.search(t)
    if not m:
        return True
    try:
        days = int(m.group(1))
    except ValueError:
        return True
    return days >= 1


def safe_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def strip_outer_quotes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r'^"(.*)"$', r"\1", regex=True)


def resolve_input_path() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in INPUT_CANDIDATES)
    raise FileNotFoundError(f"Could not find dataset file. Tried: {tried}")


def _load_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                parts = [node.text or "" for node in item.iterfind(".//a:t", ns)]
                shared_strings.append("".join(parts))

        worksheet_names = sorted(
            name for name in workbook.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )

        best_df = pd.DataFrame()
        best_score = -1
        for worksheet_name in worksheet_names:
            sheet = ET.fromstring(workbook.read(worksheet_name))
            rows = sheet.findall(".//a:sheetData/a:row", ns)
            parsed_rows: list[list[str]] = []
            for row in rows:
                values: list[str] = []
                for cell in row.findall("a:c", ns):
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find("a:v", ns)
                    value = "" if value_node is None or value_node.text is None else value_node.text
                    if cell_type == "s" and value != "":
                        value = shared_strings[int(value)]
                    values.append(value)
                parsed_rows.append(values)

            if not parsed_rows:
                continue

            header = [str(col).strip() for col in parsed_rows[0]]
            data = parsed_rows[1:]
            width = len(header)
            normalized_rows = []
            for row in data:
                if len(row) < width:
                    row = row + [""] * (width - len(row))
                elif len(row) > width:
                    row = row[:width]
                normalized_rows.append(row)

            candidate_df = pd.DataFrame(normalized_rows, columns=header)
            normalized_header = {str(col).strip().lower() for col in candidate_df.columns}
            score = int("text" in normalized_header) + int("category" in normalized_header or "label" in normalized_header) + int("priority" in normalized_header or "priority_fixed" in normalized_header or "priority_id_fixed" in normalized_header)
            if score > best_score:
                best_score = score
                best_df = candidate_df

    return best_df


def load_input_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".xlsx":
        return _load_xlsx_without_openpyxl(path)
    if suffix == ".xls":
        raise ValueError(
            "The current environment cannot read .xls files automatically. "
            "Please convert it to .csv or .xlsx first."
        )
    raise ValueError(f"Unsupported dataset file format: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        renamed[col] = _COLUMN_ALIASES.get(key, key)
    return df.rename(columns=renamed)


_EXAM_SYNTHETIC = [
    "Hall ticket is not available on the portal and exam is tomorrow. Please resolve urgently.",
    "My admit card has not been released and the exam starts tomorrow. Need immediate help.",
    "Exam timetable has not been published and the exam begins tomorrow. Very stressful.",
    "Exam centre and seating details are not released. Exam is next day. Please update.",
    "Hall ticket not yet issued. Exam is today and I cannot appear without it.",
    "Result not published yet; revaluation deadline is tomorrow. Please act fast.",
    "Registration portal is down and the exam enrollment deadline is today.",
    "Timetable not released and exam is starting today. Please share the schedule.",
    "I have not received my hall ticket and the examination is tomorrow morning.",
    "Admit card is missing from the portal and I have an exam in a few hours.",
    "Seating arrangement and venue details for tomorrow's exam still not uploaded.",
    "Exam date is tomorrow and the admit card link on the portal is broken.",
    "Hall ticket download failing; exam hall is tomorrow at 9 AM. Urgent fix needed.",
    "The exam timetable was changed without notice and my hall ticket shows wrong date.",
    "Revaluation form not working and last date is today. Please fix immediately.",
]


def main():
    in_path = resolve_input_path()
    print(f"Loading: {in_path}")
    df = normalize_columns(load_input_dataframe(in_path))

    need = {"text", "label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_priority_id = "priority_id_fixed" in df.columns
    has_priority_text = "priority" in df.columns or "priority_fixed" in df.columns
    if not has_priority_id and not has_priority_text:
        raise ValueError(
            "Missing priority column: expected one of priority_id_fixed, priority, priority_fixed"
        )

    df["text"] = strip_outer_quotes(df["text"]).str.replace(r"\s+", " ", regex=True).str.strip()
    df["label"] = safe_strip(df["label"])

    if "label_id" not in df.columns:
        print("label_id column not found. Generating label IDs from sorted label values.")
        label_names = sorted(df["label"].astype(str).str.strip().unique().tolist())
        label_to_id = {label: idx for idx, label in enumerate(label_names)}
        df["label_id"] = df["label"].map(label_to_id).astype(int)

    if "priority_fixed" in df.columns:
        df["priority_fixed"] = safe_strip(df["priority_fixed"])
    if "priority" in df.columns:
        df["priority"] = safe_strip(df["priority"])

    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    print(f"Removed empty rows: {before - len(df)}")

    before = len(df)
    df["word_count"] = df["text"].str.split().str.len().astype(int)
    df = df[df["word_count"] >= MIN_WORDS].copy()
    print(f"Removed short rows (<{MIN_WORDS} words): {before - len(df)}")

    before = len(df)
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    df = df.dropna(subset=["label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)
    print(f"Removed invalid label_id rows: {before - len(df)}")

    if FIX_EXAM_LABEL_NOISE and (df["label"] == "Academic").any() and (df["label"] == "Examination").any():
        academic_label_id = int(df.loc[df["label"] == "Academic", "label_id"].iloc[0])
        exam_mask = df["label"] == "Examination"
        noise_mask = exam_mask & df["text"].apply(_is_exam_label_noise)
        reclassified = int(noise_mask.sum())
        df.loc[noise_mask, "label"] = "Academic"
        df.loc[noise_mask, "label_id"] = academic_label_id
        print(f"Reclassified Examination -> Academic rows: {reclassified}")

    # Remap arbitrary dataset label IDs (for example 1..15) to contiguous 0..N-1 IDs
    # because the training head size is derived from the number of unique labels.
    label_order = (
        df[["label_id", "label"]]
        .drop_duplicates()
        .sort_values(["label_id", "label"])
        .reset_index(drop=True)
    )
    remapped_ids = {int(old_id): new_id for new_id, old_id in enumerate(label_order["label_id"].tolist())}
    df["label_id"] = df["label_id"].map(remapped_ids).astype(int)
    print(
        "Remapped label_id values to contiguous IDs:",
        {old_id: new_id for old_id, new_id in remapped_ids.items()},
    )

    if "priority_id_fixed" not in df.columns:
        df["priority_id_fixed"] = pd.NA

    df["priority_id_fixed"] = pd.to_numeric(df["priority_id_fixed"], errors="coerce")

    recovered = 0
    invalid_mask = df["priority_id_fixed"].isna() | (~df["priority_id_fixed"].isin([0, 1, 2]))
    if invalid_mask.any():
        src_col = "priority_fixed" if "priority_fixed" in df.columns else ("priority" if "priority" in df.columns else None)
        if src_col is not None:
            recovered_values = (
                df.loc[invalid_mask, src_col].astype(str).str.strip().str.lower().map(_PRIORITY_MAP)
            )
            recovered_mask = recovered_values.notna()
            recovered = int(recovered_mask.sum())
            if recovered:
                df.loc[recovered_values.index[recovered_mask], "priority_id_fixed"] = (
                    recovered_values.loc[recovered_mask].astype(int).values
                )

    invalid_mask = df["priority_id_fixed"].isna() | (~df["priority_id_fixed"].isin([0, 1, 2]))
    imputed = int(invalid_mask.sum())
    if imputed:
        df.loc[invalid_mask, "priority_id_fixed"] = DEFAULT_PRIORITY_ID

    df["priority_imputed"] = 0
    if imputed:
        df.loc[invalid_mask, "priority_imputed"] = 1

    df["priority_id_fixed"] = df["priority_id_fixed"].astype(int)
    print(f"Recovered priority rows from text labels: {recovered}")
    print(f"Imputed default Medium priority rows: {imputed}")

    before_override = df["priority_id_fixed"].copy()
    exam_mask = df["text"].map(_is_exam_urgent)
    water_mask = df["text"].map(_is_hostel_water_outage)
    df.loc[exam_mask, "priority_id_fixed"] = 2
    df.loc[water_mask, "priority_id_fixed"] = 2
    overridden = int((before_override != df["priority_id_fixed"]).sum())
    print(f"Priority overrides to High: {overridden}")

    if AUGMENT_EXAM_HIGH and (df["label"] == "Examination").any():
        exam_label_id = int(df.loc[df["label"] == "Examination", "label_id"].iloc[0])
        synth_rows = [
            {
                "text": text,
                "label": "Examination",
                "label_id": exam_label_id,
                "priority": "High",
                "priority_id_fixed": 2,
                "priority_imputed": 0,
                "source": "synthetic",
                "source_type": "synthetic",
                "text_len": len(text),
                "word_count": len(text.split()),
            }
            for text in _EXAM_SYNTHETIC
        ]
        synth_df = pd.DataFrame(synth_rows)
        for col in df.columns:
            if col not in synth_df.columns:
                synth_df[col] = None
        synth_df = synth_df[df.columns]
        df = pd.concat([df, synth_df], ignore_index=True)
        print(f"Added synthetic Examination/High rows: {len(synth_rows)}")

    df["text_len"] = df["text"].str.len().astype(int)
    df["word_count"] = df["text"].str.split().str.len().astype(int)

    conflict = df.groupby("text")[["label_id", "priority_id_fixed"]].nunique().reset_index()
    conflicting = conflict[(conflict["label_id"] > 1) | (conflict["priority_id_fixed"] > 1)]
    print(f"Conflicting duplicate texts: {len(conflicting)}")

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    print(f"Dropped duplicate texts: {before - len(df)}")

    print("\nFinal label distribution:")
    print(df["label"].value_counts().to_string())
    print("\nFinal priority distribution:")
    print(df["priority_id_fixed"].value_counts().sort_index().to_string())
    print("\nFinal category x priority table:")
    print(pd.crosstab(df["label"], df["priority_id_fixed"]).to_string())
    print(
        f"\nExamination + High rows: {len(df[(df['label'] == 'Examination') & (df['priority_id_fixed'] == 2)])}"
    )

    os.makedirs(OUT_PATH.parent, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset: {OUT_PATH} rows: {len(df)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    id_to_label = (
        df[["label_id", "label"]]
        .drop_duplicates()
        .sort_values("label_id")
        .set_index("label_id")["label"]
        .to_dict()
    )

    default_prio_map = {0: "Low", 1: "Medium", 2: "High"}
    id_to_priority = {}
    for src_col in ("priority_fixed", "priority"):
        if src_col in df.columns:
            temp = df[["priority_id_fixed", src_col]].drop_duplicates().copy()
            temp[src_col] = temp[src_col].astype(str).str.strip()

            def normalize_priority(value: str) -> str | None:
                value = str(value).strip().lower()
                if value in {"", "nan"}:
                    return None
                if value in {"low", "l", "0"}:
                    return "Low"
                if value in {"medium", "med", "m", "1"}:
                    return "Medium"
                if value in {"high", "h", "2"}:
                    return "High"
                return None

            temp[src_col] = temp[src_col].map(normalize_priority)
            mapped = (
                temp.dropna(subset=[src_col])
                .sort_values("priority_id_fixed")
                .groupby("priority_id_fixed")[src_col]
                .first()
                .to_dict()
            )
            id_to_priority.update({int(k): v for k, v in mapped.items()})

    for key, value in default_prio_map.items():
        id_to_priority.setdefault(key, value)

    with open(OUT_DIR / "id_to_label.json", "w", encoding="utf-8") as file:
        json.dump({str(k): v for k, v in id_to_label.items()}, file, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "id_to_priority.json", "w", encoding="utf-8") as file:
        json.dump({str(k): v for k, v in id_to_priority.items()}, file, ensure_ascii=False, indent=2)

    print(f"Saved mappings to: {OUT_DIR}")


if __name__ == "__main__":
    main()
