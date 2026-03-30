import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.helpers import clean_text
except Exception:
    clean_text = None
try:
    from src.utils.model_paths import load_project_env, resolve_backbone_source
except Exception:
    load_project_env = None
    resolve_backbone_source = None

DATA_DIR = PROJECT_ROOT / "data"
FILE_PATH = DATA_DIR / "dataset.csv"
MODEL_DIR = PROJECT_ROOT / "outputs" / "edu_classifier_multitask"

if callable(load_project_env):
    load_project_env(PROJECT_ROOT)

if not FILE_PATH.exists():
    alt_path = DATA_DIR / "dataset.xlsx"
    if alt_path.exists():
        FILE_PATH = alt_path

print("Loading dataset...")

def load_xlsx_without_openpyxl(path: Path) -> pd.DataFrame:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(node.text or "" for node in item.iterfind(".//a:t", ns)))

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


if FILE_PATH.suffix.lower() == ".xlsx":
    df = load_xlsx_without_openpyxl(FILE_PATH)
elif FILE_PATH.suffix.lower() == ".xls":
    raise ValueError("Please convert dataset.xls to dataset.csv or dataset.xlsx before running sanity_check.py")
else:
    df = pd.read_csv(FILE_PATH, low_memory=False)

print("Dataset loaded successfully\n")
normalized_columns = {str(col).strip().lower(): col for col in df.columns}
if "category" in normalized_columns and "label" not in normalized_columns:
    df = df.rename(columns={normalized_columns["category"]: "label"})
if "priority" in normalized_columns:
    current_priority_col = normalized_columns["priority"]
    if current_priority_col != "priority":
        df = df.rename(columns={current_priority_col: "priority"})
if "text" in normalized_columns:
    current_text_col = normalized_columns["text"]
    if current_text_col != "text":
        df = df.rename(columns={current_text_col: "text"})

print("Columns:", df.columns.tolist())
print("Total rows:", len(df))
print("\nFirst 3 rows:")
print(df.head(3))

required = {"text", "label"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")
print("\nRequired columns found")

if "label_id" not in df.columns:
    print("label_id not found. It will be generated from label during cleaning.")

priority_col = next(
    (candidate for candidate in ("priority_id_fixed", "priority_fixed", "priority") if candidate in df.columns),
    None,
)
if priority_col is None:
    raise ValueError("Missing priority column: expected one of priority_id_fixed, priority_fixed, priority")
print(f"Using priority column: {priority_col}")

print("\nMissing values:")
print(df.isnull().sum())

df["text"] = df["text"].astype(str)
if clean_text is not None:
    df["text"] = df["text"].map(clean_text)
else:
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

empty_text_count = (df["text"].str.len() == 0).sum()
print(f"\nEmpty text rows: {empty_text_count}")

duplicates = df.duplicated(subset=["text"]).sum()
print(f"Duplicate text rows: {duplicates}")

if "label_id" in df.columns:
    try:
        df["label_id"] = pd.to_numeric(df["label_id"], errors="raise").astype(int)
        print("\nlabel_id is integer")
    except Exception as exc:
        print("\nlabel_id is NOT integer:", exc)

if priority_col == "priority_id_fixed":
    try:
        df["priority_id_fixed"] = pd.to_numeric(df["priority_id_fixed"], errors="raise").astype(int)
        bad = (~df["priority_id_fixed"].isin([0, 1, 2])).sum()
        if bad == 0:
            print("priority_id_fixed is valid (0/1/2)")
        else:
            print(f"priority_id_fixed has invalid values (not 0/1/2): {bad}")
    except Exception as exc:
        print("\npriority_id_fixed is NOT integer:", exc)
else:
    normalized = (
        df[priority_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
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
        )
    )
    bad = normalized.isna().sum()
    if bad == 0:
        print(f"{priority_col} can be mapped cleanly to priority_id_fixed (0/1/2)")
    else:
        print(f"{priority_col} has unmapped values: {bad}")

print("\nLabel distribution:")
print(df["label"].value_counts())

print(f"\nPriority distribution ({priority_col}):")
print(df[priority_col].value_counts().sort_index())

df["text_length"] = df["text"].apply(lambda value: len(str(value).split()))
print("\nText length stats:")
print(df["text_length"].describe())

print("\nLongest text sample (first 500 chars):")
print(df.sort_values("text_length", ascending=False)["text"].iloc[0][:500])

print("\n===============================")
print("PIPELINE CHECK")
print("===============================")

pipeline_paths = {
    "dataset_clean": DATA_DIR / "dataset_clean.csv",
    "dataset_corrected": DATA_DIR / "dataset_corrected.csv",
    "train_split": DATA_DIR / "train.csv",
    "val_split": DATA_DIR / "val.csv",
    "test_split": DATA_DIR / "test.csv",
    "model_dir": MODEL_DIR,
    "model_weights_safe": MODEL_DIR / "model.safetensors",
    "model_weights_bin": MODEL_DIR / "pytorch_model.bin",
    "tokenizer_config": MODEL_DIR / "tokenizer_config.json",
    "id_to_label": MODEL_DIR / "id_to_label.json",
    "id_to_priority": MODEL_DIR / "id_to_priority.json",
    "metadata_config": MODEL_DIR / "metadata_config.json",
}

for name, path in pipeline_paths.items():
    print(f"{name}: {'OK' if path.exists() else 'MISSING'} -> {path}")

if callable(resolve_backbone_source):
    resolved_backbone, backbone_note = resolve_backbone_source(PROJECT_ROOT, MODEL_DIR)
    print(f"resolved_backbone: {resolved_backbone}")
    if backbone_note:
        print(f"backbone_note: {backbone_note}")

split_paths = [DATA_DIR / "train.csv", DATA_DIR / "val.csv", DATA_DIR / "test.csv"]
if all(path.exists() for path in split_paths):
    split_frames = {path.stem: pd.read_csv(path, low_memory=False) for path in split_paths}
    required_split_cols = {"text", "label", "label_id", "priority", "priority_id_fixed"}
    for split_name, split_df in split_frames.items():
        missing_cols = required_split_cols - set(split_df.columns)
        if missing_cols:
            raise ValueError(f"{split_name}.csv missing columns: {sorted(missing_cols)}")
        print(f"{split_name}.csv rows: {len(split_df)}")

    merged = pd.concat(split_frames.values(), ignore_index=True)
    label_pairs = (
        merged[["label_id", "label"]]
        .dropna()
        .assign(label=lambda frame: frame["label"].astype(str).str.strip())
        .drop_duplicates()
    )
    bad_label_ids = label_pairs.groupby("label_id")["label"].nunique()
    conflicts = bad_label_ids[bad_label_ids > 1]
    if not conflicts.empty:
        raise ValueError(f"Split label mapping conflicts detected: {conflicts.to_dict()}")
    print(f"split_label_count: {label_pairs['label_id'].nunique()}")

    prio_pairs = (
        merged[["priority_id_fixed", "priority"]]
        .dropna()
        .assign(priority=lambda frame: frame["priority"].astype(str).str.strip())
        .drop_duplicates()
    )
    bad_prio_ids = prio_pairs.groupby("priority_id_fixed")["priority"].nunique()
    prio_conflicts = bad_prio_ids[bad_prio_ids > 1]
    if not prio_conflicts.empty:
        raise ValueError(f"Split priority mapping conflicts detected: {prio_conflicts.to_dict()}")
    print(f"split_priority_values: {sorted(prio_pairs['priority'].unique().tolist())}")

    label_map_path = MODEL_DIR / "id_to_label.json"
    if label_map_path.exists():
        label_map = pd.read_json(label_map_path, typ="series")
        print(f"saved_label_map_count: {len(label_map)}")

print("\n===============================")
print("GPU CHECK")
print("===============================")
if torch.cuda.is_available():
    print("GPU Available:", torch.cuda.get_device_name(0))
else:
    print("GPU NOT Available")

print("\nSanity check completed successfully")
