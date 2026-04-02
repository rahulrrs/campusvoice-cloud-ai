import json
import os
import random
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SOURCE_MODEL_DIR = str(PROJECT_ROOT / "outputs" / "general_complaint_model")
FALLBACK_SOURCE_MODEL = os.getenv("BACKBONE_MODEL_NAME", "roberta-base")
OUTPUT_DIR = SOURCE_MODEL_DIR

PRIMARY_DATA_PATH = str(PROJECT_ROOT / "data" / "dataset_clean.csv")
OPTIONAL_DATA_PATHS = [
    str(PROJECT_ROOT / "data" / "train.csv"),
    str(PROJECT_ROOT / "data" / "val.csv"),
    str(PROJECT_ROOT / "data" / "pseudo_feedback.csv"),
    str(PROJECT_ROOT / "data" / "frontend_feedback.csv"),
]
GENERAL_DOMAIN_DATA_PATHS = [
    str(PROJECT_ROOT / "data" / "complaint.csv"),
    str(PROJECT_ROOT / "data" / "complaints.csv"),
]
STRICT_NARRATIVE_ONLY_PATHS = {
    str(PROJECT_ROOT / "data" / "complaint.csv"),
    str(PROJECT_ROOT / "data" / "complaints.csv"),
}

TEXT_COLUMN_CANDIDATES = (
    "text",
    "complaint",
    "description",
    "content",
    "consumer complaint narrative",
)
FALLBACK_TEXT_BUILD_COLUMNS = (
    "product",
    "sub-product",
    "issue",
    "sub-issue",
)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(str(raw).strip())


def env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    value = str(raw).strip().lower()
    if value in {"none", "null", "all", "full", "-1", "unlimited"}:
        return None
    return int(value)


def env_choice(name: str, default: str, allowed: set[str]) -> str:
    raw = os.getenv(name)
    value = (str(raw).strip().lower() if raw is not None else default).strip().lower()
    if value not in allowed:
        return default
    return value


MAX_ROWS_BY_SOURCE = {
    PRIMARY_DATA_PATH: env_optional_int("MLM_MAX_ROWS_PRIMARY", None),
    str(PROJECT_ROOT / "data" / "train.csv"): env_optional_int("MLM_MAX_ROWS_TRAIN", 50000),
    str(PROJECT_ROOT / "data" / "val.csv"): env_optional_int("MLM_MAX_ROWS_VAL", 10000),
    str(PROJECT_ROOT / "data" / "pseudo_feedback.csv"): env_optional_int("MLM_MAX_ROWS_PSEUDO_FEEDBACK", 5000),
    str(PROJECT_ROOT / "data" / "frontend_feedback.csv"): env_optional_int("MLM_MAX_ROWS_FRONTEND_FEEDBACK", 5000),
    str(PROJECT_ROOT / "data" / "complaint.csv"): env_optional_int("MLM_MAX_ROWS_COMPLAINT", None),
    str(PROJECT_ROOT / "data" / "complaints.csv"): env_optional_int("MLM_MAX_ROWS_COMPLAINTS", None),
}
CHUNKED_READ_THRESHOLD_BYTES = 100 * 1024 * 1024
CHUNK_SIZE = 50000

MAX_LENGTH = env_int("MLM_MAX_LENGTH", 256)
TRAIN_FRACTION = 0.9
MLM_PROBABILITY = 0.15
SEED = 42
EPOCHS = env_int("MLM_EPOCHS", 2)
TRAIN_BATCH_SIZE = env_int("MLM_TRAIN_BATCH_SIZE", 8)
EVAL_BATCH_SIZE = env_int("MLM_EVAL_BATCH_SIZE", 8)
LEARNING_RATE = float(os.getenv("MLM_LEARNING_RATE", "2e-5"))
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS = 100
SAMPLING_MODE = env_choice("MLM_SAMPLING_MODE", "head", {"head", "reservoir"})
RESERVOIR_SCAN_MAX_ROWS = env_optional_int("MLM_RESERVOIR_SCAN_MAX_ROWS", None)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_source_model() -> str:
    required_files = (
        "config.json",
        "tokenizer_config.json",
    )
    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
    )
    if os.path.isdir(SOURCE_MODEL_DIR):
        has_required = all(
            os.path.exists(os.path.join(SOURCE_MODEL_DIR, name))
            for name in required_files
        )
        has_weights = any(
            os.path.exists(os.path.join(SOURCE_MODEL_DIR, name))
            for name in weight_files
        )
        if has_required and has_weights:
            return SOURCE_MODEL_DIR
    return FALLBACK_SOURCE_MODEL


def build_training_arguments(num_train_rows: int) -> TrainingArguments:
    supported = inspect.signature(TrainingArguments.__init__).parameters
    steps_per_epoch = max(1, num_train_rows // TRAIN_BATCH_SIZE)
    warmup_steps = max(1, int(steps_per_epoch * EPOCHS * WARMUP_RATIO))
    kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "logging_steps": LOGGING_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "seed": SEED,
    }

    if "overwrite_output_dir" in supported:
        kwargs["overwrite_output_dir"] = True
    if "warmup_steps" in supported:
        kwargs["warmup_steps"] = warmup_steps
    elif "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = WARMUP_RATIO
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in supported:
        kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in supported:
        kwargs["logging_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def build_trainer(
    model: AutoModelForMaskedLM,
    training_args: TrainingArguments,
    train_ds: Dataset,
    eval_ds: Dataset,
    collator: DataCollatorForLanguageModeling,
    tokenizer: AutoTokenizer,
) -> Trainer:
    supported = inspect.signature(Trainer.__init__).parameters
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    if "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def sample_rows(items: list[str], max_rows: int | None) -> list[str]:
    if max_rows is None or len(items) <= max_rows:
        return items
    return random.sample(items, max_rows)


def detect_text_column(columns: list[str]) -> str:
    lowered = {col.lower(): col for col in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    if any(candidate in lowered for candidate in FALLBACK_TEXT_BUILD_COLUMNS):
        return ""
    raise ValueError(
        "Could not find a supported text column in dataset. "
        f"Checked direct text columns {TEXT_COLUMN_CANDIDATES} and fallback columns {FALLBACK_TEXT_BUILD_COLUMNS}"
    )


def build_text_series(df: pd.DataFrame, path: str | None = None) -> pd.Series:
    columns = df.columns.tolist()
    text_col = detect_text_column(columns)
    if text_col:
        return normalize_text_series(df[text_col])

    if path in STRICT_NARRATIVE_ONLY_PATHS:
        return pd.Series(dtype=str)

    lowered = {col.lower(): col for col in columns}
    parts: list[pd.Series] = []
    for candidate in FALLBACK_TEXT_BUILD_COLUMNS:
        real_col = lowered.get(candidate)
        if not real_col:
            continue
        parts.append(
            df[real_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if not parts:
        return pd.Series(dtype=str)

    combined = parts[0]
    for part in parts[1:]:
        combined = combined + " | " + part
    return normalize_text_series(combined)


def load_texts_from_csv(path: str) -> list[str]:
    if not os.path.exists(path):
        return []

    print(f"Loading source: {path}")
    max_rows = MAX_ROWS_BY_SOURCE.get(path)
    file_size = os.path.getsize(path)
    if file_size >= CHUNKED_READ_THRESHOLD_BYTES:
        print(
            f"Reading large CSV in chunks from {path} "
            f"({round(file_size / 1024**3, 2)} GB, cap={max_rows})"
        )
        return load_texts_from_large_csv(path, max_rows=max_rows)

    df = pd.read_csv(path, low_memory=False)
    texts = build_text_series(df, path=path).tolist()
    print(f"Loaded {len(texts)} usable rows from {path}")
    return sample_rows(texts, max_rows)


def normalize_text_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.dropna()
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return cleaned[cleaned.str.len() > 20]


def load_texts_from_large_csv(path: str, max_rows: int | None) -> list[str]:
    sampled: list[str] = []
    seen = 0
    chunk_idx = 0
    stop_early = max_rows is not None and SAMPLING_MODE == "head"
    reservoir_scan_limit = RESERVOIR_SCAN_MAX_ROWS if SAMPLING_MODE == "reservoir" else None

    for chunk in pd.read_csv(path, low_memory=False, chunksize=CHUNK_SIZE):
        chunk_idx += 1
        texts = build_text_series(chunk, path=path).tolist()
        for text in texts:
            seen += 1
            if max_rows is None:
                sampled.append(text)
                continue
            if len(sampled) < max_rows:
                sampled.append(text)
                if stop_early and len(sampled) >= max_rows:
                    print(
                        f"  chunk {chunk_idx}: reached cap={max_rows} using head sampling; stopping early.",
                        flush=True,
                    )
                    return sampled
                continue
            if SAMPLING_MODE == "reservoir":
                replace_at = random.randint(0, seen - 1)
                if replace_at < max_rows:
                    sampled[replace_at] = text
                if reservoir_scan_limit is not None and seen >= reservoir_scan_limit:
                    print(
                        f"  chunk {chunk_idx}: reached reservoir scan cap={reservoir_scan_limit}; stopping early.",
                        flush=True,
                    )
                    return sampled
        print(
            f"  chunk {chunk_idx}: scanned={seen} sampled={len(sampled)}",
            flush=True,
        )

    return sampled


def build_corpus() -> tuple[list[str], dict[str, int]]:
    source_counts: dict[str, int] = {}
    all_texts: list[str] = []
    loaded_paths: set[str] = set()

    include_general_domain = str(os.getenv("MLM_INCLUDE_GENERAL_DOMAIN_DATA", "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    source_paths = [PRIMARY_DATA_PATH, *OPTIONAL_DATA_PATHS]
    if include_general_domain:
        source_paths.extend(GENERAL_DOMAIN_DATA_PATHS)

    for path in source_paths:
        if path in loaded_paths:
            continue
        texts = load_texts_from_csv(path)
        if not texts:
            continue
        loaded_paths.add(path)
        source_counts[path] = len(texts)
        all_texts.extend(texts)

    if not all_texts:
        raise ValueError("No texts were loaded for MLM training.")

    seen: set[str] = set()
    deduped: list[str] = []
    for text in all_texts:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)

    return deduped, source_counts


def tokenize_batch(batch: dict[str, list[str]], tokenizer: AutoTokenizer) -> dict[str, list[list[int]]]:
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_special_tokens_mask=True,
    )


def main() -> None:
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_model = resolve_source_model()
    print(f"Using source model: {base_model}")
    print(
        "MLM config:",
        {
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "include_general_domain_data": str(
                os.getenv("MLM_INCLUDE_GENERAL_DOMAIN_DATA", "true")
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            "sampling_mode": SAMPLING_MODE,
            "reservoir_scan_max_rows": RESERVOIR_SCAN_MAX_ROWS,
        },
    )

    texts, source_counts = build_corpus()
    print(f"Loaded {len(texts)} deduplicated complaint texts for backbone pretraining.")
    for path, count in source_counts.items():
        print(f"  - {path}: {count}")

    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(train_size=TRAIN_FRACTION, seed=SEED, shuffle=True)
    train_ds = split["train"]
    eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)

    train_ds = train_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train split",
    )
    eval_ds = eval_ds.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing eval split",
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    training_args = build_training_arguments(len(train_ds))

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_ds=train_ds,
        eval_ds=eval_ds,
        collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    manifest = {
        "model_name": "general complaint model",
        "output_dir": OUTPUT_DIR,
        "source_model": base_model,
        "updated_checkpoint": SOURCE_MODEL_DIR,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "deduplicated_rows": len(texts),
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "mlm_probability": MLM_PROBABILITY,
        "learning_rate": LEARNING_RATE,
        "source_counts": source_counts,
        "source_caps": MAX_ROWS_BY_SOURCE,
        "eval_metrics": eval_metrics,
    }
    with open(os.path.join(OUTPUT_DIR, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved continued-pretrained model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
