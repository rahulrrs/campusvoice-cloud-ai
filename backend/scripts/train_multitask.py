import json
import inspect
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.complaint_ml import (
    METADATA_FEATURE_NAMES,
    build_metadata_feature_map,
    feature_map_to_vector,
    fit_metadata_scaler,
    scale_feature_vector,
)
from src.utils.model_paths import load_project_env, resolve_backbone_source

load_project_env(PROJECT_ROOT)

# ---------------- CONFIG ----------------
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "edu_classifier_multitask"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
PSEUDO_FEEDBACK_PATH = DATA_DIR / "pseudo_feedback.csv"
FRONTEND_FEEDBACK_PATH = DATA_DIR / "frontend_feedback.csv"
OUT_DIR = OUTPUT_DIR

DEFAULT_MAX_LENGTH = 192
SEED = 42
DEFAULT_EPOCHS = 6
DEFAULT_BATCH = 8
DEFAULT_GRAD_ACCUM = 2
LR = 2.0e-5
WEIGHT_DECAY = 0.01

LAMBDA_LABEL = float(os.getenv("LAMBDA_LABEL", "1.05"))
LAMBDA_PRIORITY = float(os.getenv("LAMBDA_PRIORITY", "1.30"))

LABEL_FOCAL_GAMMA = 1.5
PRIORITY_FOCAL_GAMMA = float(os.getenv("PRIORITY_FOCAL_GAMMA", "1.2"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.04"))
PRIORITY_SMOOTHING = float(os.getenv("PRIORITY_SMOOTHING", "0.02"))

MAX_PER_CLASS = 0
OVERSAMPLE_HIGH_PRIORITY = 1

USE_WEIGHTED_SAMPLER = "auto"
SAMPLER_LABEL_EXP = 0.6
SAMPLER_PRIORITY_EXP = 0.8
USE_PSEUDO_FEEDBACK = False
MAX_PSEUDO_FEEDBACK_ROWS = 5000
USE_FRONTEND_FEEDBACK = False
MAX_FRONTEND_FEEDBACK_ROWS = 10000
REVIEWED_NOTES_WEIGHT_BOOST = float(os.getenv("REVIEWED_NOTES_WEIGHT_BOOST", "1.15"))
REVIEWED_CORRECTION_WEIGHT_BOOST = float(os.getenv("REVIEWED_CORRECTION_WEIGHT_BOOST", "1.35"))
# ---------------------------------------

REVIEW_NOTE_FEATURE_NAMES = [
    "review_note_present",
    "review_note_text_len_chars",
    "review_note_word_count",
    "review_note_urgency_term_hits",
    "review_note_deadline_term_hits",
    "review_note_negative_term_hits",
    "review_note_harassment_term_hits",
    "review_note_academic_term_hits",
    "review_note_infrastructure_term_hits",
    "review_status_approved_flag",
]


def derive_id_maps_from_splits(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    merged = pd.concat([train_df, val_df], ignore_index=True)

    label_pairs = (
        merged[["label_id", "label"]]
        .dropna()
        .assign(label=lambda frame: frame["label"].astype(str).str.strip())
        .drop_duplicates()
    )
    label_counts = label_pairs.groupby("label_id")["label"].nunique()
    if (label_counts > 1).any():
        raise ValueError(
            f"Inconsistent label names for label_id values: {label_counts[label_counts > 1].to_dict()}"
        )
    id_to_label = {
        int(row["label_id"]): str(row["label"]).strip()
        for _, row in label_pairs.sort_values("label_id").iterrows()
    }

    priority_source_col = "priority" if "priority" in merged.columns else None
    id_to_priority: dict[int, str] = {}
    if priority_source_col is not None:
        prio_pairs = (
            merged[["priority_id_fixed", priority_source_col]]
            .dropna()
            .assign(priority=lambda frame: frame[priority_source_col].astype(str).str.strip())
            .drop_duplicates(subset=["priority_id_fixed", "priority"])
        )
        prio_counts = prio_pairs.groupby("priority_id_fixed")["priority"].nunique()
        if (prio_counts > 1).any():
            raise ValueError(
                f"Inconsistent priority names for priority_id_fixed values: {prio_counts[prio_counts > 1].to_dict()}"
            )
        id_to_priority = {
            int(row["priority_id_fixed"]): str(row["priority"]).strip()
            for _, row in prio_pairs.sort_values("priority_id_fixed").iterrows()
        }

    default_priority_names = {0: "Low", 1: "Medium", 2: "High"}
    for key, value in default_priority_names.items():
        id_to_priority.setdefault(key, value)

    return id_to_label, id_to_priority


def build_training_arguments(
    use_fp16: bool,
    train_rows: int,
    batch_size: int,
    num_epochs: int,
    grad_accum_steps: int,
) -> TrainingArguments:
    supported = inspect.signature(TrainingArguments.__init__).parameters
    effective_batch = max(batch_size * grad_accum_steps, 1)
    steps_per_epoch = max(1, train_rows // effective_batch)
    warmup_steps = max(1, int(steps_per_epoch * num_epochs * 0.08))

    kwargs = {
        "output_dir": OUT_DIR,
        "learning_rate": LR,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": max(batch_size, 16),
        "num_train_epochs": num_epochs,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": grad_accum_steps,
        "logging_steps": max(10, steps_per_epoch // 3),
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "fp16": use_fp16,
        "seed": SEED,
        "dataloader_num_workers": 0,
    }

    if "warmup_steps" in supported:
        kwargs["warmup_steps"] = warmup_steps
    elif "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = 0.08
    if "lr_scheduler_type" in supported:
        kwargs["lr_scheduler_type"] = "cosine"
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in supported:
        kwargs["save_strategy"] = "epoch"
    if "report_to" in supported:
        kwargs["report_to"] = ["tensorboard"]
    if "save_safetensors" in supported:
        kwargs["save_safetensors"] = True

    return TrainingArguments(**kwargs)


def choose_runtime_hyperparams(train_df: pd.DataFrame) -> dict[str, int]:
    train_rows = int(len(train_df))
    avg_words = float(train_df["text"].astype(str).str.split().str.len().mean())

    if torch.cuda.is_available():
        if train_rows <= 2500:
            batch_size = 24
            grad_accum_steps = 1
            num_epochs = 8
        elif train_rows <= 6000:
            batch_size = 20
            grad_accum_steps = 1
            num_epochs = 7
        else:
            batch_size = DEFAULT_BATCH
            grad_accum_steps = DEFAULT_GRAD_ACCUM
            num_epochs = DEFAULT_EPOCHS
    else:
        batch_size = DEFAULT_BATCH
        grad_accum_steps = DEFAULT_GRAD_ACCUM
        num_epochs = DEFAULT_EPOCHS + (1 if train_rows <= 2500 else 0)

    # Short complaint-style texts usually converge better with a slightly longer schedule.
    if avg_words <= 14:
        num_epochs += 1

    return {
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "num_epochs": num_epochs,
    }


def choose_dynamic_max_length(tokenizer, texts: pd.Series) -> int:
    sample_texts = texts.astype(str).tolist()
    if len(sample_texts) > 1500:
        sample_texts = sample_texts[:1500]
    token_lengths = [len(tokenizer.tokenize(text)) for text in sample_texts if str(text).strip()]
    if not token_lengths:
        return DEFAULT_MAX_LENGTH
    p95 = int(np.percentile(token_lengths, 95))
    if p95 <= 48:
        return 64
    if p95 <= 72:
        return 96
    if p95 <= 112:
        return 128
    if p95 <= 160:
        return 160
    return DEFAULT_MAX_LENGTH


def should_use_weighted_sampler(label_counts: pd.Series, prio_counts: pd.Series) -> bool:
    if USE_WEIGHTED_SAMPLER is True:
        return True
    if USE_WEIGHTED_SAMPLER is False:
        return False
    label_ratio = float(label_counts.max() / max(label_counts.min(), 1))
    prio_ratio = float(prio_counts.max() / max(prio_counts.min(), 1))
    return label_ratio >= 1.35 or prio_ratio >= 1.20


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_effective_num_weights(
    counts: pd.Series, num_classes: int, beta: float = 0.999
) -> torch.Tensor:
    weights = []
    for i in range(num_classes):
        c = int(counts.get(i, 0))
        if c <= 0:
            weights.append(1.0)
            continue
        effective_num = 1.0 - (beta ** c)
        w = (1.0 - beta) / max(effective_num, 1e-12)
        weights.append(w)
    t = torch.tensor(weights, dtype=torch.float32)
    return t / t.mean()


def focal_smoothed_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    num_classes = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    with torch.no_grad():
        if num_classes <= 1:
            smooth = 0.0
        else:
            smooth = label_smoothing
        target_dist = torch.full_like(log_probs, smooth / max(num_classes - 1, 1))
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smooth)

    ce = -(target_dist * log_probs).sum(dim=1)
    pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
    focal_factor = (1.0 - pt) ** gamma

    if class_weights is not None:
        ce = ce * class_weights[targets]

    return (focal_factor * ce).mean()


def build_priority_cost_matrix(num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for actual in range(num_classes):
        for predicted in range(num_classes):
            if actual == predicted:
                continue
            matrix[actual, predicted] = abs(actual - predicted) * 0.4
    if num_classes >= 3:
        matrix[2, 0] = 1.35
        matrix[2, 1] = 0.7
        matrix[1, 0] = 0.45
        matrix[0, 2] = 0.9
        matrix[0, 1] = 0.3
        matrix[1, 2] = 0.55
    return matrix


def cost_sensitive_focal_smoothed_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    gamma: float,
    label_smoothing: float,
    cost_matrix: torch.Tensor | None,
) -> torch.Tensor:
    base_loss = focal_smoothed_ce(
        logits=logits,
        targets=targets,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing,
    )
    if cost_matrix is None:
        return base_loss

    probs = torch.softmax(logits, dim=-1)
    expected_cost = (probs * cost_matrix[targets]).sum(dim=1)
    multiplier = 1.0 + expected_cost
    ce_per_item = torch.nn.functional.cross_entropy(
        logits,
        targets,
        reduction="none",
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    return (ce_per_item * multiplier).mean() * 0.5 + base_loss * 0.5


class EncoderMultiTask(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_priority: int,
        label_weights: torch.Tensor,
        priority_weights: torch.Tensor,
        label_metadata_dim: int = 0,
        priority_metadata_dim: int = 0,
        priority_cost_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.label_metadata_dim = label_metadata_dim
        self.priority_metadata_dim = priority_metadata_dim

        self.dropout = nn.Dropout(0.1)

        self.label_dropout = nn.Dropout(0.2)
        if label_metadata_dim > 0:
            self.label_meta_proj = nn.Sequential(
                nn.Linear(label_metadata_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.label_meta_proj = None
        self.label_hidden = nn.Linear(hidden, hidden // 2)
        self.label_head = nn.Linear(hidden // 2, num_labels)

        if priority_metadata_dim > 0:
            self.priority_meta_proj = nn.Sequential(
                nn.Linear(priority_metadata_dim, hidden // 4),
                nn.LayerNorm(hidden // 4),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            prio_input = hidden + (hidden // 4)
        else:
            self.priority_meta_proj = None
            prio_input = hidden

        self.prio_dropout = nn.Dropout(0.2)
        self.prio_hidden = nn.Linear(prio_input, hidden // 4)
        self.prio_head = nn.Linear(hidden // 4, num_priority)

        self.act = nn.GELU()

        self.register_buffer("label_weights", label_weights)
        self.register_buffer("priority_weights", priority_weights)
        if priority_cost_matrix is None:
            priority_cost_matrix = build_priority_cost_matrix(num_priority)
        self.register_buffer("priority_cost_matrix", priority_cost_matrix)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        priority_labels=None,
        metadata_features=None,
        **kwargs,
    ):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])

        label_input = pooled
        if self.label_metadata_dim > 0 and metadata_features is not None:
            label_input = label_input + self.label_meta_proj(metadata_features.float())
        label_logits = self.label_head(self.act(self.label_hidden(self.label_dropout(label_input))))
        priority_input = pooled
        if self.priority_metadata_dim > 0 and metadata_features is not None:
            meta = metadata_features.float()
            meta_repr = self.priority_meta_proj(meta)
            priority_input = torch.cat([priority_input, meta_repr], dim=-1)
        prio_logits = self.prio_head(self.act(self.prio_hidden(self.prio_dropout(priority_input))))

        loss = None
        if labels is not None and priority_labels is not None:
            loss_label = focal_smoothed_ce(
                logits=label_logits,
                targets=labels,
                class_weights=self.label_weights,
                gamma=LABEL_FOCAL_GAMMA,
                label_smoothing=LABEL_SMOOTHING,
            )
            loss_prio = focal_smoothed_ce(
                logits=prio_logits,
                targets=priority_labels,
                class_weights=self.priority_weights,
                gamma=PRIORITY_FOCAL_GAMMA,
                label_smoothing=PRIORITY_SMOOTHING,
            )
            loss_prio = cost_sensitive_focal_smoothed_ce(
                logits=prio_logits,
                targets=priority_labels,
                class_weights=self.priority_weights,
                gamma=PRIORITY_FOCAL_GAMMA,
                label_smoothing=PRIORITY_SMOOTHING,
                cost_matrix=self.priority_cost_matrix,
            )
            loss = (LAMBDA_LABEL * loss_label) + (LAMBDA_PRIORITY * loss_prio)

        return {
            "loss": loss,
            "label_logits": label_logits,
            "priority_logits": prio_logits,
        }


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, sample_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            priority_labels=inputs.get("priority_labels"),
            metadata_features=inputs.get("metadata_features"),
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        labels = inputs.get("labels")
        priority_labels = inputs.get("priority_labels")

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=None,
                priority_labels=None,
                metadata_features=inputs.get("metadata_features"),
            )

        label_logits = outputs["label_logits"]
        prio_logits = outputs["priority_logits"]

        loss = None
        if labels is not None and priority_labels is not None:
            with torch.no_grad():
                out_loss = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=labels,
                    priority_labels=priority_labels,
                    metadata_features=inputs.get("metadata_features"),
                )
            loss = out_loss["loss"].detach()

        if prediction_loss_only:
            return (loss, None, None)

        stacked_labels = torch.stack([labels, priority_labels], dim=1)
        return (loss, (label_logits, prio_logits), stacked_labels)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.sample_weights is None:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )


def compute_metrics_fn(eval_pred):
    (label_logits, prio_logits), labels = eval_pred

    y_label_true = labels[:, 0]
    y_prio_true = labels[:, 1]

    y_label_pred = np.argmax(label_logits, axis=1)
    y_prio_pred = np.argmax(prio_logits, axis=1)

    label_f1 = f1_score(y_label_true, y_label_pred, average="macro", zero_division=0)
    prio_f1 = f1_score(y_prio_true, y_prio_pred, average="macro", zero_division=0)

    return {
        "label_acc": accuracy_score(y_label_true, y_label_pred),
        "label_f1_macro": label_f1,
        "prio_acc": accuracy_score(y_prio_true, y_prio_pred),
        "prio_f1_macro": prio_f1,
        "f1_macro": (label_f1 + prio_f1) / 2.0,
    }


def validate_targets(df: pd.DataFrame, num_labels: int, num_priority: int, split_name: str) -> None:
    missing = df[["text", "labels", "priority_labels"]].isna().sum()
    if int(missing.sum()) > 0:
        raise ValueError(f"{split_name} has missing required values:\n{missing.to_string()}")

    bad_labels = df[~df["labels"].between(0, num_labels - 1)]
    if not bad_labels.empty:
        sample = bad_labels[["text", "labels"]].head(5).to_dict("records")
        raise ValueError(
            f"{split_name} contains out-of-range labels for 0..{num_labels - 1}: {sample}"
        )

    bad_priority = df[~df["priority_labels"].between(0, num_priority - 1)]
    if not bad_priority.empty:
        sample = bad_priority[["text", "priority_labels"]].head(5).to_dict("records")
        raise ValueError(
            f"{split_name} contains out-of-range priority labels for 0..{num_priority - 1}: {sample}"
        )


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading split datasets...")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    val_df = pd.read_csv(VAL_PATH, low_memory=False)

    required_cols = {"text", "label_id", "priority_id_fixed"}
    missing = required_cols - set(train_df.columns)
    if missing:
        raise ValueError(f"Missing columns in train.csv: {missing}")

    map_required_cols = {"text", "label", "label_id", "priority", "priority_id_fixed"}
    merged_columns = set(train_df.columns) | set(val_df.columns)
    missing_map_cols = map_required_cols - merged_columns
    if missing_map_cols:
        raise ValueError(f"Missing columns required to derive label maps from train/val: {missing_map_cols}")

    id_to_label, id_to_priority = derive_id_maps_from_splits(train_df, val_df)

    train_df = train_df[["text", "label_id", "priority_id_fixed"]].rename(
        columns={"label_id": "labels", "priority_id_fixed": "priority_labels"}
    )
    val_df = val_df[["text", "label_id", "priority_id_fixed"]].rename(
        columns={"label_id": "labels", "priority_id_fixed": "priority_labels"}
    )

    num_labels = len(id_to_label)
    num_priority = len(id_to_priority)
    print(f"Labels: {num_labels} | Priority classes: {num_priority}")

    train_df = train_df.dropna(subset=["text", "labels", "priority_labels"]).copy()
    val_df = val_df.dropna(subset=["text", "labels", "priority_labels"]).copy()
    train_df["labels"] = train_df["labels"].astype(int)
    train_df["priority_labels"] = train_df["priority_labels"].astype(int)
    val_df["labels"] = val_df["labels"].astype(int)
    val_df["priority_labels"] = val_df["priority_labels"].astype(int)
    validate_targets(train_df, num_labels=num_labels, num_priority=num_priority, split_name="train")
    validate_targets(val_df, num_labels=num_labels, num_priority=num_priority, split_name="val")

    metadata_feature_names = METADATA_FEATURE_NAMES + REVIEW_NOTE_FEATURE_NAMES

    def build_review_note_feature_map(row: pd.Series) -> dict[str, float]:
        review_notes = str(row.get("review_notes") or "").strip()
        review_status = str(row.get("review_status") or "").strip().lower()
        if not review_notes:
            return {name: 0.0 for name in REVIEW_NOTE_FEATURE_NAMES}

        note_feature_map = build_metadata_feature_map(text=review_notes)
        return {
            "review_note_present": 1.0,
            "review_note_text_len_chars": float(note_feature_map.get("text_len_chars", 0.0)),
            "review_note_word_count": float(note_feature_map.get("word_count", 0.0)),
            "review_note_urgency_term_hits": float(note_feature_map.get("urgency_term_hits", 0.0)),
            "review_note_deadline_term_hits": float(note_feature_map.get("deadline_term_hits", 0.0)),
            "review_note_negative_term_hits": float(note_feature_map.get("negative_term_hits", 0.0)),
            "review_note_harassment_term_hits": float(note_feature_map.get("harassment_term_hits", 0.0)),
            "review_note_academic_term_hits": float(note_feature_map.get("academic_term_hits", 0.0)),
            "review_note_infrastructure_term_hits": float(note_feature_map.get("infrastructure_term_hits", 0.0)),
            "review_status_approved_flag": float(review_status == "approved"),
        }

    def enrich_with_metadata(df: pd.DataFrame, scaler: dict | None = None) -> tuple[pd.DataFrame, dict | None]:
        feature_rows: list[list[float]] = []
        for _, row in df.iterrows():
            feature_map = build_metadata_feature_map(text=str(row["text"]))
            feature_map.update(build_review_note_feature_map(row))
            feature_rows.append(feature_map_to_vector(feature_map, metadata_feature_names))
        active_scaler = scaler or fit_metadata_scaler(feature_rows, metadata_feature_names)
        df = df.copy()
        df["metadata_features"] = [
            scale_feature_vector(row, active_scaler)
            for row in feature_rows
        ]
        return df, active_scaler

    if USE_FRONTEND_FEEDBACK and os.path.exists(FRONTEND_FEEDBACK_PATH):
        try:
            fb = pd.read_csv(FRONTEND_FEEDBACK_PATH, low_memory=False)
            fb_need = {"text", "label_id", "priority_id_fixed"}
            if fb_need.issubset(set(fb.columns)):
                fb = fb[["text", "label_id", "priority_id_fixed"]].rename(
                    columns={"label_id": "labels", "priority_id_fixed": "priority_labels"}
                )
                fb = fb.dropna(subset=["text", "labels", "priority_labels"]).copy()
                fb["labels"] = fb["labels"].astype(int)
                fb["priority_labels"] = fb["priority_labels"].astype(int)
                fb = fb[
                    fb["labels"].between(0, num_labels - 1)
                    & fb["priority_labels"].between(0, num_priority - 1)
                ]
                if MAX_FRONTEND_FEEDBACK_ROWS > 0 and len(fb) > MAX_FRONTEND_FEEDBACK_ROWS:
                    fb = fb.tail(MAX_FRONTEND_FEEDBACK_ROWS)
                before = len(train_df)
                train_df = pd.concat([train_df, fb], ignore_index=True)
                train_df = train_df.drop_duplicates(
                    subset=["text", "labels", "priority_labels"], keep="last"
                ).reset_index(drop=True)
                print(
                    f"Frontend feedback added: {len(train_df) - before} "
                    f"(from {len(fb)} rows in {str(FRONTEND_FEEDBACK_PATH)})"
                )
            else:
                print(f"Skipping frontend feedback: missing columns in {str(FRONTEND_FEEDBACK_PATH)}")
        except Exception as e:
            print(f"Skipping frontend feedback due to read/parse error: {e}")

    if USE_PSEUDO_FEEDBACK and os.path.exists(PSEUDO_FEEDBACK_PATH):
        try:
            fb = pd.read_csv(PSEUDO_FEEDBACK_PATH, low_memory=False)
            fb_need = {"text", "label_id", "priority_id_fixed"}
            if fb_need.issubset(set(fb.columns)):
                fb = fb[["text", "label_id", "priority_id_fixed"]].rename(
                    columns={"label_id": "labels", "priority_id_fixed": "priority_labels"}
                )
                fb = fb.dropna(subset=["text", "labels", "priority_labels"]).copy()
                fb["labels"] = fb["labels"].astype(int)
                fb["priority_labels"] = fb["priority_labels"].astype(int)
                fb = fb[
                    fb["labels"].between(0, num_labels - 1)
                    & fb["priority_labels"].between(0, num_priority - 1)
                ]
                if MAX_PSEUDO_FEEDBACK_ROWS > 0 and len(fb) > MAX_PSEUDO_FEEDBACK_ROWS:
                    fb = fb.tail(MAX_PSEUDO_FEEDBACK_ROWS)
                before = len(train_df)
                train_df = pd.concat([train_df, fb], ignore_index=True)
                train_df = train_df.drop_duplicates(
                    subset=["text", "labels", "priority_labels"], keep="last"
                ).reset_index(drop=True)
                print(
                    f"Pseudo feedback added: {len(train_df) - before} "
                    f"(from {len(fb)} rows in {str(PSEUDO_FEEDBACK_PATH)})"
                )
            else:
                print(f"Skipping pseudo feedback: missing columns in {str(PSEUDO_FEEDBACK_PATH)}")
        except Exception as e:
            print(f"Skipping pseudo feedback due to read/parse error: {e}")

    print(f"Before capping total training samples: {len(train_df)}")
    if MAX_PER_CLASS and MAX_PER_CLASS > 0:
        train_df = (
            pd.concat(
                [
                    g.sample(min(len(g), MAX_PER_CLASS), random_state=SEED)
                    for _, g in train_df.groupby("labels")
                ]
            )
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )
        print(f"After capping (max {MAX_PER_CLASS}/class): {len(train_df)}")
    else:
        print("Per-class capping disabled for this run.")

    high_rows = train_df[train_df["priority_labels"] == 2]
    print(f"High-priority rows before oversample: {len(high_rows)}")
    if OVERSAMPLE_HIGH_PRIORITY > 1 and len(high_rows) > 0:
        extra = pd.concat([high_rows] * (OVERSAMPLE_HIGH_PRIORITY - 1), ignore_index=True)
        train_df = (
            pd.concat([train_df, extra], ignore_index=True)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )
        print(f"After oversampling High x{OVERSAMPLE_HIGH_PRIORITY}: {len(train_df)}")

    label_counts = train_df["labels"].value_counts().sort_index()
    prio_counts = train_df["priority_labels"].value_counts().sort_index()

    label_w = build_effective_num_weights(label_counts, num_labels, beta=0.999)
    prio_w = build_effective_num_weights(prio_counts, num_priority, beta=0.99)

    print("Label weights (first 10):", [round(x, 4) for x in label_w.tolist()[:10]])
    print("Priority weights:", [round(x, 4) for x in prio_w.tolist()])

    label_inv = train_df["labels"].map(
        lambda x: 1.0 / max(int(label_counts.get(int(x), 1)), 1)
    )
    prio_inv = train_df["priority_labels"].map(
        lambda x: 1.0 / max(int(prio_counts.get(int(x), 1)), 1)
    )
    sample_weights = (label_inv.pow(SAMPLER_LABEL_EXP) * prio_inv.pow(SAMPLER_PRIORITY_EXP)).values
    if "review_has_notes" in train_df.columns:
        reviewed_notes_mask = pd.to_numeric(train_df["review_has_notes"], errors="coerce").fillna(0).astype(int) > 0
        if reviewed_notes_mask.any():
            sample_weights = sample_weights * np.where(reviewed_notes_mask.to_numpy(), REVIEWED_NOTES_WEIGHT_BOOST, 1.0)
            print(
                f"Applied reviewed-notes weight boost x{REVIEWED_NOTES_WEIGHT_BOOST} "
                f"to {int(reviewed_notes_mask.sum())} training rows."
            )
    if "review_was_corrected" in train_df.columns:
        reviewed_corrected_mask = (
            pd.to_numeric(train_df["review_was_corrected"], errors="coerce").fillna(0).astype(int) > 0
        )
        if reviewed_corrected_mask.any():
            sample_weights = sample_weights * np.where(
                reviewed_corrected_mask.to_numpy(),
                REVIEWED_CORRECTION_WEIGHT_BOOST,
                1.0,
            )
            print(
                f"Applied reviewed-correction weight boost x{REVIEWED_CORRECTION_WEIGHT_BOOST} "
                f"to {int(reviewed_corrected_mask.sum())} training rows."
            )
    sample_weights = sample_weights / sample_weights.mean()
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    base_model, backbone_note = resolve_backbone_source(PROJECT_ROOT, OUT_DIR)
    if backbone_note:
        print(backbone_note)
    print("Using base model:", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dynamic_max_length = choose_dynamic_max_length(tokenizer, train_df["text"])
    runtime_cfg = choose_runtime_hyperparams(train_df)
    use_weighted_sampler = should_use_weighted_sampler(label_counts, prio_counts)
    print("Dynamic max_length:", dynamic_max_length)
    print(
        "Runtime config:",
        {
            "batch_size": runtime_cfg["batch_size"],
            "grad_accum_steps": runtime_cfg["grad_accum_steps"],
            "num_epochs": runtime_cfg["num_epochs"],
            "weighted_sampler": use_weighted_sampler,
        },
    )

    train_df, metadata_scaler = enrich_with_metadata(train_df)
    val_df, _ = enrich_with_metadata(val_df, scaler=metadata_scaler)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=dynamic_max_length)

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = EncoderMultiTask(
        model_name=base_model,
        num_labels=num_labels,
        num_priority=num_priority,
        label_weights=label_w,
        priority_weights=prio_w,
        label_metadata_dim=len(metadata_feature_names),
        priority_metadata_dim=len(metadata_feature_names),
    )

    use_fp16 = torch.cuda.is_available()
    print("CUDA available:", use_fp16)
    if use_fp16:
        print("GPU:", torch.cuda.get_device_name(0))

    training_args = build_training_arguments(
        use_fp16=use_fp16,
        train_rows=len(train_df),
        batch_size=runtime_cfg["batch_size"],
        num_epochs=runtime_cfg["num_epochs"],
        grad_accum_steps=runtime_cfg["grad_accum_steps"],
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        sample_weights=sample_weights if use_weighted_sampler else None,
    )

    print("Training (label + priority multitask)...")
    trainer.train()

    print("Saving model + tokenizer + mappings...")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    with open(os.path.join(OUT_DIR, "id_to_label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_label.items()}, f, indent=2)

    with open(os.path.join(OUT_DIR, "id_to_priority.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_priority.items()}, f, indent=2)

    with open(os.path.join(OUT_DIR, "metadata_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_scaler, f, indent=2)

    print("Done")
    print(f"Training outputs saved under: {OUT_DIR}")
    print(f"TensorBoard logs: tensorboard --logdir {OUT_DIR}")


if __name__ == "__main__":
    main()
