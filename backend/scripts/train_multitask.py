import json
import inspect
import os
import random
import sys

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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------- CONFIG ----------------
TRAIN_PATH = r"data\train.csv"
VAL_PATH = r"data\val.csv"
PSEUDO_FEEDBACK_PATH = r"data\pseudo_feedback.csv"
FRONTEND_FEEDBACK_PATH = r"data\frontend_feedback.csv"
OUT_DIR = r"outputs\edu_classifier_multitask"

BASE_MODEL = r"outputs\general_complaint_model"
LEGACY_BASE_MODEL = r"outputs\distilbert_cfpb_mlm"
FALLBACK_MODEL = "distilbert-base-uncased"

MAX_LENGTH = 256
SEED = 42
EPOCHS = 10
BATCH = 16
LR = 1.8e-5
WEIGHT_DECAY = 0.01

LAMBDA_LABEL = 1.0
LAMBDA_PRIORITY = 1.8

LABEL_FOCAL_GAMMA = 1.5
PRIORITY_FOCAL_GAMMA = 1.0
LABEL_SMOOTHING = 0.05
PRIORITY_SMOOTHING = 0.03

MAX_PER_CLASS = 650
OVERSAMPLE_HIGH_PRIORITY = 2

USE_WEIGHTED_SAMPLER = True
SAMPLER_LABEL_EXP = 0.6
SAMPLER_PRIORITY_EXP = 0.8
USE_PSEUDO_FEEDBACK = True
MAX_PSEUDO_FEEDBACK_ROWS = 5000
USE_FRONTEND_FEEDBACK = True
MAX_FRONTEND_FEEDBACK_ROWS = 10000
# ---------------------------------------


def build_training_arguments(use_fp16: bool, train_rows: int) -> TrainingArguments:
    supported = inspect.signature(TrainingArguments.__init__).parameters
    steps_per_epoch = max(1, train_rows // max(BATCH, 1))
    warmup_steps = max(1, int(steps_per_epoch * EPOCHS * 0.08))

    kwargs = {
        "output_dir": OUT_DIR,
        "learning_rate": LR,
        "per_device_train_batch_size": BATCH,
        "per_device_eval_batch_size": BATCH,
        "num_train_epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 2,
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "fp16": use_fp16,
        "seed": SEED,
    }

    if "warmup_steps" in supported:
        kwargs["warmup_steps"] = warmup_steps
    elif "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = 0.08
    if "lr_scheduler_type" in supported:
        kwargs["lr_scheduler_type"] = "cosine"
    if "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"
    if "eval_steps" in supported:
        kwargs["eval_steps"] = 200
    if "save_strategy" in supported:
        kwargs["save_strategy"] = "steps"
    if "save_steps" in supported:
        kwargs["save_steps"] = 200
    if "report_to" in supported:
        kwargs["report_to"] = ["tensorboard"]

    return TrainingArguments(**kwargs)


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


class DistilBertMultiTask(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_priority: int,
        label_weights: torch.Tensor,
        priority_weights: torch.Tensor,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.label_dropout = nn.Dropout(0.2)
        self.label_hidden = nn.Linear(hidden, hidden // 2)
        self.label_head = nn.Linear(hidden // 2, num_labels)

        self.prio_dropout = nn.Dropout(0.2)
        self.prio_hidden = nn.Linear(hidden, hidden // 4)
        self.prio_head = nn.Linear(hidden // 4, num_priority)

        self.act = nn.GELU()

        self.register_buffer("label_weights", label_weights)
        self.register_buffer("priority_weights", priority_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        priority_labels=None,
        **kwargs,
    ):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])

        label_logits = self.label_head(self.act(self.label_hidden(self.label_dropout(pooled))))
        prio_logits = self.prio_head(self.act(self.prio_hidden(self.prio_dropout(pooled))))

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

    train_df = train_df[["text", "label_id", "priority_id_fixed"]].rename(
        columns={"label_id": "labels", "priority_id_fixed": "priority_labels"}
    )
    val_df = val_df[["text", "label_id", "priority_id_fixed"]].rename(
        columns={"label_id": "labels", "priority_id_fixed": "priority_labels"}
    )

    label_map_path = os.path.join(OUT_DIR, "id_to_label.json")
    prio_map_path = os.path.join(OUT_DIR, "id_to_priority.json")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Missing: {label_map_path} (run clean_dataset.py first)")
    if not os.path.exists(prio_map_path):
        raise FileNotFoundError(f"Missing: {prio_map_path} (run clean_dataset.py first)")

    with open(label_map_path, "r", encoding="utf-8") as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    with open(prio_map_path, "r", encoding="utf-8") as f:
        id_to_priority = {int(k): v for k, v in json.load(f).items()}

    num_labels = len(id_to_label)
    num_priority = len(id_to_priority)
    print(f"Labels: {num_labels} | Priority classes: {num_priority}")

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
                    f"(from {len(fb)} rows in {FRONTEND_FEEDBACK_PATH})"
                )
            else:
                print(f"Skipping frontend feedback: missing columns in {FRONTEND_FEEDBACK_PATH}")
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
                    f"(from {len(fb)} rows in {PSEUDO_FEEDBACK_PATH})"
                )
            else:
                print(f"Skipping pseudo feedback: missing columns in {PSEUDO_FEEDBACK_PATH}")
        except Exception as e:
            print(f"Skipping pseudo feedback due to read/parse error: {e}")

    print(f"Before capping total training samples: {len(train_df)}")
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

    label_w = build_effective_num_weights(label_counts, num_labels, beta=0.9995)
    prio_w = build_effective_num_weights(prio_counts, num_priority, beta=0.995)

    print("Label weights (first 10):", [round(x, 4) for x in label_w.tolist()[:10]])
    print("Priority weights:", [round(x, 4) for x in prio_w.tolist()])

    label_inv = train_df["labels"].map(
        lambda x: 1.0 / max(int(label_counts.get(int(x), 1)), 1)
    )
    prio_inv = train_df["priority_labels"].map(
        lambda x: 1.0 / max(int(prio_counts.get(int(x), 1)), 1)
    )
    sample_weights = (label_inv.pow(SAMPLER_LABEL_EXP) * prio_inv.pow(SAMPLER_PRIORITY_EXP)).values
    sample_weights = sample_weights / sample_weights.mean()
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    if os.path.isdir(BASE_MODEL):
        base_model = BASE_MODEL
    elif os.path.isdir(LEGACY_BASE_MODEL):
        base_model = LEGACY_BASE_MODEL
    else:
        base_model = FALLBACK_MODEL
    print("Using base model:", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = DistilBertMultiTask(
        model_name=base_model,
        num_labels=num_labels,
        num_priority=num_priority,
        label_weights=label_w,
        priority_weights=prio_w,
    )

    use_fp16 = torch.cuda.is_available()
    print("CUDA available:", use_fp16)
    if use_fp16:
        print("GPU:", torch.cuda.get_device_name(0))

    training_args = build_training_arguments(use_fp16=use_fp16, train_rows=len(train_df))

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        sample_weights=sample_weights if USE_WEIGHTED_SAMPLER else None,
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

    print("Done")
    print(f"Training outputs saved under: {OUT_DIR}")
    print(f"TensorBoard logs: tensorboard --logdir {OUT_DIR}")


if __name__ == "__main__":
    main()
