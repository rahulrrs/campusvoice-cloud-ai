# scripts/train_multitask.py
import os
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

# ---------------- CONFIG ----------------
TRAIN_PATH = r"data\train.csv"
VAL_PATH   = r"data\val.csv"

OUT_DIR = r"outputs\edu_classifier_multitask"

BASE_MODEL     = r"outputs\cfpb_outputs\distilbert_cfpb_mlm"
FALLBACK_MODEL = "distilbert-base-uncased"

MAX_LENGTH = 256
SEED       = 42
EPOCHS     = 8          # ↑ was 5 – give more room with early stopping
BATCH      = 16
LR         = 2e-5
WEIGHT_DECAY = 0.01

# ── Task loss weights ────────────────────────────────────────────────────────
# Priority was under-trained vs labels; boosting LAMBDA_PRIORITY forces the
# model to pay more attention to the severely imbalanced priority signal.
LAMBDA_LABEL    = 1.0
LAMBDA_PRIORITY = 2.0   # ↑ was 1.0

# ── Per-class cap for label balancing ────────────────────────────────────────
# Was 800; lowering gives smaller classes a fairer share of the gradient.
MAX_PER_CLASS = 500      # ↑ was 800

# ── Extra oversampling for High-priority rows ─────────────────────────────────
# Repeat High-priority rows so they're not drowned out by Low (60 % of data).
OVERSAMPLE_HIGH_PRIORITY = 3   # duplicate High rows 3× in training set
# ---------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading split datasets...")
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)

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

# ── mappings ─────────────────────────────────────────────────────────────────
label_map_path = os.path.join(OUT_DIR, "id_to_label.json")
prio_map_path  = os.path.join(OUT_DIR, "id_to_priority.json")

if not os.path.exists(label_map_path):
    raise FileNotFoundError(f"Missing: {label_map_path} (run clean_dataset.py first)")
if not os.path.exists(prio_map_path):
    raise FileNotFoundError(f"Missing: {prio_map_path} (run clean_dataset.py first)")

with open(label_map_path, "r", encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}
num_labels = len(id_to_label)

with open(prio_map_path, "r", encoding="utf-8") as f:
    id_to_priority = {int(k): v for k, v in json.load(f).items()}
num_priority = len(id_to_priority)

print(f"Labels: {num_labels} | Priority classes: {num_priority}")

# ── Cap dominant label classes ────────────────────────────────────────────────
print(f"Before capping total training samples: {len(train_df)}")
train_df = pd.concat([
    g.sample(min(len(g), MAX_PER_CLASS), random_state=SEED)
    for _, g in train_df.groupby("labels")
]).reset_index(drop=True)
print(f"After capping (max {MAX_PER_CLASS}/class): {len(train_df)}")

# ── Oversample High-priority rows ─────────────────────────────────────────────
high_rows = train_df[train_df["priority_labels"] == 2]
print(f"High-priority rows before oversample: {len(high_rows)}")
if OVERSAMPLE_HIGH_PRIORITY > 1 and len(high_rows) > 0:
    extra = pd.concat([high_rows] * (OVERSAMPLE_HIGH_PRIORITY - 1), ignore_index=True)
    train_df = pd.concat([train_df, extra], ignore_index=True).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)
    print(f"After oversampling High×{OVERSAMPLE_HIGH_PRIORITY}: {len(train_df)}")

# ── Class weights ─────────────────────────────────────────────────────────────
# Use sqrt-inverse-frequency so extreme imbalance doesn't dominate.
label_counts = train_df["labels"].value_counts().sort_index()
label_w = torch.ones(num_labels, dtype=torch.float)
for i in range(num_labels):
    c = int(label_counts.get(i, 0))
    label_w[i] = 1.0 if c == 0 else (len(train_df) / (num_labels * c)) ** 0.5

prio_counts = train_df["priority_labels"].value_counts().sort_index()
prio_w = torch.ones(num_priority, dtype=torch.float)
for i in range(num_priority):
    c = int(prio_counts.get(i, 0))
    # Use stronger inverse-frequency for priority due to severe imbalance
    prio_w[i] = 1.0 if c == 0 else len(train_df) / (num_priority * c)

print("Label weights   (first 10):", [round(x, 4) for x in label_w.tolist()[:10]])
print("Priority weights:", [round(x, 4) for x in prio_w.tolist()])

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)

base_model = BASE_MODEL if os.path.isdir(BASE_MODEL) else FALLBACK_MODEL
print("Using base model:", base_model)

tokenizer = AutoTokenizer.from_pretrained(base_model)

def tok_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
val_ds   = val_ds.map(tok_fn, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ── Multitask model ───────────────────────────────────────────────────────────
class DistilBertMultiTask(nn.Module):
    def __init__(self, model_name: str, num_labels: int, num_priority: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        # Shared dropout
        self.dropout = nn.Dropout(0.1)

        # Label head – slightly wider for 22 classes
        self.label_dropout = nn.Dropout(0.2)
        self.label_hidden   = nn.Linear(hidden, hidden // 2)
        self.label_head     = nn.Linear(hidden // 2, num_labels)

        # Priority head
        self.prio_dropout = nn.Dropout(0.2)
        self.prio_hidden  = nn.Linear(hidden, hidden // 4)
        self.prio_head    = nn.Linear(hidden // 4, num_priority)

        self.act = nn.GELU()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        priority_labels=None,
        **kwargs,
    ):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])

        label_logits = self.label_head(self.act(self.label_hidden(self.label_dropout(pooled))))
        prio_logits  = self.prio_head(self.act(self.prio_hidden(self.prio_dropout(pooled))))

        loss = None
        if labels is not None and priority_labels is not None:
            loss_fct_label = nn.CrossEntropyLoss(weight=label_w.to(label_logits.device))
            loss_fct_prio  = nn.CrossEntropyLoss(weight=prio_w.to(prio_logits.device))
            loss_label = loss_fct_label(label_logits, labels)
            loss_prio  = loss_fct_prio(prio_logits, priority_labels)
            loss = LAMBDA_LABEL * loss_label + LAMBDA_PRIORITY * loss_prio

        return {
            "loss": loss,
            "label_logits": label_logits,
            "priority_logits": prio_logits,
        }


model = DistilBertMultiTask(base_model, num_labels=num_labels, num_priority=num_priority)


# ── Custom Trainer ────────────────────────────────────────────────────────────
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels         = inputs.get("labels")
        priority_labels = inputs.get("priority_labels")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=labels,
            priority_labels=priority_labels,
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        labels          = inputs.get("labels")
        priority_labels = inputs.get("priority_labels")

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=None,
                priority_labels=None,
            )

        label_logits = outputs["label_logits"]
        prio_logits  = outputs["priority_logits"]

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


def compute_metrics_fn(eval_pred):
    (label_logits, prio_logits), labels = eval_pred

    y_label_true = labels[:, 0]
    y_prio_true  = labels[:, 1]

    y_label_pred = np.argmax(label_logits, axis=1)
    y_prio_pred  = np.argmax(prio_logits, axis=1)

    label_f1 = f1_score(y_label_true, y_label_pred, average="macro", zero_division=0)
    prio_f1  = f1_score(y_prio_true,  y_prio_pred,  average="macro", zero_division=0)

    return {
        "label_acc":      accuracy_score(y_label_true, y_label_pred),
        "label_f1_macro": label_f1,
        "prio_acc":       accuracy_score(y_prio_true,  y_prio_pred),
        "prio_f1_macro":  prio_f1,
        "f1_macro":       (label_f1 + prio_f1) / 2.0,
    }


use_fp16 = torch.cuda.is_available()
print("CUDA available:", use_fp16)
if use_fp16:
    print("GPU:", torch.cuda.get_device_name(0))

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=1.0,
    warmup_ratio=0.06,          # ← added: helps early convergence
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=use_fp16,
    report_to=["tensorboard"],
    seed=SEED,
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    compute_metrics=compute_metrics_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],  # ↑ was 3
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
print(f"TensorBoard logs: tensorboard --logdir {OUT_DIR}")