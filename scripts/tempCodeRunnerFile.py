import os, json, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

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
TEST_PATH  = r"data\test.csv"  # not used in training, used in eval script below

OUT_DIR = r"outputs\edu_classifier_multitask"

# Use your MLM-adapted model folder if it exists:
BASE_MODEL = r"outputs\cfpb_outputs\distilbert_cfpb_mlm"
FALLBACK_MODEL = "distilbert-base-uncased"

MAX_LENGTH = 256
SEED = 42
EPOCHS = 4
BATCH = 16
LR = 2e-5
WEIGHT_DECAY = 0.01

# loss weights: tune if needed
LAMBDA_LABEL = 1.0
LAMBDA_PRIORITY = 1.0
# ---------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("📥 Loading split datasets...")
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)

required_cols = {"text", "label_id", "priority_id"}
missing = required_cols - set(train_df.columns)
if missing:
    raise ValueError(f"Missing columns in train.csv: {missing}")

# HF expects "labels" by default; we keep both:
train_df = train_df[["text", "label_id", "priority_id"]].rename(
    columns={"label_id": "labels", "priority_id": "priority_labels"}
)
val_df = val_df[["text", "label_id", "priority_id"]].rename(
    columns={"label_id": "labels", "priority_id": "priority_labels"}
)

# --- label mapping ---
map_path = os.path.join(r"outputs\edu_classifier", "label_mapping.json")
with open(map_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)

id_to_label = {int(k): v for k, v in mapping["id_to_label"].items()}
num_labels = len(id_to_label)

# priority mapping (create from data)
priority_ids = sorted(train_df["priority_labels"].unique().tolist())
num_priority = len(priority_ids)

# build id->name if you already have it; otherwise infer typical ordering
# adjust these if your ids differ
default_priority_names = {0: "Low", 1: "Medium", 2: "High"}
id_to_priority = {int(i): default_priority_names.get(int(i), f"P{i}") for i in priority_ids}

print("🏷️ Labels:", num_labels, "| Priority classes:", num_priority)

# -------- class weights (important) --------
label_classes = np.unique(train_df["labels"])
label_w = compute_class_weight("balanced", classes=label_classes, y=train_df["labels"])
label_w = torch.tensor(label_w, dtype=torch.float)

prio_classes = np.unique(train_df["priority_labels"])
prio_w = compute_class_weight("balanced", classes=prio_classes, y=train_df["priority_labels"])
prio_w = torch.tensor(prio_w, dtype=torch.float)

# -------- datasets --------
train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)

base_model = BASE_MODEL if os.path.isdir(BASE_MODEL) else FALLBACK_MODEL
print("🧠 Using base model:", base_model)

tokenizer = AutoTokenizer.from_pretrained(base_model)

def tok_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
val_ds   = val_ds.map(tok_fn, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------- multitask model --------
class DistilBertMultiTask(nn.Module):
    def __init__(self, model_name: str, num_labels: int, num_priority: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.label_head = nn.Linear(hidden, num_labels)
        self.prio_head  = nn.Linear(hidden, num_priority)

    def forward(self, input_ids=None, attention_mask=None, labels=None, priority_labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT: use [CLS]-like first token
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        label_logits = self.label_head(pooled)
        prio_logits  = self.prio_head(pooled)

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

# -------- metrics --------
def compute_metrics_multitask(eval_pred):
    # eval_pred.predictions will be a tuple if we return multiple logits;
    # but Trainer expects "logits" by default. We'll override prediction_step in Trainer.
    raise RuntimeError("compute_metrics is handled inside MultiTaskTrainer.")

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
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
        # Make Trainer collect both logits
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=None,
                priority_labels=None,
            )
        label_logits = outputs["label_logits"]
        prio_logits  = outputs["priority_logits"]

        # Also compute loss if labels provided
        loss = None
        if "labels" in inputs and "priority_labels" in inputs:
            outputs_loss = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs["labels"],
                priority_labels=inputs["priority_labels"],
            )
            loss = outputs_loss["loss"].detach()

        if prediction_loss_only:
            return (loss, None, None)

        # Return (loss, logits, labels) where logits can be a tuple
        labels = torch.stack([inputs["labels"], inputs["priority_labels"]], dim=1)
        return (loss, (label_logits, prio_logits), labels)

    def compute_metrics(self, eval_pred):
        (label_logits, prio_logits), labels = eval_pred
        y_label_true = labels[:, 0]
        y_prio_true  = labels[:, 1]

        y_label_pred = np.argmax(label_logits, axis=1)
        y_prio_pred  = np.argmax(prio_logits, axis=1)

        return {
            "label_acc": accuracy_score(y_label_true, y_label_pred),
            "label_f1_macro": f1_score(y_label_true, y_label_pred, average="macro"),
            "prio_acc": accuracy_score(y_prio_true, y_prio_pred),
            "prio_f1_macro": f1_score(y_prio_true, y_prio_pred, average="macro"),
            # overall score you can optimize
            "f1_macro": (f1_score(y_label_true, y_label_pred, average="macro")
                         + f1_score(y_prio_true, y_prio_pred, average="macro")) / 2.0
        }

use_fp16 = torch.cuda.is_available()
print("✅ CUDA available:", use_fp16)
if use_fp16:
    print("✅ GPU:", torch.cuda.get_device_name(0))

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=use_fp16,
    report_to=["tensorboard"],  # ✅ TensorBoard
    seed=SEED,
)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("🚀 Training (multi-task)...")
trainer.train()

print("💾 Saving model + tokenizer + mappings...")
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# Save mappings for inference
with open(os.path.join(OUT_DIR, "id_to_label.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in id_to_label.items()}, f, indent=2)
with open(os.path.join(OUT_DIR, "id_to_priority.json"), "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in id_to_priority.items()}, f, indent=2)

print("✅ Done!")
print(f"📈 TensorBoard logs: run ->  tensorboard --logdir {OUT_DIR}")