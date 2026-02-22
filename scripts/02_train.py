import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from pathlib import Path
import torch

print("Loading train/val data...")

train_df = pd.read_csv("data/train.csv")
val_df   = pd.read_csv("data/val.csv")

# =========================
# 1️⃣ REMAP LABELS 1–30 → 0–29
# =========================
train_df["label"] = train_df["label"] - 1
val_df["label"]   = val_df["label"] - 1

num_labels = train_df["label"].nunique()
print("Number of labels:", num_labels)

# =========================
# 2️⃣ Convert to HF Dataset
# =========================
train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)

# =========================
# 3️⃣ Load Tokenizer + Tokenize
# =========================
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)

# Rename label column to what Trainer expects
train_ds = train_ds.rename_column("label", "labels")
val_ds   = val_ds.rename_column("label", "labels")

# Keep only columns used by the model
keep_cols = ["input_ids", "attention_mask", "labels"]
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
val_ds   = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])

# =========================
# 4️⃣ Load Model
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# =========================
# 5️⃣ Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }

# =========================
# 6️⃣ Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir="models/distilbert_complaints",
    eval_strategy="epoch",   # ✅ for your transformers version
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=torch.cuda.is_available()
)

# =========================
# 7️⃣ Trainer (NO tokenizer arg)
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# =========================
# 8️⃣ Train
# =========================
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete.")

# =========================
# 9️⃣ Save Final Model + Tokenizer
# =========================
final_dir = Path("models/distilbert_complaints_final")
final_dir.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))

print("✅ Model saved to:", final_dir)