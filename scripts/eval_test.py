import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

# ========= CONFIG =========
TEST_PATH = r"data\test.csv"
MODEL_DIR = r"outputs\edu_classifier_multitask"
MAX_LENGTH = 256
BATCH_SIZE = 32
# ==========================

id_to_label_path = os.path.join(MODEL_DIR, "id_to_label.json")
id_to_priority_path = os.path.join(MODEL_DIR, "id_to_priority.json")

if not os.path.isfile(id_to_label_path):
    raise FileNotFoundError(f"Missing {id_to_label_path} (train_multitask.py should create it)")
if not os.path.isfile(id_to_priority_path):
    raise FileNotFoundError(f"Missing {id_to_priority_path} (train_multitask.py should create it)")

with open(id_to_label_path, "r", encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}
with open(id_to_priority_path, "r", encoding="utf-8") as f:
    id_to_priority = {int(k): v for k, v in json.load(f).items()}

num_labels = len(id_to_label)
num_priority = len(id_to_priority)

print("📥 Loading:", TEST_PATH)
df = pd.read_csv(TEST_PATH)

needed = {"text", "label_id", "priority_id"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing columns in test.csv: {missing}")

df = df[["text", "label_id", "priority_id"]].copy()
df.rename(columns={"label_id": "labels", "priority_id": "priority_labels"}, inplace=True)

test_ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

def tok_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------- model definition must match train_multitask.py ----------
class DistilBertMultiTask(nn.Module):
    def __init__(self, model_dir: str, num_labels: int, num_priority: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_dir)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.label_head = nn.Linear(hidden, num_labels)
        self.prio_head  = nn.Linear(hidden, num_priority)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.label_head(pooled), self.prio_head(pooled)

model = DistilBertMultiTask(MODEL_DIR, num_labels=num_labels, num_priority=num_priority)

# load heads + backbone weights
state = torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(state, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---------- batched inference ----------
all_label_true, all_prio_true = [], []
all_label_pred, all_prio_pred = [], []

loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collator)

with torch.no_grad():
    for batch in loader:
        labels = batch.pop("labels").numpy()
        prios  = batch.pop("priority_labels").numpy()

        batch = {k: v.to(device) for k, v in batch.items()}
        label_logits, prio_logits = model(**batch)

        label_pred = torch.argmax(label_logits, dim=-1).cpu().numpy()
        prio_pred  = torch.argmax(prio_logits, dim=-1).cpu().numpy()

        all_label_true.extend(labels.tolist())
        all_prio_true.extend(prios.tolist())
        all_label_pred.extend(label_pred.tolist())
        all_prio_pred.extend(prio_pred.tolist())

# ---------- metrics ----------
label_acc = accuracy_score(all_label_true, all_label_pred)
label_f1  = f1_score(all_label_true, all_label_pred, average="macro")

prio_acc = accuracy_score(all_prio_true, all_prio_pred)
prio_f1  = f1_score(all_prio_true, all_prio_pred, average="macro")

print("\n✅ TEST RESULTS")
print(f"Label Acc: {label_acc:.4f} | Label F1-macro: {label_f1:.4f}")
print(f"Prio  Acc: {prio_acc:.4f} | Prio  F1-macro: {prio_f1:.4f}")

labels_sorted = [id_to_label[i] for i in sorted(id_to_label)]
prio_sorted   = [id_to_priority[i] for i in sorted(id_to_priority)]

report_label = classification_report(all_label_true, all_label_pred, target_names=labels_sorted, digits=4)
report_prio  = classification_report(all_prio_true, all_prio_pred, target_names=prio_sorted, digits=4)

print("\n--- Label report ---\n", report_label[:1200], "...\n")
print("\n--- Priority report ---\n", report_prio, "\n")

os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "test_report_label.txt"), "w", encoding="utf-8") as f:
    f.write(report_label)
with open(os.path.join(MODEL_DIR, "test_report_priority.txt"), "w", encoding="utf-8") as f:
    f.write(report_prio)

cm_label = confusion_matrix(all_label_true, all_label_pred)
cm_prio  = confusion_matrix(all_prio_true, all_prio_pred)

np.save(os.path.join(MODEL_DIR, "test_confusion_label.npy"), cm_label)
np.save(os.path.join(MODEL_DIR, "test_confusion_priority.npy"), cm_prio)

metrics = {
    "label_acc": float(label_acc),
    "label_f1_macro": float(label_f1),
    "prio_acc": float(prio_acc),
    "prio_f1_macro": float(prio_f1),
}
with open(os.path.join(MODEL_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("💾 Saved test metrics + reports in:", MODEL_DIR)