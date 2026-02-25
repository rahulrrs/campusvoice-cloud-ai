import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score

from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from safetensors.torch import load_file

# ========= CONFIG =========
MODEL_DIR = r"outputs\edu_classifier_multitask"
TEST_PATH = r"data\test.csv"
BATCH = 32
MAX_LENGTH = 256

# Must match what you trained with:
BASE_MODEL = r"outputs\cfpb_outputs\distilbert_cfpb_mlm"
FALLBACK_MODEL = "distilbert-base-uncased"
# ==========================

print("📥 Loading:", TEST_PATH)
df = pd.read_csv(TEST_PATH, low_memory=False)

need = {"text", "label_id", "priority_id"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing columns in test.csv: {missing}")

test_df = df[["text", "label_id", "priority_id"]].rename(
    columns={"label_id": "labels", "priority_id": "priority_labels"}
)

# --- mappings saved during training ---
label_map_path = os.path.join(MODEL_DIR, "id_to_label.json")
prio_map_path  = os.path.join(MODEL_DIR, "id_to_priority.json")

for p in [label_map_path, prio_map_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

with open(label_map_path, "r", encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}

with open(prio_map_path, "r", encoding="utf-8") as f:
    id_to_priority = {int(k): v for k, v in json.load(f).items()}

num_labels = len(id_to_label)
num_priority = len(id_to_priority)

# --- HF dataset ---
test_ds = Dataset.from_pandas(test_df)

backbone_name = BASE_MODEL if os.path.isdir(BASE_MODEL) else FALLBACK_MODEL
print("🧠 Loading backbone:", backbone_name)

tokenizer = AutoTokenizer.from_pretrained(backbone_name)

def tok_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer)
loader = DataLoader(test_ds, batch_size=BATCH, collate_fn=collator)

# --- model definition must match training ---
class DistilBertMultiTask(nn.Module):
    def __init__(self, backbone_name: str, num_labels: int, num_priority: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.label_head = nn.Linear(hidden, num_labels)
        self.prio_head  = nn.Linear(hidden, num_priority)

    # ✅ IMPORTANT: accept token_type_ids / extra kwargs safely (DistilBERT ignores them)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.label_head(pooled), self.prio_head(pooled)

model = DistilBertMultiTask(backbone_name, num_labels, num_priority)

# --- load your trained weights (safetensors) ---
weights_path = os.path.join(MODEL_DIR, "model.safetensors")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Missing: {weights_path}")

state_dict = load_file(weights_path)
model.load_state_dict(state_dict, strict=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("✅ Device:", device)

# --- evaluation ---
y_label_true, y_label_pred = [], []
y_prio_true, y_prio_pred = [], []

with torch.no_grad():
    for batch in loader:
        labels = batch.pop("labels").cpu().numpy()
        prios  = batch.pop("priority_labels").cpu().numpy()

        # ✅ Robust: pass only keys we need (avoids token_type_ids issues completely)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        attention_mask = attention_mask.to(device) if attention_mask is not None else None

        label_logits, prio_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        y_label_true.extend(labels)
        y_prio_true.extend(prios)
        y_label_pred.extend(label_logits.argmax(dim=1).cpu().numpy())
        y_prio_pred.extend(prio_logits.argmax(dim=1).cpu().numpy())

print("\n====================")
print("📊 LABEL REPORT")
print("====================")
print(classification_report(y_label_true, y_label_pred, digits=4))

print("\n====================")
print("📊 PRIORITY REPORT")
print("====================")
print(classification_report(y_prio_true, y_prio_pred, digits=4))

print("\n✅ Label Macro-F1:", f1_score(y_label_true, y_label_pred, average="macro"))
print("✅ Priority Macro-F1:", f1_score(y_prio_true, y_prio_pred, average="macro"))
print("✅ Label Accuracy:", accuracy_score(y_label_true, y_label_pred))
print("✅ Priority Accuracy:", accuracy_score(y_prio_true, y_prio_pred))