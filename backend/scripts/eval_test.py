import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score

from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding

# ========= CONFIG =========
MODEL_DIR = r"outputs\edu_classifier_multitask"
TEST_PATH = r"data\test.csv"
BATCH = 32
MAX_LENGTH = 256
FALLBACK_MODEL = "distilbert-base-uncased"
# ==========================

print("📥 Loading:", TEST_PATH)
df = pd.read_csv(TEST_PATH, low_memory=False)

# ✅ use priority_id_fixed
need = {"text", "label_id", "priority_id_fixed"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"❌ Missing columns in test.csv: {missing}")

test_df = df[["text", "label_id", "priority_id_fixed"]].rename(
    columns={"label_id": "labels", "priority_id_fixed": "priority_labels"}
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

# --- Prefer loading tokenizer from MODEL_DIR (matches training) ---
tokenizer_src = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "tokenizer_config.json")) else FALLBACK_MODEL
print("🧠 Loading tokenizer from:", tokenizer_src)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

def tok_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

test_ds = Dataset.from_pandas(test_df)
test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer)
loader = DataLoader(test_ds, batch_size=BATCH, collate_fn=collator)

# --- Load backbone from MODEL_DIR if config exists, else fallback ---
backbone_src = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "config.json")) else FALLBACK_MODEL
print("🧠 Loading backbone from:", backbone_src)

# ✅ MUST match training architecture
class DistilBertMultiTask(nn.Module):
    def __init__(self, model_name: str, num_labels: int, num_priority: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.label_dropout = nn.Dropout(0.2)
        self.prio_dropout = nn.Dropout(0.2)
        self.act = nn.ReLU()
        self.label_hidden = nn.Linear(hidden, hidden // 2)
        self.label_head = nn.Linear(hidden // 2, num_labels)
        self.prio_hidden = nn.Linear(hidden, hidden // 4)
        self.prio_head = nn.Linear(hidden // 4, num_priority)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        label_logits = self.label_head(self.act(self.label_hidden(self.label_dropout(pooled))))
        prio_logits  = self.prio_head(self.act(self.prio_hidden(self.prio_dropout(pooled))))
        return label_logits, prio_logits

model = DistilBertMultiTask(backbone_src, num_labels, num_priority)

# --- load weights (safetensors or bin) ---
weights_safetensors = os.path.join(MODEL_DIR, "model.safetensors")
weights_bin = os.path.join(MODEL_DIR, "pytorch_model.bin")

state_dict = None
if os.path.exists(weights_safetensors):
    print("📦 Loading weights:", weights_safetensors)
    from safetensors.torch import load_file
    state_dict = load_file(weights_safetensors)
elif os.path.exists(weights_bin):
    print("📦 Loading weights:", weights_bin)
    state_dict = torch.load(weights_bin, map_location="cpu")
else:
    raise FileNotFoundError(f"Missing model weights in {MODEL_DIR} (expected model.safetensors or pytorch_model.bin)")

# strict=True should work now because class matches training
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
