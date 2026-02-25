# scripts/predict.py
import os, json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

# ========= CONFIG =========
MODEL_DIR = r"outputs\edu_classifier_multitask"
MAX_LENGTH = 256

# Use the SAME backbone you trained with (if you have it). Otherwise fallback.
BACKBONE_DIR = r"outputs\cfpb_outputs\distilbert_cfpb_mlm"
FALLBACK_BACKBONE = "distilbert-base-uncased"

# confidence thresholds (tune later)
LABEL_THRESHOLD = 0.55
PRIO_THRESHOLD  = 0.50
# ==========================

# --- load mappings ---
id_to_label_path = os.path.join(MODEL_DIR, "id_to_label.json")
id_to_priority_path = os.path.join(MODEL_DIR, "id_to_priority.json")

with open(id_to_label_path, "r", encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}

with open(id_to_priority_path, "r", encoding="utf-8") as f:
    id_to_priority = {int(k): v for k, v in json.load(f).items()}

num_labels = len(id_to_label)
num_priority = len(id_to_priority)

# --- backbone + tokenizer ---
backbone_name = BACKBONE_DIR if os.path.isdir(BACKBONE_DIR) else FALLBACK_BACKBONE
print("🧠 Using backbone:", backbone_name)

tokenizer = AutoTokenizer.from_pretrained(backbone_name)

# --- model ---
class DistilBertMultiTask(nn.Module):
    def __init__(self, backbone_name: str, num_labels: int, num_priority: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.label_head = nn.Linear(hidden, num_labels)
        self.prio_head  = nn.Linear(hidden, num_priority)

    # accept extra keys safely (token_type_ids etc.)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.label_head(pooled), self.prio_head(pooled)

model = DistilBertMultiTask(backbone_name, num_labels=num_labels, num_priority=num_priority)

# --- load weights (prefer safetensors, fallback to bin) ---
safe_path = os.path.join(MODEL_DIR, "model.safetensors")
bin_path  = os.path.join(MODEL_DIR, "pytorch_model.bin")

if os.path.exists(safe_path):
    print("📦 Loading weights:", safe_path)
    state = load_file(safe_path)
elif os.path.exists(bin_path):
    print("📦 Loading weights:", bin_path)
    state = torch.load(bin_path, map_location="cpu")
else:
    raise FileNotFoundError(f"❌ No weights found. Expected:\n- {safe_path}\n- {bin_path}")

missing, unexpected = model.load_state_dict(state, strict=False)
print("✅ Weights loaded")
if missing:
    print("⚠️ Missing keys (sample):", missing[:10], "..." if len(missing) > 10 else "")
if unexpected:
    print("⚠️ Unexpected keys (sample):", unexpected[:10], "..." if len(unexpected) > 10 else "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("✅ Device:", device)

# --- prediction ---
def predict_texts(texts):
    with torch.no_grad():
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        label_logits, prio_logits = model(**enc)

        label_probs = torch.softmax(label_logits, dim=-1)
        prio_probs  = torch.softmax(prio_logits, dim=-1)

        label_ids = label_probs.argmax(dim=1).cpu().tolist()
        prio_ids  = prio_probs.argmax(dim=1).cpu().tolist()

        label_conf = label_probs.max(dim=1).values.cpu().tolist()
        prio_conf  = prio_probs.max(dim=1).values.cpu().tolist()

    results = []
    for t, lid, lconf, pid, pconf in zip(texts, label_ids, label_conf, prio_ids, prio_conf):
        label_name = id_to_label[int(lid)]
        prio_name  = id_to_priority[int(pid)]

        if lconf < LABEL_THRESHOLD:
            label_name = "Unknown"
        if pconf < PRIO_THRESHOLD:
            prio_name = "Unknown"

        results.append({
            "text": t,
            "label": label_name,
            "label_confidence": float(lconf),
            "priority": prio_name,
            "priority_confidence": float(pconf),
        })
    return results

if __name__ == "__main__":
    texts = [
        "Hostel water is not available for 2 days",
        "Exam timetable is not released",
        "Fee payment failed in portal",
        "Some random complaint that doesn't fit anything",
    ]

    preds = predict_texts(texts)
    for r in preds:
        print("\nTEXT:", r["text"])
        print(f"LABEL: {r['label']} (conf={r['label_confidence']:.3f})")
        print(f"PRIO : {r['priority']} (conf={r['priority_confidence']:.3f})")