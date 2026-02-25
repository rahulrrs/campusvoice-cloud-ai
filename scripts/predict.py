import os, json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ========= CONFIG =========
MODEL_DIR = r"outputs\edu_classifier_multitask"
MAX_LENGTH = 256

# confidence thresholds (tune later)
LABEL_THRESHOLD = 0.55
PRIO_THRESHOLD  = 0.50
# ==========================

id_to_label_path = os.path.join(MODEL_DIR, "id_to_label.json")
id_to_priority_path = os.path.join(MODEL_DIR, "id_to_priority.json")

with open(id_to_label_path, "r", encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}
with open(id_to_priority_path, "r", encoding="utf-8") as f:
    id_to_priority = {int(k): v for k, v in json.load(f).items()}

num_labels = len(id_to_label)
num_priority = len(id_to_priority)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

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

# load weights saved by Trainer
state_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
state = torch.load(state_path, map_location="cpu")
model.load_state_dict(state, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)

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

        label_probs = torch.softmax(label_logits, dim=-1).cpu().numpy()
        prio_probs  = torch.softmax(prio_logits, dim=-1).cpu().numpy()

    label_ids = label_probs.argmax(axis=1)
    prio_ids  = prio_probs.argmax(axis=1)

    label_conf = label_probs.max(axis=1)
    prio_conf  = prio_probs.max(axis=1)

    results = []
    for t, lid, lconf, pid, pconf in zip(texts, label_ids, label_conf, prio_ids, prio_conf):
        label_name = id_to_label[int(lid)]
        prio_name  = id_to_priority[int(pid)]

        # Unknown fallback
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
        "Some random complaint that doesn't fit anything"
    ]

    preds = predict_texts(texts)

    for r in preds:
        print("\nTEXT:", r["text"])
        print(f"LABEL: {r['label']} (conf={r['label_confidence']:.3f})")
        print(f"PRIO : {r['priority']} (conf={r['priority_confidence']:.3f})")