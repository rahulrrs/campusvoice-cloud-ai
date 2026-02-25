import os, json
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

# ========= CONFIG =========
MODEL_DIR = r"outputs\edu_classifier_multitask"
MAX_LENGTH = 256
LABEL_THRESHOLD = 0.55
PRIO_THRESHOLD  = 0.50

# optional mapping for routing (edit as you want)
LABEL_TO_DEPT = {
    "Academic": "Academic Affairs",
    "Faculty": "Academic Affairs",
    "Exam": "Examination Cell",
    "IT / Portal": "IT Support",
    "Fees": "Accounts",
    "Hostel": "Hostel Office",
    "Mess": "Catering/Mess",
    "Library": "Library",
    "Placement": "Career Services",
    "Transport": "Transport Office",
    "Health": "Health Center",
    "Safety": "Security",
    "Scholarship": "Scholarship Office",
    "Administration": "Admin Office",
    "Certificate": "Admin Office",
    "Discipline": "Disciplinary Committee",
    "Attendance": "Academic Affairs",
    "Infrastructure": "Maintenance",
    "Lab": "Lab Incharge",
    "Other": "Helpdesk",
    "Unknown": "Helpdesk",
}
# ==========================

# ----- load mappings -----
with open(os.path.join(MODEL_DIR, "id_to_label.json"), "r", encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(MODEL_DIR, "id_to_priority.json"), "r", encoding="utf-8") as f:
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

state = torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu")
model.load_state_dict(state, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_one(text: str):
    with torch.no_grad():
        enc = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        label_logits, prio_logits = model(**enc)

        label_probs = torch.softmax(label_logits, dim=-1).cpu().numpy()[0]
        prio_probs  = torch.softmax(prio_logits, dim=-1).cpu().numpy()[0]

    lid = int(label_probs.argmax())
    pid = int(prio_probs.argmax())
    lconf = float(label_probs.max())
    pconf = float(prio_probs.max())

    label = id_to_label[lid]
    priority = id_to_priority[pid]

    if lconf < LABEL_THRESHOLD:
        label = "Unknown"
    if pconf < PRIO_THRESHOLD:
        priority = "Unknown"

    dept = LABEL_TO_DEPT.get(label, "Helpdesk")

    return {
        "label": label,
        "label_confidence": lconf,
        "priority": priority,
        "priority_confidence": pconf,
        "department": dept,
    }

# -------- FastAPI --------
app = FastAPI(title="Complaint Routing API")

class ComplaintIn(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: ComplaintIn):
    return predict_one(payload.text)