import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.complaint_ml import build_metadata_feature_map, feature_map_to_vector, scale_feature_vector
from src.utils.model_paths import load_project_env, resolve_backbone_source

MODEL_DIR = PROJECT_ROOT / "outputs" / "edu_classifier_multitask"
load_project_env(PROJECT_ROOT)
MAX_LENGTH = 256
PREDICT_BATCH_SIZE = int(os.getenv("PREDICT_BATCH_SIZE", "32"))


id_to_label_path = MODEL_DIR / "id_to_label.json"
id_to_priority_path = MODEL_DIR / "id_to_priority.json"
if not id_to_label_path.exists():
    raise FileNotFoundError(f"Missing: {id_to_label_path}")
if not id_to_priority_path.exists():
    raise FileNotFoundError(f"Missing: {id_to_priority_path}")

with open(id_to_label_path, "r", encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}
with open(id_to_priority_path, "r", encoding="utf-8") as f:
    id_to_priority = {int(k): v for k, v in json.load(f).items()}

loaded_labels = set(id_to_label.values())
if not loaded_labels:
    raise ValueError(
        "Classifier label mapping is empty. Check id_to_label.json."
    )

num_labels = len(id_to_label)
num_priority = len(id_to_priority)

tokenizer_config_path = MODEL_DIR / "tokenizer_config.json"
if not tokenizer_config_path.exists():
    raise FileNotFoundError(
        f"Missing classifier tokenizer config: {tokenizer_config_path}"
    )
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
metadata_config_path = MODEL_DIR / "metadata_config.json"
metadata_config = None
if metadata_config_path.exists():
    with open(metadata_config_path, "r", encoding="utf-8") as f:
        metadata_config = json.load(f)


class EncoderMultiTask(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_labels: int,
        num_priority: int,
        label_metadata_dim: int = 0,
        priority_metadata_dim: int = 0,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.label_metadata_dim = label_metadata_dim
        self.priority_metadata_dim = priority_metadata_dim
        self.dropout = nn.Dropout(0.1)
        self.label_dropout = nn.Dropout(0.2)
        if label_metadata_dim > 0:
            self.label_meta_proj = nn.Sequential(
                nn.Linear(label_metadata_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.label_meta_proj = None
        self.label_hidden = nn.Linear(hidden, hidden // 2)
        self.label_head = nn.Linear(hidden // 2, num_labels)
        if priority_metadata_dim > 0:
            self.priority_meta_proj = nn.Sequential(
                nn.Linear(priority_metadata_dim, hidden // 4),
                nn.LayerNorm(hidden // 4),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            prio_input = hidden + (hidden // 4)
        else:
            self.priority_meta_proj = None
            prio_input = hidden
        self.prio_dropout = nn.Dropout(0.2)
        self.prio_hidden = nn.Linear(prio_input, hidden // 4)
        self.prio_head = nn.Linear(hidden // 4, num_priority)
        self.act = nn.GELU()

    def forward(self, input_ids=None, attention_mask=None, metadata_features=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        label_input = pooled
        if self.label_metadata_dim > 0 and metadata_features is not None:
            label_input = label_input + self.label_meta_proj(metadata_features.float())
        label_logits = self.label_head(self.act(self.label_hidden(self.label_dropout(label_input))))
        priority_input = pooled
        if self.priority_metadata_dim > 0 and metadata_features is not None:
            priority_input = torch.cat([priority_input, self.priority_meta_proj(metadata_features.float())], dim=-1)
        prio_logits = self.prio_head(self.act(self.prio_hidden(self.prio_dropout(priority_input))))
        return label_logits, prio_logits

safe_path = MODEL_DIR / "model.safetensors"
bin_path = MODEL_DIR / "pytorch_model.bin"
if safe_path.exists():
    state = load_file(str(safe_path))
elif bin_path.exists():
    state = torch.load(str(bin_path), map_location="cpu")
else:
    raise FileNotFoundError(f"No weights found. Expected:\n  {safe_path}\n  {bin_path}")

for key in ("label_weights", "priority_weights", "priority_cost_matrix"):
    state.pop(key, None)

metadata_dim = len(metadata_config.get("feature_names", [])) if isinstance(metadata_config, dict) else 0
has_label_metadata_layers = any(key.startswith("label_meta_proj.") for key in state)
has_metadata_layers = any(key.startswith("priority_meta_proj.") for key in state)
backbone_source, backbone_note = resolve_backbone_source(PROJECT_ROOT, MODEL_DIR)
if backbone_note:
    print(backbone_note)
model = EncoderMultiTask(
    backbone_name=backbone_source,
    num_labels=num_labels,
    num_priority=num_priority,
    label_metadata_dim=metadata_dim if has_label_metadata_layers else 0,
    priority_metadata_dim=metadata_dim if has_metadata_layers else 0,
)
model.load_state_dict(state, strict=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict_texts(texts):
    if isinstance(texts, str):
        texts = [texts]

    results = []
    batch_size = max(PREDICT_BATCH_SIZE, 1)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            metadata_vectors = [
                scale_feature_vector(
                    feature_map_to_vector(
                        build_metadata_feature_map(text=text),
                        metadata_config.get("feature_names") if isinstance(metadata_config, dict) else None,
                    ),
                    metadata_config,
                )
                for text in batch_texts
            ]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            metadata_tensor = torch.tensor(metadata_vectors, dtype=torch.float32, device=device)
            label_logits, prio_logits = model(**enc, metadata_features=metadata_tensor)
            label_probs = torch.softmax(label_logits, dim=-1)
            prio_probs = torch.softmax(prio_logits, dim=-1)
            label_ids = label_probs.argmax(dim=1).cpu().tolist()
            prio_ids = prio_probs.argmax(dim=1).cpu().tolist()
            label_conf = label_probs.max(dim=1).values.cpu().tolist()
            prio_conf = prio_probs.max(dim=1).values.cpu().tolist()

            for text, label_id, label_score, prio_id, prio_score in zip(
                batch_texts, label_ids, label_conf, prio_ids, prio_conf
            ):
                results.append(
                    {
                        "text": text,
                        "label": id_to_label.get(int(label_id), "Other"),
                        "label_confidence": float(label_score),
                        "priority": id_to_priority.get(int(prio_id), "Medium"),
                        "priority_confidence": float(prio_score),
                    }
                )
    return results


if __name__ == "__main__":
    texts = [
        """I submitted my assignment on time through the portal but the faculty marked it as late submission.
        I even have the confirmation screenshot showing successful upload. Because of this, my marks are affected
        and I am worried about my internal score. Kindly verify the submission logs and update my marks accordingly.""",
        """The hostel WiFi has been extremely slow for the past week making it difficult to attend online classes
        and complete project work. Many students are facing the same issue but no permanent solution has been provided.
        This is affecting our academic productivity and deadlines. Please fix the network problem urgently.""",
        """During practical sessions, there are not enough systems available in the lab and students are forced to share.
        This makes it difficult to complete experiments properly and understand the concepts. The lab infrastructure
        needs improvement so that each student gets fair access.""",
        """I applied for leave through the portal due to medical reasons but it was not approved and now my attendance
        shows shortage. I had already submitted medical proof to the department. Kindly review my leave application
        and correct the attendance records.""",
        """The classroom projector frequently stops working during lectures which interrupts teaching.
        Faculty members waste time trying to fix it and students miss important explanations.
        This issue has been reported multiple times but still not resolved. Please repair or replace the projector.""",
        """There is a delay in fee refund for students who withdrew from elective courses.
        Despite repeated visits to the accounts office, no clear timeline has been provided.
        This financial delay is causing inconvenience for many students. Kindly process the refund soon.""",
        """The campus parking area is overcrowded and vehicles are parked randomly blocking pathways.
        Recently, a student's bike was scratched due to lack of proper parking management.
        Better parking regulation and monitoring is required to avoid such incidents.""",
        """The mess menu displayed is different from what is actually served most of the time.
        Students rely on the menu but end up getting limited or repetitive food options.
        This creates dissatisfaction and complaints among hostel residents. Kindly ensure menu consistency.""",
        """My ID card stopped working for library entry and hostel access even though it is not damaged.
        I reported this to the administration but the issue is still pending.
        This causes inconvenience as I cannot access essential facilities. Please resolve the ID card issue.""",
        """Placement training sessions are scheduled during regular class hours which creates a conflict.
        Students have to choose between attending classes or placement preparation sessions.
        This affects both academic performance and career preparation. Kindly reschedule training sessions.""",
        """The washrooms in the academic block are not cleaned regularly and often lack basic hygiene supplies.
        Students find it uncomfortable to use these facilities throughout the day.
        Proper maintenance and regular cleaning should be ensured.""",
        """There was confusion during exam seating arrangement and many students were searching for their rooms
        at the last minute. This created unnecessary stress before the exam started.
        Better communication and clear instructions would help avoid such situations.""",
        """The sports facilities are not accessible after evening hours even though many students are free only then.
        Limited access discourages participation in physical activities and campus engagement.
        Kindly extend sports facility timings.""",
        """My scholarship amount has been approved but not credited to my bank account yet.
        I verified my bank details and submitted all required documents.
        This delay is affecting my ability to pay academic expenses. Please check and update the payment status.""",
        """Group study rooms in the library are often occupied without booking and staff do not monitor usage.
        Students who reserve rooms are unable to use them at scheduled times.
        A proper booking enforcement system is required to resolve this issue.""",
    ]


    preds = predict_texts(texts)
    for r in preds:
        print(f"\nTEXT: {r['text']}")
        print(f"LABEL: {r['label']} (conf={r['label_confidence']:.3f})")
        print(f"PRIO : {r['priority']} (conf={r['priority_confidence']:.3f})")
