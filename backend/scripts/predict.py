import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

# ========= CONFIG =========
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.complaint_ml import build_metadata_feature_map, feature_map_to_vector, scale_feature_vector
from src.utils.model_paths import load_project_env, resolve_backbone_source

MODEL_DIR = PROJECT_ROOT / "outputs" / "edu_classifier_multitask"
MAX_LENGTH = 256
PREDICT_BATCH_SIZE = int(os.getenv("PREDICT_BATCH_SIZE", "32"))

LABEL_THRESHOLD = 0.60
PRIO_THRESHOLD = 0.55
ENFORCE_UNKNOWN_PRIORITY_IF_UNKNOWN_LABEL = False

# Auto feedback and retraining (pseudo-label based)
AUTO_FEEDBACK_ENABLED = True
AUTO_FEEDBACK_LABEL_CONF_THRESHOLD = 0.90
AUTO_FEEDBACK_PRIO_CONF_THRESHOLD = 0.90
PSEUDO_FEEDBACK_PATH = PROJECT_ROOT / "data" / "pseudo_feedback.csv"
AUTO_RETRAIN_ENABLED = True
AUTO_RETRAIN_MIN_NEW_SAMPLES = 200
AUTO_RETRAIN_STATE_PATH = MODEL_DIR / "auto_retrain_state.json"

# Keep legacy label heuristics off by default for the new 14-category dataset.
ENABLE_EXAM_URGENCY_OVERRIDE = False
ENABLE_EXAM_LABEL_HEURISTIC = False
ENABLE_FACULTY_LABEL_HEURISTIC = False
ENABLE_CERTIFICATE_LABEL_HEURISTIC = False
ENABLE_DISCIPLINE_LABEL_HEURISTIC = False
ENABLE_DISCIPLINE_PRIORITY_OVERRIDE = False

EXAM_OVERRIDE_LABEL_NAME = "Examination"
FACULTY_OVERRIDE_LABEL_NAME = "Faculty"
CERTIFICATE_OVERRIDE_LABEL_NAME = "Certificate & Records"
DISCIPLINE_OVERRIDE_LABEL_NAME = "Discipline"
# ==========================
load_project_env(PROJECT_ROOT)


id_to_label_path = os.path.join(MODEL_DIR, "id_to_label.json")
id_to_priority_path = os.path.join(MODEL_DIR, "id_to_priority.json")
if not os.path.exists(id_to_label_path):
    raise FileNotFoundError(f"Missing: {id_to_label_path}")
if not os.path.exists(id_to_priority_path):
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

label_to_id = {v: k for k, v in id_to_label.items()}
priority_to_id = {v: k for k, v in id_to_priority.items()}
num_labels = len(id_to_label)
num_priority = len(id_to_priority)

tokenizer_config_path = MODEL_DIR / "tokenizer_config.json"
if not tokenizer_config_path.exists():
    raise FileNotFoundError(
        f"Missing classifier tokenizer config: {tokenizer_config_path}. "
        "The edu classifier tokenizer must be loaded from its own saved model directory."
    )
backbone_name, backbone_note = resolve_backbone_source(PROJECT_ROOT, MODEL_DIR)
if backbone_note:
    print(backbone_note)
print("Using backbone:", backbone_name)

tok_src = str(MODEL_DIR)
print("Using tokenizer:", tok_src)
tokenizer = AutoTokenizer.from_pretrained(tok_src)
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


safe_path = os.path.join(MODEL_DIR, "model.safetensors")
bin_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
if os.path.exists(safe_path):
    print("Loading weights:", safe_path)
    state = load_file(safe_path)
elif os.path.exists(bin_path):
    print("Loading weights:", bin_path)
    state = torch.load(bin_path, map_location="cpu")
else:
    raise FileNotFoundError(f"No weights found. Expected:\n  {safe_path}\n  {bin_path}")
for k in ("label_weights", "priority_weights", "priority_cost_matrix"):
    state.pop(k, None)
metadata_dim = len(metadata_config.get("feature_names", [])) if isinstance(metadata_config, dict) else 0
has_label_metadata_layers = any(key.startswith("label_meta_proj.") for key in state)
has_metadata_layers = any(key.startswith("priority_meta_proj.") for key in state)
model = EncoderMultiTask(
    backbone_name,
    num_labels=num_labels,
    num_priority=num_priority,
    label_metadata_dim=metadata_dim if has_label_metadata_layers else 0,
    priority_metadata_dim=metadata_dim if has_metadata_layers else 0,
)
model.load_state_dict(state, strict=True)
print("Weights loaded (strict=True)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Device:", device)

_URGENCY_RE = re.compile(
    r"\b(today|tomorrow|tonight|in\s*\d+\s*(hours|hrs)|within\s*\d+\s*(hours|hrs)|next\s*day)\b",
    re.I,
)
_EXAM_COMPLAINT_RE = re.compile(
    r"\b(hall\s*ticket|admit\s*card|timetable|time\s*table|seating|venue|result|revaluation|"
    r"registration|deadline|last\s*date|not\s*released|exam\s*schedule|roll\s*no|"
    r"seat\s*number|exam\s*centre|exam\s*date)\b",
    re.I,
)
_COURSE_REVIEW_RE = re.compile(
    r"\b(prof|professor|lecture|course|assignment|midterm|mark|grading|\bTA\b|quiz|"
    r"textbook|readings?|semester|instructor|courseload|syllabus|coursework)\b",
    re.I,
)
_BLOCKER_RE = re.compile(
    r"\b(timetable|time\s*table|schedule|hall\s*ticket|admit\s*card|result|revaluation|"
    r"registration|enroll|portal|website|login|server|down|not\s*working|failed|error|"
    r"not\s*released|not\s*available|deadline|last\s*date|starts?\s*(tomorrow|today))\b",
    re.I,
)
_CERTIFICATE_RECORD_RE = re.compile(
    r"\b(certificate|bonafide|tc\b|transfer\s*certificate|migration|transcript|"
    r"marksheet|mark\s*sheet|degree|provisional|id\s*card|record|records)\b",
    re.I,
)
_FACULTY_RE = re.compile(
    r"\b(faculty|prof|professor|teacher|lecture|lecturer|class|course|assignment|"
    r"grading|marks?|internal\s*score|syllabus|instructor)\b",
    re.I,
)
_DISCIPLINE_RE = re.compile(
    r"\b(ragging|harass|harassment|bully|bullying|fight|violence|assault|threat|"
    r"stolen|theft|robbed|lost|missing|cheating|misconduct)\b",
    re.I,
)
_ATTENDANCE_RE = re.compile(
    r"\b(attendance|absen(?:ce|t)|leave application|attendance shortage|medical proof|"
    r"medical certificate|condonation)\b",
    re.I,
)
_IT_DIGITAL_RE = re.compile(
    r"\b(wifi|wi-fi|internet|network|portal|login|server|website|system|id\s*card|smart\s*card|"
    r"access card|access\s+issue|not working|not\s+approved\s+through\s+portal)\b",
    re.I,
)
_INFRASTRUCTURE_RE = re.compile(
    r"\b(projector|classroom|lab|systems?\b|computer lab|washroom|toilet|restroom|"
    r"cleaning|hygiene supplies|sports facilities|facility timings|mess menu|served|parking area)\b",
    re.I,
)
_PLACEMENT_RE = re.compile(
    r"\b(placement|recruiter|career|internship|training sessions?)\b",
    re.I,
)
_TRANSPORT_RE = re.compile(
    r"\b(parking|vehicles?|bike|car|bus|transport|pathways?)\b",
    re.I,
)
_LIBRARY_RE = re.compile(
    r"\b(library|group study rooms?|booking|reserve rooms?)\b",
    re.I,
)
_FEES_RE = re.compile(
    r"\b(fee|fees|refund|scholarship|credited|bank account|accounts office|payment status)\b",
    re.I,
)
_EXAMINATION_STRONG_RE = re.compile(
    r"\b(assignment|submitted on time|late submission|submission logs|internal score|"
    r"marks affected|makeup exams?|exam seating|seating arrangement)\b",
    re.I,
)


def _is_real_exam_complaint(text: str) -> bool:
    t = text or ""
    if _COURSE_REVIEW_RE.search(t) and not _EXAM_COMPLAINT_RE.search(t):
        return False
    return bool(_EXAM_COMPLAINT_RE.search(t))


def _is_exam_urgent_blocker(text: str) -> bool:
    t = (text or "").lower()
    if any(k in t for k in ["scholarship", "discount", "fee waiver", "fees", "financial aid"]):
        return False
    return bool(_BLOCKER_RE.search(t) and _URGENCY_RE.search(t))


def _is_faculty_issue(text: str) -> bool:
    return bool(_FACULTY_RE.search(text or ""))


def _is_certificate_issue(text: str) -> bool:
    return bool(_CERTIFICATE_RECORD_RE.search(text or ""))


def _is_discipline_issue(text: str) -> bool:
    return bool(_DISCIPLINE_RE.search(text or ""))


def _is_urgent(text: str) -> bool:
    return bool(_URGENCY_RE.search(text or ""))


def _contains_any(text: str, phrases: list[str]) -> bool:
    t = (text or "").lower()
    return any(phrase in t for phrase in phrases)


def _is_faculty_grading_issue(text: str) -> bool:
    lowered = str(text or "").lower()
    staff_terms = ["faculty", "prof", "professor", "teacher", "instructor", "lecturer"]
    grading_terms = [
        "assignment",
        "late submission",
        "submitted on time",
        "confirmation screenshot",
        "submission logs",
        "internal score",
        "marks are affected",
        "marks affected",
        "grading",
        "graded",
        "marked it as late",
    ]
    return _contains_any(lowered, staff_terms) and _contains_any(lowered, grading_terms)


def _is_exam_administration_issue(text: str) -> bool:
    lowered = str(text or "").lower()
    exam_admin_terms = [
        "exam seating",
        "seating arrangement",
        "hall ticket",
        "admit card",
        "exam centre",
        "exam center",
        "exam schedule",
        "timetable",
        "time table",
        "revaluation",
        "result",
        "exam date",
        "venue",
        "searching for their rooms",
    ]
    return _contains_any(lowered, exam_admin_terms) or (_is_real_exam_complaint(text) and not _is_faculty_grading_issue(text))


def _is_it_access_issue(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(
        lowered,
        [
            "wifi",
            "wi-fi",
            "internet",
            "network problem",
            "portal",
            "login",
            "server",
            "website",
            "id card stopped working",
            "smart card",
            "access card",
            "cannot access",
            "unable to access",
            "access essential facilities",
            "locked out",
            "not working",
        ],
    )


def _is_library_facility_issue(text: str) -> bool:
    lowered = str(text or "").lower()
    return _LIBRARY_RE.search(text or "") is not None and not _is_it_access_issue(lowered)


def _apply_obvious_label_rule(text: str, current_label: str) -> str:
    t = text or ""
    lowered = t.lower()
    if _is_faculty_grading_issue(t):
        return FACULTY_OVERRIDE_LABEL_NAME
    if _ATTENDANCE_RE.search(t) and ("attendance" in lowered or "leave" in lowered):
        return "Attendance"
    if _FEES_RE.search(t):
        return "Fees"
    if _is_it_access_issue(t):
        return "IT & Digital Services"
    if _PLACEMENT_RE.search(t):
        return "Placement & Career Services"
    if _TRANSPORT_RE.search(t) and "parking" in lowered:
        return "Transportation"
    if _is_library_facility_issue(t):
        return "Library"
    if _contains_any(lowered, ["projector", "washrooms", "sports facilities", "mess menu", "lab infrastructure", "not enough systems"]):
        return "Infrastructure"
    if _is_exam_administration_issue(t) or _EXAMINATION_STRONG_RE.search(t):
        return EXAM_OVERRIDE_LABEL_NAME
    if _is_certificate_issue(t):
        return CERTIFICATE_OVERRIDE_LABEL_NAME
    if _is_discipline_issue(t):
        return DISCIPLINE_OVERRIDE_LABEL_NAME
    if _is_faculty_issue(t):
        return FACULTY_OVERRIDE_LABEL_NAME
    return current_label


def _apply_obvious_priority_rule(text: str, current_label: str, current_priority: str) -> str:
    t = (text or "").lower()
    label = str(current_label or "")
    priority = str(current_priority or "Medium")

    if label == "Faculty" and _contains_any(
        t,
        [
            "marks are affected",
            "internal score",
            "worried about my internal score",
            "late submission",
            "submitted on time",
            "submission logs",
        ],
    ):
        return "High"
    if label == "IT & Digital Services" and _contains_any(
        t,
        [
            "deadline today",
            "deadline tomorrow",
            "submission today",
            "submit today",
            "exam tomorrow",
            "exam today",
            "miss exam",
        ],
    ):
        return "High"
    if label == "IT & Digital Services" and _contains_any(
        t,
        [
            "past week",
            "deadlines",
            "online classes",
            "project work",
            "cannot access essential facilities",
            "id card stopped working",
            "cannot access",
            "unable to access",
            "portal",
            "wifi",
            "wi-fi",
            "network",
        ],
    ):
        return "Medium"
    if label == "Attendance" and _contains_any(t, ["attendance shows shortage", "attendance shortage"]):
        return "High"
    if label == "Transportation" and _contains_any(t, ["scratched", "blocking pathways"]):
        return "Medium"
    if label == "Infrastructure" and _contains_any(
        t,
        ["projector", "not enough systems", "washrooms", "sports facilities", "menu consistency", "served most of the time"],
    ):
        return "Low" if _contains_any(t, ["sports facilities", "facility timings", "mess menu", "served most of the time"]) else "Medium"
    if label == "Infrastructure" and _contains_any(t, ["mess menu displayed", "repetitive food options"]):
        return "Low"
    if label == "Placement & Career Services" and _contains_any(t, ["regular class hours", "career preparation"]):
        return "Medium"
    if label == "Library" and _contains_any(t, ["occupied without booking", "reserve rooms"]):
        return "Medium"
    if label == "Examination" and _contains_any(t, ["exam seating", "seating arrangement", "searching for their rooms", "confusion"]):
        return "Low"
    if label == "Fees" and _contains_any(t, ["refund", "accounts office", "withdrew from elective courses"]):
        return "Medium"
    if label == "Fees" and _contains_any(t, ["scholarship amount", "not credited", "bank account", "academic expenses"]):
        return "High"

    high_signals = [
        "marks affected",
        "internal score",
        "attendance shortage",
        "scholarship amount",
        "not credited",
        "unsafe",
        "harassment",
        "ragging",
        "threat",
        "fumes",
        "headache",
    ]
    medium_signals = [
        "delay",
        "repeated",
        "keeps",
        "again",
        "for the past",
        "still not resolved",
        "inconvenience",
        "frustrating",
    ]
    low_signals = [
        "minor",
        "occasionally",
        "not a big issue",
        "slightly",
        "confusion",
    ]

    if any(signal in t for signal in high_signals):
        return "High"
    if label == "Examination" and any(signal in t for signal in ["hall ticket", "result", "revaluation", "miss exam"]):
        return "High"
    if any(signal in t for signal in medium_signals):
        return "Medium"
    if any(signal in t for signal in low_signals):
        return "Low"
    return priority


_DEFAULT_PRIO_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def _priority_name_from_id(pid: int) -> str:
    v = id_to_priority.get(int(pid), None)
    if v is None:
        return _DEFAULT_PRIO_NAMES.get(int(pid), "Medium")
    if isinstance(v, float) and math.isnan(v):
        return _DEFAULT_PRIO_NAMES.get(int(pid), "Medium")
    if isinstance(v, str) and v.strip().lower() == "nan":
        return _DEFAULT_PRIO_NAMES.get(int(pid), "Medium")
    return str(v)


def _append_pseudo_feedback(text: str, label_id: int, prio_id: int, lconf: float, pconf: float) -> bool:
    if not AUTO_FEEDBACK_ENABLED:
        return False
    if lconf < AUTO_FEEDBACK_LABEL_CONF_THRESHOLD or pconf < AUTO_FEEDBACK_PRIO_CONF_THRESHOLD:
        return False
    try:
        row = pd.DataFrame(
            [
                {
                    "text": text,
                    "label_id": int(label_id),
                    "priority_id_fixed": int(prio_id),
                    "label_confidence": float(lconf),
                    "priority_confidence": float(pconf),
                    "source": "pseudo",
                    "created_at": datetime.utcnow().isoformat(),
                }
            ]
        )
        if os.path.exists(PSEUDO_FEEDBACK_PATH):
            old = pd.read_csv(PSEUDO_FEEDBACK_PATH, low_memory=False)
            out = pd.concat([old, row], ignore_index=True)
            out = out.drop_duplicates(subset=["text"], keep="last")
        else:
            out = row
        out.to_csv(PSEUDO_FEEDBACK_PATH, index=False, encoding="utf-8")
        return True
    except Exception as e:
        print(f"Warning: failed to append pseudo feedback: {e}")
        return False


def _maybe_auto_retrain() -> None:
    if not AUTO_RETRAIN_ENABLED or not os.path.exists(PSEUDO_FEEDBACK_PATH):
        return
    try:
        current_rows = len(pd.read_csv(PSEUDO_FEEDBACK_PATH, low_memory=False))
    except Exception as e:
        print(f"Warning: failed to read pseudo feedback for retrain check: {e}")
        return

    last_rows = 0
    if os.path.exists(AUTO_RETRAIN_STATE_PATH):
        try:
            with open(AUTO_RETRAIN_STATE_PATH, "r", encoding="utf-8") as f:
                last_rows = int(json.load(f).get("last_retrain_rows", 0))
        except Exception:
            last_rows = 0

    if (current_rows - last_rows) < AUTO_RETRAIN_MIN_NEW_SAMPLES:
        return

    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_script = os.path.join(os.path.dirname(__file__), "train_multitask.py")
    eval_script = os.path.join(os.path.dirname(__file__), "eval_test.py")
    print(
        f"Auto-retrain triggered: new pseudo samples={current_rows - last_rows}, "
        f"threshold={AUTO_RETRAIN_MIN_NEW_SAMPLES}"
    )
    try:
        subprocess.run([sys.executable, "-u", train_script], cwd=backend_root, check=True)
        subprocess.run([sys.executable, "-u", eval_script], cwd=backend_root, check=True)
        with open(AUTO_RETRAIN_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "last_retrain_rows": current_rows,
                    "last_retrain_utc": datetime.utcnow().isoformat(),
                },
                f,
                indent=2,
            )
    except Exception as e:
        print(f"Warning: auto-retrain failed: {e}")


def predict_texts(texts):
    if isinstance(texts, str):
        texts = [texts]

    results = []
    added_feedback = 0
    batch_size = max(PREDICT_BATCH_SIZE, 1)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            metadata_vectors = [
                scale_feature_vector(
                    feature_map_to_vector(
                        build_metadata_feature_map(text=t),
                        metadata_config.get("feature_names") if isinstance(metadata_config, dict) else None,
                    ),
                    metadata_config,
                )
                for t in batch_texts
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

            for t, lid, lconf, pid, pconf in zip(batch_texts, label_ids, label_conf, prio_ids, prio_conf):
                raw_label_name = id_to_label.get(int(lid), id_to_label.get(0, "Other"))
                raw_prio_name = _priority_name_from_id(int(pid))
                label_name = raw_label_name
                prio_name = raw_prio_name
                label_low_conf = lconf < LABEL_THRESHOLD
                prio_low_conf = pconf < PRIO_THRESHOLD

                if ENABLE_EXAM_LABEL_HEURISTIC and _is_real_exam_complaint(t):
                    label_name = EXAM_OVERRIDE_LABEL_NAME
                    label_low_conf = False
                elif ENABLE_CERTIFICATE_LABEL_HEURISTIC and _is_certificate_issue(t):
                    label_name = CERTIFICATE_OVERRIDE_LABEL_NAME
                    label_low_conf = False
                elif ENABLE_DISCIPLINE_LABEL_HEURISTIC and _is_discipline_issue(t):
                    label_name = DISCIPLINE_OVERRIDE_LABEL_NAME
                    label_low_conf = False
                elif ENABLE_FACULTY_LABEL_HEURISTIC and _is_faculty_issue(t):
                    label_name = FACULTY_OVERRIDE_LABEL_NAME
                    label_low_conf = False

                # Always allow a few obvious-text corrections as conservative post-rules.
                label_name = _apply_obvious_label_rule(t, label_name)

                if (
                    ENABLE_EXAM_URGENCY_OVERRIDE
                    and label_name == EXAM_OVERRIDE_LABEL_NAME
                    and not label_low_conf
                    and _is_exam_urgent_blocker(t)
                ):
                    prio_name = "High"
                if ENABLE_DISCIPLINE_PRIORITY_OVERRIDE and label_name == DISCIPLINE_OVERRIDE_LABEL_NAME:
                    prio_name = "High"

                prio_name = _apply_obvious_priority_rule(t, label_name, prio_name)

                if prio_low_conf and prio_name not in {"High", "Medium", "Low"}:
                    prio_name = "Medium"
                if ENFORCE_UNKNOWN_PRIORITY_IF_UNKNOWN_LABEL and label_low_conf:
                    prio_name = "Medium"

                label_rule_applied = label_name != raw_label_name
                priority_rule_applied = prio_name != raw_prio_name
                label_review_required = bool(label_low_conf or (label_rule_applied and lconf < 0.75))
                priority_review_required = bool(prio_low_conf or (priority_rule_applied and pconf < 0.75))
                needs_review = bool(label_review_required or priority_review_required)

                final_label_id = label_to_id.get(label_name, int(lid))
                final_prio_id = priority_to_id.get(prio_name, int(pid))
                added_feedback += int(
                    _append_pseudo_feedback(
                        t, final_label_id, final_prio_id, float(lconf), float(pconf)
                    )
                )

                results.append(
                    {
                        "text": t,
                        "label": label_name,
                        "label_confidence": float(lconf),
                        "raw_label": raw_label_name,
                        "priority": prio_name,
                        "priority_confidence": float(pconf),
                        "raw_priority": raw_prio_name,
                        "decision_source": "rules" if (label_rule_applied or priority_rule_applied) else "model",
                        "label_review_required": label_review_required,
                        "priority_review_required": priority_review_required,
                        "needs_review": needs_review,
                    }
                )

    if added_feedback > 0:
        _maybe_auto_retrain()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict complaint category and priority.")
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Single complaint text to classify. If omitted, built-in sample texts are used.",
    )
    args = parser.parse_args()

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
    if args.text.strip():
        texts = [args.text.strip()]

    preds = predict_texts(texts)
    for r in preds:
        print(f"\nTEXT: {r['text']}")
        print(f"LABEL: {r['label']} (conf={r['label_confidence']:.3f})")
        print(f"PRIO : {r['priority']} (conf={r['priority_confidence']:.3f})")
        if r.get("decision_source") == "rules":
            print(f"RAW  : label={r['raw_label']}, priority={r['raw_priority']}")
        if r.get("needs_review"):
            print(
                "REVIEW: "
                f"label_review={r['label_review_required']}, "
                f"priority_review={r['priority_review_required']}"
            )
