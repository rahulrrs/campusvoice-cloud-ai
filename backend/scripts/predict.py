# scripts/predict.py
import os
import json
import re
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

# ========= CONFIG =========
MODEL_DIR = r"outputs\edu_classifier_multitask"
MAX_LENGTH = 256

BACKBONE_DIR     = r"outputs\cfpb_outputs\distilbert_cfpb_mlm"
FALLBACK_BACKBONE = "distilbert-base-uncased"

# ── Thresholds ────────────────────────────────────────────────────────────────
# Lowered from 0.70 → 0.55 because label noise during training causes legit
# predictions to fall below 0.70 even when the model is directionally correct.
LABEL_THRESHOLD = 0.55
PRIO_THRESHOLD  = 0.45

ENFORCE_UNKNOWN_PRIORITY_IF_UNKNOWN_LABEL = True

# ── Feature flags ─────────────────────────────────────────────────────────────
ENABLE_EXAM_URGENCY_OVERRIDE        = True
ENABLE_EXAM_LABEL_HEURISTIC         = True   # NEW: force Examination label for clear exam complaints
ENABLE_HOSTEL_WATER_OVERRIDE        = True
ENABLE_IT_PORTAL_OVERRIDE           = True
ENABLE_SAFETY_THREAT_OVERRIDE       = True
ENABLE_LOST_FOUND_OVERRIDE          = True
ENABLE_IT_PRIORITY_OVERRIDE         = True
ENABLE_SCHOLARSHIP_URGENCY_OVERRIDE = True
ENABLE_RAGGING_LABEL_OVERRIDE       = True
ENABLE_TRANSPORT_PRIORITY_OVERRIDE  = True

# ── Label name constants (must match your dataset) ────────────────────────────
EXAM_OVERRIDE_LABEL_NAME        = "Examination"
HOSTEL_WATER_OVERRIDE_LABEL_NAME = "Hostel"
IT_OVERRIDE_LABEL_NAME          = "IT & Digital Services"
SAFETY_OVERRIDE_LABEL_NAME      = "Ragging / Harassment"
LOST_FOUND_OVERRIDE_LABEL_NAME  = "Lost & Found"
SCHOLARSHIP_OVERRIDE_LABEL_NAME = "Scholarship"
TRANSPORT_OVERRIDE_LABEL_NAME   = "Transport"
# ==========================

id_to_label_path    = os.path.join(MODEL_DIR, "id_to_label.json")
id_to_priority_path = os.path.join(MODEL_DIR, "id_to_priority.json")

if not os.path.exists(id_to_label_path):
    raise FileNotFoundError(f"Missing: {id_to_label_path}")
if not os.path.exists(id_to_priority_path):
    raise FileNotFoundError(f"Missing: {id_to_priority_path}")

with open(id_to_label_path, "r", encoding="utf-8") as f:
    id_to_label = {int(k): v for k, v in json.load(f).items()}

with open(id_to_priority_path, "r", encoding="utf-8") as f:
    id_to_priority = {int(k): v for k, v in json.load(f).items()}

num_labels   = len(id_to_label)
num_priority = len(id_to_priority)

# Prefer trained backbone in MODEL_DIR
if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    backbone_name = MODEL_DIR
elif os.path.isdir(BACKBONE_DIR):
    backbone_name = BACKBONE_DIR
else:
    backbone_name = FALLBACK_BACKBONE

print("Using backbone:", backbone_name)

tok_src = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "tokenizer_config.json")) else backbone_name
print("Using tokenizer:", tok_src)
tokenizer = AutoTokenizer.from_pretrained(tok_src)


# ── Model – MUST match train_multitask.py architecture ───────────────────────
class DistilBertMultiTask(nn.Module):
    def __init__(self, backbone_name: str, num_labels: int, num_priority: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.label_dropout = nn.Dropout(0.2)
        self.label_hidden  = nn.Linear(hidden, hidden // 2)
        self.label_head    = nn.Linear(hidden // 2, num_labels)

        self.prio_dropout = nn.Dropout(0.2)
        self.prio_hidden  = nn.Linear(hidden, hidden // 4)
        self.prio_head    = nn.Linear(hidden // 4, num_priority)

        self.act = nn.GELU()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        label_logits = self.label_head(self.act(self.label_hidden(self.label_dropout(pooled))))
        prio_logits  = self.prio_head(self.act(self.prio_hidden(self.prio_dropout(pooled))))
        return label_logits, prio_logits


model = DistilBertMultiTask(backbone_name, num_labels=num_labels, num_priority=num_priority)

safe_path = os.path.join(MODEL_DIR, "model.safetensors")
bin_path  = os.path.join(MODEL_DIR, "pytorch_model.bin")

if os.path.exists(safe_path):
    print("Loading weights:", safe_path)
    state = load_file(safe_path)
elif os.path.exists(bin_path):
    print("Loading weights:", bin_path)
    state = torch.load(bin_path, map_location="cpu")
else:
    raise FileNotFoundError(f"No weights found. Expected:\n  {safe_path}\n  {bin_path}")

model.load_state_dict(state, strict=True)
print("✅ Weights loaded (strict=True)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Device:", device)


# ── Regex patterns ────────────────────────────────────────────────────────────
_URGENCY_RE = re.compile(
    r"\b(today|tomorrow|tonight|in\s*\d+\s*(hours|hrs)|within\s*\d+\s*(hours|hrs)|next\s*day)\b",
    re.I,
)
# Exam-specific admin complaint keywords (must NOT be course-review style)
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
_WATER_OUTAGE_RE = re.compile(
    r"\b(no\s*water|water\s*(is\s*)?not\s*available|water\s*supply\s*.{0,20}\bdown\b|"
    r"water\s*problem|water\s*outage|water\s*cut)\b",
    re.I,
)
_DURATION_RE = re.compile(r"\b(\d+)\s*(day|days|d)\b", re.I)
_IT_PORTAL_RE = re.compile(
    r"\b(portal|website|login|server|app|payment|fee\s*payment|transaction|failed|error|"
    r"not\s*working|receipt)\b",
    re.I,
)
_SAFETY_THREAT_RE = re.compile(r"\b(harass|ragging|assault|threat|violence)\b", re.I)
_LOST_FOUND_RE    = re.compile(r"\b(stolen|theft|robbed|lost|missing|stole)\b", re.I)
_TRANSPORT_RE     = re.compile(
    r"\b(bus|transport|overcrowd|overcrowded|crowded|unsafe|rush\s*hour|peak\s*hour)\b", re.I
)


_SCHOLARSHIP_CONTEXT_RE = re.compile(
    r"\b(scholarship|financial\s*aid|stipend|bursary|fee\s*waiver|application\s*status)\b",
    re.I,
)


# ── Heuristic helpers ─────────────────────────────────────────────────────────
def _is_real_exam_complaint(text: str) -> bool:
    """True when text is a real exam-admin complaint, not a course review or scholarship query."""
    t = text or ""
    # Scholarship context trumps exam keywords like 'deadline'
    if _SCHOLARSHIP_CONTEXT_RE.search(t):
        return False
    if _COURSE_REVIEW_RE.search(t) and not _EXAM_COMPLAINT_RE.search(t):
        return False
    return bool(_EXAM_COMPLAINT_RE.search(t))


def _is_exam_urgent_blocker(text: str) -> bool:
    t = (text or "").lower()
    if any(k in t for k in ["scholarship", "discount", "fee waiver", "fees", "financial aid"]):
        return False
    return bool(_BLOCKER_RE.search(t) and _URGENCY_RE.search(t))


def _is_hostel_water_outage(text: str) -> bool:
    t = (text or "").lower()
    if not _WATER_OUTAGE_RE.search(t):
        return False
    m = _DURATION_RE.search(t)
    if not m:
        return True
    try:
        days = int(m.group(1))
    except ValueError:
        return True
    return days >= 1


def _is_it_portal_issue(text: str) -> bool:
    return bool(_IT_PORTAL_RE.search(text or ""))

def _is_safety_threat(text: str) -> bool:
    return bool(_SAFETY_THREAT_RE.search(text or ""))

def _is_lost_found(text: str) -> bool:
    return bool(_LOST_FOUND_RE.search(text or ""))

def _mentions_hostel(text: str) -> bool:
    return bool(re.search(r"\b(hostel|dorm|warden|roommate)\b", text or "", re.I))

def _is_transport_issue(text: str) -> bool:
    return bool(_TRANSPORT_RE.search(text or ""))

def _is_urgent(text: str) -> bool:
    return bool(_URGENCY_RE.search(text or ""))


_DEFAULT_PRIO_NAMES = {0: "Low", 1: "Medium", 2: "High"}

def _priority_name_from_id(pid: int) -> str:
    v = id_to_priority.get(int(pid), None)
    if v is None:
        return _DEFAULT_PRIO_NAMES.get(int(pid), "Unknown")
    if isinstance(v, float) and math.isnan(v):
        return _DEFAULT_PRIO_NAMES.get(int(pid), "Unknown")
    if isinstance(v, str) and v.strip().lower() == "nan":
        return _DEFAULT_PRIO_NAMES.get(int(pid), "Unknown")
    return str(v)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_texts(texts):
    if isinstance(texts, str):
        texts = [texts]

    with torch.no_grad():
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        label_logits, prio_logits = model(**enc)

        label_probs = torch.softmax(label_logits, dim=-1)
        prio_probs  = torch.softmax(prio_logits,  dim=-1)

        label_ids  = label_probs.argmax(dim=1).cpu().tolist()
        prio_ids   = prio_probs.argmax(dim=1).cpu().tolist()
        label_conf = label_probs.max(dim=1).values.cpu().tolist()
        prio_conf  = prio_probs.max(dim=1).values.cpu().tolist()

    results = []
    for t, lid, lconf, pid, pconf in zip(texts, label_ids, label_conf, prio_ids, prio_conf):
        label_name = id_to_label.get(int(lid), "Unknown")
        prio_name  = _priority_name_from_id(int(pid))

        label_unknown = lconf < LABEL_THRESHOLD

        # ── 1. Heuristic LABEL overrides (specific → broad) ───────────────
        # Hostel water outage (most specific – check first)
        if ENABLE_HOSTEL_WATER_OVERRIDE and _is_hostel_water_outage(t) and _mentions_hostel(t):
            label_name    = HOSTEL_WATER_OVERRIDE_LABEL_NAME
            label_unknown = False

        # Real exam complaint (NEW) – fires even if model was uncertain
        elif ENABLE_EXAM_LABEL_HEURISTIC and _is_real_exam_complaint(t):
            label_name    = EXAM_OVERRIDE_LABEL_NAME
            label_unknown = False

        # Transport issue
        elif ENABLE_TRANSPORT_PRIORITY_OVERRIDE and _is_transport_issue(t):
            label_name    = TRANSPORT_OVERRIDE_LABEL_NAME
            label_unknown = False

        # IT / portal issue
        elif ENABLE_IT_PORTAL_OVERRIDE and _is_it_portal_issue(t):
            label_name    = IT_OVERRIDE_LABEL_NAME
            label_unknown = False

        # Lost & Found (can stack with other labels, so no elif)
        if ENABLE_LOST_FOUND_OVERRIDE and _is_lost_found(t):
            label_name    = LOST_FOUND_OVERRIDE_LABEL_NAME
            label_unknown = False

        # Safety / ragging (highest urgency, apply last to win)
        if ENABLE_RAGGING_LABEL_OVERRIDE and _is_safety_threat(t):
            label_name    = SAFETY_OVERRIDE_LABEL_NAME
            label_unknown = False

        # ── 2. Priority: low-confidence → Unknown ─────────────────────────
        if pconf < PRIO_THRESHOLD:
            prio_name = "Unknown"

        if ENFORCE_UNKNOWN_PRIORITY_IF_UNKNOWN_LABEL and label_unknown:
            prio_name = "Unknown"

        # ── 3. Heuristic PRIORITY overrides ──────────────────────────────
        # Exam urgency → High
        if (
            ENABLE_EXAM_URGENCY_OVERRIDE
            and label_name == EXAM_OVERRIDE_LABEL_NAME
            and not label_unknown
            and _is_exam_urgent_blocker(t)
        ):
            prio_name = "High"

        # Hostel water outage → High
        if (
            ENABLE_HOSTEL_WATER_OVERRIDE
            and label_name == HOSTEL_WATER_OVERRIDE_LABEL_NAME
            and not label_unknown
            and _is_hostel_water_outage(t)
        ):
            prio_name = "High"

        # Safety / ragging → High
        if (
            ENABLE_SAFETY_THREAT_OVERRIDE
            and label_name == SAFETY_OVERRIDE_LABEL_NAME
            and not label_unknown
            and _is_safety_threat(t)
        ):
            prio_name = "High"

        # IT portal → Medium (not urgent by default)
        if ENABLE_IT_PRIORITY_OVERRIDE and label_name == IT_OVERRIDE_LABEL_NAME and _is_it_portal_issue(t):
            prio_name = "Medium"

        # Scholarship + urgency → High
        if (
            ENABLE_SCHOLARSHIP_URGENCY_OVERRIDE
            and label_name == SCHOLARSHIP_OVERRIDE_LABEL_NAME
            and _is_urgent(t)
        ):
            prio_name = "High"

        # Transport → Medium
        if (
            ENABLE_TRANSPORT_PRIORITY_OVERRIDE
            and label_name == TRANSPORT_OVERRIDE_LABEL_NAME
            and _is_transport_issue(t)
        ):
            prio_name = "Medium"

        results.append({
            "text":               t,
            "label":              label_name,
            "label_confidence":   float(lconf),
            "priority":           prio_name,
            "priority_confidence": float(pconf),
        })

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
        Recently, a student’s bike was scratched due to lack of proper parking management.
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
        A proper booking enforcement system is required to resolve this issue."""
    ]

    preds = predict_texts(texts)
    for r in preds:
        print(f"\nTEXT: {r['text']}")
        print(f"LABEL: {r['label']} (conf={r['label_confidence']:.3f})")
        print(f"PRIO : {r['priority']} (conf={r['priority_confidence']:.3f})")
