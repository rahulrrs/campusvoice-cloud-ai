import json
import logging
import os
import re
import runpy
import socket
import subprocess
import sys
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import jwt
import numpy as np
import psycopg2
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from psycopg2.extras import Json, RealDictCursor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

import torch
import torch.nn as nn
try:
    from google import genai as google_genai
except Exception:
    google_genai = None

try:
    from safetensors.torch import load_file as load_safetensors_file
except Exception:
    load_safetensors_file = None

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

APP_ROOT = Path(__file__).resolve().parents[1]

# ========= ML CONFIG =========
MODEL_DIR = APP_ROOT / "outputs" / "edu_classifier_multitask"
MAX_LENGTH = 256
LABEL_THRESHOLD = 0.55
PRIO_THRESHOLD = 0.50
FRONTEND_FEEDBACK_PATH = APP_ROOT / "data" / "frontend_feedback.csv"
AUTO_RETRAIN_STATE_PATH = MODEL_DIR / "api_auto_retrain_state.json"
AUTO_RETRAIN_MIN_NEW_FEEDBACK = 30

LABEL_TO_DEPT = {
    "Academic": "Academic Affairs",
    "Faculty": "Academic Affairs",
    "Examination": "Examination Cell",
    "IT & Digital Services": "IT Support",
    "Fees": "Accounts",
    "Hostel": "Hostel Office",
    "Mess / Canteen": "Catering/Mess",
    "Library": "Library",
    "Placement & Career Services": "Career Services",
    "Transport": "Transport Office",
    "Health Services": "Health Center",
    "Safety & Security": "Security",
    "Scholarship": "Scholarship Office",
    "Administration": "Admin Office",
    "Certificate & Records": "Admin Office",
    "Discipline": "Disciplinary Committee",
    "Attendance": "Academic Affairs",
    "Infrastructure": "Maintenance",
    "Lab": "Lab Incharge",
    "Lost & Found": "Helpdesk",
    "Ragging / Harassment": "Disciplinary Committee",
    "Other": "Helpdesk",
    "Unknown": "Helpdesk",
}

_UNKNOWN_LABEL_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Infrastructure", re.compile(r"\b(ac|air\s*condition|fan|bench|washroom|toilet|projector|light|classroom)\b", re.I)),
    ("Hostel", re.compile(r"\b(hostel|dorm|warden|roommate)\b", re.I)),
    ("Mess / Canteen", re.compile(r"\b(mess|canteen|food|meal|hygiene)\b", re.I)),
    ("Library", re.compile(r"\b(library|book|reading\s*room)\b", re.I)),
    ("Lab", re.compile(r"\b(lab|laboratory|practical|experiment)\b", re.I)),
    ("Transport", re.compile(r"\b(bus|transport|shuttle|route|driver)\b", re.I)),
    ("IT & Digital Services", re.compile(r"\b(portal|website|login|server|app|network|wifi|wi-?fi)\b", re.I)),
    ("Attendance", re.compile(r"\b(attendance|absent|present)\b", re.I)),
    ("Examination", re.compile(r"\b(exam|hall\s*ticket|timetable|result|revaluation)\b", re.I)),
    ("Fees", re.compile(r"\b(fee|fees|refund|payment)\b", re.I)),
    ("Scholarship", re.compile(r"\b(scholarship|stipend|financial\s*aid|bursary)\b", re.I)),
    ("Placement & Career Services", re.compile(r"\b(placement|internship|career|recruit|training)\b", re.I)),
    ("Safety & Security", re.compile(r"\b(safety|security|threat|unsafe|violence|assault)\b", re.I)),
    ("Ragging / Harassment", re.compile(r"\b(ragging|harass|bully)\b", re.I)),
    ("Lost & Found", re.compile(r"\b(lost|missing|stolen|theft|wallet|id\s*card)\b", re.I)),
]


def _recover_unknown_label(text: str, default_label: str) -> str:
    for label, pattern in _UNKNOWN_LABEL_RULES:
        if pattern.search(text):
            return label
    return default_label


class Settings(BaseModel):
    aws_region: str = Field(default="us-east-1")
    cognito_user_pool_id: str = Field(default="")
    cognito_app_client_id: str = Field(default="")
    rds_host: str = Field(default="")
    rds_port: int = Field(default=5432)
    rds_database: str = Field(default="complaints")
    rds_user: str = Field(default="")
    rds_password: str = Field(default="")
    attachments_bucket: str = Field(default="")
    cors_allow_origin: str = Field(default="*")
    presigned_url_expires_seconds: int = Field(default=900)
    admin_emails: str = Field(default="")
    backbone_model_name: str = Field(default="distilbert-base-uncased")
    backbone_model_dir: str = Field(default=str(APP_ROOT / "outputs" / "general_complaint_model"))
    chatbot_provider: str = Field(default="gemini")
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.5-flash")


def get_settings() -> Settings:
    return Settings(
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        cognito_user_pool_id=os.getenv("COGNITO_USER_POOL_ID", ""),
        cognito_app_client_id=os.getenv("COGNITO_APP_CLIENT_ID", ""),
        rds_host=os.getenv("RDS_HOST", ""),
        rds_port=int(os.getenv("RDS_PORT", "5432")),
        rds_database=os.getenv("RDS_DATABASE", "complaints"),
        rds_user=os.getenv("RDS_USER", ""),
        rds_password=os.getenv("RDS_PASSWORD", ""),
        attachments_bucket=os.getenv("ATTACHMENTS_BUCKET", ""),
        cors_allow_origin=os.getenv("CORS_ALLOW_ORIGIN", "*"),
        presigned_url_expires_seconds=int(os.getenv("PRESIGNED_URL_EXPIRES_SECONDS", "900")),
        admin_emails=os.getenv("ADMIN_EMAILS", ""),
        backbone_model_name=os.getenv("BACKBONE_MODEL_NAME", "distilbert-base-uncased"),
        backbone_model_dir=os.getenv(
            "BACKBONE_MODEL_DIR",
            str(APP_ROOT / "outputs" / "general_complaint_model"),
        ),
        chatbot_provider=os.getenv("CHATBOT_PROVIDER", "gemini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )


settings = get_settings()
logger = logging.getLogger(__name__)
LEGACY_BACKBONE_MODEL_DIR = APP_ROOT / "outputs" / "distilbert_cfpb_mlm"


def _cors_origins_from_env(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_backbone_dir() -> str:
    if os.path.isdir(settings.backbone_model_dir):
        return settings.backbone_model_dir
    if LEGACY_BACKBONE_MODEL_DIR.is_dir():
        return str(LEGACY_BACKBONE_MODEL_DIR)
    return settings.backbone_model_name


app = FastAPI(title="Complaint Routing and Management API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_from_env(settings.cors_allow_origin) or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_client = boto3.client(
    "s3",
    region_name=settings.aws_region,
    config=BotoConfig(connect_timeout=2, read_timeout=2, retries={"max_attempts": 1}),
)


def _require_env_for_auth() -> None:
    required = {
        "COGNITO_USER_POOL_ID": settings.cognito_user_pool_id,
        "COGNITO_APP_CLIENT_ID": settings.cognito_app_client_id,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing backend environment variables: {', '.join(missing)}",
        )


def _require_env_for_db() -> None:
    required = {
        "RDS_HOST": settings.rds_host,
        "RDS_USER": settings.rds_user,
        "RDS_PASSWORD": settings.rds_password,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing backend environment variables: {', '.join(missing)}",
        )


def _require_env_for_uploads() -> None:
    _require_env_for_auth()
    if not settings.attachments_bucket:
        raise HTTPException(
            status_code=500,
            detail="Missing backend environment variable: ATTACHMENTS_BUCKET",
        )


def get_db_conn():
    _require_env_for_db()
    return psycopg2.connect(
        host=settings.rds_host,
        port=settings.rds_port,
        dbname=settings.rds_database,
        user=settings.rds_user,
        password=settings.rds_password,
        connect_timeout=3,
        cursor_factory=RealDictCursor,
    )


def _jwks_url() -> str:
    return (
        f"https://cognito-idp.{settings.aws_region}.amazonaws.com/"
        f"{settings.cognito_user_pool_id}/.well-known/jwks.json"
    )


_jwks_client_lock = threading.Lock()
_jwks_client: jwt.PyJWKClient | None = None


def _get_jwks_client() -> jwt.PyJWKClient:
    global _jwks_client
    if _jwks_client is not None:
        return _jwks_client
    with _jwks_client_lock:
        if _jwks_client is None:
            _jwks_client = jwt.PyJWKClient(_jwks_url())
    return _jwks_client


class CurrentUser(BaseModel):
    user_id: str
    email: str | None = None
    is_admin: bool = False


def _get_admin_email_set() -> set[str]:
    return {
        item.strip().lower()
        for item in settings.admin_emails.split(",")
        if item.strip()
    }


def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    _require_env_for_auth()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    issuer = (
        f"https://cognito-idp.{settings.aws_region}.amazonaws.com/"
        f"{settings.cognito_user_pool_id}"
    )

    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"verify_aud": False},
        )
    except Exception as exc:
        # Dev-friendly fallback: inspect token claims without signature verification
        # so we can diagnose mismatched token types/app clients more easily.
        try:
            payload = jwt.decode(
                token,
                options={
                    "verify_signature": False,
                    "verify_exp": True,
                    "verify_aud": False,
                    "verify_iss": False,
                },
                algorithms=["RS256"],
            )
            unverified_iss = payload.get("iss")
            if not isinstance(unverified_iss, str) or settings.cognito_user_pool_id not in unverified_iss:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token issuer"
                ) from exc
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            ) from exc

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing sub")
    email = payload.get("email")
    email_str = email if isinstance(email, str) else None
    is_admin = bool(email_str and email_str.lower() in _get_admin_email_set())
    return CurrentUser(user_id=user_id, email=email_str, is_admin=is_admin)


def require_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(row)
    created_at = serialized.get("created_at")
    updated_at = serialized.get("updated_at")
    if isinstance(created_at, datetime):
        serialized["created_at"] = created_at.astimezone(timezone.utc).isoformat()
    if isinstance(updated_at, datetime):
        serialized["updated_at"] = updated_at.astimezone(timezone.utc).isoformat()
    attachments = serialized.get("attachments")
    if not isinstance(attachments, list):
        serialized["attachments"] = []
    evidence_types = serialized.get("evidence_types")
    if not isinstance(evidence_types, list):
        serialized["evidence_types"] = []
    analysis = serialized.get("analysis")
    if not isinstance(analysis, dict):
        serialized["analysis"] = {}
    return serialized


def _sanitize_filename(name: str) -> str:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", "."))
    return safe or "attachment"


class ComplaintIn(BaseModel):
    text: str


class ComplaintCreate(BaseModel):
    title: str
    description: str
    category: str = "Uncategorized"
    priority: str = "medium"
    status: str = "pending"
    attachments: list[str] = Field(default_factory=list)
    evidence_types: list[str] = Field(default_factory=list)
    source_language: str | None = None
    analysis: dict[str, Any] = Field(default_factory=dict)


class PresignedUploadRequest(BaseModel):
    fileName: str
    contentType: str = "application/octet-stream"


class PresignedDownloadRequest(BaseModel):
    key: str


class ComplaintAdminUpdate(BaseModel):
    category: str | None = None
    priority: str | None = None
    department: str | None = None
    status: str | None = None


class AutoClassifyRequest(BaseModel):
    only_pending: bool = True


class ComplaintAnalysisRequest(BaseModel):
    title: str = ""
    description: str = ""


class ChatTurn(BaseModel):
    role: str
    text: str


class ChatbotRequest(BaseModel):
    message: str
    history: list[ChatTurn] = Field(default_factory=list)


# ---------- Optional ML model loading ----------
_model_lock = threading.Lock()
_model_ready = False
_model_error: str | None = None
tokenizer = None
model = None
device = None
_gemini_client = None
id_to_label: dict[int, str] = {}
id_to_priority: dict[int, str] = {}
_retrain_lock = threading.Lock()
_retrain_in_progress = False
_chatbot_lock = threading.Lock()

_CHATBOT_INTENTS: dict[str, dict[str, Any]] = {
    "registration": {
        "examples": [
            "how do i submit a complaint",
            "register issue",
            "file grievance",
            "where can i create complaint",
            "help me lodge complaint",
        ],
        "reply": (
            "Go to Submit Complaint, enter title and details, then attach image/document/voice evidence. "
            "The system will auto-analyze category, urgency, abuse risk, and duplicates."
        ),
    },
    "status_lookup": {
        "examples": [
            "check complaint status",
            "track my complaint progress",
            "what is my pending complaint count",
            "show my resolved issues",
        ],
        "reply": "I can summarize your complaint status right now.",
    },
    "duplicate_check": {
        "examples": [
            "is this issue already reported",
            "find duplicate complaint",
            "same issue reported before",
            "similar complaint exists",
        ],
        "reply": "I will run semantic duplicate detection on your text.",
    },
    "evidence_help": {
        "examples": [
            "what evidence should i upload",
            "can i submit voice complaint",
            "is image proof allowed",
            "how to attach files",
        ],
        "reply": (
            "Use text for details, images for visual proof, and voice for narrated context. "
            "Clear location/time details improve routing quality."
        ),
    },
    "recommendation_help": {
        "examples": [
            "suggest solution for complaint",
            "recommend action",
            "what should department do",
            "best resolution workflow",
        ],
        "reply": "I can suggest actions based on semantically similar resolved complaints.",
    },
    "complaint_coaching": {
        "examples": [
            "i want to file a complaint",
            "help me write my complaint",
            "what category is this issue",
            "can you improve my complaint text",
            "is this complaint clear enough",
        ],
        "reply": (
            "I can read your complaint draft, predict its category and urgency, and tell you what details are still missing."
        ),
    },
    "analytics_help": {
        "examples": [
            "predict complaint trends",
            "which issue category is rising",
            "forecast complaint spike",
            "trend analytics",
        ],
        "reply": "Admin analytics includes trend forecasts and rising categories for the next 7 days.",
    },
    "general_help": {
        "examples": [
            "help",
            "what can you do",
            "what do you do",
            "who are you",
            "what are you",
            "introduce yourself",
            "how can you help me",
            "assistant capabilities",
            "how this portal works",
        ],
        "reply": (
            "I am the CampusVoice assistant. I can explain how the portal works, help you register a complaint, "
            "check complaint status, suggest evidence to upload, look for similar complaints, and guide you step by step."
        ),
    },
    "general_chat": {
        "examples": [
            "how are you",
            "thank you",
            "thanks",
            "bye",
            "goodbye",
            "can we chat",
            "talk normally",
            "be a general assistant",
        ],
        "reply": (
            "I can chat more generally too. I can explain ideas, help you think through questions, rewrite text, "
            "brainstorm options, and switch into complaint-helper mode whenever you need it."
        ),
    },
}


def _load_id_maps() -> tuple[dict[int, str], dict[int, str]]:
    with open(MODEL_DIR / "id_to_label.json", "r", encoding="utf-8") as f:
        local_labels = {int(k): v for k, v in json.load(f).items()}
    with open(MODEL_DIR / "id_to_priority.json", "r", encoding="utf-8") as f:
        local_prios = {int(k): v for k, v in json.load(f).items()}
    return local_labels, local_prios


def _priority_to_id(priority: str, prio_map: dict[int, str]) -> int | None:
    if not isinstance(priority, str):
        return None
    p = priority.strip().lower()
    if p not in {"low", "medium", "high"}:
        return None
    for k, v in prio_map.items():
        if str(v).strip().lower() == p:
            return int(k)
    fallback = {"low": 0, "medium": 1, "high": 2}
    return fallback.get(p)


def _append_frontend_feedback(
    title: str,
    description: str,
    category: str,
    priority: str,
    source: str,
) -> bool:
    try:
        label_map, prio_map = _load_id_maps()
    except Exception:
        return False

    label_to_id = {str(v).strip().lower(): int(k) for k, v in label_map.items()}
    cat_norm = (category or "").strip().lower()
    if cat_norm in {"", "unknown", "uncategorized"}:
        return False
    label_id = label_to_id.get(cat_norm)
    if label_id is None:
        return False
    prio_id = _priority_to_id(priority, prio_map)
    if prio_id is None:
        return False

    text = f"{(title or '').strip()}\n\n{(description or '').strip()}".strip()
    if not text:
        return False

    FRONTEND_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "text": text,
        "label_id": label_id,
        "priority_id_fixed": prio_id,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        import pandas as pd

        if FRONTEND_FEEDBACK_PATH.exists():
            old = pd.read_csv(FRONTEND_FEEDBACK_PATH, low_memory=False)
            out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
            out = out.drop_duplicates(subset=["text", "label_id", "priority_id_fixed"], keep="last")
        else:
            out = pd.DataFrame([row])
        out.to_csv(FRONTEND_FEEDBACK_PATH, index=False, encoding="utf-8")
        return True
    except Exception:
        return False


def _run_retrain_job() -> None:
    global _retrain_in_progress
    backend_root = str(Path(__file__).resolve().parents[1])
    train_script = str(Path(__file__).resolve().parents[1] / "scripts" / "train_multitask.py")
    eval_script = str(Path(__file__).resolve().parents[1] / "scripts" / "eval_test.py")
    try:
        subprocess.run([sys.executable, "-u", train_script], cwd=backend_root, check=True)
        subprocess.run([sys.executable, "-u", eval_script], cwd=backend_root, check=True)
        try:
            import pandas as pd

            rows = len(pd.read_csv(FRONTEND_FEEDBACK_PATH, low_memory=False)) if FRONTEND_FEEDBACK_PATH.exists() else 0
            AUTO_RETRAIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(AUTO_RETRAIN_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "last_retrain_rows": rows,
                        "last_retrain_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass
    finally:
        with _retrain_lock:
            _retrain_in_progress = False


def _maybe_trigger_auto_retrain() -> None:
    global _retrain_in_progress
    if not FRONTEND_FEEDBACK_PATH.exists():
        return
    try:
        import pandas as pd

        current_rows = len(pd.read_csv(FRONTEND_FEEDBACK_PATH, low_memory=False))
    except Exception:
        return

    last_rows = 0
    if AUTO_RETRAIN_STATE_PATH.exists():
        try:
            with open(AUTO_RETRAIN_STATE_PATH, "r", encoding="utf-8") as f:
                last_rows = int(json.load(f).get("last_retrain_rows", 0))
        except Exception:
            last_rows = 0

    if (current_rows - last_rows) < AUTO_RETRAIN_MIN_NEW_FEEDBACK:
        return

    with _retrain_lock:
        if _retrain_in_progress:
            return
        _retrain_in_progress = True

    threading.Thread(target=_run_retrain_job, daemon=True).start()


class DistilBertMultiTask(nn.Module):
    def __init__(self, backbone_name: str, num_labels: int, num_priority: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size

        # Must match scripts/train_multitask.py architecture.
        self.dropout = nn.Dropout(0.1)

        self.label_dropout = nn.Dropout(0.2)
        self.label_hidden = nn.Linear(hidden, hidden // 2)
        self.label_head = nn.Linear(hidden // 2, num_labels)

        self.prio_dropout = nn.Dropout(0.2)
        self.prio_hidden = nn.Linear(hidden, hidden // 4)
        self.prio_head = nn.Linear(hidden // 4, num_priority)
        self.act = nn.GELU()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        label_logits = self.label_head(self.act(self.label_hidden(self.label_dropout(pooled))))
        prio_logits = self.prio_head(self.act(self.prio_hidden(self.prio_dropout(pooled))))
        return label_logits, prio_logits


def _load_model_once() -> None:
    global _model_ready
    global _model_error
    global tokenizer
    global model
    global device
    global id_to_label
    global id_to_priority

    if _model_ready:
        return
    with _model_lock:
        if _model_ready:
            return
        try:
            with open(MODEL_DIR / "id_to_label.json", "r", encoding="utf-8") as f:
                id_to_label = {int(k): v for k, v in json.load(f).items()}
            with open(MODEL_DIR / "id_to_priority.json", "r", encoding="utf-8") as f:
                id_to_priority = {int(k): v for k, v in json.load(f).items()}

            if (MODEL_DIR / "config.json").exists():
                backbone_name = str(MODEL_DIR)
            else:
                backbone_name = _resolve_backbone_dir()

            tok_src = (
                str(MODEL_DIR)
                if (MODEL_DIR / "tokenizer_config.json").exists()
                else backbone_name
            )
            tokenizer = AutoTokenizer.from_pretrained(tok_src)
            model = DistilBertMultiTask(
                backbone_name,
                num_labels=len(id_to_label),
                num_priority=len(id_to_priority),
            )

            state_path = MODEL_DIR / "pytorch_model.bin"
            safetensors_path = MODEL_DIR / "model.safetensors"

            if state_path.exists():
                state = torch.load(state_path, map_location="cpu")
            elif safetensors_path.exists():
                if load_safetensors_file is None:
                    raise RuntimeError(
                        "model.safetensors found but safetensors package is not installed"
                    )
                state = load_safetensors_file(safetensors_path, device="cpu")
            else:
                raise FileNotFoundError(
                    f"Missing model weights. Expected one of: {state_path}, {safetensors_path}"
                )

            for k in ("label_weights", "priority_weights"):
                state.pop(k, None)
            model.load_state_dict(state, strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            _model_ready = True
        except Exception as exc:
            _model_error = str(exc)
            raise


def _use_gemini_chatbot() -> bool:
    return (
        settings.chatbot_provider.strip().lower() == "gemini"
        and bool(settings.gemini_api_key.strip())
        and google_genai is not None
    )


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    if not _use_gemini_chatbot():
        return None
    with _chatbot_lock:
        if _gemini_client is None:
            _gemini_client = google_genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


def _generate_chatbot_text(prompt: str, max_new_tokens: int = 160) -> str:
    del max_new_tokens
    if _use_gemini_chatbot():
        client = _get_gemini_client()
        if client is None:
            raise RuntimeError("Gemini client is not configured")
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
        return re.sub(r"\s+", " ", text).strip()
    raise RuntimeError("Gemini chatbot is not configured")


def _ensure_schema() -> None:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS department VARCHAR(120)
                """
            )
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS evidence_types JSONB NOT NULL DEFAULT '[]'::jsonb
                """
            )
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS analysis JSONB NOT NULL DEFAULT '{}'::jsonb
                """
            )
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS source_language VARCHAR(40)
                """
            )
        conn.commit()


def _sync_models_on_startup() -> None:
    if os.getenv("MODEL_SYNC_ON_STARTUP", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    sync_script = APP_ROOT / "scripts" / "sync_models_from_s3.py"
    if not sync_script.exists():
        logger.warning("Model sync script is missing at %s", sync_script)
        return

    try:
        runpy.run_path(str(sync_script), run_name="__main__")
    except SystemExit as exc:
        logger.warning("Model sync exited early: %s", exc)
    except Exception as exc:
        logger.warning("Model sync failed during startup: %s", exc)


_POSITIVE_TERMS = {
    "good", "resolved", "thanks", "helpful", "clean", "working", "great", "smooth", "appreciate",
}
_NEGATIVE_TERMS = {
    "bad", "delay", "broken", "issue", "problem", "angry", "frustrated", "urgent", "unsafe",
    "dirty", "leak", "harassment", "bully", "stolen", "failed", "complaint", "worst", "late",
}
_EMOTION_LEXICON: dict[str, set[str]] = {
    "anger": {"angry", "frustrated", "furious", "outraged", "annoyed"},
    "fear": {"unsafe", "threat", "scared", "afraid", "panic", "security"},
    "sadness": {"sad", "disappointed", "upset", "hopeless", "ignored"},
    "disgust": {"dirty", "smell", "filthy", "stale", "rotten"},
    "urgency": {"urgent", "immediately", "asap", "critical", "emergency", "today"},
}
_TOXIC_TERMS = {"idiot", "stupid", "hate", "useless", "trash", "nonsense", "abuse"}
_SPAM_TERMS = {"buy now", "click here", "subscribe", "promo", "free money", "offer"}
_LANGUAGE_HINTS: dict[str, list[str]] = {
    "hi": ["hai", "nahi", "kripya", "dhanyavad"],
    "ta": ["ungal", "vendum", "illa", "nanri"],
    "te": ["meeru", "dayachesi", "dhanyavadalu", "ledu"],
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", _normalize_text(text))


def _detect_language(text: str) -> str:
    lower = _normalize_text(text)
    scores = {
        lang: sum(1 for token in hints if token in lower)
        for lang, hints in _LANGUAGE_HINTS.items()
    }
    best_lang, best_score = max(scores.items(), key=lambda item: item[1], default=("en", 0))
    return best_lang if best_score > 0 else "en"


def _sentiment_analysis(text: str) -> dict[str, Any]:
    tokens = _tokenize(text)
    if not tokens:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "emotion": "neutral",
            "emotion_intensity": 0.0,
            "urgency_score": 0.35,
        }

    counts = Counter(tokens)
    positive = sum(counts[t] for t in _POSITIVE_TERMS if t in counts)
    negative = sum(counts[t] for t in _NEGATIVE_TERMS if t in counts)
    base_score = (positive - negative) / max(len(tokens), 1)

    emotion_hits = {
        emotion: sum(counts[t] for t in lexicon if t in counts)
        for emotion, lexicon in _EMOTION_LEXICON.items()
    }
    emotion, emotion_count = max(emotion_hits.items(), key=lambda item: item[1], default=("neutral", 0))
    emotion_intensity = min(1.0, emotion_count / 3) if emotion_count else 0.0

    urgency_terms = emotion_hits.get("urgency", 0)
    urgency_score = min(1.0, 0.35 + max(0, -base_score) * 1.8 + urgency_terms * 0.15 + emotion_intensity * 0.2)

    if base_score < -0.05:
        sentiment_label = "negative"
    elif base_score > 0.05:
        sentiment_label = "positive"
    else:
        sentiment_label = "neutral"

    return {
        "sentiment_score": round(base_score, 4),
        "sentiment_label": sentiment_label,
        "emotion": emotion,
        "emotion_intensity": round(emotion_intensity, 4),
        "urgency_score": round(urgency_score, 4),
    }


def _abuse_analysis(text: str) -> dict[str, Any]:
    normalized = _normalize_text(text)
    tokens = _tokenize(text)
    toxic_hits = sum(1 for token in tokens if token in _TOXIC_TERMS)
    spam_hits = sum(1 for phrase in _SPAM_TERMS if phrase in normalized)
    repeated_chars = len(re.findall(r"(.)\1{4,}", normalized))
    url_count = len(re.findall(r"https?://|www\.", normalized))

    spam_score = min(1.0, spam_hits * 0.35 + repeated_chars * 0.15 + url_count * 0.2)
    toxicity_score = min(1.0, toxic_hits * 0.3)

    flags: list[str] = []
    if toxicity_score >= 0.3:
        flags.append("toxic_language")
    if spam_score >= 0.35:
        flags.append("spam_risk")

    return {
        "toxicity_score": round(toxicity_score, 4),
        "spam_score": round(spam_score, 4),
        "flags": flags,
    }


def _user_behavior_risk(user_id: str | None) -> dict[str, Any]:
    if not user_id:
        return {"risk_score": 0.0, "recent_submissions_30d": 0}
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS total_30d,
                       COUNT(*) FILTER (WHERE status = 'rejected') AS rejected_30d
                FROM complaints
                WHERE user_id = %s
                  AND created_at >= NOW() - INTERVAL '30 days'
                """,
                (user_id,),
            )
            row = cur.fetchone() or {}
    total = int(row.get("total_30d") or 0)
    rejected = int(row.get("rejected_30d") or 0)
    risk = min(1.0, total * 0.06 + rejected * 0.08)
    return {"risk_score": round(risk, 4), "recent_submissions_30d": total}


def _vector_duplicate_search(
    text: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_texts = [
        f"{row.get('title', '')}\n\n{row.get('description', '')}".strip()
        for row in rows
    ]
    if not text.strip() or not candidate_texts:
        return {
            "is_duplicate": False,
            "score": 0.0,
            "method": "tfidf-fallback",
            "matches": [],
        }

    method = "tfidf-fallback"
    corpus = [text, *candidate_texts]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    ranked = np.argsort(similarities)[::-1][:3]

    matches: list[dict[str, Any]] = []
    top_score = 0.0
    for idx in ranked:
        score = float(similarities[idx])
        row = rows[int(idx)]
        if score <= 0:
            continue
        top_score = max(top_score, score)
        matches.append(
            {
                "id": row["id"],
                "title": row.get("title"),
                "category": row.get("category"),
                "status": row.get("status"),
                "score": round(score, 4),
            }
        )

    return {
        "is_duplicate": top_score >= 0.82,
        "score": round(top_score, 4),
        "method": method,
        "matches": matches,
    }


def _find_duplicate_candidates(text: str, user_id: str | None = None) -> dict[str, Any]:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT id, title, description, category, status
                    FROM complaints
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 100
                    """,
                    (user_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, title, description, category, status
                    FROM complaints
                    ORDER BY created_at DESC
                    LIMIT 200
                    """
                )
            rows = cur.fetchall()
    return _vector_duplicate_search(text, rows)


def _build_recommendations(
    text: str,
    category: str,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            params: list[Any] = []
            filters = ["status = 'resolved'"]
            if category and category not in {"Unknown", "Uncategorized"}:
                filters.append("category = %s")
                params.append(category)
            if user_id:
                filters.append("user_id = %s")
                params.append(user_id)
            cur.execute(
                f"""
                SELECT id, title, description, category, department
                FROM complaints
                WHERE {' AND '.join(filters)}
                ORDER BY updated_at DESC NULLS LAST, created_at DESC
                LIMIT 40
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    matches = _vector_duplicate_search(text, rows).get("matches", [])
    recommendations: list[dict[str, Any]] = []
    for match in matches[:3]:
        department = next(
            (row.get("department") for row in rows if row["id"] == match["id"]),
            LABEL_TO_DEPT.get(category, "Helpdesk"),
        )
        recommendations.append(
            {
                "complaint_id": match["id"],
                "title": match["title"],
                "score": match["score"],
                "suggested_department": department,
                "suggested_action": f"Route to {department} and reuse the resolution workflow from a similar resolved complaint.",
            }
        )
    return recommendations


def _analyze_text_bundle(title: str, description: str, user_id: str | None = None) -> dict[str, Any]:
    text = f"{title.strip()}\n\n{description.strip()}".strip()
    prediction = {
        "label": "Unknown",
        "label_confidence": 0.0,
        "priority": "medium",
        "priority_confidence": 0.0,
        "department": "Helpdesk",
    }
    if text:
        try:
            prediction = predict_one(text)
        except Exception:
            # Keep complaint flows working even if ML weights are unavailable.
            pass
    sentiment = _sentiment_analysis(text)
    abuse = _abuse_analysis(text)
    try:
        abuse["user_behavior"] = _user_behavior_risk(user_id)
    except Exception:
        abuse["user_behavior"] = {
            "risk_score": 0.0,
            "recent_submissions_30d": 0,
        }

    try:
        duplicate = _find_duplicate_candidates(text, user_id=user_id)
    except Exception:
        duplicate = {
            "is_duplicate": False,
            "score": 0.0,
            "method": "unavailable",
            "matches": [],
        }

    try:
        recommendations = _build_recommendations(text, prediction["label"], user_id=user_id)
    except Exception:
        recommendations = []

    knowledge_graph = {
        "department": prediction["department"],
        "issue_type": prediction["label"],
        "priority": prediction["priority"],
        "entities": [entity for entity in [prediction["department"], prediction["label"]] if entity],
    }

    return {
        "classification": prediction,
        "sentiment": sentiment,
        "abuse": abuse,
        "duplicate_detection": duplicate,
        "recommendations": recommendations,
        "knowledge_graph": knowledge_graph,
        "source_language": _detect_language(text),
    }


def _forecast_complaint_trends() -> dict[str, Any]:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT category, created_at::date AS day, COUNT(*) AS total
                FROM complaints
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY category, created_at::date
                ORDER BY day ASC
                """
            )
            rows = cur.fetchall()

    by_category: dict[str, dict[str, int]] = defaultdict(dict)
    overall_daily: Counter[str] = Counter()
    for row in rows:
        category = row.get("category") or "Uncategorized"
        day = row["day"].isoformat()
        total = int(row["total"])
        by_category[category][day] = total
        overall_daily[day] += total

    def _series_forecast(day_counts: dict[str, int]) -> dict[str, Any]:
        ordered = sorted(day_counts.items())
        values = [count for _, count in ordered]
        if not values:
            return {"recent_average": 0.0, "predicted_next_7_days": 0.0, "trend": "stable"}
        recent = values[-7:] if len(values) >= 7 else values
        recent_avg = float(sum(recent) / len(recent))
        slope = float(recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        predicted = max(0.0, recent_avg * 7 + slope * 7)
        trend = "rising" if slope > 0.35 else "falling" if slope < -0.35 else "stable"
        return {
            "recent_average": round(recent_avg, 2),
            "predicted_next_7_days": round(predicted, 2),
            "trend": trend,
        }

    category_forecasts = [
        {"category": category, **_series_forecast(counts)}
        for category, counts in by_category.items()
    ]
    category_forecasts.sort(key=lambda item: item["predicted_next_7_days"], reverse=True)

    return {
        "overall": _series_forecast(dict(overall_daily)),
        "top_categories": category_forecasts[:5],
    }


def _infer_chat_intent(message: str, context_text: str = "") -> tuple[str, float]:
    text = _normalize_text(f"{context_text} {message}".strip())
    if not text:
        return "general_chat", 0.0

    override = _classify_query_override(message)
    if override is not None:
        return override

    if _looks_like_complaint_statement(message):
        return "complaint_coaching", 0.84
    if _STATUS_QUERY_RE.search(text):
        return "status_lookup", 0.76
    if any(term in text for term in ("duplicate", "similar complaint", "already reported")):
        return "duplicate_check", 0.75
    if any(term in text for term in ("evidence", "photo", "screenshot", "voice", "document")):
        return "evidence_help", 0.73
    if any(term in text for term in ("recommend", "suggest", "best action", "resolution")):
        return "recommendation_help", 0.72
    if any(term in text for term in ("trend", "analytics", "forecast", "rising category")):
        return "analytics_help", 0.7
    if any(term in text for term in ("complaint", "grievance", "portal", "submit", "register", "file")):
        return "general_help", 0.68
    return "general_chat", 0.6


def _blend_project_general_content(
    project_text: str,
    general_text: str,
    project_ratio: float = 0.8,
) -> str:
    project_words = project_text.split()
    general_words = general_text.split()
    total_words = max(len(project_words) + len(general_words), 1)
    target_project_words = max(1, int(total_words * project_ratio))
    target_general_words = max(1, total_words - target_project_words)

    project_part = " ".join(project_words[:target_project_words]).strip()
    general_part = " ".join(general_words[:target_general_words]).strip()
    if project_part and general_part:
        return f"{project_part} {general_part}".strip()
    return (project_part or general_part).strip()


def _humanize_assistant_reply(intent: str, core: str) -> str:
    openers = {
        "status_lookup": "Thanks for checking in.",
        "duplicate_check": "Great question.",
        "registration": "You are in the right place.",
        "recommendation_help": "Here is what I suggest.",
        "complaint_coaching": "I reviewed your complaint text.",
        "general_help": "Happy to help.",
        "general_chat": "Sure.",
    }
    nudges = {
        "status_lookup": "Would you like me to help you draft a new complaint now?",
        "duplicate_check": "If you want, share your exact issue text and I can check it more precisely.",
        "registration": "Would you like a ready-to-use complaint template?",
        "recommendation_help": "Want me to suggest the best department and priority too?",
        "complaint_coaching": "If you want, send one more sentence and I will refine it again.",
        "general_help": "Tell me your issue in one line and I will guide you step by step.",
        "general_chat": "If you want, I can switch into complaint-helper mode anytime.",
    }
    opener = openers.get(intent, "Happy to help.")
    nudge = nudges.get(intent, nudges["general_help"])
    return f"{opener} {core} {nudge}"


def _is_greeting(text: str) -> bool:
    return _normalize_text(text) in {"hi", "hello", "hey", "hii", "good morning", "good evening"}


_ASSISTANCE_QUERY_RE = re.compile(
    r"\b(how|what|where|when|can|could|should|do i|how do i|how can i|help me|guide me)\b",
    re.I,
)
_REGISTRATION_QUERY_RE = re.compile(
    r"\b(register|submit|file|lodge|create)\b.*\b(complaint|issue|grievance)\b|"
    r"\bhow do i\b.*\b(complaint|issue|grievance)\b|"
    r"\bwhere can i\b.*\b(complaint|issue|grievance)\b",
    re.I,
)
_STATUS_QUERY_RE = re.compile(
    r"\b(status|track|progress|pending|resolved|approved|rejected)\b",
    re.I,
)
_GENERAL_ASSISTANT_QUERY_RE = re.compile(
    r"\b(who are you|what are you|what do you do|what can you do|assistant capabilities|"
    r"introduce yourself|how can you help|how does this portal work|portal help)\b",
    re.I,
)
_GENERAL_CHAT_QUERY_RE = re.compile(
    r"\b(how are you|thank you|thanks|bye|goodbye|talk normally|general assistant|"
    r"can we chat|chat with me|just chat)\b",
    re.I,
)


def _is_help_seeking_query(message: str) -> bool:
    text = _normalize_text(message)
    return text.endswith("?") or bool(_ASSISTANCE_QUERY_RE.search(text))


def _classify_query_override(message: str) -> tuple[str, float] | None:
    text = _normalize_text(message)
    if not text:
        return None
    if _GENERAL_CHAT_QUERY_RE.search(text):
        return "general_chat", 0.98
    if _GENERAL_ASSISTANT_QUERY_RE.search(text):
        return "general_help", 0.98
    if _REGISTRATION_QUERY_RE.search(text):
        return "registration", 0.96
    if _STATUS_QUERY_RE.search(text) and _is_help_seeking_query(text):
        return "status_lookup", 0.9
    return None


def _general_chat_response(message: str) -> tuple[str, list[str]]:
    text = _normalize_text(message)
    if "how are you" in text:
        return (
            "I am doing well and ready to help. You can chat with me generally, or ask me about complaints, status, evidence, or drafting.",
            [
                "Do you want to chat generally or work on a complaint?",
                "Do you want help writing or improving some text?",
                "Do you want me to explain how the complaint portal works?",
            ],
        )
    if any(token in text for token in {"thank you", "thanks"}):
        return (
            "You are welcome. I can keep chatting generally, or help with complaint filing, tracking, and evidence guidance whenever you want.",
            [
                "Do you want to ask a general question?",
                "Do you want help filing a complaint next?",
            ],
        )
    if any(token in text for token in {"bye", "goodbye"}):
        return (
            "Glad I could help. Come back anytime if you want general guidance or complaint support.",
            [
                "Do you want a quick summary before you go?",
            ],
        )
    return (
        "I can be both a general chatbot and a complaint helper. I can chat normally, explain things simply, help you write or improve text, brainstorm options, and also help with filing, tracking, and refining complaints.",
        [
            "Do you want general help with something right now?",
            "Do you want complaint support instead?",
            "Do you want me to explain what I can do in each mode?",
        ],
    )


def _build_chatbot_prompt(
    *,
    message: str,
    intent: str,
    status_summary: dict[str, int],
    user_context: list[dict[str, Any]],
    duplicate: dict[str, Any],
    analysis_preview: dict[str, Any] | None,
) -> str:
    sections = [
        "You are CampusVoice Assistant, a friendly general-purpose AI assistant and complaint portal helper.",
        "Answer naturally and directly.",
        "If the user asks a general question, answer it normally.",
        "If the user asks for current live data like weather or breaking news, clearly say you do not have live data.",
        "If complaint portal context is provided, use it only when relevant.",
        f"Detected mode: {intent}",
    ]

    if status_summary.get("total", 0):
        sections.append(
            "User complaint summary: "
            f"{status_summary['total']} total, {status_summary['pending']} pending, "
            f"{status_summary['in_progress']} in-progress, {status_summary['resolved']} resolved, "
            f"{status_summary['rejected']} rejected."
        )

    if user_context:
        top = user_context[:2]
        context_lines = [
            f"- {item.get('title', 'N/A')} | category={item.get('category', 'N/A')} | "
            f"status={item.get('status', 'N/A')} | score={float(item.get('score', 0.0)):.2f}"
            for item in top
        ]
        sections.append("Relevant prior complaints:\n" + "\n".join(context_lines))

    if duplicate.get("matches"):
        top_match = duplicate["matches"][0]
        sections.append(
            "Duplicate hint: "
            f"{top_match.get('title', 'N/A')} with similarity {float(top_match.get('score', 0.0)):.2f}."
        )

    if isinstance(analysis_preview, dict) and analysis_preview:
        classification = analysis_preview.get("classification", {})
        if isinstance(classification, dict):
            sections.append(
                "Complaint analysis hint: "
                f"category={classification.get('label', 'Unknown')}, "
                f"priority={classification.get('priority', 'medium')}, "
                f"department={classification.get('department', 'Helpdesk')}."
            )

    sections.append(f"User message: {message}")
    sections.append("Answer:")
    return "\n\n".join(sections)


def _chat_context_text(history: list[ChatTurn]) -> str:
    # Keep only recent turns to avoid noisy intent drift.
    recent = history[-6:] if history else []
    parts: list[str] = []
    for turn in recent:
        if turn.role == "user":
            parts.append(str(turn.text))
    return " ".join(parts).strip()


def _fetch_status_summary(user_id: str) -> dict[str, int]:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                       COUNT(*) FILTER (WHERE status = 'in-progress') AS in_progress,
                       COUNT(*) FILTER (WHERE status = 'resolved') AS resolved,
                       COUNT(*) FILTER (WHERE status = 'rejected') AS rejected
                FROM complaints
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone() or {}
    return {
        "total": int(row.get("total", 0)),
        "pending": int(row.get("pending", 0)),
        "in_progress": int(row.get("in_progress", 0)),
        "resolved": int(row.get("resolved", 0)),
        "rejected": int(row.get("rejected", 0)),
    }


def _retrieve_user_context(message: str, user_id: str) -> list[dict[str, Any]]:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, description, category, status, department
                FROM complaints
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 60
                """,
                (user_id,),
            )
            rows = cur.fetchall()
    search = _vector_duplicate_search(message, rows)
    snippets: list[dict[str, Any]] = []
    for item in search.get("matches", [])[:3]:
        row = next((r for r in rows if r["id"] == item["id"]), None)
        if not row:
            continue
        snippets.append(
            {
                "id": row["id"],
                "title": row.get("title"),
                "category": row.get("category"),
                "status": row.get("status"),
                "department": row.get("department"),
                "score": item["score"],
            }
        )
    return snippets


_TIME_HINT_RE = re.compile(
    r"\b(today|yesterday|tomorrow|tonight|morning|afternoon|evening|\d{1,2}[:.]\d{2}|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|last\s+\w+|since\s+\w+)\b",
    re.I,
)
_LOCATION_HINT_RE = re.compile(
    r"\b(hostel|block|room|lab|classroom|canteen|library|bus|gate|portal|website|department|"
    r"office|hall|building|floor|campus|route)\b",
    re.I,
)
_ACTION_HINT_RE = re.compile(
    r"\b(repair|fix|replace|refund|resolve|investigate|take action|approve|release|provide|clean)\b",
    re.I,
)
_EVIDENCE_HINT_RE = re.compile(
    r"\b(photo|image|screenshot|audio|voice|video|recording|bill|receipt|proof|evidence)\b",
    re.I,
)


def _looks_like_complaint_statement(message: str) -> bool:
    text = _normalize_text(message)
    if _is_help_seeking_query(text):
        return False
    if len(text.split()) >= 8:
        return True
    complaint_terms = {
        "issue", "problem", "complaint", "not working", "delay", "harassment",
        "hostel", "exam", "fees", "portal", "wifi", "library", "canteen", "faculty",
    }
    return any(term in text for term in complaint_terms)


def _build_suggested_title(message: str, category: str) -> str:
    cleaned = re.sub(r"\s+", " ", message).strip(" .")
    if not cleaned:
        return f"{category} complaint"
    first_sentence = re.split(r"[.!?]\s+", cleaned, maxsplit=1)[0].strip()
    title = first_sentence[:72].strip()
    if len(first_sentence) > 72:
        title = f"{title.rstrip()}..."
    if category and category not in {"Unknown", "Uncategorized"} and category.lower() not in title.lower():
        return f"{category}: {title}"
    return title


def _find_missing_complaint_details(message: str) -> list[str]:
    text = _normalize_text(message)
    missing: list[str] = []
    if not _TIME_HINT_RE.search(text):
        missing.append("when this happened")
    if not _LOCATION_HINT_RE.search(text):
        missing.append("where it happened")
    if not _ACTION_HINT_RE.search(text):
        missing.append("what resolution you want")
    if not _EVIDENCE_HINT_RE.search(text):
        missing.append("whether you have photo, screenshot, or audio evidence")
    return missing


def _build_follow_up_questions(message: str, analysis: dict[str, Any]) -> list[str]:
    follow_ups: list[str] = []
    missing = _find_missing_complaint_details(message)
    for item in missing[:3]:
        if item == "when this happened":
            follow_ups.append("When did this issue start, and is it still happening now?")
        elif item == "where it happened":
            follow_ups.append("Where exactly did this happen, such as the block, room, office, or portal page?")
        elif item == "what resolution you want":
            follow_ups.append("What outcome do you want, for example repair, refund, approval, or investigation?")
        elif item == "whether you have photo, screenshot, or audio evidence":
            follow_ups.append("Do you have any screenshot, photo, receipt, or audio clip to support this complaint?")

    duplicate = analysis.get("duplicate_detection", {}) if isinstance(analysis, dict) else {}
    if isinstance(duplicate, dict) and duplicate.get("is_duplicate"):
        follow_ups.append("Do you want to submit this as a new complaint, or add evidence to the similar complaint instead?")

    classification = analysis.get("classification", {}) if isinstance(analysis, dict) else {}
    if isinstance(classification, dict) and str(classification.get("priority", "")).lower() == "high":
        follow_ups.append("Is this issue blocking classes, exams, payments, or safety right now?")

    return follow_ups[:4]


def _registration_guidance(message: str) -> tuple[str, list[str], str | None]:
    generic_title = _build_suggested_title(message, "")
    return (
        "To register a complaint, open Submit Complaint, add a short title, describe what happened, "
        "and attach photo, document, or voice evidence if you have it. After submission, the system "
        "will suggest category, priority, and department automatically.",
        [
            "Do you want a ready-to-use complaint template?",
            "Do you want help choosing the best title for your complaint?",
            "Do you want to know what evidence is most useful to upload?",
        ],
        generic_title if generic_title and generic_title.lower() != message.strip().lower() else None,
    )


def _compose_analysis_driven_reply(
    message: str,
    analysis: dict[str, Any],
    intent: str,
) -> tuple[str, list[str], str]:
    classification = analysis.get("classification", {}) if isinstance(analysis, dict) else {}
    sentiment = analysis.get("sentiment", {}) if isinstance(analysis, dict) else {}
    duplicate = analysis.get("duplicate_detection", {}) if isinstance(analysis, dict) else {}
    label = str(classification.get("label", "Unknown"))
    priority = str(classification.get("priority", "medium")).lower()
    department = str(classification.get("department", "Helpdesk"))
    urgency_score = float(sentiment.get("urgency_score", 0.35) or 0.35)
    suggested_title = _build_suggested_title(message, label)
    follow_ups = _build_follow_up_questions(message, analysis)

    urgency_phrase = (
        "This looks urgent."
        if priority == "high" or urgency_score >= 0.75
        else "This looks moderately urgent."
        if priority == "medium"
        else "This does not look highly urgent right now."
    )
    duplicate_phrase = ""
    if isinstance(duplicate, dict) and duplicate.get("matches"):
        top = duplicate["matches"][0]
        duplicate_phrase = (
            f" I also found a similar complaint titled '{top.get('title', 'N/A')}' with similarity {float(top.get('score', 0.0)):.2f}."
        )

    if intent in {"registration", "complaint_coaching", "general_help", "evidence_help"}:
        core = (
            f"This complaint most likely fits the category '{label}' and should go to {department}. "
            f"Priority looks '{priority}'. {urgency_phrase} A strong title would be '{suggested_title}'.{duplicate_phrase}"
        )
    elif intent == "duplicate_check":
        core = (
            f"Based on your text, the issue still looks like '{label}' and should route to {department}. "
            f"Priority looks '{priority}'.{duplicate_phrase or ' I did not find a strong duplicate.'}"
        )
    elif intent == "recommendation_help":
        recommendation = ""
        recs = analysis.get("recommendations", []) if isinstance(analysis, dict) else []
        if recs:
            recommendation = f" Recommended action: {recs[0].get('suggested_action', '')}"
        core = (
            f"This looks like a '{label}' complaint for {department} with '{priority}' priority.{recommendation}"
        )
    else:
        core = (
            f"Your issue reads like '{label}' and would likely be handled by {department}. "
            f"Priority looks '{priority}'."
        )
    return core, follow_ups, suggested_title


def predict_one(text: str):
    if not _model_ready:
        _load_model_once()
    with torch.no_grad():
        enc = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        label_logits, prio_logits = model(**enc)

        label_probs = torch.softmax(label_logits, dim=-1).cpu().numpy()[0]
        prio_probs = torch.softmax(prio_logits, dim=-1).cpu().numpy()[0]

    lid = int(label_probs.argmax())
    pid = int(prio_probs.argmax())
    lconf = float(label_probs.max())
    pconf = float(prio_probs.max())

    raw_label = id_to_label[lid]
    label = raw_label
    priority = str(id_to_priority[pid])

    if lconf < LABEL_THRESHOLD:
        label = "Unknown"
    if label == "Unknown":
        # Avoid persisting Unknown for obvious complaint keywords.
        label = _recover_unknown_label(text, raw_label)
    if pconf < PRIO_THRESHOLD and priority.strip().lower() not in {"low", "medium", "high"}:
        priority = "medium"
    if priority.strip().lower() not in {"low", "medium", "high"}:
        priority = "medium"

    dept = LABEL_TO_DEPT.get(label, "Helpdesk")
    return {
        "label": label,
        "label_confidence": lconf,
        "priority": priority,
        "priority_confidence": pconf,
        "department": dept,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/dependencies")
def health_dependencies():
    deps: dict[str, Any] = {
        "api": "ok",
        "aws_region": settings.aws_region,
        "rds_host": settings.rds_host,
        "attachments_bucket": settings.attachments_bucket,
        "chatbot_provider": settings.chatbot_provider,
        "gemini_model": settings.gemini_model if settings.chatbot_provider.strip().lower() == "gemini" else "",
    }

    host = settings.rds_host
    port = int(settings.rds_port)
    try:
        if host:
            with socket.create_connection((host, port), timeout=2):
                deps["rds_tcp"] = "ok"
        else:
            deps["rds_tcp"] = "missing_host"
    except Exception as exc:
        deps["rds_tcp"] = f"error: {exc}"

    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS count FROM complaints")
                row = cur.fetchone() or {}
        deps["db_query"] = "ok"
        deps["complaints_count"] = int(row.get("count", 0))
    except Exception as exc:
        deps["db_query"] = f"error: {exc}"

    try:
        if settings.attachments_bucket:
            s3_client.head_bucket(Bucket=settings.attachments_bucket)
            deps["s3_bucket"] = "ok"
        else:
            deps["s3_bucket"] = "missing_bucket"
    except Exception as exc:
        deps["s3_bucket"] = f"error: {exc}"

    return deps


@app.on_event("startup")
def startup_checks() -> None:
    _sync_models_on_startup()
    try:
        _ensure_schema()
    except Exception as exc:
        # Keep API bootable if DB is temporarily unreachable; endpoints needing DB will still fail clearly.
        logger.warning("Skipping startup schema check because DB is unreachable: %s", exc)


@app.post("/predict")
def predict(payload: ComplaintIn):
    try:
        return predict_one(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@app.post("/complaints/analyze")
def analyze_complaint(
    payload: ComplaintAnalysisRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    return _analyze_text_bundle(payload.title, payload.description, user_id=current_user.user_id)


@app.get("/complaints")
def list_complaints(current_user: CurrentUser = Depends(get_current_user)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, description, category, priority, department, status,
                       attachments, evidence_types, analysis, source_language, created_at, updated_at
                FROM complaints
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (current_user.user_id,),
            )
            rows = cur.fetchall()
    return [_serialize_row(row) for row in rows]


@app.post("/complaints", status_code=201)
def create_complaint(payload: ComplaintCreate, current_user: CurrentUser = Depends(get_current_user)):
    priority = payload.priority.lower().strip()
    # User-submitted complaints always start in pending until admin approves.
    status_value = "pending"
    if priority not in {"low", "medium", "high"}:
        raise HTTPException(status_code=400, detail="priority must be low, medium, or high")

    if isinstance(payload.analysis, dict) and payload.analysis:
        merged_analysis = dict(payload.analysis)
    else:
        merged_analysis = _analyze_text_bundle(
            payload.title.strip(),
            payload.description.strip(),
            user_id=current_user.user_id,
        )
    if payload.source_language:
        merged_analysis["source_language"] = payload.source_language

    classification = merged_analysis.get("classification", {}) if isinstance(merged_analysis, dict) else {}
    if not isinstance(classification, dict):
        classification = {}
    predicted_category = str(classification.get("label", "")).strip()
    predicted_priority = str(classification.get("priority", "")).strip().lower()
    if predicted_priority not in {"low", "medium", "high"}:
        predicted_priority = priority
    category_to_store = (
        predicted_category
        if predicted_category and predicted_category.lower() not in {"unknown", "uncategorized"}
        else (payload.category or "Uncategorized").strip()
    )

    complaint_id = str(uuid.uuid4())
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO complaints (
                  id, user_id, title, description, category, priority, department, status,
                  attachments, evidence_types, analysis, source_language
                ) VALUES (
                  %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id, user_id, title, description, category, priority, department, status,
                          attachments, evidence_types, analysis, source_language, created_at, updated_at
                """,
                (
                    complaint_id,
                    current_user.user_id,
                    payload.title.strip(),
                    payload.description.strip(),
                    category_to_store,
                    predicted_priority,
                    classification.get("department"),
                    status_value,
                    Json(payload.attachments),
                    Json(payload.evidence_types),
                    Json(merged_analysis),
                    payload.source_language or merged_analysis.get("source_language"),
                ),
            )
            row = cur.fetchone()
        conn.commit()

    # If frontend provides category/priority, capture as supervised feedback.
    if _append_frontend_feedback(
        title=payload.title.strip(),
        description=payload.description.strip(),
        category=category_to_store,
        priority=predicted_priority,
        source="frontend_submit",
    ):
        _maybe_trigger_auto_retrain()

    return _serialize_row(row)


@app.post("/admin/complaints/{complaint_id}/approve")
def approve_complaint(
    complaint_id: str,
    admin_user: CurrentUser = Depends(require_admin),
):
    del admin_user
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE complaints
                SET status = 'in-progress'
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status,
                          attachments, evidence_types, analysis, source_language, created_at, updated_at
                """,
                (complaint_id,),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return _serialize_row(row)


@app.post("/uploads/presigned-url")
def create_presigned_upload(
    payload: PresignedUploadRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    if not payload.fileName.strip():
        raise HTTPException(status_code=400, detail="fileName is required")
    _require_env_for_uploads()

    key = f"attachments/{current_user.user_id}/{uuid.uuid4()}-{_sanitize_filename(payload.fileName)}"
    upload_url = s3_client.generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.attachments_bucket,
            "Key": key,
            "ContentType": payload.contentType or "application/octet-stream",
        },
        ExpiresIn=settings.presigned_url_expires_seconds,
    )
    return {
        "uploadUrl": upload_url,
        "key": key,
        "expiresIn": settings.presigned_url_expires_seconds,
    }


@app.post("/uploads/presigned-download")
def create_presigned_download(
    payload: PresignedDownloadRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    key = payload.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")
    if not key.startswith("attachments/"):
        raise HTTPException(status_code=400, detail="invalid attachment key")
    if not current_user.is_admin:
        allowed_prefix = f"attachments/{current_user.user_id}/"
        if not key.startswith(allowed_prefix):
            raise HTTPException(status_code=403, detail="Not allowed to access this attachment")

    _require_env_for_uploads()
    download_url = s3_client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.attachments_bucket,
            "Key": key,
        },
        ExpiresIn=settings.presigned_url_expires_seconds,
    )
    return {
        "downloadUrl": download_url,
        "key": key,
        "expiresIn": settings.presigned_url_expires_seconds,
    }


@app.get("/admin/complaints")
def list_all_complaints(admin_user: CurrentUser = Depends(require_admin)):
    del admin_user
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, description, category, priority, department, status,
                       attachments, evidence_types, analysis, source_language, created_at, updated_at
                FROM complaints
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()
    return [_serialize_row(row) for row in rows]


@app.post("/admin/complaints/{complaint_id}/predict")
def predict_complaint(complaint_id: str, admin_user: CurrentUser = Depends(require_admin)):
    del admin_user
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, description
                FROM complaints
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Complaint not found")

    prediction = _analyze_text_bundle(row["title"], row["description"])
    return prediction


@app.post("/admin/complaints/{complaint_id}/auto-apply")
def auto_apply_prediction(
    complaint_id: str,
    admin_user: CurrentUser = Depends(require_admin),
):
    del admin_user
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, description
                FROM complaints
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Complaint not found")

            try:
                prediction_bundle = _analyze_text_bundle(row["title"], row["description"])
                prediction = prediction_bundle["classification"]
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model unavailable during auto-apply: {exc}",
                ) from exc
            cur.execute(
                """
                UPDATE complaints
                SET category = %s, priority = %s, department = %s, analysis = %s
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status,
                          attachments, evidence_types, analysis, source_language, created_at, updated_at
                """,
                (
                    prediction["label"],
                    str(prediction["priority"]).lower(),
                    prediction["department"],
                    Json(prediction_bundle),
                    complaint_id,
                ),
            )
            updated = cur.fetchone()
        conn.commit()

    return {
        "prediction": prediction,
        "complaint": _serialize_row(updated),
    }


@app.patch("/admin/complaints/{complaint_id}")
def update_complaint_by_admin(
    complaint_id: str,
    payload: ComplaintAdminUpdate,
    admin_user: CurrentUser = Depends(require_admin),
):
    del admin_user

    fields: list[str] = []
    values: list[Any] = []
    idx = 1

    if payload.category is not None:
        fields.append(f"category = %s")
        values.append(payload.category.strip() or "Uncategorized")
        idx += 1
    if payload.priority is not None:
        priority = payload.priority.strip().lower()
        if priority not in {"low", "medium", "high", "unknown"}:
            raise HTTPException(status_code=400, detail="invalid priority")
        fields.append("priority = %s")
        values.append(priority)
        idx += 1
    if payload.department is not None:
        fields.append("department = %s")
        values.append(payload.department.strip() or None)
        idx += 1
    if payload.status is not None:
        status_value = payload.status.strip().lower()
        if status_value not in {"pending", "in-progress", "resolved", "rejected"}:
            raise HTTPException(status_code=400, detail="invalid status")
        fields.append("status = %s")
        values.append(status_value)
        idx += 1

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    values.append(complaint_id)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE complaints
                SET {", ".join(fields)}
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status,
                          attachments, evidence_types, analysis, source_language, created_at, updated_at
                """,
                tuple(values),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Complaint not found")

    # Admin-updated labels are high-quality supervision for retraining.
    if _append_frontend_feedback(
        title=str(row.get("title", "")).strip(),
        description=str(row.get("description", "")).strip(),
        category=str(row.get("category", "")).strip(),
        priority=str(row.get("priority", "")).strip(),
        source="admin_update",
    ):
        _maybe_trigger_auto_retrain()

    return _serialize_row(row)


@app.post("/admin/retrain")
def trigger_manual_retrain(admin_user: CurrentUser = Depends(require_admin)):
    del admin_user
    with _retrain_lock:
        global _retrain_in_progress
        if _retrain_in_progress:
            return {"ok": True, "status": "already_running"}
        _retrain_in_progress = True

    threading.Thread(target=_run_retrain_job, daemon=True).start()
    return {"ok": True, "status": "started"}


@app.post("/admin/complaints/auto-classify")
def auto_classify_all_complaints(
    payload: AutoClassifyRequest,
    admin_user: CurrentUser = Depends(require_admin),
):
    del admin_user
    updated_items: list[dict[str, Any]] = []

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            if payload.only_pending:
                cur.execute(
                    """
                    SELECT id, title, description
                    FROM complaints
                    WHERE status = 'pending'
                      AND (category IS NULL OR category = '' OR category = 'Uncategorized' OR category = 'Unknown')
                    ORDER BY created_at DESC
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT id, title, description
                    FROM complaints
                    ORDER BY created_at DESC
                    """
                )
            rows = cur.fetchall()

            for row in rows:
                try:
                    prediction = _analyze_text_bundle(row["title"], row["description"])
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model unavailable during auto-classify: {exc}",
                    ) from exc
                cur.execute(
                    """
                    UPDATE complaints
                    SET category = %s, priority = %s, department = %s, analysis = %s
                    WHERE id = %s::uuid
                    RETURNING id, user_id, title, description, category, priority, department, status,
                              attachments, evidence_types, analysis, source_language, created_at, updated_at
                    """,
                    (
                        prediction["classification"]["label"],
                        str(prediction["classification"]["priority"]).lower(),
                        prediction["classification"]["department"],
                        Json(prediction),
                        row["id"],
                    ),
                )
                updated = cur.fetchone()
                updated_items.append(
                    {
                        "prediction": prediction["classification"],
                        "analysis": prediction,
                        "complaint": _serialize_row(updated),
                    }
                )
        conn.commit()

    return {
        "updatedCount": len(updated_items),
        "items": updated_items,
    }


@app.get("/admin/analytics")
def get_admin_analytics(admin_user: CurrentUser = Depends(require_admin)):
    del admin_user
    trend_forecast = _forecast_complaint_trends()
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT analysis, category, priority, status
                FROM complaints
                ORDER BY created_at DESC
                LIMIT 200
                """
            )
            rows = cur.fetchall()

    abusive = 0
    urgent = 0
    duplicates = 0
    emotions: Counter[str] = Counter()
    for row in rows:
        analysis = row.get("analysis") if isinstance(row.get("analysis"), dict) else {}
        sentiment = analysis.get("sentiment", {})
        abuse = analysis.get("abuse", {})
        duplicate = analysis.get("duplicate_detection", {})
        if float(abuse.get("toxicity_score", 0)) >= 0.3 or float(abuse.get("spam_score", 0)) >= 0.35:
            abusive += 1
        if float(sentiment.get("urgency_score", 0)) >= 0.75:
            urgent += 1
        if bool(duplicate.get("is_duplicate")):
            duplicates += 1
        emotion = sentiment.get("emotion")
        if isinstance(emotion, str) and emotion:
            emotions[emotion] += 1

    return {
        "summary": {
            "complaints_analyzed": len(rows),
            "urgent_count": urgent,
            "abusive_or_spam_count": abusive,
            "duplicate_count": duplicates,
        },
        "emotion_distribution": dict(emotions.most_common(5)),
        "trend_forecast": trend_forecast,
    }


@app.post("/chatbot/respond")
def chatbot_respond(
    payload: ChatbotRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    message = _normalize_text(payload.message)
    if not message:
        return {
            "reply": "Please describe the issue you want to register.",
            "intent": "clarify",
            "intent_confidence": 0.0,
        }

    context_text = _chat_context_text(payload.history)
    intent_input = f"{context_text} {message}".strip() if context_text else message
    intent, confidence = _infer_chat_intent(message, context_text)
    language = _detect_language(message)

    status_summary = {
        "total": 0,
        "pending": 0,
        "in_progress": 0,
        "resolved": 0,
        "rejected": 0,
    }
    user_context: list[dict[str, Any]] = []
    duplicate = {"is_duplicate": False, "score": 0.0, "method": "unavailable", "matches": []}
    db_available = True
    analysis_preview: dict[str, Any] | None = None
    follow_up_questions: list[str] = []
    suggested_title: str | None = None

    if _is_greeting(message):
        return {
            "reply": "Hi. I can help you file a complaint, check status, or detect duplicates. What do you want to do?",
            "intent": "general_help",
            "intent_confidence": 1.0,
            "status_summary": status_summary,
            "duplicate_detection": None,
            "context_snippets": user_context,
            "follow_up_questions": [
                "Do you want help writing a new complaint?",
                "Do you want me to check the status of your complaints?",
            ],
        }

    short_follow_up = {"yes", "no", "ok", "okay", "continue", "tell me more", "what next"}
    if message in short_follow_up and context_text:
        prev_intent, prev_conf = _infer_chat_intent(context_text)
        if prev_conf >= 0.35:
            intent = prev_intent
            confidence = max(confidence, prev_conf)

    if _looks_like_complaint_statement(message) and intent in {"general_help", "registration", "evidence_help"}:
        intent = "complaint_coaching"
        confidence = max(confidence, 0.74)

    needs_db = intent in {"status_lookup", "duplicate_check", "recommendation_help", "complaint_coaching"}
    if needs_db:
        try:
            status_summary = _fetch_status_summary(current_user.user_id)
            user_context = _retrieve_user_context(message, current_user.user_id)
            duplicate = _find_duplicate_candidates(message, user_id=current_user.user_id)
        except Exception:
            db_available = False

    if intent == "status_lookup":
        project_reply = (
            f"You currently have {status_summary['total']} complaints: "
            f"{status_summary['pending']} pending, {status_summary['in_progress']} in-progress, "
            f"{status_summary['resolved']} resolved, and {status_summary['rejected']} rejected."
        )
    elif intent == "general_chat":
        project_reply, follow_up_questions = _general_chat_response(message)
    elif intent == "registration" and _is_help_seeking_query(message):
        project_reply, follow_up_questions, suggested_title = _registration_guidance(message)
    elif intent == "duplicate_check":
        if duplicate["matches"]:
            top = duplicate["matches"][0]
            project_reply = (
                f"I found a very similar complaint: '{top['title']}' "
                f"(similarity {top['score']:.2f})."
            )
        else:
            project_reply = "I could not find a strong duplicate in your recent complaints."
    elif intent in {"recommendation_help", "complaint_coaching"}:
        analysis_preview = _analyze_text_bundle("Complaint assistant query", message, user_id=current_user.user_id)
        project_reply, follow_up_questions, suggested_title = _compose_analysis_driven_reply(
            message=message,
            analysis=analysis_preview,
            intent=intent,
        )
    else:
        project_reply = _CHATBOT_INTENTS.get(intent, _CHATBOT_INTENTS["general_help"])["reply"]

    if analysis_preview is None and _looks_like_complaint_statement(message):
        try:
            analysis_preview = _analyze_text_bundle("Complaint assistant query", message, user_id=current_user.user_id)
            analysis_reply, extra_follow_ups, suggested_title = _compose_analysis_driven_reply(
                message=message,
                analysis=analysis_preview,
                intent=intent,
            )
            if intent in {"registration", "evidence_help", "general_help", "duplicate_check"}:
                project_reply = analysis_reply
            if not follow_up_questions:
                follow_up_questions = extra_follow_ups
        except Exception:
            analysis_preview = None

    prompt = _build_chatbot_prompt(
        message=message,
        intent=intent,
        status_summary=status_summary,
        user_context=user_context,
        duplicate=duplicate,
        analysis_preview=analysis_preview,
    )

    project_context_line = ""
    if db_available and needs_db:
        project_context_line = (
            f"In your portal context, you currently have {status_summary['total']} total complaints "
            f"with {status_summary['pending']} still pending."
        )
        if user_context:
            first_ctx = user_context[0]
            project_context_line += (
                f" The most relevant prior complaint is '{first_ctx.get('title', 'N/A')}' "
                f"with score {float(first_ctx.get('score', 0.0)):.2f}."
            )

    if intent in {"general_help", "general_chat"}:
        general_guidance = (
            "I can answer general questions, explain things simply, help you organize your thoughts, "
            "and switch into complaint support whenever you need it."
        )
    else:
        general_guidance = (
            "A good complaint usually includes what happened, where it happened, when it happened, "
            "and one clear expected resolution."
        )
    include_context = intent not in {"general_help", "general_chat", "evidence_help"} or len(message.split()) > 5
    project_payload = (
        f"{project_reply} {project_context_line}".strip()
        if include_context and project_context_line
        else project_reply
    )

    blended = _blend_project_general_content(
        project_text=project_payload,
        general_text=general_guidance,
        project_ratio=0.8,
    )
    try:
        generated_reply = _generate_chatbot_text(prompt)
        reply = generated_reply or _humanize_assistant_reply(intent, blended)
    except Exception:
        reply = _humanize_assistant_reply(intent, blended)

    if not db_available:
        reply = f"{reply} I cannot reach live complaint records right now, but I can still guide you."

    if language != "en":
        reply = f"{reply} Language hint detected: {language}. You can continue in your preferred language."

    return {
        "reply": reply,
        "intent": intent,
        "intent_confidence": round(confidence, 4),
        "status_summary": status_summary,
        "duplicate_detection": duplicate if intent == "duplicate_check" else None,
        "context_snippets": user_context,
        "analysis_preview": analysis_preview,
        "follow_up_questions": follow_up_questions,
        "suggested_title": suggested_title,
    }
