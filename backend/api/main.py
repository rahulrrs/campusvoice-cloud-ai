import csv
import io
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
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches
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
from fastapi.responses import PlainTextResponse
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
    super_admin_emails: str = Field(default="")
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
        super_admin_emails=os.getenv("SUPER_ADMIN_EMAILS", ""),
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
    role: str = "user"
    is_admin: bool = False
    is_super_admin: bool = False


_STATUS_INPUT_ALIASES = {
    "submitted": "submitted",
    "pending": "pending",
    "in_progress": "in_progress",
    "in-progress": "in_progress",
    "resolved": "resolved",
    "rejected": "rejected",
}
_STATUS_API_LABELS = {
    "submitted": "submitted",
    "pending": "pending",
    "in_progress": "in-progress",
    "resolved": "resolved",
    "rejected": "rejected",
}
_FAQ_ITEMS = [
    {
        "id": "faq-1",
        "question": "Can I submit a complaint anonymously?",
        "answer": "Yes. Anonymous submission is on by default. You can choose to reveal your identity before submitting if you want follow-up tied to your name.",
    },
    {
        "id": "faq-2",
        "question": "How do I track my complaint?",
        "answer": "Open your dashboard and select a complaint to view its status and timeline from submission to resolution.",
    },
    {
        "id": "faq-3",
        "question": "What statuses can a complaint have?",
        "answer": "Complaints move through submitted, pending, in progress, and resolved. Older records may still show rejected for backward compatibility.",
    },
    {
        "id": "faq-4",
        "question": "Can I upload evidence?",
        "answer": "Yes. You can attach images, documents, and audio files. Voice notes recorded in the browser are also supported.",
    },
    {
        "id": "faq-5",
        "question": "How long does it take for a complaint to be reviewed?",
        "answer": "Review time depends on category and urgency. High-priority complaints are surfaced faster, and you can monitor each stage from the dashboard timeline.",
    },
    {
        "id": "faq-6",
        "question": "Can I reopen a resolved complaint?",
        "answer": "Yes. If a resolved issue still persists, open the complaint detail page and submit a reopen reason so the team can review it again.",
    },
    {
        "id": "faq-7",
        "question": "Who can see my complaint details?",
        "answer": "Only the relevant admins and support staff handling the issue should see the complaint details. Anonymous submissions hide your identity from regular workflow views.",
    },
    {
        "id": "faq-8",
        "question": "Why does my complaint show pending sync?",
        "answer": "Pending sync appears when you submit while offline or while the network is unstable. The app stores the complaint safely and syncs it automatically when connectivity returns.",
    },
    {
        "id": "faq-9",
        "question": "Can I get updates after submitting a complaint?",
        "answer": "Yes. Use the Notifications page and the complaint conversation panel to follow replies, status changes, assignment progress, and resolution updates.",
    },
    {
        "id": "faq-10",
        "question": "What kind of evidence is most useful?",
        "answer": "Clear descriptions, exact time and location, and any supporting screenshots, photos, documents, or audio notes help the system and admins review complaints faster.",
    },
]


def _get_admin_email_set() -> set[str]:
    return {
        item.strip().lower()
        for item in settings.admin_emails.split(",")
        if item.strip()
    }


def _get_super_admin_email_set() -> set[str]:
    return {
        item.strip().lower()
        for item in settings.super_admin_emails.split(",")
        if item.strip()
    }


def _ensure_admin_access_tables(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_users (
            id UUID PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            role VARCHAR(20) NOT NULL DEFAULT 'admin',
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            invite_token TEXT,
            invite_expires_at TIMESTAMPTZ,
            invited_by TEXT,
            accepted_by_user_id TEXT,
            accepted_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_admin_users_email
        ON admin_users (email)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_admin_users_status_role
        ON admin_users (status, role)
        """
    )


def _normalize_admin_role(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "super_admin":
        return "super_admin"
    if normalized == "admin":
        return "admin"
    return "user"


def _normalize_admin_status(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"pending", "active", "suspended", "revoked"}:
        return normalized
    return "pending"


def _serialize_admin_access_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(row)
    for dt_key in ("invite_expires_at", "accepted_at", "created_at", "updated_at"):
        value = serialized.get(dt_key)
        if isinstance(value, datetime):
            serialized[dt_key] = value.astimezone(timezone.utc).isoformat()
    serialized["role"] = _normalize_admin_role(serialized.get("role"))
    serialized["status"] = _normalize_admin_status(serialized.get("status"))
    serialized["email"] = str(serialized.get("email") or "").strip().lower()
    return serialized


def _get_db_access_record(email: str | None) -> dict[str, Any] | None:
    normalized_email = str(email or "").strip().lower()
    if not normalized_email:
        return None
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                _ensure_admin_access_tables(cur)
                cur.execute(
                    """
                    SELECT id, email, role, status, invite_token, invite_expires_at, invited_by,
                           accepted_by_user_id, accepted_at, created_at, updated_at
                    FROM admin_users
                    WHERE lower(email) = %s
                    """,
                    (normalized_email,),
                )
                row = cur.fetchone()
        return row
    except Exception:
        return None


def _resolve_access_from_sources(email: str | None) -> tuple[str, bool, bool]:
    normalized_email = str(email or "").strip().lower()
    if not normalized_email:
        return ("user", False, False)

    if normalized_email in _get_super_admin_email_set():
        return ("super_admin", True, True)

    db_row = _get_db_access_record(normalized_email)
    if db_row:
        role = _normalize_admin_role(db_row.get("role"))
        status_value = _normalize_admin_status(db_row.get("status"))
        if status_value == "active" and role in {"admin", "super_admin"}:
            return (role, True, role == "super_admin")

    if normalized_email in _get_admin_email_set():
        return ("admin", True, False)
    return ("user", False, False)


def _list_pending_invites_for_email(email: str | None) -> list[dict[str, Any]]:
    normalized_email = str(email or "").strip().lower()
    if not normalized_email:
        return []
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                _ensure_admin_access_tables(cur)
                cur.execute(
                    """
                    SELECT id, email, role, status, invite_token, invite_expires_at, invited_by,
                           accepted_by_user_id, accepted_at, created_at, updated_at
                    FROM admin_users
                    WHERE lower(email) = %s
                      AND status = 'pending'
                    ORDER BY created_at DESC
                    """,
                    (normalized_email,),
                )
                rows = cur.fetchall()
        return [_serialize_admin_access_row(row) for row in rows]
    except Exception:
        return []


class AdminAccessRecord(BaseModel):
    id: str
    email: str
    role: str
    status: str
    invite_token: str | None = None
    invite_expires_at: str | None = None
    invited_by: str | None = None
    accepted_by_user_id: str | None = None
    accepted_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class AccessProfileResponse(BaseModel):
    user_id: str
    email: str | None = None
    role: str
    is_admin: bool
    is_super_admin: bool
    pending_invites: list[AdminAccessRecord] = Field(default_factory=list)


class AdminInviteRequest(BaseModel):
    email: str
    role: str = Field(default="admin")


class AdminInviteAcceptRequest(BaseModel):
    token: str


class AdminAccessUpdateRequest(BaseModel):
    role: str | None = None
    status: str | None = None


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
    role, is_admin, is_super_admin = _resolve_access_from_sources(email_str)
    return CurrentUser(
        user_id=user_id,
        email=email_str,
        role=role,
        is_admin=is_admin,
        is_super_admin=is_super_admin,
    )


def require_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def require_super_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if not current_user.is_super_admin:
        raise HTTPException(status_code=403, detail="Super admin access required")
    return current_user


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(row)
    for dt_key in (
        "created_at",
        "updated_at",
        "submitted_at",
        "pending_at",
        "in_progress_at",
        "resolved_at",
        "last_student_update_at",
        "last_public_admin_update_at",
        "last_user_viewed_updates_at",
        "last_admin_viewed_updates_at",
        "reopened_at",
        "sla_due_at",
    ):
        value = serialized.get(dt_key)
        if isinstance(value, datetime):
            serialized[dt_key] = value.astimezone(timezone.utc).isoformat()
    attachments = serialized.get("attachments")
    if not isinstance(attachments, list):
        serialized["attachments"] = []
    evidence_types = serialized.get("evidence_types")
    if not isinstance(evidence_types, list):
        serialized["evidence_types"] = []
    analysis = serialized.get("analysis")
    if not isinstance(analysis, dict):
        serialized["analysis"] = {}
    decision_reason = serialized.get("decision_reason")
    if not isinstance(decision_reason, dict):
        serialized["decision_reason"] = {}
    fairness_flags = serialized.get("fairness_flags")
    if not isinstance(fairness_flags, list):
        serialized["fairness_flags"] = []
    serialized["status"] = _STATUS_API_LABELS.get(str(serialized.get("status", "")).strip(), serialized.get("status", "pending"))
    serialized["is_anonymous"] = bool(serialized.get("is_anonymous", True))
    serialized["reopen_count"] = int(serialized.get("reopen_count", 0) or 0)
    serialized["resolution_summary"] = str(serialized.get("resolution_summary") or "").strip() or None
    serialized["requires_human_review"] = bool(serialized.get("requires_human_review", False))
    serialized["risk_score"] = float(serialized.get("risk_score", 0) or 0)
    serialized["routing_confidence"] = float(serialized.get("routing_confidence", 0) or 0)
    serialized["decision_state"] = str(serialized.get("decision_state") or "submitted")
    serialized["decision_source"] = str(serialized.get("decision_source") or "system")
    serialized["quarantined_reason"] = str(serialized.get("quarantined_reason") or "").strip() or None
    serialized["auto_route_version"] = str(serialized.get("auto_route_version") or "rules-v1")
    serialized["escalation_level"] = str(serialized.get("escalation_level") or "").strip() or None
    last_admin_update = row.get("last_public_admin_update_at")
    last_user_seen = row.get("last_user_viewed_updates_at")
    last_student_update = row.get("last_student_update_at")
    last_admin_seen = row.get("last_admin_viewed_updates_at")
    serialized["has_unread_updates_for_user"] = bool(
        isinstance(last_admin_update, datetime)
        and (not isinstance(last_user_seen, datetime) or last_admin_update > last_user_seen)
    )
    serialized["has_unread_updates_for_admin"] = bool(
        isinstance(last_student_update, datetime)
        and (not isinstance(last_admin_seen, datetime) or last_student_update > last_admin_seen)
    )
    return serialized


def _serialize_user_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized = _serialize_row(row)
    serialized.pop("analysis", None)
    return serialized


def _serialize_update_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(row)
    value = serialized.get("created_at")
    if isinstance(value, datetime):
        serialized["created_at"] = value.astimezone(timezone.utc).isoformat()
    serialized["is_internal"] = bool(serialized.get("is_internal", False))
    serialized["author_role"] = str(serialized.get("author_role", "")).strip() or "system"
    return serialized


def _serialize_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(row)
    value = serialized.get("created_at")
    if isinstance(value, datetime):
        serialized["created_at"] = value.astimezone(timezone.utc).isoformat()
    for key in ("previous_state", "new_state", "reason"):
        if not isinstance(serialized.get(key), dict):
            serialized[key] = {}
    serialized["actor_type"] = str(serialized.get("actor_type") or "system")
    serialized["event_type"] = str(serialized.get("event_type") or "unknown")
    serialized["actor_id"] = str(serialized.get("actor_id") or "").strip() or None
    serialized["model_version"] = str(serialized.get("model_version") or "").strip() or None
    serialized["rule_version"] = str(serialized.get("rule_version") or "").strip() or None
    return serialized


def _notification_item(
    *,
    complaint_id: str,
    title: str,
    category: str,
    status_value: str,
    timestamp: datetime | None,
    group_key: str,
    group_label: str,
    department: str | None = None,
    priority: str | None = None,
    preview: str | None = None,
) -> dict[str, Any]:
    return {
        "complaint_id": complaint_id,
        "title": title,
        "category": category,
        "status": _STATUS_API_LABELS.get(status_value, status_value),
        "timestamp": timestamp.astimezone(timezone.utc).isoformat() if isinstance(timestamp, datetime) else None,
        "group_key": group_key,
        "group_label": group_label,
        "department": department,
        "priority": priority,
        "preview": preview,
    }


def _build_admin_complaint_filters(
    *,
    status: str | None = None,
    department: str | None = None,
    assigned_to: str | None = None,
    review_state: str | None = None,
    q: str | None = None,
) -> tuple[list[str], list[Any]]:
    filters: list[str] = []
    params: list[Any] = []

    if status:
        normalized_status = _normalize_status_input(status)
        if normalized_status == "pending":
            filters.append("status IN ('submitted', 'pending')")
        else:
            filters.append("status = %s")
            params.append(normalized_status)

    if department and department.strip().lower() != "all":
        filters.append("department = %s")
        params.append(department.strip())

    if assigned_to and assigned_to.strip().lower() != "all":
        filters.append("assigned_to = %s")
        params.append(assigned_to.strip())

    if q and q.strip():
        filters.append("(title ILIKE %s OR description ILIKE %s OR category ILIKE %s)")
        like = f"%{q.strip()}%"
        params.extend([like, like, like])

    review_state_norm = str(review_state or "").strip().lower()
    attention_sql = """
        (
          COALESCE((analysis->'abuse'->>'toxicity_score')::float, 0) >= 0.35
          OR COALESCE((analysis->'abuse'->>'spam_score')::float, 0) >= 0.35
          OR COALESCE((analysis->'duplicate_detection'->>'is_duplicate')::boolean, false)
          OR jsonb_array_length(COALESCE(analysis->'submission_guard'->'warnings', '[]'::jsonb)) > 0
        )
    """
    if review_state_norm == "needs_attention":
        filters.append(attention_sql)
    elif review_state_norm == "human_review":
        filters.append("COALESCE(requires_human_review, FALSE)")
    elif review_state_norm == "escalated":
        filters.append("decision_state = 'escalated'")
    elif review_state_norm == "quarantined":
        filters.append("decision_state = 'quarantined'")
    elif review_state_norm == "duplicates":
        filters.append("COALESCE((analysis->'duplicate_detection'->>'is_duplicate')::boolean, false)")
    elif review_state_norm == "clean":
        filters.append(f"NOT {attention_sql}")

    return filters, params


def _complaint_report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "title": str(row.get("title") or ""),
        "category": str(row.get("category") or ""),
        "priority": str(row.get("priority") or ""),
        "department": str(row.get("department") or ""),
        "assigned_to": str(row.get("assigned_to") or ""),
        "status": _STATUS_API_LABELS.get(str(row.get("status") or ""), str(row.get("status") or "")),
        "is_anonymous": bool(row.get("is_anonymous")),
        "reopen_count": int(row.get("reopen_count") or 0),
        "submitted_at": row.get("submitted_at").isoformat() if row.get("submitted_at") else "",
        "pending_at": row.get("pending_at").isoformat() if row.get("pending_at") else "",
        "in_progress_at": row.get("in_progress_at").isoformat() if row.get("in_progress_at") else "",
        "resolved_at": row.get("resolved_at").isoformat() if row.get("resolved_at") else "",
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else "",
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else "",
    }


def _normalize_status_input(raw_status: str | None) -> str:
    normalized = str(raw_status or "").strip().lower()
    mapped = _STATUS_INPUT_ALIASES.get(normalized)
    if mapped is None:
        raise HTTPException(status_code=400, detail="invalid status")
    return mapped


def _status_timestamp_updates(status_value: str) -> tuple[list[str], list[Any]]:
    now = datetime.now(timezone.utc)
    updates: list[str] = []
    values: list[Any] = []
    if status_value == "submitted":
        updates.append("submitted_at = COALESCE(submitted_at, %s)")
        values.append(now)
    elif status_value == "pending":
        updates.append("pending_at = COALESCE(pending_at, %s)")
        values.append(now)
    elif status_value == "in_progress":
        updates.append("in_progress_at = COALESCE(in_progress_at, %s)")
        values.append(now)
    elif status_value == "resolved":
        updates.append("resolved_at = COALESCE(resolved_at, %s)")
        values.append(now)
    return updates, values


def _sanitize_filename(name: str) -> str:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", "."))
    return safe or "attachment"


def _category_alias_map() -> dict[str, list[str]]:
    categories = _dataset_category_names()
    if not categories:
        categories = [item for item in LABEL_TO_DEPT.keys() if item not in {"Unknown", "Other"}]

    aliases: dict[str, list[str]] = {category: [category.lower()] for category in categories}
    aliases.setdefault("Ragging / Harassment", []).extend(["harassment", "harrassment", "harrssment", "ragging", "bullying"])
    aliases.setdefault("Safety & Security", []).extend(["security", "safety", "unsafe", "theft", "stolen"])
    aliases.setdefault("IT & Digital Services", []).extend(["wifi", "wi fi", "internet", "portal", "website", "login", "server"])
    aliases.setdefault("Infrastructure", []).extend(["infrastructure", "maintenance", "water leak", "electricity", "washroom"])
    aliases.setdefault("Hostel", []).extend(["hostel", "dorm", "warden", "room"])
    aliases.setdefault("Fees", []).extend(["fees", "fee", "payment", "refund"])
    aliases.setdefault("Examination", []).extend(["exam", "examination", "result", "hall ticket"])
    aliases.setdefault("Transportation", []).extend(["transport", "transportation", "bus", "shuttle", "driver"])
    aliases.setdefault("Placement & Career Services", []).extend(["placement", "placements", "internship", "career"])
    aliases.setdefault("Certificate & Records", []).extend(["certificate", "records", "transcript", "bonafide"])
    return {category: _dedupe_keep_order(values) for category, values in aliases.items()}


def _recover_fuzzy_category_label(text: str, default_label: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return default_label

    alias_map = _category_alias_map()
    for category, aliases in alias_map.items():
        if any(alias in normalized for alias in aliases):
            return category

    vocabulary: dict[str, str] = {}
    for category, aliases in alias_map.items():
        for alias in aliases:
            vocabulary[alias] = category

    tokens = [token for token in _tokenize(text) if len(token) >= 4]
    for token in tokens:
        match = get_close_matches(token, vocabulary.keys(), n=1, cutoff=0.82)
        if match:
            return vocabulary[match[0]]

    return default_label


def _attachment_metadata_analysis(
    attachment_keys: list[str],
    evidence_types: list[str],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    image_count = 0

    for key in attachment_keys:
        file_name = key.split("/")[-1]
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        is_image = ext in {"jpg", "jpeg", "png", "webp", "gif", "bmp", "heic"}
        if is_image:
            image_count += 1
        entry_warnings: list[str] = []
        if is_image and re.fullmatch(r"(img|image|photo|scan)[-_]?\d*", file_name.rsplit(".", 1)[0].lower()):
            entry_warnings.append("Generic image filename; context may need manual verification.")
        entries.append(
            {
                "file_name": file_name,
                "extension": ext or "unknown",
                "kind": "image" if is_image else "file",
                "warnings": entry_warnings,
            }
        )

    if image_count:
        warnings.append("Image verification uses basic metadata checks only; manual review may still be needed.")
    if "image" in evidence_types and image_count == 0:
        warnings.append("Image evidence was declared but no image attachment key was found.")

    return {
        "attachments": entries,
        "image_count": image_count,
        "warnings": warnings,
    }


def _submission_guard(analysis: dict[str, Any]) -> dict[str, Any]:
    abuse = analysis.get("abuse", {}) if isinstance(analysis, dict) else {}
    duplicate = analysis.get("duplicate_detection", {}) if isinstance(analysis, dict) else {}
    user_behavior = abuse.get("user_behavior", {}) if isinstance(abuse, dict) else {}
    toxicity = float(abuse.get("toxicity_score", 0.0) or 0.0)
    spam = float(abuse.get("spam_score", 0.0) or 0.0)
    behavior_risk = float(user_behavior.get("risk_score", 0.0) or 0.0)

    warnings: list[str] = []
    reasons: list[str] = []

    if toxicity >= 0.35:
        warnings.append("Please rewrite the complaint in respectful language so the team can review it quickly.")
    if spam >= 0.35:
        warnings.append("The complaint text looks repetitive or promotional. Please keep it factual and specific.")
    if bool(duplicate.get("is_duplicate")) and float(duplicate.get("score", 0.0) or 0.0) >= 0.94:
        warnings.append("This looks very similar to an earlier complaint. Consider updating the existing complaint instead.")

    if toxicity >= 0.75:
        reasons.append("The complaint could not be submitted because it contains strongly abusive language.")
    if spam >= 0.72 or (spam >= 0.55 and behavior_risk >= 0.45):
        reasons.append("The complaint could not be submitted because it looks like spam or repeated non-genuine content.")

    return {
        "allow_submission": not reasons,
        "warnings": warnings,
        "reasons": reasons,
    }


def _clamp_score(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _calculate_sla_due_at(priority: str, category: str, risk_score: float) -> datetime:
    normalized_priority = priority.strip().lower()
    hours = 72
    if normalized_priority == "high":
        hours = 12
    elif normalized_priority == "medium":
        hours = 36

    if category in {"Ragging / Harassment", "Safety & Security"}:
        hours = min(hours, 6)
    elif risk_score >= 0.8:
        hours = min(hours, 8)
    elif risk_score >= 0.65:
        hours = min(hours, 18)

    return datetime.now(timezone.utc) + timedelta(hours=hours)


def _detect_fairness_flags(
    *,
    category: str,
    is_anonymous: bool,
    abuse_score: float,
    spam_score: float,
    urgency_score: float,
    duplicate_score: float,
) -> list[str]:
    flags: list[str] = []
    sensitive_categories = {"Ragging / Harassment", "Safety & Security", "Health Services"}
    if category in sensitive_categories:
        flags.append("sensitive-category")
    if is_anonymous and category in sensitive_categories:
        flags.append("anonymous-sensitive")
    if spam_score >= 0.35 and category in sensitive_categories:
        flags.append("review-spam-on-sensitive")
    if abuse_score >= 0.35 and urgency_score >= 0.55:
        flags.append("emotionally-charged-urgent")
    if duplicate_score >= 0.9 and urgency_score >= 0.55:
        flags.append("possible-repeat-urgent-issue")
    return flags


def _build_automation_decision(
    *,
    analysis: dict[str, Any],
    fallback_priority: str,
    fallback_category: str,
    is_anonymous: bool,
) -> dict[str, Any]:
    classification = analysis.get("classification", {}) if isinstance(analysis, dict) else {}
    sentiment = analysis.get("sentiment", {}) if isinstance(analysis, dict) else {}
    abuse = analysis.get("abuse", {}) if isinstance(analysis, dict) else {}
    duplicate = analysis.get("duplicate_detection", {}) if isinstance(analysis, dict) else {}

    label = str(classification.get("label") or fallback_category or "Uncategorized").strip()
    if label.lower() in {"unknown", "uncategorized"}:
        label = fallback_category or "Uncategorized"

    priority = str(classification.get("priority") or fallback_priority or "medium").strip().lower()
    if priority not in {"low", "medium", "high"}:
        priority = fallback_priority if fallback_priority in {"low", "medium", "high"} else "medium"

    department = str(classification.get("department") or LABEL_TO_DEPT.get(label, "Helpdesk")).strip() or "Helpdesk"
    label_confidence = _clamp_score(float(classification.get("label_confidence", 0.0) or 0.0))
    priority_confidence = _clamp_score(float(classification.get("priority_confidence", 0.0) or 0.0))
    urgency_score = _clamp_score(float(sentiment.get("urgency_score", 0.0) or 0.0))
    toxicity_score = _clamp_score(float(abuse.get("toxicity_score", 0.0) or 0.0))
    spam_score = _clamp_score(float(abuse.get("spam_score", 0.0) or 0.0))
    duplicate_score = _clamp_score(float(duplicate.get("score", 0.0) or 0.0))
    is_duplicate = bool(duplicate.get("is_duplicate", False))

    routing_confidence = round((label_confidence * 0.65) + (priority_confidence * 0.35), 4)
    risk_score = round(
        _clamp_score(
            (urgency_score * 0.36)
            + (toxicity_score * 0.18)
            + (spam_score * 0.12)
            + (duplicate_score * 0.12)
            + ((1.0 - routing_confidence) * 0.12)
            + (0.10 if label in {"Ragging / Harassment", "Safety & Security"} else 0.0)
        ),
        4,
    )

    fairness_flags = _detect_fairness_flags(
        category=label,
        is_anonymous=is_anonymous,
        abuse_score=toxicity_score,
        spam_score=spam_score,
        urgency_score=urgency_score,
        duplicate_score=duplicate_score,
    )

    requires_human_review = bool(
        routing_confidence < 0.6
        or spam_score >= 0.55
        or toxicity_score >= 0.55
        or "review-spam-on-sensitive" in fairness_flags
    )

    quarantined_reason: str | None = None
    escalation_level: str | None = None
    decision_state = "routed"
    workflow_status = "pending"

    if label in {"Ragging / Harassment", "Safety & Security"} or risk_score >= 0.8:
        escalation_level = "high"
        decision_state = "escalated"
    elif risk_score >= 0.65:
        escalation_level = "medium"
        decision_state = "escalated"

    if spam_score >= 0.72 and routing_confidence < 0.55:
        quarantined_reason = "High spam likelihood with low routing confidence."
        decision_state = "quarantined"
        workflow_status = "submitted"
        requires_human_review = True
    elif requires_human_review:
        decision_state = "in_review"
        workflow_status = "submitted"

    decision_reason = {
        "category": label,
        "priority": priority,
        "department": department,
        "routing_confidence": routing_confidence,
        "risk_score": risk_score,
        "drivers": {
            "urgency_score": urgency_score,
            "toxicity_score": toxicity_score,
            "spam_score": spam_score,
            "duplicate_score": duplicate_score,
            "is_duplicate": is_duplicate,
        },
        "explanation": (
            f"Auto-routed to {department} from category {label} with {priority} priority."
            if decision_state == "routed"
            else f"Flagged for {decision_state.replace('_', ' ')} because confidence or risk signals require supervision."
        ),
    }

    return {
        "category": label,
        "priority": priority,
        "department": department,
        "status": workflow_status,
        "decision_state": decision_state,
        "risk_score": risk_score,
        "routing_confidence": routing_confidence,
        "decision_source": "system",
        "decision_reason": decision_reason,
        "fairness_flags": fairness_flags,
        "requires_human_review": requires_human_review,
        "escalation_level": escalation_level,
        "sla_due_at": _calculate_sla_due_at(priority, label, risk_score),
        "quarantined_reason": quarantined_reason,
        "auto_route_version": "rules-v1",
    }


def _write_complaint_audit_log(
    cur: RealDictCursor,
    *,
    complaint_id: str,
    actor_type: str,
    actor_id: str | None,
    event_type: str,
    previous_state: dict[str, Any] | None = None,
    new_state: dict[str, Any] | None = None,
    reason: dict[str, Any] | None = None,
    model_version: str | None = None,
    rule_version: str | None = None,
) -> None:
    cur.execute(
        """
        INSERT INTO complaint_audit_log (
          id, complaint_id, actor_type, actor_id, event_type,
          previous_state, new_state, reason, model_version, rule_version
        ) VALUES (
          %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """,
        (
            str(uuid.uuid4()),
            complaint_id,
            actor_type,
            actor_id,
            event_type,
            Json(previous_state or {}),
            Json(new_state or {}),
            Json(reason or {}),
            model_version,
            rule_version,
        ),
    )


def _validate_upload_request(
    file_name: str,
    content_type: str,
    file_size: int | None,
) -> dict[str, Any]:
    safe_name = _sanitize_filename(file_name)
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    size = max(int(file_size or 0), 0)
    warnings: list[str] = []

    allowed_image = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "heic"}
    allowed_audio = {"mp3", "wav", "ogg", "m4a", "aac", "webm"}
    allowed_docs = {"pdf", "doc", "docx", "txt"}

    if content_type.startswith("image/"):
        if ext and ext not in allowed_image:
            warnings.append("The image file extension does not fully match the content type.")
        if size and size > 15 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Images larger than 15 MB are not allowed.")
        kind = "image"
    elif content_type.startswith("audio/"):
        if ext and ext not in allowed_audio:
            warnings.append("The audio file extension does not fully match the content type.")
        if size and size > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio files larger than 25 MB are not allowed.")
        kind = "audio"
    else:
        if ext and ext not in allowed_docs:
            raise HTTPException(status_code=400, detail="Only image, audio, PDF, DOC, DOCX, and TXT files are supported.")
        if size and size > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Documents larger than 20 MB are not allowed.")
        kind = "document"

    if kind == "image" and re.fullmatch(r"(img|image|photo|scan)[-_]?\d*", safe_name.rsplit(".", 1)[0].lower()):
        warnings.append("Image verification is limited because the filename is very generic.")

    return {"file_name": safe_name, "kind": kind, "warnings": warnings}


class ComplaintIn(BaseModel):
    text: str


class ComplaintCreate(BaseModel):
    title: str
    description: str
    category: str = "Uncategorized"
    priority: str = "medium"
    status: str = "submitted"
    is_anonymous: bool = True
    attachments: list[str] = Field(default_factory=list)
    evidence_types: list[str] = Field(default_factory=list)
    source_language: str | None = None
    analysis: dict[str, Any] = Field(default_factory=dict)


class FAQItem(BaseModel):
    id: str
    question: str
    answer: str


class PresignedUploadRequest(BaseModel):
    fileName: str
    contentType: str = "application/octet-stream"
    fileSize: int | None = None


class PresignedDownloadRequest(BaseModel):
    key: str


class ComplaintAdminUpdate(BaseModel):
    category: str | None = None
    priority: str | None = None
    department: str | None = None
    status: str | None = None
    decision_state: str | None = None
    assigned_to: str | None = None
    admin_notes: str | None = None
    resolution_summary: str | None = None


class AutoClassifyRequest(BaseModel):
    only_pending: bool = True


class ComplaintAnalysisRequest(BaseModel):
    title: str = ""
    description: str = ""


class ComplaintUpdateCreate(BaseModel):
    body: str


class AdminComplaintUpdateCreate(BaseModel):
    body: str
    is_internal: bool = False


class ComplaintReopenRequest(BaseModel):
    reason: str


class NotificationMarkReadRequest(BaseModel):
    complaint_id: str | None = None
    mark_all: bool = False


class ComplaintAuditLogItem(BaseModel):
    id: str
    complaint_id: str
    actor_type: str
    actor_id: str | None = None
    event_type: str
    previous_state: dict[str, Any] = Field(default_factory=dict)
    new_state: dict[str, Any] = Field(default_factory=dict)
    reason: dict[str, Any] = Field(default_factory=dict)
    model_version: str | None = None
    rule_version: str | None = None
    created_at: str


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
    "category_list": {
        "examples": [
            "list all categories",
            "what categories are available",
            "show complaint categories",
            "list all complaint types",
        ],
        "reply": "I can list all available complaint categories.",
    },
    "priority_policy": {
        "examples": [
            "is there priority categorization",
            "what priority levels are there",
            "how do you know priority",
            "is there low medium high",
        ],
        "reply": "Yes. Complaints are grouped into clear priority levels.",
    },
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


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _dataset_category_names() -> list[str]:
    dataset_candidates = [
        APP_ROOT / "data" / "dataset_clean.csv",
        APP_ROOT / "data" / "dataset.csv",
    ]
    for path in dataset_candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["label"], low_memory=False)
            labels = _dedupe_keep_order(
                sorted({str(v).strip() for v in df["label"].dropna().tolist() if str(v).strip()})
            )
            if labels:
                return labels
        except Exception:
            continue

    categories = _dedupe_keep_order(
        [str(v).strip() for _, v in sorted(id_to_label.items()) if str(v).strip()]
    )
    if categories:
        return categories

    label_map_path = MODEL_DIR / "id_to_label.json"
    if label_map_path.exists():
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                loaded = {int(k): v for k, v in json.load(f).items()}
            categories = _dedupe_keep_order(
                [str(v).strip() for _, v in sorted(loaded.items()) if str(v).strip()]
            )
            if categories:
                return categories
        except Exception:
            pass

    return []


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
                DO $$
                BEGIN
                  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'complaint_status') THEN
                    CREATE TYPE complaint_status AS ENUM ('submitted', 'pending', 'in_progress', 'resolved', 'rejected');
                  END IF;
                END $$;
                """
            )
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
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS assigned_to TEXT,
                ADD COLUMN IF NOT EXISTS admin_notes TEXT
                """
            )
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS last_student_update_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS last_public_admin_update_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS last_user_viewed_updates_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS last_admin_viewed_updates_at TIMESTAMPTZ
                """
            )
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS resolution_summary TEXT,
                ADD COLUMN IF NOT EXISTS reopened_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS reopen_count INTEGER NOT NULL DEFAULT 0
                """
            )
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS decision_state VARCHAR(30) NOT NULL DEFAULT 'submitted',
                ADD COLUMN IF NOT EXISTS risk_score NUMERIC NOT NULL DEFAULT 0,
                ADD COLUMN IF NOT EXISTS routing_confidence NUMERIC NOT NULL DEFAULT 0,
                ADD COLUMN IF NOT EXISTS decision_source TEXT NOT NULL DEFAULT 'system',
                ADD COLUMN IF NOT EXISTS decision_reason JSONB NOT NULL DEFAULT '{}'::jsonb,
                ADD COLUMN IF NOT EXISTS fairness_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
                ADD COLUMN IF NOT EXISTS requires_human_review BOOLEAN NOT NULL DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS escalation_level VARCHAR(20),
                ADD COLUMN IF NOT EXISTS sla_due_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS quarantined_reason TEXT,
                ADD COLUMN IF NOT EXISTS auto_route_version TEXT NOT NULL DEFAULT 'rules-v1'
                """
            )
            cur.execute(
                """
                UPDATE complaints
                SET last_student_update_at = COALESCE(last_student_update_at, submitted_at, created_at)
                WHERE last_student_update_at IS NULL
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS complaint_updates (
                  id UUID PRIMARY KEY,
                  complaint_id UUID NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
                  author_role VARCHAR(20) NOT NULL,
                  author_id TEXT,
                  body TEXT NOT NULL,
                  is_internal BOOLEAN NOT NULL DEFAULT FALSE,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_complaint_updates_complaint_created_at
                ON complaint_updates (complaint_id, created_at ASC)
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS complaint_audit_log (
                  id UUID PRIMARY KEY,
                  complaint_id UUID NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
                  actor_type VARCHAR(20) NOT NULL,
                  actor_id TEXT,
                  event_type VARCHAR(80) NOT NULL,
                  previous_state JSONB NOT NULL DEFAULT '{}'::jsonb,
                  new_state JSONB NOT NULL DEFAULT '{}'::jsonb,
                  reason JSONB NOT NULL DEFAULT '{}'::jsonb,
                  model_version TEXT,
                  rule_version TEXT,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_complaint_audit_log_complaint_created_at
                ON complaint_audit_log (complaint_id, created_at DESC)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_complaints_decision_state_created_at
                ON complaints (decision_state, created_at DESC)
                """
            )
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS is_anonymous BOOLEAN NOT NULL DEFAULT TRUE
                """
            )
            cur.execute(
                """
                ALTER TABLE complaints
                ADD COLUMN IF NOT EXISTS submitted_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS pending_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS in_progress_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ
                """
            )
            cur.execute(
                """
                UPDATE complaints
                SET submitted_at = COALESCE(submitted_at, created_at)
                WHERE submitted_at IS NULL
                """
            )
            cur.execute(
                """
                UPDATE complaints
                SET pending_at = COALESCE(pending_at, created_at)
                WHERE pending_at IS NULL
                  AND status::text IN ('pending')
                """
            )
            cur.execute(
                """
                UPDATE complaints
                SET in_progress_at = COALESCE(in_progress_at, updated_at, created_at)
                WHERE in_progress_at IS NULL
                  AND status::text IN ('in-progress', 'in_progress')
                """
            )
            cur.execute(
                """
                UPDATE complaints
                SET resolved_at = COALESCE(resolved_at, updated_at, created_at)
                WHERE resolved_at IS NULL
                  AND status::text IN ('resolved', 'rejected')
                """
            )
            cur.execute(
                """
                DO $$
                BEGIN
                  IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'complaints'
                      AND column_name = 'status'
                      AND udt_name <> 'complaint_status'
                  ) THEN
                    ALTER TABLE complaints
                      ALTER COLUMN status DROP DEFAULT;

                    ALTER TABLE complaints
                      ALTER COLUMN status TYPE complaint_status
                      USING (
                        CASE
                          WHEN status = 'in-progress' THEN 'in_progress'::complaint_status
                          WHEN status = 'in_progress' THEN 'in_progress'::complaint_status
                          WHEN status = 'resolved' THEN 'resolved'::complaint_status
                          WHEN status = 'submitted' THEN 'submitted'::complaint_status
                          WHEN status = 'rejected' THEN 'rejected'::complaint_status
                          ELSE 'pending'::complaint_status
                        END
                      );
                  END IF;
                END $$;
                """
            )
            cur.execute(
                """
                ALTER TABLE complaints
                ALTER COLUMN status SET DEFAULT 'submitted'
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
    submission_guard = _submission_guard(
        {
            "abuse": abuse,
            "duplicate_detection": duplicate,
        }
    )

    return {
        "classification": prediction,
        "sentiment": sentiment,
        "abuse": abuse,
        "duplicate_detection": duplicate,
        "recommendations": recommendations,
        "knowledge_graph": knowledge_graph,
        "submission_guard": submission_guard,
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

    today = datetime.now(timezone.utc).date()
    last_30_days = [
        (today - timedelta(days=offset)).isoformat()
        for offset in range(29, -1, -1)
    ]

    by_category: dict[str, dict[str, int]] = defaultdict(dict)
    overall_daily: Counter[str] = Counter()
    for row in rows:
        category = row.get("category") or "Uncategorized"
        day = row["day"].isoformat()
        total = int(row["total"])
        by_category[category][day] = total
        overall_daily[day] += total

    def _series_forecast(day_counts: dict[str, int]) -> dict[str, Any]:
        values = [int(day_counts.get(day, 0)) for day in last_30_days]
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
        "status_lookup": "I checked that for you.",
        "duplicate_check": "I looked into that.",
        "registration": "You're in the right place.",
        "recommendation_help": "Here is the best next step.",
        "complaint_coaching": "I can help with that.",
        "general_help": "I'm here with you.",
        "general_chat": "Of course.",
    }
    nudges = {
        "status_lookup": "If you want, I can help you submit the next one too.",
        "duplicate_check": "Send the exact issue text if you want a closer check.",
        "registration": "I can also help you write it in a clear way.",
        "recommendation_help": "I can also suggest the department and priority.",
        "complaint_coaching": "Send one more line and I can make it clearer.",
        "general_help": "Tell me the issue in one line and I will guide you.",
        "general_chat": "If you want, I can switch to complaint help anytime.",
    }
    opener = openers.get(intent, "Happy to help.")
    nudge = nudges.get(intent, nudges["general_help"])
    return f"{opener} {core} {nudge}"


def _clean_assistant_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    cleaned = "\n".join([line for line in lines if line])
    cleaned = re.sub(r"\s*([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
    return cleaned.strip()


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
    if re.search(r"\b(list|show|tell me|what are)\b.*\b(categories|category|complaint types)\b", text):
        return "category_list", 0.99
    if re.search(r"\b(priority|urgent|urgency)\b.*\b(level|categorization|category|categories)\b", text) or (
        "priority" in text and any(token in text for token in {"how", "what", "is there"})
    ):
        return "priority_policy", 0.97
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
            "I'm doing well. I can chat normally, or help with complaints, status, evidence, and drafting.",
            [
                "Do you want normal chat or complaint help?",
                "Do you want help writing something?",
                "Do you want me to explain the portal quickly?",
            ],
        )
    if any(token in text for token in {"thank you", "thanks"}):
        return (
            "You're welcome. I can keep chatting, or help you file, track, or improve a complaint.",
            [
                "Do you want to ask something else?",
                "Do you want help with a complaint next?",
            ],
        )
    if any(token in text for token in {"bye", "goodbye"}):
        return (
            "Glad I could help. Come back anytime if you need support.",
            [
                "Do you want a quick summary before you go?",
            ],
        )
    return (
        "I can chat normally and I can help with complaints too. I can explain things simply, help you write clearly, and guide you through filing or tracking a complaint.",
        [
            "Do you want general help right now?",
            "Do you want complaint help instead?",
            "Do you want a quick overview of what I can do?",
        ],
    )


def _format_category_list() -> str:
    categories = _dataset_category_names()
    if not categories:
        return "I could not load the category list from the dataset right now."
    return "Here are the complaint categories:\n- " + "\n- ".join(categories)


def _priority_policy_response() -> tuple[str, list[str]]:
    return (
        "Yes. We use three priority levels:\n- High: safety risks, harassment, urgent service breakdowns, or issues seriously affecting studies\n- Medium: important issues that need attention soon but are not critical right now\n- Low: routine issues, minor delays, or general service requests",
        [
            "Do you want me to tell you which priority your issue may fall under?",
            "Do you want help writing an urgent complaint clearly?",
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
        "You are CampusVoice Assistant, a warm, emotionally aware assistant for a student complaint portal.",
        "Write in plain, natural English.",
        "Keep the reply short, clear, and supportive.",
        "Do not use markdown, stars, bold markers, or headings.",
        "Use short dash bullets only if the user explicitly asks for a list, categories, options, or steps.",
        "Do not use phrases like 'acceptable complaints cover' or long policy-style explanations unless asked.",
        "Acknowledge the user's feeling when appropriate, but do it briefly and naturally.",
        "Prefer 2 to 5 short sentences.",
        "If the user asks a general question, answer it normally.",
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
                       COUNT(*) FILTER (WHERE status IN ('submitted', 'pending')) AS pending,
                       COUNT(*) FILTER (WHERE status = 'in_progress') AS in_progress,
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
        "Go to Submit Complaint, add a short title, explain what happened, and attach photo, document, or voice evidence if you have it. "
        "After you submit it, the system can suggest the category, priority, and department.",
        [
            "Do you want a simple complaint template?",
            "Do you want help choosing a better title?",
            "Do you want to know what evidence helps most?",
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
        else "This does not seem highly urgent right now."
    )
    duplicate_phrase = ""
    if isinstance(duplicate, dict) and duplicate.get("matches"):
        top = duplicate["matches"][0]
        duplicate_phrase = (
            f" I also found a similar complaint: '{top.get('title', 'N/A')}'."
        )

    if intent in {"registration", "complaint_coaching", "general_help", "evidence_help"}:
        core = (
            f"This issue most likely fits '{label}' and should go to {department}. "
            f"The priority looks '{priority}'. {urgency_phrase} A clear title could be '{suggested_title}'.{duplicate_phrase}"
        )
    elif intent == "duplicate_check":
        core = (
            f"Your issue still looks like '{label}' and should go to {department}. "
            f"The priority looks '{priority}'.{duplicate_phrase or ' I did not find a strong duplicate.'}"
        )
    elif intent == "recommendation_help":
        recommendation = ""
        recs = analysis.get("recommendations", []) if isinstance(analysis, dict) else []
        if recs:
            recommendation = f" Best next step: {recs[0].get('suggested_action', '')}"
        core = (
            f"This looks like a '{label}' issue for {department} with '{priority}' priority.{recommendation}"
        )
    else:
        core = (
            f"Your issue looks like '{label}' and would likely be handled by {department}. "
            f"The priority looks '{priority}'."
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
    if label in {"Unknown", "Other", "Uncategorized"} or lconf < 0.7:
        label = _recover_fuzzy_category_label(text, label)
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


@app.get("/me/access", response_model=AccessProfileResponse)
def get_my_access_profile(current_user: CurrentUser = Depends(get_current_user)):
    pending_rows = _list_pending_invites_for_email(current_user.email)
    return AccessProfileResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        role=current_user.role,
        is_admin=current_user.is_admin,
        is_super_admin=current_user.is_super_admin,
        pending_invites=[AdminAccessRecord(**item) for item in pending_rows],
    )


@app.get("/super-admin/admin-users", response_model=list[AdminAccessRecord])
def list_admin_users(super_admin: CurrentUser = Depends(require_super_admin)):
    del super_admin
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            _ensure_admin_access_tables(cur)
            cur.execute(
                """
                SELECT id, email, role, status, invite_token, invite_expires_at, invited_by,
                       accepted_by_user_id, accepted_at, created_at, updated_at
                FROM admin_users
                ORDER BY
                    CASE role WHEN 'super_admin' THEN 0 ELSE 1 END,
                    CASE status WHEN 'active' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
                    created_at DESC
                """
            )
            rows = cur.fetchall()
    return [AdminAccessRecord(**_serialize_admin_access_row(row)) for row in rows]


@app.post("/super-admin/admin-users/invite", response_model=AdminAccessRecord, status_code=201)
def invite_admin_user(
    payload: AdminInviteRequest,
    super_admin: CurrentUser = Depends(require_super_admin),
):
    normalized_email = str(payload.email or "").strip().lower()
    if not normalized_email:
        raise HTTPException(status_code=400, detail="Email is required")
    role = _normalize_admin_role(payload.role)
    if role not in {"admin", "super_admin"}:
        raise HTTPException(status_code=400, detail="Role must be admin or super_admin")

    invite_token = uuid.uuid4().hex
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            _ensure_admin_access_tables(cur)
            cur.execute(
                """
                INSERT INTO admin_users (
                    id, email, role, status, invite_token, invite_expires_at, invited_by
                )
                VALUES (%s::uuid, %s, %s, 'pending', %s, %s, %s)
                ON CONFLICT (email) DO UPDATE SET
                    role = EXCLUDED.role,
                    status = 'pending',
                    invite_token = EXCLUDED.invite_token,
                    invite_expires_at = EXCLUDED.invite_expires_at,
                    invited_by = EXCLUDED.invited_by,
                    accepted_by_user_id = NULL,
                    accepted_at = NULL
                RETURNING id, email, role, status, invite_token, invite_expires_at, invited_by,
                          accepted_by_user_id, accepted_at, created_at, updated_at
                """,
                (
                    str(uuid.uuid4()),
                    normalized_email,
                    role,
                    invite_token,
                    expires_at,
                    super_admin.email or super_admin.user_id,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return AdminAccessRecord(**_serialize_admin_access_row(row))


@app.post("/admin-access/accept-invite", response_model=AdminAccessRecord)
def accept_admin_invite(
    payload: AdminInviteAcceptRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    token = str(payload.token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Invite token is required")
    if not current_user.email:
        raise HTTPException(status_code=400, detail="Signed-in account is missing an email address")

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            _ensure_admin_access_tables(cur)
            cur.execute(
                """
                SELECT id, email, role, status, invite_token, invite_expires_at, invited_by,
                       accepted_by_user_id, accepted_at, created_at, updated_at
                FROM admin_users
                WHERE invite_token = %s
                """,
                (token,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Invite not found")
            if str(row.get("email") or "").strip().lower() != current_user.email.strip().lower():
                raise HTTPException(
                    status_code=403,
                    detail="This invite is tied to a different email address",
                )
            if _normalize_admin_status(row.get("status")) != "pending":
                raise HTTPException(status_code=400, detail="This invite is no longer pending")
            expires_at = row.get("invite_expires_at")
            if isinstance(expires_at, datetime) and expires_at < datetime.now(timezone.utc):
                raise HTTPException(status_code=400, detail="This invite has expired")

            cur.execute(
                """
                UPDATE admin_users
                SET status = 'active',
                    accepted_by_user_id = %s,
                    accepted_at = NOW(),
                    invite_token = NULL,
                    invite_expires_at = NULL
                WHERE id = %s::uuid
                RETURNING id, email, role, status, invite_token, invite_expires_at, invited_by,
                          accepted_by_user_id, accepted_at, created_at, updated_at
                """,
                (current_user.user_id, row["id"]),
            )
            updated = cur.fetchone()
        conn.commit()
    return AdminAccessRecord(**_serialize_admin_access_row(updated))


@app.patch("/super-admin/admin-users/{access_id}", response_model=AdminAccessRecord)
def update_admin_user_access(
    access_id: str,
    payload: AdminAccessUpdateRequest,
    super_admin: CurrentUser = Depends(require_super_admin),
):
    del super_admin
    next_role = _normalize_admin_role(payload.role) if payload.role is not None else None
    next_status = _normalize_admin_status(payload.status) if payload.status is not None else None
    if next_role is None and next_status is None:
        raise HTTPException(status_code=400, detail="Provide at least one field to update")
    if next_role is not None and next_role not in {"admin", "super_admin"}:
        raise HTTPException(status_code=400, detail="Role must be admin or super_admin")

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            _ensure_admin_access_tables(cur)
            cur.execute(
                """
                SELECT id, email, role, status, invite_token, invite_expires_at, invited_by,
                       accepted_by_user_id, accepted_at, created_at, updated_at
                FROM admin_users
                WHERE id = %s::uuid
                """,
                (access_id,),
            )
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Admin access record not found")

            role_value = next_role or _normalize_admin_role(existing.get("role"))
            status_value = next_status or _normalize_admin_status(existing.get("status"))
            clear_invite = status_value == "active"
            cur.execute(
                """
                UPDATE admin_users
                SET role = %s,
                    status = %s,
                    invite_token = CASE WHEN %s THEN NULL ELSE invite_token END,
                    invite_expires_at = CASE WHEN %s THEN NULL ELSE invite_expires_at END
                WHERE id = %s::uuid
                RETURNING id, email, role, status, invite_token, invite_expires_at, invited_by,
                          accepted_by_user_id, accepted_at, created_at, updated_at
                """,
                (role_value, status_value, clear_invite, clear_invite, access_id),
            )
            updated = cur.fetchone()
        conn.commit()
    return AdminAccessRecord(**_serialize_admin_access_row(updated))


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
def list_complaints(
    status: str | None = None,
    category: str | None = None,
    current_user: CurrentUser = Depends(get_current_user),
):
    filters = ["user_id = %s"]
    params: list[Any] = [current_user.user_id]

    if status:
        normalized_status = _normalize_status_input(status)
        if normalized_status == "pending":
            filters.append("status IN ('submitted', 'pending')")
        else:
            filters.append("status = %s")
            params.append(normalized_status)

    if category and category.strip().lower() != "all":
        filters.append("category = %s")
        params.append(category.strip())

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, user_id, title, description, category, priority, department, status,
                       is_anonymous, attachments, evidence_types, analysis, source_language,
                       decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                       fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version,
                       last_student_update_at, last_public_admin_update_at, last_user_viewed_updates_at, last_admin_viewed_updates_at,
                       resolution_summary, reopened_at, reopen_count,
                       submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
                FROM complaints
                WHERE {" AND ".join(filters)}
                ORDER BY created_at DESC
                """,
                tuple(params),
            )
            rows = cur.fetchall()
    return [_serialize_user_row(row) for row in rows]


@app.get("/complaints/{complaint_id}")
def get_complaint_detail(
    complaint_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, description, category, priority, department, status,
                       is_anonymous, attachments, evidence_types, analysis, source_language,
                       decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                       fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version,
                       last_student_update_at, last_public_admin_update_at, last_user_viewed_updates_at, last_admin_viewed_updates_at,
                       resolution_summary, reopened_at, reopen_count,
                       submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
                FROM complaints
                WHERE id = %s::uuid AND user_id = %s
                """,
                (complaint_id, current_user.user_id),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Complaint not found")

    return _serialize_user_row(row)


@app.get("/complaints/{complaint_id}/updates")
def list_complaint_updates(
    complaint_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM complaints
                WHERE id = %s::uuid AND user_id = %s
                """,
                (complaint_id, current_user.user_id),
            )
            complaint = cur.fetchone()
            if not complaint:
                raise HTTPException(status_code=404, detail="Complaint not found")

            cur.execute(
                """
                UPDATE complaints
                SET last_user_viewed_updates_at = NOW()
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )

            cur.execute(
                """
                SELECT id, complaint_id, author_role, author_id, body, is_internal, created_at
                FROM complaint_updates
                WHERE complaint_id = %s::uuid
                  AND is_internal = FALSE
                ORDER BY created_at ASC
                """,
                (complaint_id,),
            )
            rows = cur.fetchall()
    return [_serialize_update_row(row) for row in rows]


@app.post("/complaints/{complaint_id}/updates", status_code=201)
def create_complaint_update(
    complaint_id: str,
    payload: ComplaintUpdateCreate,
    current_user: CurrentUser = Depends(get_current_user),
):
    body = payload.body.strip()
    if not body:
        raise HTTPException(status_code=400, detail="Update message is required")

    update_id = str(uuid.uuid4())
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM complaints
                WHERE id = %s::uuid AND user_id = %s
                """,
                (complaint_id, current_user.user_id),
            )
            complaint = cur.fetchone()
            if not complaint:
                raise HTTPException(status_code=404, detail="Complaint not found")

            cur.execute(
                """
                INSERT INTO complaint_updates (
                  id, complaint_id, author_role, author_id, body, is_internal
                ) VALUES (
                  %s::uuid, %s::uuid, %s, %s, %s, FALSE
                )
                RETURNING id, complaint_id, author_role, author_id, body, is_internal, created_at
                """,
                (update_id, complaint_id, "student", current_user.user_id, body),
            )
            row = cur.fetchone()
            cur.execute(
                """
                UPDATE complaints
                SET last_student_update_at = NOW()
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )
        conn.commit()
    return _serialize_update_row(row)


@app.post("/complaints/{complaint_id}/reopen")
def reopen_complaint(
    complaint_id: str,
    payload: ComplaintReopenRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    reason = payload.reason.strip()
    if not reason:
        raise HTTPException(status_code=400, detail="Reopen reason is required")

    update_id = str(uuid.uuid4())
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status
                FROM complaints
                WHERE id = %s::uuid AND user_id = %s
                """,
                (complaint_id, current_user.user_id),
            )
            complaint = cur.fetchone()
            if not complaint:
                raise HTTPException(status_code=404, detail="Complaint not found")
            if str(complaint.get("status")) != "resolved":
                raise HTTPException(status_code=400, detail="Only resolved complaints can be reopened")

            cur.execute(
                """
                UPDATE complaints
                SET status = 'pending',
                    decision_state = 'reopened',
                    requires_human_review = FALSE,
                    reopened_at = NOW(),
                    reopen_count = COALESCE(reopen_count, 0) + 1,
                    pending_at = NOW(),
                    last_student_update_at = NOW()
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status,
                          assigned_to, admin_notes, is_anonymous, attachments, evidence_types, analysis, source_language,
                          decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                          fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version,
                          last_student_update_at, last_public_admin_update_at, last_user_viewed_updates_at, last_admin_viewed_updates_at,
                          resolution_summary, reopened_at, reopen_count,
                          submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
                """,
                (complaint_id,),
            )
            row = cur.fetchone()
            _write_complaint_audit_log(
                cur,
                complaint_id=complaint_id,
                actor_type="student",
                actor_id=current_user.user_id,
                event_type="complaint_reopened",
                previous_state={"status": "resolved", "decision_state": "resolved"},
                new_state={"status": "pending", "decision_state": "reopened"},
                reason={"reopen_reason": reason},
            )

            cur.execute(
                """
                INSERT INTO complaint_updates (
                  id, complaint_id, author_role, author_id, body, is_internal
                ) VALUES (
                  %s::uuid, %s::uuid, %s, %s, %s, FALSE
                )
                """,
                (
                    update_id,
                    complaint_id,
                    "student",
                    current_user.user_id,
                    f"Complaint reopened: {reason}",
                ),
            )
        conn.commit()

    return _serialize_user_row(row)


@app.post("/complaints", status_code=201)
def create_complaint(payload: ComplaintCreate, current_user: CurrentUser = Depends(get_current_user)):
    priority = payload.priority.lower().strip()
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
    merged_analysis["attachment_checks"] = _attachment_metadata_analysis(
        payload.attachments,
        payload.evidence_types,
    )
    submission_guard = merged_analysis.get("submission_guard", {})
    if isinstance(submission_guard, dict) and not bool(submission_guard.get("allow_submission", True)):
        reasons = submission_guard.get("reasons", [])
        detail = reasons[0] if isinstance(reasons, list) and reasons else "Complaint could not be submitted."
        raise HTTPException(status_code=400, detail=str(detail))

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
    automation = _build_automation_decision(
        analysis=merged_analysis,
        fallback_priority=predicted_priority,
        fallback_category=category_to_store,
        is_anonymous=bool(payload.is_anonymous),
    )
    status_value = str(automation["status"])

    complaint_id = str(uuid.uuid4())
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO complaints (
                  id, user_id, title, description, category, priority, department, status,
                  is_anonymous, attachments, evidence_types, analysis, source_language,
                  submitted_at, pending_at, last_student_update_at,
                  decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                  fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version
                ) VALUES (
                  %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id, user_id, title, description, category, priority, department, status,
                          is_anonymous, attachments, evidence_types, analysis, source_language,
                          decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                          fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version,
                          last_student_update_at, last_public_admin_update_at, last_user_viewed_updates_at, last_admin_viewed_updates_at,
                          resolution_summary, reopened_at, reopen_count,
                          submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
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
                    bool(payload.is_anonymous),
                    Json(payload.attachments),
                    Json(payload.evidence_types),
                    Json(merged_analysis),
                    payload.source_language or merged_analysis.get("source_language"),
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc) if status_value == "pending" else None,
                    datetime.now(timezone.utc),
                    automation["decision_state"],
                    automation["risk_score"],
                    automation["routing_confidence"],
                    automation["decision_source"],
                    Json(automation["decision_reason"]),
                    Json(automation["fairness_flags"]),
                    automation["requires_human_review"],
                    automation["escalation_level"],
                    automation["sla_due_at"],
                    automation["quarantined_reason"],
                    automation["auto_route_version"],
                ),
            )
            row = cur.fetchone()
            _write_complaint_audit_log(
                cur,
                complaint_id=complaint_id,
                actor_type="system",
                actor_id=current_user.user_id,
                event_type="automation_intake",
                previous_state={},
                new_state={
                    "status": status_value,
                    "decision_state": automation["decision_state"],
                    "department": automation["department"],
                    "priority": automation["priority"],
                },
                reason=automation["decision_reason"],
                model_version=str(settings.backbone_model_name),
                rule_version=str(automation["auto_route_version"]),
            )
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

    return _serialize_user_row(row)


@app.post("/admin/complaints/{complaint_id}/approve")
def approve_complaint(
    complaint_id: str,
    admin_user: CurrentUser = Depends(require_admin),
):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, decision_state
                FROM complaints
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )
            previous_row = cur.fetchone()
            cur.execute(
                """
                UPDATE complaints
                SET status = 'pending',
                    decision_state = CASE
                      WHEN decision_state = 'quarantined' THEN 'in_review'
                      ELSE 'routed'
                    END,
                    requires_human_review = FALSE,
                    pending_at = COALESCE(pending_at, NOW())
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status,
                          assigned_to, admin_notes, is_anonymous, attachments, evidence_types, analysis, source_language,
                          decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                          fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version,
                          submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
                """,
                (complaint_id,),
            )
            row = cur.fetchone()
            if row:
                _write_complaint_audit_log(
                    cur,
                    complaint_id=complaint_id,
                    actor_type="admin",
                    actor_id=admin_user.email or admin_user.user_id,
                    event_type="manual_route_override",
                    previous_state={
                        "status": previous_row.get("status") if previous_row else None,
                        "decision_state": previous_row.get("decision_state") if previous_row else None,
                    },
                    new_state={
                        "status": row.get("status"),
                        "decision_state": row.get("decision_state"),
                    },
                    reason={"message": "Admin moved complaint into active routed queue."},
                    rule_version="rules-v1",
                )
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
    validation = _validate_upload_request(payload.fileName, payload.contentType or "application/octet-stream", payload.fileSize)

    key = f"attachments/{current_user.user_id}/{uuid.uuid4()}-{validation['file_name']}"
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
        "warnings": validation["warnings"],
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
def list_all_complaints(
    status: str | None = None,
    department: str | None = None,
    assigned_to: str | None = None,
    review_state: str | None = None,
    q: str | None = None,
    admin_user: CurrentUser = Depends(require_admin),
):
    del admin_user
    filters, params = _build_admin_complaint_filters(
        status=status,
        department=department,
        assigned_to=assigned_to,
        review_state=review_state,
        q=q,
    )
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, user_id, title, description, category, priority, department, status,
                       assigned_to, admin_notes, is_anonymous, attachments, evidence_types, analysis, source_language,
                       decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                       fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version,
                       last_student_update_at, last_public_admin_update_at, last_user_viewed_updates_at, last_admin_viewed_updates_at,
                       resolution_summary, reopened_at, reopen_count,
                       submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
                FROM complaints
                {where_clause}
                ORDER BY created_at DESC
                """,
                tuple(params),
            )
            rows = cur.fetchall()
    return [_serialize_row(row) for row in rows]


@app.get("/admin/reports/complaints")
def export_complaints_report(
    format: str = "csv",
    status: str | None = None,
    department: str | None = None,
    assigned_to: str | None = None,
    review_state: str | None = None,
    q: str | None = None,
    admin_user: CurrentUser = Depends(require_admin),
):
    del admin_user
    filters, params = _build_admin_complaint_filters(
        status=status,
        department=department,
        assigned_to=assigned_to,
        review_state=review_state,
        q=q,
    )
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, title, category, priority, department, assigned_to, status,
                       is_anonymous, reopen_count,
                       submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
                FROM complaints
                {where_clause}
                ORDER BY created_at DESC
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    report_rows = [_complaint_report_row(row) for row in rows]
    export_format = format.strip().lower()
    if export_format == "json":
        return {"items": report_rows, "count": len(report_rows)}
    if export_format != "csv":
        raise HTTPException(status_code=400, detail="format must be csv or json")

    output = io.StringIO()
    fieldnames = [
        "id",
        "title",
        "category",
        "priority",
        "department",
        "assigned_to",
        "status",
        "is_anonymous",
        "reopen_count",
        "submitted_at",
        "pending_at",
        "in_progress_at",
        "resolved_at",
        "created_at",
        "updated_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(report_rows)
    return PlainTextResponse(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="campusvoice-complaints-report.csv"'},
    )


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


@app.get("/admin/complaints/{complaint_id}/updates")
def list_admin_complaint_updates(
    complaint_id: str,
    admin_user: CurrentUser = Depends(require_admin),
):
    del admin_user
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM complaints
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )
            complaint = cur.fetchone()
            if not complaint:
                raise HTTPException(status_code=404, detail="Complaint not found")

            cur.execute(
                """
                UPDATE complaints
                SET last_admin_viewed_updates_at = NOW()
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )

            cur.execute(
                """
                SELECT id, complaint_id, author_role, author_id, body, is_internal, created_at
                FROM complaint_updates
                WHERE complaint_id = %s::uuid
                ORDER BY created_at ASC
                """,
                (complaint_id,),
            )
            rows = cur.fetchall()
    return [_serialize_update_row(row) for row in rows]


@app.post("/admin/complaints/{complaint_id}/updates", status_code=201)
def create_admin_complaint_update(
    complaint_id: str,
    payload: AdminComplaintUpdateCreate,
    admin_user: CurrentUser = Depends(require_admin),
):
    body = payload.body.strip()
    if not body:
        raise HTTPException(status_code=400, detail="Update message is required")

    update_id = str(uuid.uuid4())
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM complaints
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )
            complaint = cur.fetchone()
            if not complaint:
                raise HTTPException(status_code=404, detail="Complaint not found")

            cur.execute(
                """
                INSERT INTO complaint_updates (
                  id, complaint_id, author_role, author_id, body, is_internal
                ) VALUES (
                  %s::uuid, %s::uuid, %s, %s, %s, %s
                )
                RETURNING id, complaint_id, author_role, author_id, body, is_internal, created_at
                """,
                (
                    update_id,
                    complaint_id,
                    "admin",
                    admin_user.email or admin_user.user_id,
                    body,
                    bool(payload.is_internal),
                ),
            )
            row = cur.fetchone()
            if payload.is_internal:
                cur.execute(
                    """
                    UPDATE complaints
                    SET last_admin_viewed_updates_at = NOW()
                    WHERE id = %s::uuid
                    """,
                    (complaint_id,),
                )
            else:
                cur.execute(
                    """
                    UPDATE complaints
                    SET last_public_admin_update_at = NOW(),
                        last_admin_viewed_updates_at = NOW()
                    WHERE id = %s::uuid
                    """,
                    (complaint_id,),
                )
        conn.commit()
    return _serialize_update_row(row)


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
            automation = _build_automation_decision(
                analysis=prediction_bundle,
                fallback_priority=str(prediction.get("priority") or "medium").lower(),
                fallback_category=str(prediction.get("label") or "Uncategorized"),
                is_anonymous=False,
            )
            cur.execute(
                """
                UPDATE complaints
                SET category = %s, priority = %s, department = %s, analysis = %s,
                    decision_state = %s, risk_score = %s, routing_confidence = %s, decision_source = %s,
                    decision_reason = %s, fairness_flags = %s, requires_human_review = %s,
                    escalation_level = %s, sla_due_at = %s, quarantined_reason = %s, auto_route_version = %s
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status,
                          assigned_to, admin_notes, is_anonymous, attachments, evidence_types, analysis, source_language,
                          decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                          fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version,
                          last_student_update_at, last_public_admin_update_at, last_user_viewed_updates_at, last_admin_viewed_updates_at,
                          submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
                """,
                (
                    prediction["label"],
                    str(prediction["priority"]).lower(),
                    prediction["department"],
                    Json(prediction_bundle),
                    automation["decision_state"],
                    automation["risk_score"],
                    automation["routing_confidence"],
                    automation["decision_source"],
                    Json(automation["decision_reason"]),
                    Json(automation["fairness_flags"]),
                    automation["requires_human_review"],
                    automation["escalation_level"],
                    automation["sla_due_at"],
                    automation["quarantined_reason"],
                    automation["auto_route_version"],
                    complaint_id,
                ),
            )
            updated = cur.fetchone()
            _write_complaint_audit_log(
                cur,
                complaint_id=complaint_id,
                actor_type="system",
                actor_id=admin_user.email or admin_user.user_id,
                event_type="automation_refreshed",
                previous_state={},
                new_state={
                    "category": prediction["label"],
                    "priority": str(prediction["priority"]).lower(),
                    "department": prediction["department"],
                    "decision_state": automation["decision_state"],
                },
                reason=automation["decision_reason"],
                model_version=str(settings.backbone_model_name),
                rule_version=str(automation["auto_route_version"]),
            )
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
    if payload.assigned_to is not None:
        fields.append("assigned_to = %s")
        values.append(payload.assigned_to.strip() or None)
        idx += 1
    if payload.admin_notes is not None:
        fields.append("admin_notes = %s")
        values.append(payload.admin_notes.strip() or None)
        idx += 1
    if payload.resolution_summary is not None:
        fields.append("resolution_summary = %s")
        values.append(payload.resolution_summary.strip() or None)
        idx += 1
    if payload.decision_state is not None:
        decision_state = payload.decision_state.strip().lower().replace("-", "_")
        if decision_state not in {"submitted", "auto_classified", "routed", "in_review", "escalated", "resolved", "reopened", "quarantined"}:
            raise HTTPException(status_code=400, detail="invalid decision_state")
        fields.append("decision_state = %s")
        values.append(decision_state)
        idx += 1
    if payload.status is not None:
        status_value = payload.status.strip().lower()
        status_value = _normalize_status_input(status_value)
        if status_value == "resolved" and not (
            (payload.resolution_summary and payload.resolution_summary.strip())
            or any(field.startswith("resolution_summary =") for field in fields)
        ):
            raise HTTPException(status_code=400, detail="resolution summary is required when resolving a complaint")
        fields.append("status = %s")
        values.append(status_value)
        if payload.decision_state is None:
            inferred_decision_state = {
                "submitted": "submitted",
                "pending": "routed",
                "in_progress": "routed",
                "resolved": "resolved",
                "rejected": "quarantined",
            }[status_value]
            fields.append("decision_state = %s")
            values.append(inferred_decision_state)
        timestamp_fields, timestamp_values = _status_timestamp_updates(status_value)
        fields.extend(timestamp_fields)
        values.extend(timestamp_values)
        idx += 1

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    values.append(complaint_id)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, decision_state, assigned_to, department
                FROM complaints
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )
            previous_row = cur.fetchone()
            cur.execute(
                f"""
                UPDATE complaints
                SET {", ".join(fields)}
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status,
                          assigned_to, admin_notes, is_anonymous, attachments, evidence_types, analysis, source_language,
                          decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                          fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version,
                          last_student_update_at, last_public_admin_update_at, last_user_viewed_updates_at, last_admin_viewed_updates_at,
                          resolution_summary, reopened_at, reopen_count,
                          submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
                """,
                tuple(values),
            )
            row = cur.fetchone()
            if row:
                _write_complaint_audit_log(
                    cur,
                    complaint_id=complaint_id,
                    actor_type="admin",
                    actor_id=admin_user.email or admin_user.user_id,
                    event_type="admin_case_update",
                    previous_state={
                        "status": previous_row.get("status") if previous_row else None,
                        "decision_state": previous_row.get("decision_state") if previous_row else None,
                        "assigned_to": previous_row.get("assigned_to") if previous_row else None,
                        "department": previous_row.get("department") if previous_row else None,
                    },
                    new_state={
                        "status": row.get("status"),
                        "decision_state": row.get("decision_state"),
                        "assigned_to": row.get("assigned_to"),
                        "department": row.get("department"),
                    },
                    reason={
                        "fields_updated": [
                            key
                            for key, value in {
                                "category": payload.category,
                                "priority": payload.priority,
                                "department": payload.department,
                                "status": payload.status,
                                "decision_state": payload.decision_state,
                                "assigned_to": payload.assigned_to,
                                "admin_notes": payload.admin_notes,
                                "resolution_summary": payload.resolution_summary,
                            }.items()
                            if value is not None
                        ]
                    },
                    rule_version="rules-v1",
                )
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


@app.get("/faq", response_model=list[FAQItem])
def list_faq():
    return _FAQ_ITEMS


@app.get("/notifications")
def list_notifications(current_user: CurrentUser = Depends(get_current_user)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            if current_user.is_admin:
                cur.execute(
                    """
                    SELECT id, title, category, status, department, priority,
                           last_student_update_at, last_admin_viewed_updates_at,
                           assigned_to, submitted_at
                    FROM complaints
                    ORDER BY created_at DESC
                    """
                )
                rows = cur.fetchall()

                student_updates = [
                    _notification_item(
                        complaint_id=str(row["id"]),
                        title=str(row.get("title") or "Complaint"),
                        category=str(row.get("category") or "Uncategorized"),
                        status_value=str(row.get("status") or "pending"),
                        timestamp=row.get("last_student_update_at"),
                        group_key="student_updates",
                        group_label="New Student Updates",
                        department=row.get("department"),
                        priority=row.get("priority"),
                        preview="A student added a new update.",
                    )
                    for row in rows
                    if isinstance(row.get("last_student_update_at"), datetime)
                    and (
                        not isinstance(row.get("last_admin_viewed_updates_at"), datetime)
                        or row["last_student_update_at"] > row["last_admin_viewed_updates_at"]
                    )
                ]
                awaiting_assignment = [
                    _notification_item(
                        complaint_id=str(row["id"]),
                        title=str(row.get("title") or "Complaint"),
                        category=str(row.get("category") or "Uncategorized"),
                        status_value=str(row.get("status") or "pending"),
                        timestamp=row.get("submitted_at"),
                        group_key="awaiting_assignment",
                        group_label="Awaiting Assignment",
                        department=row.get("department"),
                        priority=row.get("priority"),
                        preview="This complaint is active and still unassigned.",
                    )
                    for row in rows
                    if str(row.get("status") or "") in {"submitted", "pending", "in_progress"}
                    and not str(row.get("assigned_to") or "").strip()
                    and (
                        not isinstance(row.get("last_admin_viewed_updates_at"), datetime)
                        or (
                            isinstance(row.get("submitted_at"), datetime)
                            and row["submitted_at"] > row["last_admin_viewed_updates_at"]
                        )
                    )
                ]
                urgent_queue = [
                    _notification_item(
                        complaint_id=str(row["id"]),
                        title=str(row.get("title") or "Complaint"),
                        category=str(row.get("category") or "Uncategorized"),
                        status_value=str(row.get("status") or "pending"),
                        timestamp=row.get("submitted_at"),
                        group_key="urgent_queue",
                        group_label="Urgent Queue",
                        department=row.get("department"),
                        priority=row.get("priority"),
                        preview="This complaint is marked high priority and still active.",
                    )
                    for row in rows
                    if str(row.get("priority") or "").lower() == "high"
                    and str(row.get("status") or "") in {"submitted", "pending", "in_progress"}
                    and (
                        not isinstance(row.get("last_admin_viewed_updates_at"), datetime)
                        or (
                            isinstance(row.get("submitted_at"), datetime)
                            and row["submitted_at"] > row["last_admin_viewed_updates_at"]
                        )
                    )
                ]
                groups = [
                    {"key": "student_updates", "label": "New Student Updates", "items": student_updates},
                    {"key": "awaiting_assignment", "label": "Awaiting Assignment", "items": awaiting_assignment},
                    {"key": "urgent_queue", "label": "Urgent Queue", "items": urgent_queue},
                ]
            else:
                cur.execute(
                    """
                    SELECT id, title, category, status, department, priority,
                           last_public_admin_update_at, last_user_viewed_updates_at, resolved_at
                    FROM complaints
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (current_user.user_id,),
                )
                rows = cur.fetchall()
                new_updates = [
                    _notification_item(
                        complaint_id=str(row["id"]),
                        title=str(row.get("title") or "Complaint"),
                        category=str(row.get("category") or "Uncategorized"),
                        status_value=str(row.get("status") or "pending"),
                        timestamp=row.get("last_public_admin_update_at"),
                        group_key="new_updates",
                        group_label="New Updates",
                        department=row.get("department"),
                        priority=row.get("priority"),
                        preview="A new public update is available from the complaint team.",
                    )
                    for row in rows
                    if isinstance(row.get("last_public_admin_update_at"), datetime)
                    and (
                        not isinstance(row.get("last_user_viewed_updates_at"), datetime)
                        or row["last_public_admin_update_at"] > row["last_user_viewed_updates_at"]
                    )
                ]
                resolved_recently = [
                    _notification_item(
                        complaint_id=str(row["id"]),
                        title=str(row.get("title") or "Complaint"),
                        category=str(row.get("category") or "Uncategorized"),
                        status_value=str(row.get("status") or "resolved"),
                        timestamp=row.get("resolved_at"),
                        group_key="resolved_recently",
                        group_label="Recently Resolved",
                        department=row.get("department"),
                        priority=row.get("priority"),
                        preview="This complaint was recently marked resolved.",
                    )
                    for row in rows
                    if str(row.get("status") or "") == "resolved"
                    and isinstance(row.get("resolved_at"), datetime)
                    and (
                        not isinstance(row.get("last_user_viewed_updates_at"), datetime)
                        or row["resolved_at"] > row["last_user_viewed_updates_at"]
                    )
                ]
                groups = [
                    {"key": "new_updates", "label": "New Updates", "items": new_updates},
                    {"key": "resolved_recently", "label": "Recently Resolved", "items": resolved_recently},
                ]

    normalized_groups = [
        {
            "key": group["key"],
            "label": group["label"],
            "count": len(group["items"]),
            "items": group["items"],
        }
        for group in groups
        if group["items"]
    ]
    unique_complaint_ids = {
        str(item.get("complaint_id"))
        for group in normalized_groups
        for item in group["items"]
        if item.get("complaint_id")
    }
    return {
        "total": len(unique_complaint_ids),
        "groups": normalized_groups,
    }


@app.post("/notifications/mark-read")
def mark_notifications_read(
    payload: NotificationMarkReadRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    now = datetime.now(timezone.utc)
    viewed_column = "last_admin_viewed_updates_at" if current_user.is_admin else "last_user_viewed_updates_at"

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            if payload.mark_all:
                if current_user.is_admin:
                    cur.execute(
                        f"""
                        UPDATE complaints
                        SET {viewed_column} = %s
                        WHERE
                          (last_student_update_at IS NOT NULL AND ({viewed_column} IS NULL OR last_student_update_at > {viewed_column}))
                          OR (
                            status IN ('submitted', 'pending', 'in_progress')
                            AND assigned_to IS NULL
                            AND submitted_at IS NOT NULL
                            AND ({viewed_column} IS NULL OR submitted_at > {viewed_column})
                          )
                          OR (
                            priority = 'high'
                            AND status IN ('submitted', 'pending', 'in_progress')
                            AND submitted_at IS NOT NULL
                            AND ({viewed_column} IS NULL OR submitted_at > {viewed_column})
                          )
                        """
                        ,
                        (now,),
                    )
                else:
                    cur.execute(
                        f"""
                        UPDATE complaints
                        SET {viewed_column} = %s
                        WHERE user_id = %s
                          AND (
                            (last_public_admin_update_at IS NOT NULL AND ({viewed_column} IS NULL OR last_public_admin_update_at > {viewed_column}))
                            OR (resolved_at IS NOT NULL AND status = 'resolved' AND ({viewed_column} IS NULL OR resolved_at > {viewed_column}))
                          )
                        """
                        ,
                        (now, current_user.user_id),
                    )
            elif payload.complaint_id:
                if current_user.is_admin:
                    cur.execute(
                        f"""
                        UPDATE complaints
                        SET {viewed_column} = %s
                        WHERE id = %s::uuid
                        """
                        ,
                        (now, payload.complaint_id),
                    )
                else:
                    cur.execute(
                        f"""
                        UPDATE complaints
                        SET {viewed_column} = %s
                        WHERE id = %s::uuid AND user_id = %s
                        """
                        ,
                        (now, payload.complaint_id, current_user.user_id),
                    )
            else:
                raise HTTPException(status_code=400, detail="complaint_id or mark_all is required")
        conn.commit()

    return {"ok": True}


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
                    WHERE status IN ('submitted', 'pending')
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
                automation = _build_automation_decision(
                    analysis=prediction,
                    fallback_priority=str(prediction["classification"]["priority"]).lower(),
                    fallback_category=str(prediction["classification"]["label"]),
                    is_anonymous=False,
                )
                cur.execute(
                    """
                    UPDATE complaints
                    SET category = %s, priority = %s, department = %s, analysis = %s,
                        decision_state = %s, risk_score = %s, routing_confidence = %s, decision_source = %s,
                        decision_reason = %s, fairness_flags = %s, requires_human_review = %s,
                        escalation_level = %s, sla_due_at = %s, quarantined_reason = %s, auto_route_version = %s
                    WHERE id = %s::uuid
                    RETURNING id, user_id, title, description, category, priority, department, status,
                              assigned_to, admin_notes, is_anonymous, attachments, evidence_types, analysis, source_language,
                              decision_state, risk_score, routing_confidence, decision_source, decision_reason,
                              fairness_flags, requires_human_review, escalation_level, sla_due_at, quarantined_reason, auto_route_version,
                              last_student_update_at, last_public_admin_update_at, last_user_viewed_updates_at, last_admin_viewed_updates_at,
                              submitted_at, pending_at, in_progress_at, resolved_at, created_at, updated_at
                    """,
                    (
                        prediction["classification"]["label"],
                        str(prediction["classification"]["priority"]).lower(),
                        prediction["classification"]["department"],
                        Json(prediction),
                        automation["decision_state"],
                        automation["risk_score"],
                        automation["routing_confidence"],
                        automation["decision_source"],
                        Json(automation["decision_reason"]),
                        Json(automation["fairness_flags"]),
                        automation["requires_human_review"],
                        automation["escalation_level"],
                        automation["sla_due_at"],
                        automation["quarantined_reason"],
                        automation["auto_route_version"],
                        row["id"],
                    ),
                )
                updated = cur.fetchone()
                _write_complaint_audit_log(
                    cur,
                    complaint_id=str(row["id"]),
                    actor_type="system",
                    actor_id=admin_user.email or admin_user.user_id,
                    event_type="bulk_automation_refresh",
                    new_state={
                        "category": prediction["classification"]["label"],
                        "priority": str(prediction["classification"]["priority"]).lower(),
                        "department": prediction["classification"]["department"],
                        "decision_state": automation["decision_state"],
                    },
                    reason=automation["decision_reason"],
                    model_version=str(settings.backbone_model_name),
                    rule_version=str(automation["auto_route_version"]),
                )
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


@app.get("/admin/audit-log")
def list_admin_audit_log(
    limit: int = 50,
    admin_user: CurrentUser = Depends(require_admin),
):
    del admin_user
    safe_limit = max(1, min(limit, 200))
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, complaint_id, actor_type, actor_id, event_type,
                       previous_state, new_state, reason, model_version, rule_version, created_at
                FROM complaint_audit_log
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (safe_limit,),
            )
            rows = cur.fetchall()
    return [_serialize_audit_row(row) for row in rows]


@app.get("/admin/complaints/{complaint_id}/audit-log")
def list_complaint_audit_log(
    complaint_id: str,
    admin_user: CurrentUser = Depends(require_admin),
):
    del admin_user
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, complaint_id, actor_type, actor_id, event_type,
                       previous_state, new_state, reason, model_version, rule_version, created_at
                FROM complaint_audit_log
                WHERE complaint_id = %s::uuid
                ORDER BY created_at DESC
                """,
                (complaint_id,),
            )
            rows = cur.fetchall()
    return [_serialize_audit_row(row) for row in rows]


@app.get("/admin/analytics")
def get_admin_analytics(admin_user: CurrentUser = Depends(require_admin)):
    del admin_user
    trend_forecast = _forecast_complaint_trends()
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT analysis, category, priority, status, department, assigned_to,
                       decision_state, risk_score, requires_human_review, fairness_flags,
                       escalation_level, quarantined_reason, sla_due_at
                FROM complaints
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()

    abusive = 0
    urgent = 0
    duplicates = 0
    auto_routed = 0
    human_review = 0
    escalated = 0
    quarantined = 0
    overdue_sla = 0
    fairness_alerts: Counter[str] = Counter()
    total_risk = 0.0
    emotions: Counter[str] = Counter()
    department_workload: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0,
        "active": 0,
        "urgent": 0,
        "unassigned": 0,
    })
    assignee_workload: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0,
        "active": 0,
        "resolved": 0,
    })
    for row in rows:
        analysis = row.get("analysis") if isinstance(row.get("analysis"), dict) else {}
        sentiment = analysis.get("sentiment", {})
        abuse = analysis.get("abuse", {})
        duplicate = analysis.get("duplicate_detection", {})
        status_value = str(row.get("status") or "")
        decision_state = str(row.get("decision_state") or "")
        department = str(row.get("department") or "Unassigned Department")
        assignee = str(row.get("assigned_to") or "Unassigned")
        if float(abuse.get("toxicity_score", 0)) >= 0.3 or float(abuse.get("spam_score", 0)) >= 0.35:
            abusive += 1
        if float(sentiment.get("urgency_score", 0)) >= 0.75:
            urgent += 1
        if bool(duplicate.get("is_duplicate")):
            duplicates += 1
        if decision_state == "routed":
            auto_routed += 1
        if bool(row.get("requires_human_review")):
            human_review += 1
        if decision_state == "escalated":
            escalated += 1
        if decision_state == "quarantined" or row.get("quarantined_reason"):
            quarantined += 1
        if isinstance(row.get("sla_due_at"), datetime) and row["sla_due_at"] < datetime.now(timezone.utc) and status_value not in {"resolved", "rejected"}:
            overdue_sla += 1
        for flag in row.get("fairness_flags") or []:
            fairness_alerts[str(flag)] += 1
        total_risk += float(row.get("risk_score") or 0.0)
        emotion = sentiment.get("emotion")
        if isinstance(emotion, str) and emotion:
            emotions[emotion] += 1

        department_workload[department]["total"] += 1
        if status_value in {"submitted", "pending", "in_progress"}:
            department_workload[department]["active"] += 1
        if float(sentiment.get("urgency_score", 0)) >= 0.75:
            department_workload[department]["urgent"] += 1
        if assignee == "Unassigned":
            department_workload[department]["unassigned"] += 1

        assignee_workload[assignee]["total"] += 1
        if status_value in {"submitted", "pending", "in_progress"}:
            assignee_workload[assignee]["active"] += 1
        if status_value == "resolved":
            assignee_workload[assignee]["resolved"] += 1

    return {
        "summary": {
            "complaints_analyzed": len(rows),
            "urgent_count": urgent,
            "abusive_or_spam_count": abusive,
            "duplicate_count": duplicates,
            "auto_routed_count": auto_routed,
            "human_review_count": human_review,
            "escalated_count": escalated,
            "quarantined_count": quarantined,
            "overdue_sla_count": overdue_sla,
            "average_risk_score": round(total_risk / len(rows), 4) if rows else 0.0,
        },
        "workload": {
            "departments": sorted(
                [{"department": key, **value} for key, value in department_workload.items()],
                key=lambda item: (item["active"], item["urgent"], item["total"]),
                reverse=True,
            )[:8],
            "assignees": sorted(
                [{"assignee": key, **value} for key, value in assignee_workload.items()],
                key=lambda item: (item["active"], item["total"]),
                reverse=True,
            )[:8],
        },
        "emotion_distribution": dict(emotions.most_common(5)),
        "fairness_summary": {
            "alert_count": sum(fairness_alerts.values()),
            "top_flags": [
                {"flag": flag, "count": count}
                for flag, count in fairness_alerts.most_common(6)
            ],
        },
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
    elif intent == "category_list":
        project_reply = _format_category_list()
        follow_up_questions = [
            "Do you want help choosing the right category for your issue?",
            "Do you want me to turn your issue into a complaint draft?",
        ]
    elif intent == "priority_policy":
        project_reply, follow_up_questions = _priority_policy_response()
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

    if intent in {"category_list", "priority_policy"}:
        reply = _clean_assistant_text(project_payload)
        follow_up_questions = [_clean_assistant_text(question) for question in follow_up_questions if question]
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

    reply = _clean_assistant_text(reply)
    follow_up_questions = [_clean_assistant_text(question) for question in follow_up_questions if question]

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
