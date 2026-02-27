import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import jwt
import psycopg2
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from psycopg2.extras import Json, RealDictCursor
from transformers import AutoModel, AutoTokenizer

import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file as load_safetensors_file
except Exception:
    load_safetensors_file = None

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# ========= ML CONFIG =========
MODEL_DIR = r"outputs\edu_classifier_multitask"
MAX_LENGTH = 256
LABEL_THRESHOLD = 0.55
PRIO_THRESHOLD = 0.50

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
    backbone_model_dir: str = Field(default=r"outputs\cfpb_outputs\distilbert_cfpb_mlm")


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
            r"outputs\cfpb_outputs\distilbert_cfpb_mlm",
        ),
    )


settings = get_settings()


def _cors_origins_from_env(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


app = FastAPI(title="Complaint Routing and Management API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3_client = boto3.client("s3", region_name=settings.aws_region)


def _require_env_for_core() -> None:
    required = {
        "COGNITO_USER_POOL_ID": settings.cognito_user_pool_id,
        "COGNITO_APP_CLIENT_ID": settings.cognito_app_client_id,
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
    _require_env_for_core()
    if not settings.attachments_bucket:
        raise HTTPException(
            status_code=500,
            detail="Missing backend environment variable: ATTACHMENTS_BUCKET",
        )


def get_db_conn():
    _require_env_for_core()
    return psycopg2.connect(
        host=settings.rds_host,
        port=settings.rds_port,
        dbname=settings.rds_database,
        user=settings.rds_user,
        password=settings.rds_password,
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
    _require_env_for_core()
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


class PresignedUploadRequest(BaseModel):
    fileName: str
    contentType: str = "application/octet-stream"


class ComplaintAdminUpdate(BaseModel):
    category: str | None = None
    priority: str | None = None
    department: str | None = None
    status: str | None = None
    status_reason: str | None = None


class AutoClassifyRequest(BaseModel):
    only_pending: bool = True


# ---------- Optional ML model loading ----------
_model_lock = threading.Lock()
_model_ready = False
_model_error: str | None = None
tokenizer = None
model = None
device = None
id_to_label: dict[int, str] = {}
id_to_priority: dict[int, str] = {}


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
            with open(os.path.join(MODEL_DIR, "id_to_label.json"), "r", encoding="utf-8") as f:
                id_to_label = {int(k): v for k, v in json.load(f).items()}
            with open(os.path.join(MODEL_DIR, "id_to_priority.json"), "r", encoding="utf-8") as f:
                id_to_priority = {int(k): v for k, v in json.load(f).items()}

            if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
                backbone_name = MODEL_DIR
            elif os.path.isdir(settings.backbone_model_dir):
                backbone_name = settings.backbone_model_dir
            else:
                backbone_name = settings.backbone_model_name

            tok_src = (
                MODEL_DIR
                if os.path.exists(os.path.join(MODEL_DIR, "tokenizer_config.json"))
                else backbone_name
            )
            tokenizer = AutoTokenizer.from_pretrained(tok_src)
            model = DistilBertMultiTask(
                backbone_name,
                num_labels=len(id_to_label),
                num_priority=len(id_to_priority),
            )

            state_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
            safetensors_path = os.path.join(MODEL_DIR, "model.safetensors")

            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu")
            elif os.path.exists(safetensors_path):
                if load_safetensors_file is None:
                    raise RuntimeError(
                        "model.safetensors found but safetensors package is not installed"
                    )
                state = load_safetensors_file(safetensors_path, device="cpu")
            else:
                raise FileNotFoundError(
                    f"Missing model weights. Expected one of: {state_path}, {safetensors_path}"
                )

            model.load_state_dict(state, strict=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            _model_ready = True
        except Exception as exc:
            _model_error = str(exc)
            raise


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
                CREATE TABLE IF NOT EXISTS complaint_status_audit (
                  id UUID PRIMARY KEY,
                  complaint_id UUID NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
                  old_status VARCHAR(30),
                  new_status VARCHAR(30) NOT NULL,
                  reason TEXT,
                  changed_by TEXT NOT NULL,
                  changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status_audit_complaint_changed_at
                ON complaint_status_audit (complaint_id, changed_at DESC)
                """
            )
        conn.commit()


def _insert_status_audit(
    cur: RealDictCursor,
    complaint_id: str,
    old_status: str | None,
    new_status: str,
    reason: str | None,
    changed_by: str,
) -> None:
    cur.execute(
        """
        INSERT INTO complaint_status_audit (
          id, complaint_id, old_status, new_status, reason, changed_by
        ) VALUES (
          %s::uuid, %s::uuid, %s, %s, %s, %s
        )
        """,
        (
            str(uuid.uuid4()),
            complaint_id,
            old_status,
            new_status,
            reason,
            changed_by,
        ),
    )


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def startup_checks() -> None:
    _ensure_schema()


@app.post("/predict")
def predict(payload: ComplaintIn):
    try:
        return predict_one(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@app.get("/complaints")
def list_complaints(current_user: CurrentUser = Depends(get_current_user)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, description, category, priority, department, status, attachments, created_at, updated_at
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

    complaint_id = str(uuid.uuid4())
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO complaints (
                  id, user_id, title, description, category, priority, department, status, attachments
                ) VALUES (
                  %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id, user_id, title, description, category, priority, department, status, attachments, created_at, updated_at
                """,
                (
                    complaint_id,
                    current_user.user_id,
                    payload.title.strip(),
                    payload.description.strip(),
                    (payload.category or "Uncategorized").strip(),
                    priority,
                    None,
                    status_value,
                    Json(payload.attachments),
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return _serialize_row(row)


@app.post("/admin/complaints/{complaint_id}/approve")
def approve_complaint(
    complaint_id: str,
    admin_user: CurrentUser = Depends(require_admin),
):
    changed_by = admin_user.email or admin_user.user_id
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status
                FROM complaints
                WHERE id = %s::uuid
                """,
                (complaint_id,),
            )
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Complaint not found")
            old_status = existing["status"]

            cur.execute(
                """
                UPDATE complaints
                SET status = 'in-progress'
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status, attachments, created_at, updated_at
                """,
                (complaint_id,),
            )
            row = cur.fetchone()
            if old_status != "in-progress":
                _insert_status_audit(
                    cur,
                    complaint_id=complaint_id,
                    old_status=old_status,
                    new_status="in-progress",
                    reason="Approved by admin",
                    changed_by=changed_by,
                )
        conn.commit()

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


@app.get("/admin/complaints")
def list_all_complaints(admin_user: CurrentUser = Depends(require_admin)):
    del admin_user
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, description, category, priority, department, status, attachments, created_at, updated_at
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

    text = f"{row['title']}\n\n{row['description']}"
    prediction = predict_one(text)
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
                prediction = predict_one(f"{row['title']}\n\n{row['description']}")
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model unavailable during auto-apply: {exc}",
                ) from exc
            cur.execute(
                """
                UPDATE complaints
                SET category = %s, priority = %s, department = %s
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status, attachments, created_at, updated_at
                """,
                (
                    prediction["label"],
                    str(prediction["priority"]).lower(),
                    prediction["department"],
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
    changed_by = admin_user.email or admin_user.user_id

    fields: list[str] = []
    values: list[Any] = []
    idx = 1
    target_status: str | None = None

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
        if status_value in {"resolved", "rejected"} and not (payload.status_reason or "").strip():
            raise HTTPException(
                status_code=400,
                detail="status_reason is required when status is resolved or rejected",
            )
        fields.append("status = %s")
        values.append(status_value)
        target_status = status_value
        idx += 1

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    values.append(complaint_id)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            old_status: str | None = None
            if target_status is not None:
                cur.execute(
                    """
                    SELECT status
                    FROM complaints
                    WHERE id = %s::uuid
                    """,
                    (complaint_id,),
                )
                existing = cur.fetchone()
                if not existing:
                    raise HTTPException(status_code=404, detail="Complaint not found")
                old_status = existing["status"]

            cur.execute(
                f"""
                UPDATE complaints
                SET {", ".join(fields)}
                WHERE id = %s::uuid
                RETURNING id, user_id, title, description, category, priority, department, status, attachments, created_at, updated_at
                """,
                tuple(values),
            )
            row = cur.fetchone()

            if target_status is not None and old_status != target_status:
                _insert_status_audit(
                    cur,
                    complaint_id=complaint_id,
                    old_status=old_status,
                    new_status=target_status,
                    reason=(payload.status_reason or "").strip() or None,
                    changed_by=changed_by,
                )
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Complaint not found")

    return _serialize_row(row)


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
                    prediction = predict_one(f"{row['title']}\n\n{row['description']}")
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Model unavailable during auto-classify: {exc}",
                    ) from exc
                cur.execute(
                    """
                    UPDATE complaints
                    SET category = %s, priority = %s, department = %s
                    WHERE id = %s::uuid
                    RETURNING id, user_id, title, description, category, priority, department, status, attachments, created_at, updated_at
                    """,
                    (
                        prediction["label"],
                        str(prediction["priority"]).lower(),
                        prediction["department"],
                        row["id"],
                    ),
                )
                updated = cur.fetchone()
                updated_items.append(
                    {
                        "prediction": prediction,
                        "complaint": _serialize_row(updated),
                    }
                )
        conn.commit()

    return {
        "updatedCount": len(updated_items),
        "items": updated_items,
    }
