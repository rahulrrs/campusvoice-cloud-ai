import os
import uuid
from datetime import UTC, datetime
from typing import Any

import boto3

from .common import get_claim_user_id, parse_json_body, response
from .db import create_complaint, list_complaints_by_user

s3 = boto3.client("s3")


def _sanitize_filename(name: str) -> str:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", "."))
    return safe or "attachment"


def health_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    return response(
        200,
        {
            "status": "ok",
            "service": "complaints-api",
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


def list_complaints_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    user_id = get_claim_user_id(event)
    if not user_id:
        return response(401, {"message": "Unauthorized"})

    complaints = list_complaints_by_user(user_id)
    return response(200, complaints)


def create_complaint_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    user_id = get_claim_user_id(event)
    if not user_id:
        return response(401, {"message": "Unauthorized"})

    body = parse_json_body(event)
    title = str(body.get("title", "")).strip()
    description = str(body.get("description", "")).strip()
    category = str(body.get("category", "")).strip()
    priority = str(body.get("priority", "medium")).strip().lower()
    status = str(body.get("status", "pending")).strip().lower()
    attachments = body.get("attachments", [])

    if not title or not description or not category:
        return response(400, {"message": "title, description and category are required"})
    if priority not in {"low", "medium", "high"}:
        return response(400, {"message": "priority must be low, medium or high"})
    if status not in {"pending", "in-progress", "resolved", "rejected"}:
        return response(400, {"message": "invalid status"})
    if not isinstance(attachments, list) or any(not isinstance(k, str) for k in attachments):
        return response(400, {"message": "attachments must be a list of string keys"})

    complaint = create_complaint(
        complaint_id=str(uuid.uuid4()),
        user_id=user_id,
        title=title,
        description=description,
        category=category,
        priority=priority,
        status=status,
        attachments=attachments,
    )
    return response(201, complaint)


def create_presigned_upload_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    user_id = get_claim_user_id(event)
    if not user_id:
        return response(401, {"message": "Unauthorized"})

    bucket = os.getenv("ATTACHMENTS_BUCKET")
    if not bucket:
        return response(500, {"message": "Missing ATTACHMENTS_BUCKET environment variable"})

    body = parse_json_body(event)
    file_name = str(body.get("fileName", "")).strip()
    content_type = str(body.get("contentType", "application/octet-stream")).strip()
    if not file_name:
        return response(400, {"message": "fileName is required"})

    key = f"attachments/{user_id}/{uuid.uuid4()}-{_sanitize_filename(file_name)}"

    upload_url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=900,
    )

    return response(
        200,
        {
            "uploadUrl": upload_url,
            "key": key,
            "expiresIn": 900,
        },
    )
