import json
from typing import Any

import boto3

from .common import get_env


_client = boto3.client("rds-data")


def _execute(sql: str, parameters: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return _client.execute_statement(
        resourceArn=get_env("RDS_CLUSTER_ARN"),
        secretArn=get_env("RDS_SECRET_ARN"),
        database=get_env("RDS_DATABASE"),
        sql=sql,
        parameters=parameters or [],
        includeResultMetadata=True,
    )


def _to_field(value: Any) -> dict[str, Any]:
    if value is None:
        return {"isNull": True}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"longValue": value}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def _record_to_dict(columns: list[dict[str, Any]], values: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, column in enumerate(columns):
        label = column.get("label") or column.get("name")
        value_field = values[index]
        if "isNull" in value_field and value_field["isNull"]:
            result[label] = None
        elif "stringValue" in value_field:
            result[label] = value_field["stringValue"]
        elif "longValue" in value_field:
            result[label] = value_field["longValue"]
        elif "doubleValue" in value_field:
            result[label] = value_field["doubleValue"]
        elif "booleanValue" in value_field:
            result[label] = value_field["booleanValue"]
        else:
            result[label] = None
    return result


def list_complaints_by_user(user_id: str) -> list[dict[str, Any]]:
    result = _execute(
        """
        SELECT id, user_id, title, description, category, priority, status, attachments, created_at, updated_at
        FROM complaints
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        """,
        [{"name": "user_id", "value": _to_field(user_id)}],
    )
    columns = result.get("columnMetadata", [])
    rows = result.get("records", [])
    items = [_record_to_dict(columns, row) for row in rows]
    for item in items:
        raw = item.get("attachments")
        if isinstance(raw, str):
            try:
                item["attachments"] = json.loads(raw)
            except json.JSONDecodeError:
                item["attachments"] = []
        elif raw is None:
            item["attachments"] = []
    return items


def create_complaint(
    *,
    complaint_id: str,
    user_id: str,
    title: str,
    description: str,
    category: str,
    priority: str,
    status: str,
    attachments: list[str],
) -> dict[str, Any]:
    result = _execute(
        """
        INSERT INTO complaints (
          id, user_id, title, description, category, priority, status, attachments
        )
        VALUES (
          :id, :user_id, :title, :description, :category, :priority, :status, :attachments
        )
        RETURNING id, user_id, title, description, category, priority, status, attachments, created_at, updated_at
        """,
        [
            {"name": "id", "value": _to_field(complaint_id)},
            {"name": "user_id", "value": _to_field(user_id)},
            {"name": "title", "value": _to_field(title)},
            {"name": "description", "value": _to_field(description)},
            {"name": "category", "value": _to_field(category)},
            {"name": "priority", "value": _to_field(priority)},
            {"name": "status", "value": _to_field(status)},
            {"name": "attachments", "value": _to_field(json.dumps(attachments))},
        ],
    )

    columns = result.get("columnMetadata", [])
    rows = result.get("records", [])
    if not rows:
        raise RuntimeError("Insert succeeded but no complaint was returned")
    item = _record_to_dict(columns, rows[0])
    raw = item.get("attachments")
    if isinstance(raw, str):
        try:
            item["attachments"] = json.loads(raw)
        except json.JSONDecodeError:
            item["attachments"] = []
    elif raw is None:
        item["attachments"] = []
    return item
