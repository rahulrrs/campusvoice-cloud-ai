import math
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np


_WORD_RE = re.compile(r"[a-zA-Z']+")
_URGENCY_TERMS = {
    "urgent",
    "immediately",
    "asap",
    "critical",
    "emergency",
    "deadline",
    "today",
    "tomorrow",
    "soon",
}
_DEADLINE_TERMS = {
    "deadline",
    "last date",
    "tomorrow",
    "today",
    "exam",
    "result",
    "hall ticket",
    "submission",
    "attendance shortage",
}
_NEGATIVE_TERMS = {
    "delay",
    "broken",
    "issue",
    "problem",
    "angry",
    "frustrated",
    "unsafe",
    "harassment",
    "bully",
    "stolen",
    "failed",
    "worst",
    "late",
    "error",
    "not working",
}
_HARASSMENT_TERMS = {
    "harassment",
    "ragging",
    "unsafe",
    "threat",
    "violence",
    "assault",
    "bully",
}
_ACADEMIC_TERMS = {
    "exam",
    "attendance",
    "faculty",
    "teacher",
    "assignment",
    "grading",
    "certificate",
    "transcript",
}
_INFRASTRUCTURE_TERMS = {
    "wifi",
    "network",
    "projector",
    "lab",
    "classroom",
    "hostel",
    "washroom",
    "bus",
    "transport",
}
_REPEATED_ISSUE_TERMS = {
    "repeated",
    "again",
    "again and again",
    "every day",
    "every week",
    "still",
    "keeps happening",
    "continues",
    "recurring",
    "multiple complaints",
}
_FINANCIAL_IMPACT_TERMS = {
    "fees",
    "refund",
    "payment",
    "scholarship",
    "bank account",
    "financial",
    "money",
    "credited",
    "dues",
    "tuition",
}
_ACCESS_BLOCK_TERMS = {
    "unable to access",
    "cannot access",
    "blocked",
    "not working",
    "portal stopped",
    "login failed",
    "cannot log in",
    "locked out",
    "not loading",
    "freezing",
}
_SAFETY_RISK_TERMS = {
    "unsafe",
    "threat",
    "violence",
    "harassment",
    "ragging",
    "fumes",
    "headache",
    "security",
    "theft",
    "stolen",
}
_ACADEMIC_IMPACT_TERMS = {
    "marks affected",
    "internal score",
    "grades",
    "attendance shortage",
    "miss exam",
    "hall ticket",
    "result",
    "assignment",
    "assessment",
    "cannot attend",
}
_DURATION_TERMS = {
    "entire week",
    "past week",
    "for 2 days",
    "for two days",
    "for days",
    "for a week",
    "since day one",
    "for the past",
    "ongoing",
}

METADATA_FEATURE_NAMES = [
    "text_len_chars",
    "word_count",
    "urgency_term_hits",
    "deadline_term_hits",
    "negative_term_hits",
    "harassment_term_hits",
    "academic_term_hits",
    "infrastructure_term_hits",
    "repeated_issue_term_hits",
    "financial_impact_term_hits",
    "access_block_term_hits",
    "safety_risk_term_hits",
    "academic_impact_term_hits",
    "duration_term_hits",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
    "digit_ratio",
    "attachment_count",
    "image_attachment_count",
    "audio_attachment_count",
    "document_attachment_count",
    "evidence_type_count",
    "attachment_context_count",
    "has_attachment_text",
    "anonymous_flag",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_text(text))


def classify_attachment_kind(file_name: str) -> str:
    lower = str(file_name or "").lower()
    if re.search(r"\.(png|jpg|jpeg|gif|webp|bmp|svg|heic|tiff?)$", lower):
        return "image"
    if re.search(r"\.(mp3|wav|ogg|m4a|aac|webm|mp4)$", lower):
        return "audio"
    if re.search(r"\.(pdf|doc|docx|txt|rtf|csv|xls|xlsx|ppt|pptx)$", lower):
        return "document"
    return "other"


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _term_hits(text: str, phrases: set[str]) -> int:
    normalized = normalize_text(text)
    return sum(1 for phrase in phrases if phrase in normalized)


def build_metadata_feature_map(
    *,
    text: str,
    attachments: list[str] | None = None,
    evidence_types: list[str] | None = None,
    attachment_contexts: list[dict[str, Any]] | None = None,
    is_anonymous: bool = False,
    submitted_at: Any = None,
) -> dict[str, float]:
    normalized = normalize_text(text)
    words = tokenize(text)
    attachment_list = [str(item) for item in (attachments or []) if str(item).strip()]
    evidence_list = [str(item) for item in (evidence_types or []) if str(item).strip()]
    context_list = [item for item in (attachment_contexts or []) if isinstance(item, dict)]

    image_count = 0
    audio_count = 0
    document_count = 0
    for attachment in attachment_list:
        kind = classify_attachment_kind(attachment)
        if kind == "image":
            image_count += 1
        elif kind == "audio":
            audio_count += 1
        elif kind == "document":
            document_count += 1

    attachment_text_present = 0
    for item in context_list:
        if any(str(item.get(field) or "").strip() for field in ("ocr_text", "transcript_text", "image_summary")):
            attachment_text_present = 1
            break

    dt = parse_datetime(submitted_at)
    hour_angle = 2 * math.pi * (dt.hour / 24.0)
    weekday_angle = 2 * math.pi * (dt.weekday() / 7.0)
    uppercase_chars = sum(1 for ch in str(text or "") if ch.isupper())
    alpha_chars = sum(1 for ch in str(text or "") if ch.isalpha())
    digit_chars = sum(1 for ch in str(text or "") if ch.isdigit())

    features = {
        "text_len_chars": float(len(str(text or ""))),
        "word_count": float(len(words)),
        "urgency_term_hits": float(_term_hits(normalized, _URGENCY_TERMS)),
        "deadline_term_hits": float(_term_hits(normalized, _DEADLINE_TERMS)),
        "negative_term_hits": float(_term_hits(normalized, _NEGATIVE_TERMS)),
        "harassment_term_hits": float(_term_hits(normalized, _HARASSMENT_TERMS)),
        "academic_term_hits": float(_term_hits(normalized, _ACADEMIC_TERMS)),
        "infrastructure_term_hits": float(_term_hits(normalized, _INFRASTRUCTURE_TERMS)),
        "repeated_issue_term_hits": float(_term_hits(normalized, _REPEATED_ISSUE_TERMS)),
        "financial_impact_term_hits": float(_term_hits(normalized, _FINANCIAL_IMPACT_TERMS)),
        "access_block_term_hits": float(_term_hits(normalized, _ACCESS_BLOCK_TERMS)),
        "safety_risk_term_hits": float(_term_hits(normalized, _SAFETY_RISK_TERMS)),
        "academic_impact_term_hits": float(_term_hits(normalized, _ACADEMIC_IMPACT_TERMS)),
        "duration_term_hits": float(_term_hits(normalized, _DURATION_TERMS)),
        "exclamation_count": float(str(text or "").count("!")),
        "question_count": float(str(text or "").count("?")),
        "uppercase_ratio": float(uppercase_chars / max(alpha_chars, 1)),
        "digit_ratio": float(digit_chars / max(len(str(text or "")), 1)),
        "attachment_count": float(len(attachment_list)),
        "image_attachment_count": float(image_count),
        "audio_attachment_count": float(audio_count),
        "document_attachment_count": float(document_count),
        "evidence_type_count": float(len(set(evidence_list))),
        "attachment_context_count": float(len(context_list)),
        "has_attachment_text": float(attachment_text_present),
        "anonymous_flag": float(bool(is_anonymous)),
        "hour_sin": float(math.sin(hour_angle)),
        "hour_cos": float(math.cos(hour_angle)),
        "weekday_sin": float(math.sin(weekday_angle)),
        "weekday_cos": float(math.cos(weekday_angle)),
    }
    return features


def feature_map_to_vector(feature_map: dict[str, float], feature_names: list[str] | None = None) -> list[float]:
    names = feature_names or METADATA_FEATURE_NAMES
    return [float(feature_map.get(name, 0.0)) for name in names]


def fit_metadata_scaler(feature_rows: list[list[float]], feature_names: list[str] | None = None) -> dict[str, Any]:
    names = feature_names or METADATA_FEATURE_NAMES
    matrix = np.asarray(feature_rows, dtype=np.float32)
    if matrix.size == 0:
        matrix = np.zeros((1, len(names)), dtype=np.float32)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    return {
        "feature_names": names,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def scale_feature_vector(vector: list[float], scaler: dict[str, Any] | None) -> list[float]:
    if not scaler:
        return [float(value) for value in vector]
    mean = scaler.get("mean") or []
    std = scaler.get("std") or []
    scaled: list[float] = []
    for index, value in enumerate(vector):
        mu = float(mean[index]) if index < len(mean) else 0.0
        sigma = float(std[index]) if index < len(std) else 1.0
        if abs(sigma) < 1.0e-6:
            sigma = 1.0
        scaled.append(float((float(value) - mu) / sigma))
    return scaled


def build_explainability_payload(
    *,
    text: str,
    classification: dict[str, Any],
    feature_map: dict[str, float],
    duplicate_detection: dict[str, Any] | None = None,
    multimodal_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rationale_items: list[dict[str, Any]] = []

    for label, feature_name, reason in [
        ("priority", "urgency_term_hits", "Urgency keywords increase handling priority."),
        ("priority", "deadline_term_hits", "Deadline or time-bound wording suggests escalation risk."),
        ("priority", "access_block_term_hits", "Loss of access raises priority because progress is blocked."),
        ("priority", "financial_impact_term_hits", "Financial impact can raise complaint severity."),
        ("priority", "safety_risk_term_hits", "Safety-related language increases severity."),
        ("priority", "academic_impact_term_hits", "Direct academic impact can justify higher priority."),
        ("priority", "attachment_count", "Evidence attachments strengthen urgency and actionability."),
        ("risk", "harassment_term_hits", "Safety or harassment language increases review sensitivity."),
        ("routing", "academic_term_hits", "Academic terms reinforce academic routing decisions."),
        ("routing", "infrastructure_term_hits", "Infrastructure terms reinforce facilities or IT routing."),
    ]:
        value = float(feature_map.get(feature_name, 0.0))
        if value > 0:
            rationale_items.append(
                {
                    "target": label,
                    "feature": feature_name,
                    "value": round(value, 4),
                    "reason": reason,
                }
            )

    duplicate_score = float((duplicate_detection or {}).get("score", 0.0) or 0.0)
    if duplicate_score > 0:
        rationale_items.append(
            {
                "target": "duplicate_detection",
                "feature": "semantic_similarity",
                "value": round(duplicate_score, 4),
                "reason": "Semantic similarity was used to compare this complaint with prior complaints.",
            }
        )

    attachment_summary = str((multimodal_evidence or {}).get("summary") or "").strip()
    if attachment_summary:
        rationale_items.append(
            {
                "target": "evidence",
                "feature": "multimodal_context",
                "value": 1.0,
                "reason": attachment_summary,
            }
        )

    rationale_items = rationale_items[:6]
    confidence = float(classification.get("priority_confidence", 0.0) or 0.0)
    if confidence >= 0.85:
        confidence_band = "high"
    elif confidence >= 0.6:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    label = str(classification.get("label") or "Unknown")
    priority = str(classification.get("priority") or "medium")
    summary_bits = [f"Routed as {label}", f"priority {priority}", f"{confidence_band} confidence"]
    if duplicate_score >= 0.82:
        summary_bits.append("strong semantic duplicate match")

    return {
        "summary": ", ".join(summary_bits) + ".",
        "confidence_band": confidence_band,
        "rationale_items": rationale_items,
        "text_preview": str(text or "")[:220],
    }
