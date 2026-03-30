import os
from pathlib import Path


def load_project_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_backbone_source(project_root: Path, model_dir: Path) -> tuple[str, str | None]:
    default_backbone_dir = project_root / "outputs" / "general_complaint_model"
    fallback_model_name = os.getenv("BACKBONE_MODEL_NAME", "roberta-base").strip() or "roberta-base"

    configured_value = os.getenv("BACKBONE_MODEL_DIR", "").strip()
    candidate = Path(configured_value) if configured_value else default_backbone_dir

    try:
        if candidate.resolve() == model_dir.resolve():
            candidate = default_backbone_dir
            return (
                str(candidate),
                "BACKBONE_MODEL_DIR points to edu_classifier_multitask; using general_complaint_model instead.",
            )
    except Exception:
        pass

    if candidate.is_dir():
        return str(candidate), None

    if default_backbone_dir.is_dir():
        if configured_value and candidate != default_backbone_dir:
            return (
                str(default_backbone_dir),
                f"Configured backbone directory not found. Falling back to: {default_backbone_dir}",
            )
        return str(default_backbone_dir), None

    return fallback_model_name, (
        f"Backbone directory not found. Falling back to model name: {fallback_model_name}"
    )
