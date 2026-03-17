import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = APP_ROOT / "outputs"


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def required_model_dirs() -> list[str]:
    raw = os.getenv(
        "MODEL_REQUIRED_DIRS",
        "distilbert_cfpb_mlm,edu_classifier_multitask",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


def local_model_missing(model_dir: str) -> bool:
    path = OUTPUTS_DIR / model_dir
    if not path.exists():
        return True
    if not any(path.iterdir()):
        return True
    return False


def should_sync() -> bool:
    return env_flag("MODEL_SYNC_ON_STARTUP", False)


def sync_mode() -> str:
    return os.getenv("MODEL_SYNC_MODE", "missing").strip().lower()


def download_prefix(
    s3_client,
    *,
    bucket: str,
    prefix: str,
    local_root: Path,
) -> int:
    paginator = s3_client.get_paginator("list_objects_v2")
    downloaded = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            relative = key[len(prefix):].lstrip("/")
            if not relative:
                continue

            local_path = local_root / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(bucket, key, str(local_path))
            downloaded += 1

    return downloaded


def main() -> None:
    if not should_sync():
        print("Model sync disabled. Skipping S3 model download.")
        return

    bucket = os.getenv("MODEL_S3_BUCKET", "").strip()
    prefix_base = os.getenv("MODEL_S3_PREFIX", "models").strip().strip("/")
    if not bucket:
        print("MODEL_SYNC_ON_STARTUP is enabled but MODEL_S3_BUCKET is missing. Skipping model sync.")
        return

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    mode = sync_mode()
    s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))

    try:
        total_downloaded = 0
        for model_dir in required_model_dirs():
            if mode == "missing" and not local_model_missing(model_dir):
                print(f"Model '{model_dir}' already exists locally. Skipping.")
                continue

            remote_prefix = f"{prefix_base}/{model_dir}"
            local_root = OUTPUTS_DIR / model_dir
            print(f"Syncing s3://{bucket}/{remote_prefix} -> {local_root}")
            downloaded = download_prefix(
                s3_client,
                bucket=bucket,
                prefix=remote_prefix,
                local_root=local_root,
            )
            print(f"Downloaded {downloaded} files for '{model_dir}'.")
            total_downloaded += downloaded

        print(f"Model sync complete. Total downloaded files: {total_downloaded}")
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Failed to sync model artifacts from S3: {exc}") from exc


if __name__ == "__main__":
    main()
