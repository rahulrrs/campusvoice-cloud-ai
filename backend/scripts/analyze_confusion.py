import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.complaint_ml import build_metadata_feature_map, feature_map_to_vector, scale_feature_vector
from src.utils.model_paths import load_project_env, resolve_backbone_source

MODEL_DIR = PROJECT_ROOT / "outputs" / "edu_classifier_multitask"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
OUT_DIR = MODEL_DIR / "analysis"
MAX_LENGTH = 256
BATCH = 32
load_project_env(PROJECT_ROOT)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def print_top_confusions(cm: np.ndarray, names: list[str], top_k: int, title: str) -> None:
    rows = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                rows.append((count, i, j))

    rows.sort(reverse=True, key=lambda item: item[0])
    print(f"\nTop {min(top_k, len(rows))} confusions for {title}:")
    if not rows:
        print("No off-diagonal errors found.")
        return

    for count, i, j in rows[:top_k]:
        true_name = names[i] if i < len(names) else str(i)
        pred_name = names[j] if j < len(names) else str(j)
        print(f"  true='{true_name}' predicted='{pred_name}' count={count}")


def save_html_heatmap(cm_df: pd.DataFrame, out_path: Path, title: str) -> None:
    values = cm_df.values.astype(float)
    vmax = float(values.max()) if values.size else 0.0
    denom = vmax if vmax > 0.0 else 1.0

    html = []
    html.append("<!doctype html>")
    html.append("<html><head><meta charset='utf-8'>")
    html.append(f"<title>{title}</title>")
    html.append(
        "<style>"
        "body{font-family:Segoe UI,Tahoma,Arial,sans-serif;margin:20px;background:#fafafa;color:#111;}"
        "h1{font-size:18px;margin:0 0 12px 0;}"
        ".wrap{overflow:auto;border:1px solid #ddd;background:#fff;max-height:85vh;}"
        "table{border-collapse:collapse;font-size:12px;}"
        "th,td{border:1px solid #e1e1e1;padding:4px 6px;text-align:center;white-space:nowrap;}"
        "thead th{position:sticky;top:0;background:#f0f0f0;z-index:2;}"
        "tbody th{position:sticky;left:0;background:#f8f8f8;z-index:1;text-align:left;}"
        ".diag{outline:2px solid #2f7d32;outline-offset:-2px;}"
        "</style>"
    )
    html.append("</head><body>")
    html.append(f"<h1>{title}</h1>")
    html.append("<div class='wrap'><table>")

    html.append("<thead><tr><th>True \\ Pred</th>")
    for col in cm_df.columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead><tbody>")

    for i, row_name in enumerate(cm_df.index):
        html.append(f"<tr><th>{row_name}</th>")
        for j, col_name in enumerate(cm_df.columns):
            val = int(cm_df.loc[row_name, col_name])
            ratio = float(val) / denom
            blue = int(255 - (140 * ratio))
            bg = f"rgb({blue},{blue},{255})"
            cls = "diag" if i == j else ""
            html.append(f"<td class='{cls}' style='background:{bg};'>{val}</td>")
        html.append("</tr>")

    html.append("</tbody></table></div></body></html>")
    out_path.write_text("".join(html), encoding="utf-8")


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
            priority_input = torch.cat(
                [priority_input, self.priority_meta_proj(metadata_features.float())], dim=-1
            )
        prio_logits = self.prio_head(self.act(self.prio_hidden(self.prio_dropout(priority_input))))
        return label_logits, prio_logits


def main() -> None:
    ensure_dir(OUT_DIR)

    print(f"Loading: {TEST_PATH}")
    df = pd.read_csv(TEST_PATH, low_memory=False)
    required_cols = {"text", "label_id", "priority_id_fixed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in test.csv: {missing}")

    test_df = df[["text", "label_id", "priority_id_fixed"]].rename(
        columns={"label_id": "labels", "priority_id_fixed": "priority_labels"}
    )

    label_map_path = MODEL_DIR / "id_to_label.json"
    prio_map_path = MODEL_DIR / "id_to_priority.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing mapping: {label_map_path}")
    if not prio_map_path.exists():
        raise FileNotFoundError(f"Missing mapping: {prio_map_path}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    with open(prio_map_path, "r", encoding="utf-8") as f:
        id_to_priority = {int(k): v for k, v in json.load(f).items()}

    label_ids = sorted(id_to_label)
    priority_ids = sorted(id_to_priority)
    if label_ids != list(range(len(label_ids))):
        raise ValueError(f"Label ids must be contiguous starting at 0. Found: {label_ids}")
    if priority_ids != list(range(len(priority_ids))):
        raise ValueError(
            f"Priority ids must be contiguous starting at 0. Found: {priority_ids}"
        )

    num_labels = len(label_ids)
    num_priority = len(priority_ids)
    label_names = [id_to_label[i] for i in label_ids]
    priority_names = [id_to_priority[i] for i in priority_ids]

    tokenizer_config_path = MODEL_DIR / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        raise FileNotFoundError(f"Missing classifier tokenizer config: {tokenizer_config_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    metadata_config_path = MODEL_DIR / "metadata_config.json"
    metadata_config = None
    if metadata_config_path.exists():
        with open(metadata_config_path, "r", encoding="utf-8") as f:
            metadata_config = json.load(f)

    test_df = test_df.copy()
    test_df["metadata_features"] = [
        scale_feature_vector(
            feature_map_to_vector(
                build_metadata_feature_map(text=str(text)),
                metadata_config.get("feature_names") if isinstance(metadata_config, dict) else None,
            ),
            metadata_config,
        )
        for text in test_df["text"].astype(str).tolist()
    ]

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_ds, batch_size=BATCH, collate_fn=collator)

    safe_path = MODEL_DIR / "model.safetensors"
    bin_path = MODEL_DIR / "pytorch_model.bin"
    if safe_path.exists():
        state = load_file(str(safe_path))
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {MODEL_DIR}")

    for key in ("label_weights", "priority_weights", "priority_cost_matrix"):
        state.pop(key, None)
    metadata_dim = len(metadata_config.get("feature_names", [])) if isinstance(metadata_config, dict) else 0
    has_label_metadata_layers = any(key.startswith("label_meta_proj.") for key in state)
    has_metadata_layers = any(key.startswith("priority_meta_proj.") for key in state)
    backbone_source, backbone_note = resolve_backbone_source(PROJECT_ROOT, MODEL_DIR)
    if backbone_note:
        print(backbone_note)
    model = EncoderMultiTask(
        backbone_source,
        num_labels,
        num_priority,
        label_metadata_dim=metadata_dim if has_label_metadata_layers else 0,
        priority_metadata_dim=metadata_dim if has_metadata_layers else 0,
    )
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Device: {device}")
    print(f"Backbone: {backbone_source}")
    print(f"Classifier dir: {MODEL_DIR}")

    y_label_true, y_label_pred = [], []
    y_prio_true, y_prio_pred = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").cpu().numpy()
            prios = batch.pop("priority_labels").cpu().numpy()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            metadata_features = batch.get("metadata_features")
            metadata_features = metadata_features.to(device) if metadata_features is not None else None

            label_logits, prio_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata_features=metadata_features,
            )

            y_label_true.extend(labels.tolist())
            y_prio_true.extend(prios.tolist())
            y_label_pred.extend(label_logits.argmax(dim=1).cpu().numpy().tolist())
            y_prio_pred.extend(prio_logits.argmax(dim=1).cpu().numpy().tolist())

    label_cm = confusion_matrix(y_label_true, y_label_pred, labels=label_ids)
    prio_cm = confusion_matrix(y_prio_true, y_prio_pred, labels=priority_ids)

    label_cm_df = pd.DataFrame(label_cm, index=label_names, columns=label_names)
    prio_cm_df = pd.DataFrame(prio_cm, index=priority_names, columns=priority_names)

    label_cm_path = OUT_DIR / "label_confusion_matrix.csv"
    prio_cm_path = OUT_DIR / "priority_confusion_matrix.csv"
    label_html_path = OUT_DIR / "label_confusion_matrix.html"
    prio_html_path = OUT_DIR / "priority_confusion_matrix.html"

    label_cm_df.to_csv(label_cm_path, encoding="utf-8")
    prio_cm_df.to_csv(prio_cm_path, encoding="utf-8")
    save_html_heatmap(label_cm_df, label_html_path, "Label Confusion Matrix")
    save_html_heatmap(prio_cm_df, prio_html_path, "Priority Confusion Matrix")

    print("\nSaved:")
    print(f"  {label_cm_path}")
    print(f"  {prio_cm_path}")
    print(f"  {label_html_path}")
    print(f"  {prio_html_path}")

    print_top_confusions(label_cm, label_names, top_k=15, title="LABEL")
    print_top_confusions(prio_cm, priority_names, top_k=10, title="PRIORITY")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        label_fig = plt.figure(figsize=(14, 12))
        sns.heatmap(label_cm_df, cmap="Blues", linewidths=0.2)
        plt.title("Label Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        label_png = OUT_DIR / "label_confusion_matrix.png"
        label_fig.tight_layout()
        label_fig.savefig(label_png, dpi=180)
        plt.close(label_fig)

        prio_fig = plt.figure(figsize=(6, 5))
        sns.heatmap(prio_cm_df, annot=True, fmt="d", cmap="Greens", linewidths=0.2)
        plt.title("Priority Confusion Matrix")
        plt.ylabel("True priority")
        plt.xlabel("Predicted priority")
        prio_png = OUT_DIR / "priority_confusion_matrix.png"
        prio_fig.tight_layout()
        prio_fig.savefig(prio_png, dpi=180)
        plt.close(prio_fig)

        print(f"  {label_png}")
        print(f"  {prio_png}")
    except Exception as e:
        print(f"\nSkipping PNG plots (matplotlib/seaborn unavailable): {e}")


if __name__ == "__main__":
    main()
