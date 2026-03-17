import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

# ========= CONFIG =========
MODEL_DIR = r"outputs\edu_classifier_multitask"
TEST_PATH = r"data\test.csv"
OUT_DIR = r"outputs\edu_classifier_multitask\analysis"
BATCH = 32
MAX_LENGTH = 256
FALLBACK_MODEL = "distilbert-base-uncased"
# ==========================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def print_top_confusions(cm: np.ndarray, names: list[str], top_k: int, title: str) -> None:
    rows = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                rows.append((c, i, j))

    rows.sort(reverse=True, key=lambda x: x[0])
    print(f"\nTop {min(top_k, len(rows))} confusions for {title}:")
    if not rows:
        print("No off-diagonal errors found.")
        return

    for c, i, j in rows[:top_k]:
        true_name = names[i] if i < len(names) else str(i)
        pred_name = names[j] if j < len(names) else str(j)
        print(f"  true='{true_name}' predicted='{pred_name}' count={c}")


def save_html_heatmap(cm_df: pd.DataFrame, out_path: str, title: str) -> None:
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
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(html))


def main():
    ensure_dir(OUT_DIR)

    print(f"Loading: {TEST_PATH}")
    df = pd.read_csv(TEST_PATH, low_memory=False)
    need = {"text", "label_id", "priority_id_fixed"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in test.csv: {missing}")

    test_df = df[["text", "label_id", "priority_id_fixed"]].rename(
        columns={"label_id": "labels", "priority_id_fixed": "priority_labels"}
    )

    label_map_path = os.path.join(MODEL_DIR, "id_to_label.json")
    prio_map_path = os.path.join(MODEL_DIR, "id_to_priority.json")
    for p in [label_map_path, prio_map_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing mapping: {p}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    with open(prio_map_path, "r", encoding="utf-8") as f:
        id_to_priority = {int(k): v for k, v in json.load(f).items()}

    num_labels = len(id_to_label)
    num_priority = len(id_to_priority)

    label_names = [id_to_label[i] for i in range(num_labels)]
    priority_names = [id_to_priority[i] for i in range(num_priority)]

    tok_src = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "tokenizer_config.json")) else FALLBACK_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(test_ds, batch_size=BATCH, collate_fn=collator)

    backbone_src = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "config.json")) else FALLBACK_MODEL

    class DistilBertMultiTask(nn.Module):
        def __init__(self, model_name: str, n_labels: int, n_prio: int):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.label_dropout = nn.Dropout(0.2)
            self.prio_dropout = nn.Dropout(0.2)
            self.act = nn.GELU()
            self.label_hidden = nn.Linear(hidden, hidden // 2)
            self.label_head = nn.Linear(hidden // 2, n_labels)
            self.prio_hidden = nn.Linear(hidden, hidden // 4)
            self.prio_head = nn.Linear(hidden // 4, n_prio)

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.dropout(out.last_hidden_state[:, 0])
            label_logits = self.label_head(self.act(self.label_hidden(self.label_dropout(pooled))))
            prio_logits = self.prio_head(self.act(self.prio_hidden(self.prio_dropout(pooled))))
            return label_logits, prio_logits

    model = DistilBertMultiTask(backbone_src, num_labels, num_priority)
    safe_path = os.path.join(MODEL_DIR, "model.safetensors")
    bin_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
    if os.path.exists(safe_path):
        state = load_file(safe_path)
    elif os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError("No model weights found.")
    model_keys = set(model.state_dict().keys())
    state = {k: v for k, v in state.items() if k in model_keys}
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Device: {device}")

    y_label_true, y_label_pred = [], []
    y_prio_true, y_prio_pred = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").cpu().numpy()
            prios = batch.pop("priority_labels").cpu().numpy()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            label_logits, prio_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            y_label_true.extend(labels.tolist())
            y_prio_true.extend(prios.tolist())
            y_label_pred.extend(label_logits.argmax(dim=1).cpu().numpy().tolist())
            y_prio_pred.extend(prio_logits.argmax(dim=1).cpu().numpy().tolist())

    label_cm = confusion_matrix(y_label_true, y_label_pred, labels=list(range(num_labels)))
    prio_cm = confusion_matrix(y_prio_true, y_prio_pred, labels=list(range(num_priority)))

    label_cm_df = pd.DataFrame(label_cm, index=label_names, columns=label_names)
    prio_cm_df = pd.DataFrame(prio_cm, index=priority_names, columns=priority_names)

    label_cm_path = os.path.join(OUT_DIR, "label_confusion_matrix.csv")
    prio_cm_path = os.path.join(OUT_DIR, "priority_confusion_matrix.csv")
    label_cm_df.to_csv(label_cm_path, encoding="utf-8")
    prio_cm_df.to_csv(prio_cm_path, encoding="utf-8")
    label_html_path = os.path.join(OUT_DIR, "label_confusion_matrix.html")
    prio_html_path = os.path.join(OUT_DIR, "priority_confusion_matrix.html")
    save_html_heatmap(label_cm_df, label_html_path, "Label Confusion Matrix")
    save_html_heatmap(prio_cm_df, prio_html_path, "Priority Confusion Matrix")

    print("\nSaved:")
    print(f"  {label_cm_path}")
    print(f"  {prio_cm_path}")
    print(f"  {label_html_path}")
    print(f"  {prio_html_path}")

    print_top_confusions(label_cm, label_names, top_k=15, title="LABEL")
    print_top_confusions(prio_cm, priority_names, top_k=10, title="PRIORITY")

    # Optional heatmaps if matplotlib is installed.
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        label_fig = plt.figure(figsize=(14, 12))
        sns.heatmap(label_cm_df, cmap="Blues", linewidths=0.2)
        plt.title("Label Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        label_png = os.path.join(OUT_DIR, "label_confusion_matrix.png")
        label_fig.tight_layout()
        label_fig.savefig(label_png, dpi=180)
        plt.close(label_fig)

        prio_fig = plt.figure(figsize=(6, 5))
        sns.heatmap(prio_cm_df, annot=True, fmt="d", cmap="Greens", linewidths=0.2)
        plt.title("Priority Confusion Matrix")
        plt.ylabel("True priority")
        plt.xlabel("Predicted priority")
        prio_png = os.path.join(OUT_DIR, "priority_confusion_matrix.png")
        prio_fig.tight_layout()
        prio_fig.savefig(prio_png, dpi=180)
        plt.close(prio_fig)

        print(f"  {label_png}")
        print(f"  {prio_png}")
    except Exception as e:
        print(f"\nSkipping PNG plots (matplotlib/seaborn unavailable): {e}")


if __name__ == "__main__":
    main()
