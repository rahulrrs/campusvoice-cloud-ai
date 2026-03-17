import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:
    torch = None
    AutoModel = None
    AutoTokenizer = None


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
MODEL_DIR = ROOT / "outputs" / "edu_classifier_multitask"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)
    required = {"text", "label_id"}
    if not required.issubset(train_df.columns) or not required.issubset(test_df.columns):
        raise ValueError("train.csv and test.csv must contain columns: text, label_id")
    train_df = train_df.dropna(subset=["text", "label_id"]).copy()
    test_df = test_df.dropna(subset=["text", "label_id"]).copy()
    train_df["label_id"] = train_df["label_id"].astype(int)
    test_df["label_id"] = test_df["label_id"].astype(int)
    return train_df, test_df


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _run_logistic_regression(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("lr", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ]
    )
    clf.fit(train_df["text"], train_df["label_id"])
    pred = clf.predict(test_df["text"])
    return _metrics(test_df["label_id"].to_numpy(), pred)


def _run_linear_svm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("svm", LinearSVC()),
        ]
    )
    clf.fit(train_df["text"], train_df["label_id"])
    pred = clf.predict(test_df["text"])
    return _metrics(test_df["label_id"].to_numpy(), pred)


def _run_distilbert(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float] | None:
    if torch is None or AutoModel is None or AutoTokenizer is None:
        return None
    if not (MODEL_DIR / "model.safetensors").exists():
        return None
    if not (MODEL_DIR / "id_to_label.json").exists():
        return None

    with open(MODEL_DIR / "id_to_label.json", "r", encoding="utf-8") as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    num_labels = len(id_to_label)

    tok_src = str(MODEL_DIR if (MODEL_DIR / "tokenizer_config.json").exists() else "distilbert-base-uncased")
    backbone_src = str(MODEL_DIR if (MODEL_DIR / "config.json").exists() else "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    class DistilLabelOnly(torch.nn.Module):
        def __init__(self, model_name: str, out_dim: int):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self.dropout = torch.nn.Dropout(0.1)
            self.label_dropout = torch.nn.Dropout(0.2)
            self.label_hidden = torch.nn.Linear(hidden, hidden // 2)
            self.label_head = torch.nn.Linear(hidden // 2, out_dim)
            self.act = torch.nn.GELU()

        def forward(self, input_ids=None, attention_mask=None):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.dropout(out.last_hidden_state[:, 0])
            return self.label_head(self.act(self.label_hidden(self.label_dropout(pooled))))

    model = DistilLabelOnly(backbone_src, num_labels)
    from safetensors.torch import load_file

    state = load_file(str(MODEL_DIR / "model.safetensors"), device="cpu")
    filtered = {
        key: value
        for key, value in state.items()
        if key.startswith("backbone.") or key.startswith("dropout.") or key.startswith("label_")
    }
    model.load_state_dict(filtered, strict=False)
    model.eval()

    y_true = test_df["label_id"].to_numpy()
    y_pred = []
    for text in test_df["text"].tolist():
        enc = tokenizer(text, truncation=True, padding=False, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
            )
        y_pred.append(int(logits.argmax(dim=1).item()))

    return _metrics(y_true, np.array(y_pred))


def main() -> None:
    train_df, test_df = _load_data()
    results = {
        "dataset": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "num_labels": int(train_df["label_id"].nunique()),
            "avg_text_len_train": float(train_df["text"].str.len().mean()),
            "avg_text_len_test": float(test_df["text"].str.len().mean()),
        },
        "models": {
            "logistic_regression": _run_logistic_regression(train_df, test_df),
            "linear_svm": _run_linear_svm(train_df, test_df),
            "distilbert_finetuned": _run_distilbert(train_df, test_df),
        },
    }

    out_path = ROOT / "outputs" / "baseline_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nSaved baseline comparison to: {out_path}")


if __name__ == "__main__":
    main()
