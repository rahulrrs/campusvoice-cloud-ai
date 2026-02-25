import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def safe_training_args(TrainingArguments, **kwargs):
    """
    Transformers version differences:
    - some versions use evaluation_strategy
    - newer versions accept eval_strategy
    We try eval_strategy first, then fallback.
    """
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        if "eval_strategy" in str(e):
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
            return TrainingArguments(**kwargs)
        raise