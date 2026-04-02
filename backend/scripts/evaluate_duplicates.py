import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.semantic_duplicates import SemanticDuplicateEngine


DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "dataset_corrected.csv",
    PROJECT_ROOT / "data" / "dataset_clean.csv",
]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "duplicate_eval.json"
RANDOM_SEED = 42
MAX_PAIRS = 40
NEGATIVES_PER_QUERY = 9
PAIR_MIN_SIMILARITY = 0.95
MIN_TEXT_LEN = 40


def resolve_dataset_path() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in DATASET_CANDIDATES)
    raise FileNotFoundError(f"No duplicate-eval dataset found. Tried: {tried}")


def normalize_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_dataset() -> pd.DataFrame:
    path = resolve_dataset_path()
    df = pd.read_csv(path, low_memory=False)
    required = {"text", "label", "priority"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    out = df[["text", "label", "priority"]].dropna().copy().reset_index(drop=True)
    out["norm_text"] = out["text"].map(normalize_text)
    out = out[out["norm_text"].str.len() >= MIN_TEXT_LEN].reset_index(drop=True)
    return out


def mine_near_duplicate_pairs(df: pd.DataFrame) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    for (_, _), group in df.groupby(["label", "priority"]):
        texts = group["norm_text"].tolist()
        if len(texts) < 2:
            continue

        indices = group.index.to_list()
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        similarities = linear_kernel(matrix, matrix)

        for local_i in range(len(texts)):
            similarities[local_i, local_i] = -1.0
            local_j = int(similarities[local_i].argmax())
            score = float(similarities[local_i, local_j])
            if score < PAIR_MIN_SIMILARITY:
                continue

            row_i = indices[local_i]
            row_j = indices[local_j]
            pair_key = tuple(sorted((row_i, row_j)))
            if pair_key in seen:
                continue

            raw_i = str(df.at[row_i, "text"])
            raw_j = str(df.at[row_j, "text"])
            if raw_i == raw_j:
                continue

            seen.add(pair_key)
            pairs.append((row_i, row_j, score))

    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:MAX_PAIRS]


def build_row(df: pd.DataFrame, index: int) -> dict[str, object]:
    return {
        "id": int(index),
        "title": "",
        "description": str(df.at[index, "text"]),
        "category": str(df.at[index, "label"]),
        "status": "closed",
    }


def evaluate_pairs(df: pd.DataFrame, pairs: list[tuple[int, int, float]]) -> dict[str, object]:
    random.seed(RANDOM_SEED)
    engine = SemanticDuplicateEngine(duplicate_threshold=0.82, ann_k=5)

    all_indices = df.index.tolist()
    top1_hits = 0
    threshold_positive_hits = 0
    threshold_negative_hits = 0
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    examples: list[dict[str, object]] = []
    method_counts: dict[str, int] = {}

    for query_index, duplicate_index, seed_similarity in pairs:
        query_text = str(df.at[query_index, "text"])
        negative_pool = [
            idx for idx in all_indices if str(df.at[idx, "label"]) != str(df.at[query_index, "label"])
        ]
        negative_indices = random.sample(negative_pool, NEGATIVES_PER_QUERY)

        positive_rows = [build_row(df, duplicate_index)] + [build_row(df, idx) for idx in negative_indices]
        random.shuffle(positive_rows)
        positive_result = engine.search(query_text, positive_rows)
        method = str(positive_result.get("method") or "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1

        if positive_result.get("matches"):
            top_id = int(positive_result["matches"][0]["id"])
            if top_id == duplicate_index:
                top1_hits += 1

        if bool(positive_result.get("is_duplicate")):
            threshold_positive_hits += 1
        positive_scores.append(float(positive_result.get("score", 0.0) or 0.0))

        negative_rows = [build_row(df, idx) for idx in negative_indices]
        negative_result = engine.search(query_text, negative_rows)
        if not bool(negative_result.get("is_duplicate")):
            threshold_negative_hits += 1
        negative_scores.append(float(negative_result.get("score", 0.0) or 0.0))

        if len(examples) < 5:
            examples.append(
                {
                    "label": str(df.at[query_index, "label"]),
                    "priority": str(df.at[query_index, "priority"]),
                    "seed_similarity": round(seed_similarity, 4),
                    "query_preview": query_text[:160],
                    "duplicate_preview": str(df.at[duplicate_index, "text"])[:160],
                    "positive_score": round(float(positive_result.get("score", 0.0) or 0.0), 4),
                    "negative_score": round(float(negative_result.get("score", 0.0) or 0.0), 4),
                }
            )

    pair_count = len(pairs)
    if pair_count == 0:
        raise ValueError("No near-duplicate probe pairs were found for evaluation.")

    metrics = {
        "pair_count": pair_count,
        "negatives_per_query": NEGATIVES_PER_QUERY,
        "precision_at_1": round(top1_hits / pair_count, 4),
        "positive_recall_at_threshold": round(threshold_positive_hits / pair_count, 4),
        "threshold_accuracy": round((threshold_positive_hits + threshold_negative_hits) / (2 * pair_count), 4),
        "avg_positive_score": round(sum(positive_scores) / len(positive_scores), 4),
        "avg_negative_score": round(sum(negative_scores) / len(negative_scores), 4),
    }

    return {
        "dataset": {
            "source_path": str(resolve_dataset_path()),
            "rows": int(len(df)),
        },
        "protocol": {
            "description": (
                "Preliminary duplicate-retrieval sanity check using high-overlap, non-identical complaint pairs "
                "mined within the same label/priority group. Each positive query is evaluated against one mined "
                "duplicate plus distractors from other categories, and an equal number of negative-only queries."
            ),
            "pair_mining": {
                "similarity_model": "char_wb_tfidf_3_5",
                "min_similarity": PAIR_MIN_SIMILARITY,
                "max_pairs": MAX_PAIRS,
                "min_text_len": MIN_TEXT_LEN,
            },
            "duplicate_threshold": 0.82,
        },
        "engine": {
            "class_name": "SemanticDuplicateEngine",
            "method_counts": method_counts,
        },
        "metrics": metrics,
        "examples": examples,
    }


def main() -> None:
    df = load_dataset()
    pairs = mine_near_duplicate_pairs(df)
    report = evaluate_pairs(df, pairs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved duplicate evaluation to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
