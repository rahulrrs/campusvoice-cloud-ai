from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import hnswlib
except Exception:
    hnswlib = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class SemanticDuplicateEngine:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        duplicate_threshold: float = 0.82,
        ann_k: int = 5,
    ) -> None:
        self.model_name = model_name
        self.duplicate_threshold = duplicate_threshold
        self.ann_k = ann_k
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:
            return None
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def _semantic_search(self, text: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        model = self._load_model()
        if model is None:
            raise RuntimeError("sentence-transformers unavailable")

        candidate_texts = [
            f"{row.get('title', '')}\n\n{row.get('description', '')}".strip()
            for row in rows
        ]
        embeddings = model.encode(
            [text, *candidate_texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query = embeddings[0]
        corpus = embeddings[1:]

        if corpus.size == 0:
            return {
                "is_duplicate": False,
                "score": 0.0,
                "method": "sentence-transformer+hnsw" if hnswlib is not None else "sentence-transformer+cosine",
                "matches": [],
            }

        if hnswlib is not None and len(rows) >= 8:
            index = hnswlib.Index(space="cosine", dim=int(corpus.shape[1]))
            index.init_index(max_elements=len(rows), ef_construction=100, M=16)
            index.add_items(corpus, np.arange(len(rows)))
            index.set_ef(max(self.ann_k, 20))
            ids, distances = index.knn_query(query, k=min(self.ann_k, len(rows)))
            idxs = ids[0].tolist()
            sims = [float(1.0 - dist) for dist in distances[0].tolist()]
            method = "sentence-transformer+hnsw"
        else:
            similarities = corpus @ query
            idxs = np.argsort(-similarities)[: self.ann_k].tolist()
            sims = [float(similarities[idx]) for idx in idxs]
            method = "sentence-transformer+cosine"

        matches: list[dict[str, Any]] = []
        top_score = 0.0
        for idx, score in zip(idxs, sims):
            row = rows[idx]
            top_score = max(top_score, score)
            if score < 0.55:
                continue
            matches.append(
                {
                    "id": row["id"],
                    "title": row.get("title"),
                    "category": row.get("category"),
                    "status": row.get("status"),
                    "score": round(score, 4),
                }
            )

        return {
            "is_duplicate": top_score >= self.duplicate_threshold,
            "score": round(top_score, 4),
            "method": method,
            "matches": matches,
        }

    def _tfidf_fallback(self, text: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        candidate_texts = [
            f"{row.get('title', '')}\n\n{row.get('description', '')}".strip()
            for row in rows
        ]
        corpus = [text, *candidate_texts]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        ranked_indices = np.argsort(-similarities)[: self.ann_k]
        matches: list[dict[str, Any]] = []
        top_score = 0.0
        for idx in ranked_indices.tolist():
            score = float(similarities[idx])
            row = rows[idx]
            top_score = max(top_score, score)
            if score < 0.45:
                continue
            matches.append(
                {
                    "id": row["id"],
                    "title": row.get("title"),
                    "category": row.get("category"),
                    "status": row.get("status"),
                    "score": round(score, 4),
                }
            )
        return {
            "is_duplicate": top_score >= self.duplicate_threshold,
            "score": round(top_score, 4),
            "method": "tfidf-fallback",
            "matches": matches,
        }

    def search(self, text: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not str(text or "").strip() or not rows:
            return {
                "is_duplicate": False,
                "score": 0.0,
                "method": "sentence-transformer+hnsw" if SentenceTransformer is not None and hnswlib is not None else "tfidf-fallback",
                "matches": [],
            }
        try:
            return self._semantic_search(text, rows)
        except Exception:
            return self._tfidf_fallback(text, rows)
