from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class FaissSearchResult:
    image_id: int
    score: float
    rank: int


class FaissVectorStore:
    def __init__(self, index_path: str | Path, dimension: int):
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.ids_path = self.index_path.with_suffix(self.index_path.suffix + ".ids.json")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = self._load_or_create_index()
        self.ids = self._load_ids()

    def add(self, image_id: int, embedding: np.ndarray) -> None:
        vector = self._prepare_vector(embedding)
        self.index.add(vector)
        self.ids.append(int(image_id))
        self.save()

    def search(self, embedding: np.ndarray, top_k: int) -> list[FaissSearchResult]:
        if self.index.ntotal == 0:
            return []

        vector = self._prepare_vector(embedding)
        limit = min(max(top_k, 1), self.index.ntotal)
        scores, indices = self.index.search(vector, limit)
        results: list[FaissSearchResult] = []
        for rank, (score, faiss_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if faiss_idx < 0:
                continue
            results.append(FaissSearchResult(image_id=self.ids[int(faiss_idx)], score=float(score), rank=rank))
        return results

    def rebuild(self, rows: list[tuple[int, np.ndarray]]) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.ids = []
        if rows:
            vectors = np.vstack([self._prepare_vector(vector)[0] for _, vector in rows]).astype(np.float32)
            self.index.add(vectors)
            self.ids = [int(image_id) for image_id, _ in rows]
        self.save()

    def remove_by_id(self, image_id: int, remaining_rows: list[tuple[int, np.ndarray]]) -> None:
        self.rebuild(remaining_rows)

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.ids_path.write_text(json.dumps(self.ids, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_or_create_index(self):
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.d != self.dimension:
                raise ValueError(f"FAISS dimension mismatch: expected {self.dimension}, got {index.d}")
            return index
        return faiss.IndexFlatIP(self.dimension)

    def _load_ids(self) -> list[int]:
        if self.ids_path.exists():
            return [int(v) for v in json.loads(self.ids_path.read_text(encoding="utf-8"))]
        return []

    def _prepare_vector(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.array(embedding, dtype=np.float32, copy=True).reshape(1, -1)
        if vector.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {vector.shape[1]}")
        faiss.normalize_L2(vector)
        return vector
