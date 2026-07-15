from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBS_DIR = ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))


class _FakeIndexFlatIP:
    def __init__(self, dimension: int):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, np.asarray(vectors, dtype=np.float32)])

    def search(self, vector: np.ndarray, top_k: int):
        scores = self.vectors @ vector[0]
        order = np.argsort(-scores)[:top_k]
        return scores[order].reshape(1, -1), order.astype(np.int64).reshape(1, -1)


class _FakeFaiss(types.ModuleType):
    def __init__(self):
        super().__init__("faiss")
        self._indexes: dict[str, _FakeIndexFlatIP] = {}

    def IndexFlatIP(self, dimension: int) -> _FakeIndexFlatIP:
        return _FakeIndexFlatIP(dimension)

    def normalize_L2(self, vector: np.ndarray) -> None:
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        vector /= np.where(norm == 0, 1, norm)

    def write_index(self, index: _FakeIndexFlatIP, path: str) -> None:
        clone = _FakeIndexFlatIP(index.d)
        clone.add(index.vectors)
        self._indexes[path] = clone
        Path(path).touch()

    def read_index(self, path: str) -> _FakeIndexFlatIP:
        return self._indexes[path]


class VectorStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["faiss"] = _FakeFaiss()
        cls.module = importlib.import_module("vector_search.faiss_store")
        importlib.reload(cls.module)

    def test_rebuild_persists_ids_and_searches_by_cosine_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "faces.faiss"
            store = self.module.FaissVectorStore(index_path, dimension=3)

            store.rebuild(
                [
                    (10, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
                    (20, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
                ]
            )

            results = store.search(np.array([0.9, 0.1, 0.0], dtype=np.float32), top_k=5)
            self.assertEqual([result.image_id for result in results], [10, 20])
            self.assertEqual([result.rank for result in results], [1, 2])

            reloaded = self.module.FaissVectorStore(index_path, dimension=3)
            self.assertEqual(reloaded.ids, [10, 20])

    def test_rejects_wrong_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = self.module.FaissVectorStore(Path(temp_dir) / "faces.faiss", dimension=3)
            store.add(1, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "Embedding dimension mismatch"):
                store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)


if __name__ == "__main__":
    unittest.main()
