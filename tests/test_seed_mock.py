from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "apps" / "api"
LIBS_DIR = ROOT / "libs"
for path in (API_DIR, LIBS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class _FakeIndexFlatIP:
    def __init__(self, dimension: int):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, np.asarray(vectors, dtype=np.float32)])


class _FakeFaiss(types.ModuleType):
    def __init__(self):
        super().__init__("faiss")

    def IndexFlatIP(self, dimension: int) -> _FakeIndexFlatIP:
        return _FakeIndexFlatIP(dimension)

    def normalize_L2(self, vector: np.ndarray) -> None:
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        vector /= np.where(norm == 0, 1, norm)

    def write_index(self, index: _FakeIndexFlatIP, path: str) -> None:
        Path(path).touch()

    def read_index(self, path: str) -> _FakeIndexFlatIP:
        return _FakeIndexFlatIP(8)


class SeedMockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["faiss"] = _FakeFaiss()
        for module_name in ("vector_search", "vector_search.faiss_store", "scripts.seed_mock"):
            sys.modules.pop(module_name, None)
        cls.module = importlib.import_module("scripts.seed_mock")
        importlib.reload(cls.module)

    def test_seed_mock_creates_images_db_rows_and_faiss_ids(self) -> None:
        from app.db import connect, list_embeddings, migrate, upsert_image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sqlite_path = root / "data" / "index" / "same_girl.sqlite3"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            conn = connect(sqlite_path)
            try:
                migrate(conn)
                upsert_image(
                    conn,
                    image_path="/outside/existing.png",
                    source_site="outside",
                    shop_name="outside",
                    profile_url=None,
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    det_score=1.0,
                    face_count=1,
                    width=64,
                    height=64,
                    embedding=np.ones(8, dtype=np.float32),
                )
            finally:
                conn.close()

            result = self.module.seed_mock(
                data_dir=root / "data",
                sqlite_path=sqlite_path,
                faiss_path=root / "data" / "index" / "faces.faiss",
                embedding_dim=8,
                variants_per_group=2,
                reset=True,
            )

            self.assertEqual(result["inserted"], 6)
            self.assertEqual(result["total_vectors"], 7)
            self.assertTrue(result["reset"])
            self.assertTrue((root / "data" / "images" / "mock_seed" / "group-01-variant-01.png").exists())
            self.assertTrue((root / "data" / "index" / "faces.faiss.ids.json").exists())

            conn = connect(sqlite_path)
            try:
                migrate(conn)
                rows = list_embeddings(conn)
                outside = conn.execute("SELECT COUNT(*) AS count FROM images WHERE source_site = 'outside'").fetchone()
                seeded = conn.execute(
                    "SELECT id, width, height FROM images WHERE image_path LIKE '%group-01-variant-01.png'"
                ).fetchone()
            finally:
                conn.close()

            self.assertEqual(len(rows), 7)
            self.assertEqual(outside["count"], 1)
            self.assertEqual((seeded["width"], seeded["height"]), (192, 192))
            self.assertEqual({embedding.shape[0] for _, embedding in rows}, {8})

            from embedding import MockEmbedder

            embedder = MockEmbedder(dimension=8)
            query = embedder.embed_image_path(root / "data" / "images" / "mock_seed" / "group-01-variant-01.png")
            seeded_embedding = dict(rows)[seeded["id"]]
            np.testing.assert_allclose(query.embedding, seeded_embedding)


if __name__ == "__main__":
    unittest.main()
