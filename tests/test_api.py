from __future__ import annotations

import sys
import tempfile
import types
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "apps" / "api"
LIBS_DIR = ROOT / "libs"
for path in (API_DIR, LIBS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _install_external_dependency_fakes() -> None:
    faiss = types.ModuleType("faiss")
    faiss.IndexFlatIP = lambda dimension: types.SimpleNamespace(d=dimension, ntotal=0)
    faiss.read_index = lambda path: faiss.IndexFlatIP(512)
    faiss.write_index = lambda index, path: None
    faiss.normalize_L2 = lambda vector: None
    sys.modules.setdefault("faiss", faiss)

    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.imread = lambda path: None
    cv2.imdecode = lambda buffer, flags: None
    sys.modules.setdefault("cv2", cv2)

    insightface = types.ModuleType("insightface")
    insightface_app = types.ModuleType("insightface.app")

    class FaceAnalysis:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, *args, **kwargs):
            pass

        def get(self, image):
            return []

    insightface_app.FaceAnalysis = FaceAnalysis
    sys.modules.setdefault("insightface", insightface)
    sys.modules.setdefault("insightface.app", insightface_app)


_install_external_dependency_fakes()

from app import main as api_main
from app.db import connect, migrate, upsert_image


class _FakeEmbedder:
    def __init__(self, embedding: np.ndarray | None = None):
        self.embedding = embedding if embedding is not None else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def embed_image_bytes(self, image_bytes: bytes):
        if image_bytes == b"noface":
            raise api_main.NoFaceDetectedError("No face detected")
        if image_bytes == b"bad":
            raise ValueError("Uploaded file is not a readable image")
        return types.SimpleNamespace(
            embedding=self.embedding,
            detection=types.SimpleNamespace(bbox=(1.0, 2.0, 11.0, 12.0), det_score=0.99),
            face_count=1,
        )


class _FakeStore:
    def __init__(self, hits=None):
        self.hits = list(hits or [])
        self.rebuild_rows = None

    def search(self, embedding: np.ndarray, top_k: int):
        return self.hits[:top_k]

    def rebuild(self, rows):
        self.rebuild_rows = list(rows)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        api_main.settings.sqlite_path = self.root / "index.sqlite3"
        api_main.settings.faiss_path = self.root / "faces.faiss"
        api_main.settings.embedding_dim = 3
        with closing(connect(api_main.settings.sqlite_path)) as conn:
            migrate(conn)
        self.client = TestClient(api_main.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temp_dir.cleanup()

    def _insert_image(self, image_path: Path) -> int:
        with closing(connect(api_main.settings.sqlite_path)) as conn:
            migrate(conn)
            return upsert_image(
                conn,
                image_path=str(image_path),
                source_site="local",
                shop_name="default",
                profile_url=None,
                bbox=[1.0, 2.0, 11.0, 12.0],
                det_score=0.95,
                face_count=1,
                width=320,
                height=240,
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )

    def test_health_uses_configured_paths(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sqlite"], str(api_main.settings.sqlite_path))

    def test_index_stats_returns_vector_count(self) -> None:
        image_path = self.root / "result.jpg"
        image_path.write_bytes(b"image")
        self._insert_image(image_path)

        response = self.client.get("/index/stats")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"vectors": 1})

    def test_search_returns_ranked_images(self) -> None:
        image_path = self.root / "result.jpg"
        image_path.write_bytes(b"image")
        image_id = self._insert_image(image_path)
        hit = types.SimpleNamespace(image_id=image_id, score=0.88, rank=1)
        store = _FakeStore([hit])

        with patch.object(api_main, "get_embedder", return_value=_FakeEmbedder()), patch.object(
            api_main, "get_store", return_value=store
        ):
            response = self.client.post("/search?top_k=5", files={"file": ("query.jpg", b"image", "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["query"]["face_count"], 1)
        self.assertEqual(body["results"][0]["image"]["id"], image_id)
        self.assertEqual(body["results"][0]["score"], 0.88)
        self.assertEqual(body["results"][0]["image"]["width"], 320)
        self.assertEqual(body["results"][0]["image"]["height"], 240)

    def test_search_validates_uploads_and_no_face(self) -> None:
        text_response = self.client.post("/search", files={"file": ("query.txt", b"text", "text/plain")})
        self.assertEqual(text_response.status_code, 400)

        with patch.object(api_main, "get_embedder", return_value=_FakeEmbedder()):
            no_face = self.client.post("/search", files={"file": ("query.jpg", b"noface", "image/jpeg")})
            bad_image = self.client.post("/search", files={"file": ("query.jpg", b"bad", "image/jpeg")})

        self.assertEqual(no_face.status_code, 422)
        self.assertEqual(bad_image.status_code, 400)

    def test_image_file_delete_and_rebuild(self) -> None:
        image_path = self.root / "stored.jpg"
        image_path.write_bytes(b"stored image")
        image_id = self._insert_image(image_path)
        store = _FakeStore()

        file_response = self.client.get(f"/images/{image_id}/file")
        self.assertEqual(file_response.status_code, 200)
        self.assertEqual(file_response.content, b"stored image")

        with patch.object(api_main, "get_store", return_value=store):
            rebuild_response = self.client.post("/admin/rebuild-faiss")
            delete_response = self.client.delete(f"/images/{image_id}")
            missing_response = self.client.delete(f"/images/{image_id}")

        self.assertEqual(rebuild_response.status_code, 200)
        self.assertEqual(rebuild_response.json(), {"vectors": 1})
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json(), {"removed": image_id})
        self.assertEqual(missing_response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
