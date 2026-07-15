from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]
    det_score: float


@dataclass(frozen=True)
class FaceEmbeddingResult:
    embedding: np.ndarray
    detection: FaceDetection
    face_count: int


class NoFaceDetectedError(ValueError):
    pass


def deterministic_mock_embedding(group_index: int, variant_index: int, dimension: int) -> np.ndarray:
    base_rng = np.random.default_rng(10_000 + group_index)
    noise_rng = np.random.default_rng(20_000 + group_index * 100 + variant_index)
    base = base_rng.normal(size=dimension).astype(np.float32)
    noise = noise_rng.normal(scale=0.035, size=dimension).astype(np.float32)
    vector = base + noise
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return (vector / norm).astype(np.float32)


def _read_png_text(image_bytes: bytes, keyword: str) -> str | None:
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return None

    offset = 8
    keyword_bytes = keyword.encode("latin-1")
    while offset + 12 <= len(image_bytes):
        length = int.from_bytes(image_bytes[offset : offset + 4], "big")
        chunk_type = image_bytes[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        if data_end + 4 > len(image_bytes):
            return None
        data = image_bytes[data_start:data_end]
        if chunk_type == b"tEXt":
            key, separator, value = data.partition(b"\x00")
            if separator and key == keyword_bytes:
                return value.decode("latin-1")
        offset = data_end + 4
    return None


class MockEmbedder:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def embed_image_path(self, image_path: str | Path) -> FaceEmbeddingResult:
        return self.embed_image_bytes(Path(image_path).read_bytes())

    def embed_image_bytes(self, image_bytes: bytes) -> FaceEmbeddingResult:
        marker = _read_png_text(image_bytes, "sgs_mock")
        if marker is None:
            raise NoFaceDetectedError("No mock embedding metadata found")
        try:
            group_index_text, variant_index_text = marker.split(",", 1)
            group_index = int(group_index_text)
            variant_index = int(variant_index_text)
        except ValueError as exc:
            raise ValueError("Invalid mock embedding metadata") from exc

        return FaceEmbeddingResult(
            embedding=deterministic_mock_embedding(group_index, variant_index, self.dimension),
            detection=FaceDetection(bbox=(36.0, 36.0, 156.0, 156.0), det_score=1.0),
            face_count=1,
        )


class InsightFaceEmbedder:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        det_thresh: float = 0.5,
        providers: Iterable[str] | None = None,
    ):
        from insightface.app import FaceAnalysis

        self.model_name = model_name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.providers = list(providers or ["CPUExecutionProvider"])
        self._app = FaceAnalysis(name=model_name, providers=self.providers)
        self._app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)

    def embed_image_path(self, image_path: str | Path) -> FaceEmbeddingResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.embed_bgr(image)

    def embed_image_bytes(self, image_bytes: bytes) -> FaceEmbeddingResult:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Uploaded file is not a readable image")
        return self.embed_bgr(image)

    def embed_bgr(self, image: np.ndarray) -> FaceEmbeddingResult:
        faces = self._app.get(image)
        if not faces:
            raise NoFaceDetectedError("No face detected")

        face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("InsightFace returned a zero-length embedding")
        embedding = embedding / norm

        return FaceEmbeddingResult(
            embedding=embedding,
            detection=FaceDetection(
                bbox=tuple(float(v) for v in face.bbox.tolist()),
                det_score=float(face.det_score),
            ),
            face_count=len(faces),
        )


@lru_cache(maxsize=1)
def get_embedder(
    model_name: str = "buffalo_l", det_size: int = 640, det_thresh: float = 0.5
) -> InsightFaceEmbedder | MockEmbedder:
    if model_name == "mock":
        return MockEmbedder()
    return InsightFaceEmbedder(model_name=model_name, det_size=det_size, det_thresh=det_thresh)
