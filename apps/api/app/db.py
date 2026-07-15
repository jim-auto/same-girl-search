from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np


def connect(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL UNIQUE,
            source_site TEXT NOT NULL,
            shop_name TEXT NOT NULL,
            profile_url TEXT,
            bbox_json TEXT NOT NULL,
            det_score REAL NOT NULL,
            face_count INTEGER NOT NULL,
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(images)")}
    for column in ("width", "height"):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE images ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_source_shop ON images(source_site, shop_name)")
    conn.commit()


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def upsert_image(
    conn: sqlite3.Connection,
    *,
    image_path: str,
    source_site: str,
    shop_name: str,
    profile_url: str | None,
    bbox: Iterable[float],
    det_score: float,
    face_count: int,
    width: int,
    height: int,
    embedding: np.ndarray,
) -> int:
    embedding_dim = int(np.asarray(embedding).shape[0])
    conn.execute(
        """
        INSERT INTO images (
            image_path, source_site, shop_name, profile_url, bbox_json,
            det_score, face_count, width, height, embedding, embedding_dim
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(image_path) DO UPDATE SET
            source_site=excluded.source_site,
            shop_name=excluded.shop_name,
            profile_url=excluded.profile_url,
            bbox_json=excluded.bbox_json,
            det_score=excluded.det_score,
            face_count=excluded.face_count,
            width=excluded.width,
            height=excluded.height,
            embedding=excluded.embedding,
            embedding_dim=excluded.embedding_dim,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            image_path,
            source_site,
            shop_name,
            profile_url,
            json.dumps(list(bbox)),
            float(det_score),
            int(face_count),
            int(width),
            int(height),
            embedding_to_blob(embedding),
            embedding_dim,
        ),
    )
    row = conn.execute("SELECT id FROM images WHERE image_path = ?", (image_path,)).fetchone()
    conn.commit()
    return int(row["id"])


def get_image(conn: sqlite3.Connection, image_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()


def list_embeddings(conn: sqlite3.Connection) -> list[tuple[int, np.ndarray]]:
    rows = conn.execute("SELECT id, embedding FROM images ORDER BY id ASC").fetchall()
    return [(int(row["id"]), blob_to_embedding(row["embedding"])) for row in rows]


def delete_image(conn: sqlite3.Connection, image_id: int) -> bool:
    cursor = conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    conn.commit()
    return cursor.rowcount > 0


def row_to_indexed_image(row: sqlite3.Row) -> dict:
    return {
        "id": int(row["id"]),
        "image_path": row["image_path"],
        "source_site": row["source_site"],
        "shop_name": row["shop_name"],
        "profile_url": row["profile_url"],
        "bbox": json.loads(row["bbox_json"]),
        "det_score": float(row["det_score"]),
        "face_count": int(row["face_count"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
    }

