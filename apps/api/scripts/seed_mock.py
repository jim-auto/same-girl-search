from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from contextlib import closing
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
API_DIR = ROOT / "apps" / "api"
for path in (ROOT, API_DIR, ROOT / "libs"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.config import get_settings
from app.db import connect, list_embeddings, migrate, upsert_image
from embedding import deterministic_mock_embedding
from vector_search import FaissVectorStore


MOCK_GROUPS = (
    ("mock-site-a", "demo-red", (214, 62, 76)),
    ("mock-site-b", "demo-teal", (32, 148, 139)),
    ("mock-site-c", "demo-gold", (224, 164, 58)),
)

MOCK_IMAGE_SIZE = 192


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_demo_png(
    path: Path,
    *,
    rgb: tuple[int, int, int],
    label_index: int,
    group_index: int,
    variant_index: int,
    size: int = MOCK_IMAGE_SIZE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    r, g, b = rgb
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            band = 28 if ((x // 24) + (y // 24) + label_index) % 2 == 0 else -18
            accent = 42 if abs(x - y) < 5 or abs((size - x) - y) < 5 else 0
            row.extend(
                (
                    max(0, min(255, r + band + accent)),
                    max(0, min(255, g + band)),
                    max(0, min(255, b + band - accent // 2)),
                )
            )
        rows.append(bytes(row))

    raw = b"".join(rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + _png_chunk(b"tEXt", f"sgs_mock\x00{group_index},{variant_index}".encode("latin-1"))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _embedding(group_index: int, variant_index: int, dimension: int) -> np.ndarray:
    return deterministic_mock_embedding(group_index, variant_index, dimension)


def seed_mock(
    *,
    data_dir: Path,
    sqlite_path: Path,
    faiss_path: Path,
    embedding_dim: int,
    variants_per_group: int = 4,
    reset: bool = False,
) -> dict:
    image_dir = data_dir / "images" / "mock_seed"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    inserted = 0

    with closing(connect(sqlite_path)) as conn:
        migrate(conn)
        if reset:
            mock_sites = [source_site for source_site, _, _ in MOCK_GROUPS]
            conn.executemany("DELETE FROM images WHERE source_site = ?", [(site,) for site in mock_sites])
            conn.commit()
            for path in image_dir.glob("group-*-variant-*.png"):
                if path.is_file():
                    path.unlink()

        for group_index, (source_site, shop_name, rgb) in enumerate(MOCK_GROUPS):
            for variant_index in range(variants_per_group):
                filename = f"group-{group_index + 1:02d}-variant-{variant_index + 1:02d}.png"
                image_path = image_dir / filename
                _write_demo_png(
                    image_path,
                    rgb=rgb,
                    label_index=group_index * variants_per_group + variant_index,
                    group_index=group_index,
                    variant_index=variant_index,
                )
                upsert_image(
                    conn,
                    image_path=str(image_path.resolve()),
                    source_site=source_site,
                    shop_name=shop_name,
                    profile_url=None,
                    bbox=[36.0, 36.0, 156.0, 156.0],
                    det_score=1.0,
                    face_count=1,
                    width=MOCK_IMAGE_SIZE,
                    height=MOCK_IMAGE_SIZE,
                    embedding=_embedding(group_index, variant_index, embedding_dim),
                )
                inserted += 1

        rows = list_embeddings(conn)

    FaissVectorStore(faiss_path, embedding_dim).rebuild(rows)
    return {
        "inserted": inserted,
        "total_vectors": len(rows),
        "reset": reset,
        "image_dir": str(image_dir),
        "sqlite": str(sqlite_path),
        "faiss": str(faiss_path),
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Seed deterministic mock images and embeddings.")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--sqlite-path", type=Path, default=settings.sqlite_path)
    parser.add_argument("--faiss-path", type=Path, default=settings.faiss_path)
    parser.add_argument("--embedding-dim", type=int, default=settings.embedding_dim)
    parser.add_argument("--variants-per-group", type=int, default=4)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing mock-seed DB rows (source_site in mock-site-a/b/c) and mock_seed PNGs before reseeding. Leaves other indexed images untouched.",
    )
    args = parser.parse_args()

    result = seed_mock(
        data_dir=args.data_dir,
        sqlite_path=args.sqlite_path,
        faiss_path=args.faiss_path,
        embedding_dim=args.embedding_dim,
        variants_per_group=args.variants_per_group,
        reset=args.reset,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
