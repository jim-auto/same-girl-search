from __future__ import annotations

import mimetypes
import sys
from contextlib import closing
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[3]
LIBS = ROOT / "libs"
for path in (ROOT, LIBS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embedding import NoFaceDetectedError, get_embedder
from vector_search import FaissVectorStore

from .config import get_settings
from .db import connect, delete_image, get_image, list_embeddings, migrate, row_to_indexed_image, upsert_image
from .models import IndexFolderRequest, IndexFolderResponse, SearchResponse

settings = get_settings()
app = FastAPI(title="same-girl-search API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    with closing(connect(settings.sqlite_path)) as conn:
        migrate(conn)


def get_store() -> FaissVectorStore:
    return FaissVectorStore(settings.faiss_path, settings.embedding_dim)


def image_files(image_dir: Path, recursive: bool) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(image_dir.rglob(pattern) if recursive else image_dir.glob(pattern))
    return sorted(set(files))


def image_dimensions(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sqlite": str(settings.sqlite_path), "faiss": str(settings.faiss_path)}


@app.get("/index/stats")
def index_stats() -> dict:
    with closing(connect(settings.sqlite_path)) as conn:
        rows = list_embeddings(conn)
    return {"vectors": len(rows)}


@app.post("/index/folder", response_model=IndexFolderResponse)
def index_folder(request: IndexFolderRequest) -> IndexFolderResponse:
    folder = Path(request.image_dir)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"image_dir is not a directory: {folder}")

    embedder = get_embedder(settings.model_name, settings.det_size, settings.det_thresh)
    store = get_store()
    indexed = 0
    skipped = 0
    errors: list[str] = []

    with closing(connect(settings.sqlite_path)) as conn:
        migrate(conn)
        for path in image_files(folder, request.recursive):
            try:
                result = embedder.embed_image_path(path)
                width, height = image_dimensions(path)
                profile_url = f"{request.profile_url_prefix.rstrip('/')}/{path.name}" if request.profile_url_prefix else None
                image_id = upsert_image(
                    conn,
                    image_path=str(path.resolve()),
                    source_site=request.source_site,
                    shop_name=request.shop_name,
                    profile_url=profile_url,
                    bbox=result.detection.bbox,
                    det_score=result.detection.det_score,
                    face_count=result.face_count,
                    width=width,
                    height=height,
                    embedding=result.embedding,
                )
                indexed += 1
            except NoFaceDetectedError:
                skipped += 1
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        store.rebuild(list_embeddings(conn))

    return IndexFolderResponse(indexed=indexed, skipped=skipped, errors=errors)


@app.post("/search", response_model=SearchResponse)
async def search(file: UploadFile = File(...), top_k: int = Query(10, ge=1, le=100)) -> SearchResponse:
    content_type = file.content_type or mimetypes.guess_type(file.filename or "")[0]
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    image_bytes = await file.read()
    try:
        query = get_embedder(settings.model_name, settings.det_size, settings.det_thresh).embed_image_bytes(image_bytes)
    except NoFaceDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_store()
    hits = store.search(query.embedding, top_k=top_k)
    results = []
    with closing(connect(settings.sqlite_path)) as conn:
        for hit in hits:
            row = get_image(conn, hit.image_id)
            if row is None:
                continue
            results.append({"rank": hit.rank, "score": hit.score, "image": row_to_indexed_image(row)})

    return SearchResponse(
        query={
            "bbox": list(query.detection.bbox),
            "det_score": query.detection.det_score,
            "face_count": query.face_count,
        },
        results=results,
    )


@app.get("/images/{image_id}/file")
def image_file(image_id: int):
    with closing(connect(settings.sqlite_path)) as conn:
        row = get_image(conn, image_id)
    if row is None:
        raise HTTPException(status_code=404, detail="image not found")
    path = Path(row["image_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"image file missing: {path}")
    return FileResponse(path)


@app.delete("/images/{image_id}")
def remove_image(image_id: int) -> dict:
    with closing(connect(settings.sqlite_path)) as conn:
        removed = delete_image(conn, image_id)
        if not removed:
            raise HTTPException(status_code=404, detail="image not found")
        get_store().rebuild(list_embeddings(conn))
    return {"removed": image_id}


@app.post("/admin/rebuild-faiss")
def rebuild_faiss() -> dict:
    with closing(connect(settings.sqlite_path)) as conn:
        rows = list_embeddings(conn)
    get_store().rebuild(rows)
    return {"vectors": len(rows)}
