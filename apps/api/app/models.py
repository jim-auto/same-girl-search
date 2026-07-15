from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class IndexFolderRequest(BaseModel):
    image_dir: str
    source_site: str = "local"
    shop_name: str = "default"
    profile_url_prefix: str | None = None
    recursive: bool = True


class IndexedImage(BaseModel):
    id: int
    image_path: str
    source_site: str
    shop_name: str
    profile_url: str | None
    bbox: list[float]
    det_score: float
    face_count: int
    width: int
    height: int


class IndexFolderResponse(BaseModel):
    indexed: int
    skipped: int
    errors: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    rank: int
    score: float
    image: IndexedImage


class SearchResponse(BaseModel):
    query: dict
    results: list[SearchResult]

