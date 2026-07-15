from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SGS_")

    data_dir: Path = Path("../../data")
    sqlite_path: Path = Path("../../data/index/same_girl.sqlite3")
    faiss_path: Path = Path("../../data/index/faces.faiss")
    model_name: str = "buffalo_l"
    det_size: int = 640
    det_thresh: float = 0.1
    embedding_dim: int = 512
    cors_origins: str = "http://localhost:3000"


def get_settings() -> Settings:
    settings = Settings()
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.faiss_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

