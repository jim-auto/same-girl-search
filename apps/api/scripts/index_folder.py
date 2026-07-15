from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
API_DIR = ROOT / "apps" / "api"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))
if str(ROOT / "libs") not in sys.path:
    sys.path.insert(0, str(ROOT / "libs"))

from app.main import index_folder
from app.models import IndexFolderRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a local image folder into SQLite + FAISS.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--site", default="local")
    parser.add_argument("--shop", default="default")
    parser.add_argument("--profile-url-prefix")
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    response = index_folder(
        IndexFolderRequest(
            image_dir=args.image_dir,
            source_site=args.site,
            shop_name=args.shop,
            profile_url_prefix=args.profile_url_prefix,
            recursive=not args.no_recursive,
        )
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
