# same-girl-search

画像から同一人物の可能性が高いプロフィールを横断検索するための OSS MVP です。

このリポジトリは、ローカルで実行できる end-to-end の実装を優先しています。アップロード画像から顔 embedding を生成し、SQLite にメタデータ、FAISS にベクトルを保存して、FastAPI と Next.js UI から近傍検索します。

## Scope

MVP でできること:

- 単一フォルダの画像をインデックスする
- `insightface` + `onnxruntime` で顔検出と face embedding を作成する
- FAISS による cosine similarity 検索を行う
- SQLite に画像パス、サイト、店舗名、プロフィール URL、顔 bbox を保存する
- Web UI から画像をアップロードして top-k 類似候補を見る
- Admin mode を有効にした Web UI から indexed image を削除する
- Mock seed 画像では bbox overlay を表示する

まだ含めないこと:

- サイト別 crawler の本実装
- 自動スクレイピング
- clustering / grouping
- duplicate suppression
- active learning

## Ethics and Lawful Use

このプロジェクトは、公開画像を無制限に収集・追跡するためのものではありません。運用時は必ず以下を守ってください。

- 各サイトの Terms of Service と robots.txt を確認する
- 明示的な rate limit と User-Agent を設定する
- 削除依頼に対応できるよう、ソース単位・画像単位で削除可能にする
- 必要以上の個人情報を保存しない
- 顔認識結果は「同一人物の可能性」であり、断定に使わない
- 法域ごとの個人情報保護法、肖像権、プライバシー権を確認する

## Repository Layout

```text
apps/
  api/                 FastAPI backend
  web/                 Next.js frontend
libs/
  embedding/           insightface embedding pipeline
  vector_search/       FAISS index manager
  crawler/             lawful crawler interfaces and robots helpers
data/
  images/              local image corpus
  index/               sqlite and FAISS files
docs/
```

## Requirements

- Docker and Docker Compose
- Optional local development:
  - Python 3.11
  - Node.js 20+

## Quick Start

```bash
docker compose up --build
```

Open:

- Web UI: <http://localhost:3000>
- API docs: <http://localhost:8000/docs>

The first embedding call downloads InsightFace model files into the container cache. This can take a while.

Compose includes healthchecks for both services. The web container waits for the API healthcheck before starting.

## Seed Deterministic Mock Data

For UI/API development without collecting real-person imagery, seed deterministic placeholder images and embeddings:

```bash
docker compose run --rm api python scripts/seed_mock.py
```

This creates PNG placeholders under `data/images/mock_seed`, inserts metadata and deterministic embeddings into SQLite, and rebuilds the FAISS index. The generated images are non-person mock assets; they are intended for layout, API, and result-card testing, not for validating face-recognition quality.

Use `--reset` to clear stale mock records before reseeding (only rows with `source_site` in `mock-site-a/b/c` are removed; other indexed images are left untouched):

```bash
docker compose run --rm api python scripts/seed_mock.py --reset
```

For local development without InsightFace installed, run the API in mock embedding mode:

```powershell
.\.venv\Scripts\python.exe .\apps\api\scripts\seed_mock.py `
  --data-dir .\data `
  --sqlite-path .\data\index\same_girl.sqlite3 `
  --faiss-path .\data\index\faces.faiss `
  --reset
```

```powershell
$env:SGS_DATA_DIR = (Resolve-Path '.\data').Path
$env:SGS_SQLITE_PATH = (Resolve-Path '.\data\index\same_girl.sqlite3').Path
$env:SGS_FAISS_PATH = (Resolve-Path '.\data\index\faces.faiss').Path
$env:SGS_MODEL_NAME = 'mock'
$env:PYTHONPATH = (Resolve-Path '.').Path + ';' + (Resolve-Path '.\libs').Path
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir apps\api --host 127.0.0.1 --port 8001
```

Then start the web UI against that local API:

```powershell
cd apps\web
$env:NEXT_PUBLIC_API_BASE_URL = 'http://127.0.0.1:8001'
npm run dev -- -p 3001
```

Smoke test the running API:

```powershell
.\.venv\Scripts\python.exe .\apps\api\scripts\smoke_test.py `
  --base-url http://127.0.0.1:8001 `
  --query-image .\data\images\mock_seed\group-01-variant-01.png
```

## GitHub Pages

This repository includes a static project page under `docs/` and a deployment workflow at `.github/workflows/pages.yml`.

To publish it:

1. Push the repository to GitHub.
2. Open `Settings` -> `Pages`.
3. Set `Source` to `GitHub Actions`.
4. Push to `main` or run the `Deploy GitHub Pages` workflow manually.

The site will be available at:

```text
https://<owner>.github.io/<repo>/
```

## Index Local Images

Put images under `data/images`.

Example folder layout:

```text
data/images/
  site-a/shop-1/profile-001.jpg
  site-a/shop-1/profile-002.jpg
```

Run indexing:

```bash
docker compose run --rm api python scripts/index_folder.py \
  --image-dir /workspace/data/images \
  --site local \
  --shop default
```

You can also call the API:

```bash
curl -X POST http://localhost:8000/index/folder \
  -H "Content-Type: application/json" \
  -d '{"image_dir":"/workspace/data/images","source_site":"local","shop_name":"default"}'
```

## Search

From the UI, drag or select an image and submit.

API example:

```bash
curl -X POST "http://localhost:8000/search?top_k=10" \
  -F "file=@query.jpg"
```

## Metadata

SQLite stores:

- image path
- source site
- shop name
- profile URL
- detected face bbox
- embedding vector as a BLOB
- timestamps

FAISS stores normalized embeddings for cosine similarity via inner product search.

## Development

API:

```bash
cd apps/api
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Web:

```bash
cd apps/web
npm install
npm run dev
```

Tests:

```bash
python -m unittest discover tests
cd apps/web
npm run lint
npm run build
```

Docker config check:

```bash
docker compose config
```

## API Endpoints

- `GET /health`
- `GET /index/stats`
- `POST /index/folder`
- `POST /search`
- `DELETE /images/{image_id}`
- `POST /admin/rebuild-faiss`

## Scaling Path

The current implementation keeps metadata in SQLite and vectors in FAISS files. The intended growth path is:

1. Add crawler implementations under `libs/crawler`.
2. Add source-level deletion and ingestion jobs.
3. Add clustering over existing embeddings.
4. Move storage to Postgres or object storage when the corpus grows.
5. Add ANN benchmark scripts before changing the vector index type.
