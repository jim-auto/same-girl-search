# same-girl-search Current Status for Review

## Project Summary

`same-girl-search` is a local OSS MVP for image-to-image face similarity search.

Intended stack:

- FastAPI backend
- Next.js frontend
- SQLite metadata store
- FAISS vector index
- InsightFace + ONNX Runtime for real face embeddings
- Docker Compose for local end-to-end execution

Safety constraint:

- Do not scrape or store identifiable real-person imagery without explicit consent.
- Demo/test data should use synthetic, permissively licensed, user-provided, or non-person mock assets.

## Current Local State

The project can currently run locally in a deterministic mock mode without InsightFace installed.

Running services in this workspace:

- API: `http://127.0.0.1:8001`
- Web: `http://127.0.0.1:3001`

Mock seed data exists under:

```text
data/images/mock_seed/
```

Local index files exist under:

```text
data/index/
```

The mock index currently has:

- `vectors`: 13
- 12 deterministic mock PNG records
- 1 previously indexed synthetic image record

## Important Constraint

`insightface==0.7.3` could not be installed in the current Windows local environment because it requires Microsoft C++ Build Tools:

```text
Microsoft Visual C++ 14.0 or greater is required
```

Workaround implemented:

- `SGS_MODEL_NAME=mock`
- Mock PNG files include deterministic embedding metadata in PNG `tEXt` chunks.
- The API can search mock seed images without importing InsightFace.
- Real InsightFace mode should still work in Docker or on a machine with the C++ toolchain installed.

## Recent Implemented Changes

### Mock Seed Mode

Added:

```text
apps/api/scripts/seed_mock.py
```

Behavior:

- Generates simple non-person PNG placeholders.
- Embeds mock metadata into PNG `tEXt` chunks.
- Inserts deterministic embeddings into SQLite.
- Rebuilds FAISS from stored embeddings.
- Supports `--reset` to clear existing DB rows first and create a mock-only demo index.

Run locally:

```powershell
.\.venv\Scripts\python.exe .\apps\api\scripts\seed_mock.py `
  --data-dir .\data `
  --sqlite-path .\data\index\same_girl.sqlite3 `
  --faiss-path .\data\index\faces.faiss `
  --reset
```

### Mock Embedder

Updated:

```text
libs/embedding/face_embedder.py
libs/embedding/__init__.py
```

Behavior:

- InsightFace import is now lazy.
- `get_embedder("mock")` returns a `MockEmbedder`.
- `MockEmbedder` reads seeded PNG metadata and produces the same deterministic embedding used by `seed_mock.py`.

This allows `/search` to work locally with mock seed images.

### FAISS Read-Only Array Fix

Updated:

```text
libs/vector_search/faiss_store.py
```

Change:

- `_prepare_vector()` now uses a writable copy:

```python
np.array(embedding, dtype=np.float32, copy=True)
```

Reason:

- SQLite BLOB-backed numpy arrays can be read-only.
- FAISS normalization mutates the array.

### API Smoke Test

Added:

```text
apps/api/scripts/smoke_test.py
tests/test_smoke_test.py
```

Checks:

- `GET /health`
- `POST /admin/rebuild-faiss`
- `POST /search` rejects `text/plain` upload with HTTP 400
- Optional query-image search

Example:

```powershell
.\.venv\Scripts\python.exe .\apps\api\scripts\smoke_test.py `
  --base-url http://127.0.0.1:8001 `
  --query-image .\data\images\mock_seed\group-01-variant-01.png `
  --json
```

Recent result:

```json
{
  "base_url": "http://127.0.0.1:8001",
  "vectors": 13,
  "search_results": 5,
  "top_score": 1.0
}
```

### Docker Dev Experience

Added:

```text
.dockerignore
```

Updated:

```text
docker-compose.yml
apps/web/Dockerfile
apps/web/.dockerignore
```

Changes:

- Root `.dockerignore` reduces API build context.
- Web Dockerfile now uses `npm ci` instead of `npm install`.
- Compose has API healthcheck using `/health`.
- Compose has Web healthcheck using a local page fetch.
- Web service waits for API healthcheck.

Note:

- Docker is not currently available on PATH in this shell, so `docker compose config/build/up` could not be rerun after these latest Docker changes.

## Tests and Verification

Passing locally:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
.\.venv\Scripts\python.exe -m compileall apps/api libs tests
cd apps\web
npm run lint
npm run build
```

Also passing against the running mock API:

```powershell
.\.venv\Scripts\python.exe .\apps\api\scripts\smoke_test.py `
  --base-url http://127.0.0.1:8001 `
  --query-image .\data\images\mock_seed\group-01-variant-01.png `
  --json
```

## Existing Features

Backend endpoints:

- `GET /health`
- `GET /index/stats`
- `POST /index/folder`
- `POST /search`
- `GET /images/{image_id}/file`
- `DELETE /images/{image_id}`
- `POST /admin/rebuild-faiss`

Frontend:

- Drag/select image
- Query preview
- Top-k input
- Search submit
- Query metadata display
- Result cards with image, rank, score, site, shop, path
- API connection error state
- Empty-index and no-result states
- Index vector count
- Visible lawful-use / mock-data notice
- Result image open action
- Admin mode delete action with stats refresh
- Bbox overlay toggle for mock seed images

Tests:

- `tests/test_db.py`
- `tests/test_vector_store.py`
- `tests/test_api.py`
- `tests/test_seed_mock.py`
- `tests/test_smoke_test.py`

## Known Issues / Open Questions

### 1. InsightFace Local Install

Question:

- Should the project continue to treat real InsightFace mode as Docker-first, or should Windows local setup be supported by documenting/installing Microsoft C++ Build Tools?

Current state:

- Mock mode works locally.
- Real mode requires Docker or C++ Build Tools.

### 2. Docker Mode

Question:

- Should Compose remain a development stack using `next dev`, or shift to a production-like demo stack using `npm run build` + `next start` or standalone output?

Current state:

- Web container still runs `npm run dev`.
- Dockerfile now uses `npm ci`.
- Healthchecks were added.

### 3. Mock Mode Design

Question:

- Is embedding metadata inside PNG `tEXt` chunks acceptable for deterministic local demo data?
- Or should mock search use a separate manifest file instead?

Tradeoff:

- PNG metadata makes query images self-contained.
- A manifest may be more explicit and easier to inspect.

### 4. Search UX

Recently implemented:

- Empty-index state distinct from no-results state.
- Clear API connection error state.
- Result image open button.
- Visible index vector count.
- Visible lawful-use / mock-data notice.
- Admin mode delete button.
- Bbox overlay toggle. Current scaling assumes 192x192 mock seed images.

Still open:

- Store image dimensions so bbox overlay is precise for arbitrary real images.

### 5. Data Policy

Question:

- Should the README and UI more strongly state that this is a lawful-use/synthetic-demo MVP and not intended for scraping real people?

Current state:

- README contains lawful-use guidance.
- UI currently has a short functional description but no visible policy notice.

## Suggested Next Steps

1. Decide Docker mode:
   - development stack with `next dev`
   - or production-like demo stack with built Next output

2. Improve frontend states:
   - API connection error
   - empty index
   - no results
   - result image open action

3. Add admin-only delete UI if useful.

4. Re-run Docker checks once Docker is available on PATH:

```bash
docker compose config
docker compose build
docker compose up -d
docker compose ps
```

5. Decide whether `seed_mock.py --reset` should clear the whole DB, as it does now, or only remove records from `data/images/mock_seed`.
