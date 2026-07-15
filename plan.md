# same-girl-search Development Plan

## Current Status

This repository is an OSS MVP for local image-to-image face similarity search. The intended local stack is:

- FastAPI backend
- Next.js frontend
- SQLite metadata store
- FAISS vector index
- InsightFace + ONNX Runtime embedding pipeline
- Docker Compose for end-to-end local execution

The app can be built and run locally with Docker Desktop where Docker is available. A small synthetic face corpus has been downloaded from Wikimedia Commons, indexed through the real Docker API container, and searched successfully. The current shell does not have `docker` on PATH, so Docker checks are documented but not rerun in this session.

## Implemented So Far

### Backend

- FastAPI app exists under `apps/api/app`.
- Main endpoints:
  - `GET /health`
  - `GET /index/stats`
  - `POST /index/folder`
  - `POST /search`
  - `GET /images/{image_id}/file`
  - `DELETE /images/{image_id}`
- `POST /admin/rebuild-faiss`
- SQLite schema and helper functions exist in `apps/api/app/db.py`.
- FAISS wrapper exists in `libs/vector_search`.
- InsightFace embedding pipeline exists in `libs/embedding`.
- `SGS_MODEL_NAME=mock` now uses deterministic mock embeddings from seeded PNG metadata, so local UI/API demos can run without InsightFace installed.
- `index_folder.py` can now run in Docker because `apps/api` is added to `sys.path`.
- `seed_mock.py` creates deterministic placeholder PNGs and embeddings, inserts them into SQLite, and rebuilds FAISS for safe demo data.
- `seed_mock.py --reset` clears existing DB rows first, producing a deterministic mock-only demo index.
- `smoke_test.py` verifies a running API via health, FAISS rebuild, invalid upload rejection, and optional query-image search.
- SQLite connections in API routes are closed with `contextlib.closing`, avoiding Windows file-lock problems and long-lived handles.

### Frontend

- Next.js UI exists under `apps/web`.
- UI supports image selection/drop, top-k input, search submission, preview, query metadata, and result cards.
- UI now distinguishes API connection errors, empty index, no search yet, and no-result searches.
- UI shows index vector count and a visible lawful-use / mock-data notice.
- Result cards include an image open action.
- Result cards include delete actions when admin mode is enabled, and deletion refreshes index stats.
- Result cards include a bbox overlay toggle. Current overlay scaling is correct for 192x192 mock seed images; real-image precision needs image dimension metadata.
- `next` has been updated from `15.1.4` to `15.5.18`.
- `package-lock.json` has been generated.
- `eslint.config.mjs` has been added.
- `npm run lint` is now non-interactive via `eslint .`.
- `next.config.ts` sets `outputFileTracingRoot` to the repo root.
- `apps/web/.dockerignore` excludes `node_modules`, `.next`, and local dev logs from Docker build context.
- `apps/web/Dockerfile` uses `npm ci` for reproducible container installs.

### Package Naming Fix

The original repo had `libs/vector-search`, but Python imports used `vector_search`.

This was fixed by moving the package to:

```text
libs/vector_search/
```

Docs and README references were updated to match.

### Tests

The test suite now uses Python standard `unittest`, avoiding extra test dependencies.

Added:

- `tests/test_db.py`
  - SQLite migrate/upsert/list/delete behavior
- `tests/test_vector_store.py`
  - FAISS wrapper behavior using a fake in-memory FAISS module
- `tests/test_api.py`
  - FastAPI endpoint tests with fake external dependencies
  - Covers health, search success, search validation errors, image file retrieval, delete, and rebuild
- `tests/test_seed_mock.py`
  - Mock seed script behavior using a fake in-memory FAISS module
  - Covers generated image files, DB rows, embedding dimensions, and FAISS id output

Current passing commands:

```bash
python -m unittest discover tests
python -m compileall apps/api libs tests
cd apps/web
npm run lint
npm run build
```

### Docker

Docker Desktop has been installed per-user on this Windows machine.

Verified:

```bash
docker --version
docker compose version
docker run --rm hello-world
docker compose config
docker compose build
docker compose up -d
```

Verified services:

- Web: `http://localhost:3000`
- API health: `http://localhost:8000/health`

The Docker build completed successfully for both `api` and `web`.

Recent Docker dev updates:

- Root `.dockerignore` added to reduce API build context size.
- API healthcheck added via `GET /health`.
- Web healthcheck added via local page fetch.
- Web service now waits for API healthcheck in Compose.

### Synthetic Test Images

Synthetic / AI-generated face images were downloaded from Wikimedia Commons into:

```text
data/images/synthetic/commons/
```

Attribution and license metadata is recorded in:

```text
data/images/synthetic/commons/ATTRIBUTION.md
```

The real Docker API container was used to index the corpus:

```bash
docker compose run --rm api python scripts/index_folder.py \
  --image-dir /workspace/data/images/synthetic/commons \
  --site commons \
  --shop synthetic
```

Result:

- `indexed`: 1
- `skipped`: 6
- `errors`: []

Only `Woman_1.jpg` was detected by InsightFace in the current set. Searching with that image returned a successful self-match with score `1.0`.

## Safety and Data Policy

This project should not be used to collect, track, or identify real people from adult services, social media, or other public websites without explicit consent.

Allowed and recommended:

- Synthetic AI-generated faces
- Public-domain or permissively licensed synthetic images
- User-provided images where consent and usage scope are clear
- Mock data for UI/API development
- A seed/demo mode that avoids real biometric data

The README already includes lawful-use guidance. Future work should keep this limitation explicit.

## Known Issues

### Low Detection Rate on Synthetic Commons Images

The first synthetic corpus contains 7 images, but InsightFace detected only 1 face.

Likely causes:

- Some files are illustrations or stylized faces.
- Some generated faces may be too artificial for the detector.
- Some images may not have frontal, high-confidence faces.

Next step should be to use a better synthetic face source or generate controlled face images locally.

### Docker Web Container Runs Dev Server

`apps/web/Dockerfile` currently runs:

```dockerfile
CMD ["npm", "run", "dev"]
```

This is acceptable for local MVP development, but not ideal for production-like usage.

Potential improvement:

- Use `npm ci`
- Run `npm run build`
- Start with `next start` or standalone output

This should be done after deciding whether Compose is meant for development or production-like demo.

### NPM Audit Warnings

`npm audit` still reports Next/PostCSS-related moderate findings. The automatic fix suggests an invalid/breaking downgrade in this environment, so it was not applied.

Current build and lint pass.

### InsightFace Model Download

The first real embedding call downloads `buffalo_l` into the container cache. This takes time and may be repeated depending on container/cache lifecycle.

Potential improvement:

- Mount a Docker volume for InsightFace model cache
- Document the first-run delay
- Consider a smaller model if acceptable

## Next Recommended Work

### 1. Build a Better Safe Demo Corpus

Goal: produce enough detectable synthetic faces for useful search demos.

Options:

- Generate synthetic portraits using an image generation tool, with consistent front-facing faces.
- Use only explicitly synthetic / public-domain faces.
- Keep attribution or generation notes in `data/images/synthetic/...`.

Acceptance criteria:

- At least 10 images indexed successfully.
- At least 3 groups with visually similar variants if possible.
- `/search` returns multiple candidates for a query.
- No real adult-service or scraped personal images are used.

### 2. Add a Mock Seed Mode

Goal: allow demo and UI development without any real face model or images.

Status: implemented.

Implemented design:

- `apps/api/scripts/seed_mock.py`
- Inserts placeholder image metadata and deterministic embeddings into SQLite.
- Rebuilds FAISS from all stored embeddings.
- Generates simple non-human PNG placeholder assets under `data/images/mock_seed`.
- Embeds mock metadata in those PNGs so `/search` can use the same deterministic vectors when `SGS_MODEL_NAME=mock`.
- Supports `--reset` to clear existing DB rows before seeding, making demos reproducible.

Acceptance criteria:

- `docker compose run --rm api python scripts/seed_mock.py` creates a searchable local index.
- `docker compose run --rm api python scripts/seed_mock.py --reset` creates a mock-only local index.
- Local API can run with `SGS_MODEL_NAME=mock` without installing InsightFace.
- Tests verify seed output shape.

### 3. Improve Docker Dev Experience

Potential changes:

Status: partially implemented.

Implemented:

- Root `.dockerignore` for API build context.
- Kept `apps/web/.dockerignore`.
- Use `npm ci` instead of `npm install` in Web Dockerfile.
- Added API and Web healthchecks to Compose.
- Web waits for API healthcheck.

Still open:

- Decide whether web container should run dev mode or production-like mode.

Acceptance criteria:

- `docker compose config` validates the updated Compose file when Docker is available.
- `docker compose build` sends smaller contexts.
- `docker compose up -d` starts both services reliably.
- Healthcheck status reflects API and Web readiness.

### 4. Add API Smoke Script

Goal: simple command to verify a running Compose stack.

Status: implemented.

File:

```text
apps/api/scripts/smoke_test.py
```

Checks:

- `GET /health`
- `POST /admin/rebuild-faiss`
- `POST /search` rejects non-image upload with 400
- Optional: search with known synthetic/mock image if indexed

Acceptance criteria:

- Can run from host or container.
- Non-zero exit on failure.
- Useful output for debugging.

### 5. Improve Search Result UX

Frontend improvements:

Status: partially implemented.

Implemented:

- Show empty-index state separately from no results.
- Show API connection error clearly.
- Add result image open button.
- Show index vector count.
- Add visible lawful-use / mock-data notice near the top of the UI.

Still open:

- Store image dimensions during indexing and use them for precise bbox overlay on arbitrary real images.

Acceptance criteria:

- UI remains compact and task-focused.
- Mobile layout does not overlap or overflow.
- `npm run lint` and `npm run build` pass.

## Immediate Next Step

The best next step is now:

```text
Finish the remaining search result UX controls.
```

Mock seed mode, reset seeding, API smoke script, Docker polish, core frontend states, admin delete controls, and mock bbox overlay are now implemented. The next useful work is image dimension metadata for precise overlays on arbitrary images.

Recommended next task:

```text
Store image dimensions in SQLite and return them from the API.
```

## Commands Reference

### Run Tests

```bash
python -m unittest discover tests
python -m compileall apps/api libs tests
cd apps/web
npm run lint
npm run build
```

### Run Docker Stack

```bash
docker compose up -d
docker compose ps
curl http://localhost:8000/health
```

### Smoke Test Running API

```bash
python apps/api/scripts/smoke_test.py --base-url http://localhost:8000
```

With mock seed data:

```bash
python apps/api/scripts/smoke_test.py \
  --base-url http://localhost:8000 \
  --query-image data/images/mock_seed/group-01-variant-01.png
```

### Index Synthetic Commons Images

```bash
docker compose run --rm api python scripts/index_folder.py \
  --image-dir /workspace/data/images/synthetic/commons \
  --site commons \
  --shop synthetic
```

### Search with Indexed Synthetic Image

PowerShell:

```powershell
$file = Resolve-Path 'data\images\synthetic\commons\Woman_1.jpg'
curl.exe -X POST "http://localhost:8000/search?top_k=5" -F "file=@$file"
```

### Stop Docker Stack

```bash
docker compose down
```
