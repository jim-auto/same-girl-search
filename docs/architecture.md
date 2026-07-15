# Architecture Notes

The MVP is intentionally direct:

- `libs/embedding` owns model loading, face detection, and normalized embedding output.
- `libs/vector_search` owns FAISS persistence and ID mapping.
- `apps/api` owns HTTP, SQLite persistence, ingestion orchestration, and deletion.
- `apps/web` owns query upload and result rendering.

The index contract is:

1. SQLite `images.id` is the canonical ID.
2. FAISS row IDs map to SQLite IDs through a JSON sidecar file.
3. Embeddings are normalized before insertion, so FAISS inner product equals cosine similarity.

Rebuild FAISS from SQLite if the JSON sidecar is missing or if records are deleted.
