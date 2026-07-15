from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def _print_ok(message: str) -> None:
    print(f"OK  {message}")


def _print_fail(message: str) -> None:
    print(f"ERR {message}", file=sys.stderr)


def _assert_status(response: httpx.Response, expected: int, label: str) -> None:
    if response.status_code != expected:
        raise RuntimeError(f"{label}: expected HTTP {expected}, got {response.status_code}: {response.text[:500]}")


def run_smoke(base_url: str, query_image: Path | None = None, timeout: float = 10.0) -> dict:
    base_url = base_url.rstrip("/")
    summary: dict[str, object] = {"base_url": base_url}

    with httpx.Client(timeout=timeout) as client:
        health = client.get(f"{base_url}/health")
        _assert_status(health, 200, "GET /health")
        health_body = health.json()
        summary["health"] = health_body
        _print_ok(f"GET /health -> {health_body.get('status', 'unknown')}")

        stats = client.get(f"{base_url}/index/stats")
        _assert_status(stats, 200, "GET /index/stats")
        stats_body = stats.json()
        summary["stats_vectors"] = stats_body.get("vectors")
        _print_ok(f"GET /index/stats -> vectors={stats_body.get('vectors')}")

        rebuild = client.post(f"{base_url}/admin/rebuild-faiss")
        _assert_status(rebuild, 200, "POST /admin/rebuild-faiss")
        rebuild_body = rebuild.json()
        summary["vectors"] = rebuild_body.get("vectors")
        _print_ok(f"POST /admin/rebuild-faiss -> vectors={rebuild_body.get('vectors')}")

        invalid = client.post(
            f"{base_url}/search",
            files={"file": ("not-image.txt", b"this is not an image", "text/plain")},
        )
        _assert_status(invalid, 400, "POST /search rejects text upload")
        _print_ok("POST /search rejects text/plain upload")

        if query_image is not None:
            if not query_image.exists():
                raise RuntimeError(f"query image does not exist: {query_image}")
            with query_image.open("rb") as file:
                search = client.post(
                    f"{base_url}/search?top_k=5",
                    files={"file": (query_image.name, file, "image/png")},
                )
            _assert_status(search, 200, "POST /search with query image")
            search_body = search.json()
            results = search_body.get("results", [])
            if not results:
                raise RuntimeError("POST /search with query image returned no results")
            summary["search_results"] = len(results)
            summary["top_score"] = results[0].get("score")
            _print_ok(f"POST /search query image -> results={len(results)}, top_score={results[0].get('score')}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a running same-girl-search API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query-image", type=Path)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary on success.")
    args = parser.parse_args()

    try:
        summary = run_smoke(args.base_url, query_image=args.query_image, timeout=args.timeout)
    except Exception as exc:
        _print_fail(str(exc))
        return 1

    if args.json:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
