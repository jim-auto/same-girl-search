from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "apps" / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


class SmokeTestScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = importlib.import_module("scripts.smoke_test")

    def test_run_smoke_checks_expected_endpoints(self) -> None:
        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.method, request.url.path))
            if request.method == "GET" and request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            if request.method == "GET" and request.url.path == "/index/stats":
                return httpx.Response(200, json={"vectors": 3})
            if request.method == "POST" and request.url.path == "/admin/rebuild-faiss":
                return httpx.Response(200, json={"vectors": 3})
            if request.method == "POST" and request.url.path == "/search":
                body = request.content
                if b"not-image.txt" in body:
                    return httpx.Response(400, json={"detail": "file must be an image"})
                return httpx.Response(200, json={"query": {}, "results": [{"rank": 1, "score": 1.0, "image": {"id": 1}}]})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        with tempfile.TemporaryDirectory() as temp_dir:
            query_image = Path(temp_dir) / "query.png"
            query_image.write_bytes(b"png")
            client_cls = httpx.Client
            with patch.object(httpx, "Client", lambda timeout: client_cls(transport=transport, timeout=timeout)):
                summary = self.module.run_smoke("http://api.test", query_image=query_image)

        self.assertEqual(summary["vectors"], 3)
        self.assertEqual(summary["search_results"], 1)
        self.assertEqual(
            calls,
            [
                ("GET", "/health"),
                ("GET", "/index/stats"),
                ("POST", "/admin/rebuild-faiss"),
                ("POST", "/search"),
                ("POST", "/search"),
            ],
        )

    def test_run_smoke_fails_on_unexpected_status(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(503, text="not ready"))
        client_cls = httpx.Client
        with patch.object(httpx, "Client", lambda timeout: client_cls(transport=transport, timeout=timeout)):
            with self.assertRaisesRegex(RuntimeError, "expected HTTP 200"):
                self.module.run_smoke("http://api.test")


if __name__ == "__main__":
    unittest.main()
