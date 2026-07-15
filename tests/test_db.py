from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "apps" / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.db import delete_image, get_image, list_embeddings, migrate, row_to_indexed_image, upsert_image, connect


class DatabaseTests(unittest.TestCase):
    def test_upsert_list_and_delete_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "index.sqlite3"
            embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            conn = connect(db_path)
            try:
                with conn:
                    migrate(conn)
                    image_id = upsert_image(
                        conn,
                        image_path="/images/a.jpg",
                        source_site="site-a",
                        shop_name="shop-a",
                        profile_url="https://example.test/profile/a",
                        bbox=[1.0, 2.0, 11.0, 12.0],
                        det_score=0.97,
                        face_count=1,
                        width=320,
                        height=240,
                        embedding=embedding,
                    )

                    row = get_image(conn, image_id)
                    self.assertIsNotNone(row)
                    indexed = row_to_indexed_image(row)
                    self.assertEqual(indexed["image_path"], "/images/a.jpg")
                    self.assertEqual(indexed["bbox"], [1.0, 2.0, 11.0, 12.0])
                    self.assertEqual(indexed["width"], 320)
                    self.assertEqual(indexed["height"], 240)

                    rows = list_embeddings(conn)
                    self.assertEqual([row_id for row_id, _ in rows], [image_id])
                    np.testing.assert_array_equal(rows[0][1], embedding)

                    updated_id = upsert_image(
                        conn,
                        image_path="/images/a.jpg",
                        source_site="site-b",
                        shop_name="shop-b",
                        profile_url=None,
                        bbox=[2.0, 3.0, 12.0, 13.0],
                        det_score=0.88,
                        face_count=2,
                        width=320,
                        height=240,
                        embedding=np.array([4.0, 5.0, 6.0], dtype=np.float32),
                    )
                    self.assertEqual(updated_id, image_id)
                    updated = row_to_indexed_image(get_image(conn, image_id))
                    self.assertEqual(updated["source_site"], "site-b")
                    self.assertEqual(updated["face_count"], 2)

                    self.assertTrue(delete_image(conn, image_id))
                    self.assertIsNone(get_image(conn, image_id))
                    self.assertFalse(delete_image(conn, image_id))
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
