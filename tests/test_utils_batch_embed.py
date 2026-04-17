"""
Unit tests for embed_texts_batch sub-batching logic in utils.py.

Verifies that embed_texts_batch splits large text lists into sequential
API calls of at most _EMBED_MAX_BATCH_SIZE (100) items each.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch, call


class TestEmbedTextsBatchSubBatching(unittest.TestCase):
    """Tests for the 100-item sub-batching logic in embed_texts_batch."""

    def _make_fake_embed_result(self, n: int):
        """Return a fake embed_content result with n embeddings of dimension 4."""
        fake_result = MagicMock()
        fake_result.embeddings = [MagicMock(values=[0.1, 0.2, 0.3, 0.4]) for _ in range(n)]
        return fake_result

    def _run(self, coro):
        return asyncio.run(coro)

    def test_single_batch_under_limit(self):
        """50 texts → exactly 1 API call, 50 results returned."""
        texts = [f"text {i}" for i in range(50)]

        fake_client = MagicMock()
        fake_client.models.embed_content.return_value = self._make_fake_embed_result(50)

        with patch("utils._get_client", return_value=fake_client), \
             patch("utils.HAS_GENAI", True):
            from utils import embed_texts_batch
            results = self._run(embed_texts_batch(texts))

        self.assertEqual(fake_client.models.embed_content.call_count, 1)
        self.assertEqual(len(results), 50)

    def test_exactly_100_texts_is_one_call(self):
        """Exactly 100 texts → 1 API call (boundary condition)."""
        texts = [f"text {i}" for i in range(100)]

        fake_client = MagicMock()
        fake_client.models.embed_content.return_value = self._make_fake_embed_result(100)

        with patch("utils._get_client", return_value=fake_client), \
             patch("utils.HAS_GENAI", True):
            from utils import embed_texts_batch
            results = self._run(embed_texts_batch(texts))

        self.assertEqual(fake_client.models.embed_content.call_count, 1)
        self.assertEqual(len(results), 100)

    def test_150_texts_splits_into_two_calls(self):
        """150 texts → 2 API calls (100 + 50), 150 results returned."""
        texts = [f"text {i}" for i in range(150)]

        fake_client = MagicMock()
        # First call returns 100 embeddings, second returns 50
        fake_client.models.embed_content.side_effect = [
            self._make_fake_embed_result(100),
            self._make_fake_embed_result(50),
        ]

        with patch("utils._get_client", return_value=fake_client), \
             patch("utils.HAS_GENAI", True):
            from utils import embed_texts_batch
            results = self._run(embed_texts_batch(texts))

        self.assertEqual(fake_client.models.embed_content.call_count, 2)
        self.assertEqual(len(results), 150)
        # Verify each call received the correct batch size
        first_call_contents = fake_client.models.embed_content.call_args_list[0]
        second_call_contents = fake_client.models.embed_content.call_args_list[1]
        self.assertEqual(len(first_call_contents.kwargs["contents"]), 100)
        self.assertEqual(len(second_call_contents.kwargs["contents"]), 50)

    def test_300_texts_splits_into_three_calls(self):
        """300 texts → 3 API calls (100 + 100 + 100), 300 results returned."""
        texts = [f"text {i}" for i in range(300)]

        fake_client = MagicMock()
        fake_client.models.embed_content.side_effect = [
            self._make_fake_embed_result(100),
            self._make_fake_embed_result(100),
            self._make_fake_embed_result(100),
        ]

        with patch("utils._get_client", return_value=fake_client), \
             patch("utils.HAS_GENAI", True):
            from utils import embed_texts_batch
            results = self._run(embed_texts_batch(texts))

        self.assertEqual(fake_client.models.embed_content.call_count, 3)
        self.assertEqual(len(results), 300)

    def test_failed_sub_batch_fills_empty_placeholders(self):
        """If the second sub-batch raises, results for that slice are [] (index-aligned)."""
        texts = [f"text {i}" for i in range(150)]

        fake_client = MagicMock()
        fake_client.models.embed_content.side_effect = [
            self._make_fake_embed_result(100),
            Exception("500 Internal Server Error"),
        ]

        with patch("utils._get_client", return_value=fake_client), \
             patch("utils.HAS_GENAI", True):
            from utils import embed_texts_batch
            results = self._run(embed_texts_batch(texts))

        # Total length preserved for zip() alignment in librarian
        self.assertEqual(len(results), 150)
        # First 100 should have real embeddings
        self.assertEqual(results[0], [0.1, 0.2, 0.3, 0.4])
        # Last 50 should be empty placeholders
        for r in results[100:]:
            self.assertEqual(r, [])

    def test_empty_texts_returns_empty(self):
        """Empty input list → empty output, no API call made."""
        fake_client = MagicMock()

        with patch("utils._get_client", return_value=fake_client), \
             patch("utils.HAS_GENAI", True):
            from utils import embed_texts_batch
            results = self._run(embed_texts_batch([]))

        fake_client.models.embed_content.assert_not_called()
        self.assertEqual(results, [])

    def test_no_genai_returns_empty(self):
        """When HAS_GENAI is False, returns [] without touching client."""
        fake_client = MagicMock()

        with patch("utils._get_client", return_value=fake_client), \
             patch("utils.HAS_GENAI", False):
            from utils import embed_texts_batch
            results = self._run(embed_texts_batch(["hello", "world"]))

        fake_client.models.embed_content.assert_not_called()
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
