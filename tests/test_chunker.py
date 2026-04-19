"""
Unit tests for chunker.py — Pre-ingestion structural document splitting.

13 tests across 4 classes. Two tests are EXPECTED TO FAIL before fixes are applied:
  - TestMarkdownStrategy.test_markdown_h1_headings
      (regex currently #{2,3}, H1 '#' not matched — Fix 4)
  - TestPostProcessing.test_merge_small_chunks_backward_tail
      (backward/tail merge not implemented — Fix 3)
"""

import json
import unittest
from unittest.mock import patch

from chunker import _merge_small_chunks, chunk_document


class TestMarkdownStrategy(unittest.TestCase):
    """Tests for the Markdown heading split strategy."""

    def test_markdown_chunks_headings(self):
        """## headings split into named chunks — should PASS now."""
        # Use bodies large enough to exceed CHUNK_MIN_CHARS (default 200) so
        # _merge_small_chunks does not collapse the two sections into one.
        body = "x" * 250
        text = f"## Introduction\n{body}\n\n## Details\n{body}"
        with patch("chunker.CHUNK_MIN_CHARS", 200):
            chunks = chunk_document(text, "doc.md")
        titles = [c["section_title"] for c in chunks]
        self.assertIn("Introduction", titles)
        self.assertIn("Details", titles)
        self.assertGreaterEqual(len(chunks), 2)

    def test_markdown_h1_headings(self):
        """# H1 headings split into chunks.

        EXPECTED TO FAIL before Fix 4 — regex is currently #{2,3} so H1 '#' is
        not matched and the document falls through to plaintext strategy instead.
        """
        # Use long enough bodies so merging doesn't combine them into one chunk.
        body = "x" * 300
        text = f"# Section One\n{body}\n\n# Section Two\n{body}"
        with patch("chunker.CHUNK_MIN_CHARS", 50):
            chunks = chunk_document(text, "doc.md")
        titles = [c["section_title"] for c in chunks]
        # After Fix 4 both H1 headings should appear as section titles.
        self.assertIn("Section One", titles)
        self.assertIn("Section Two", titles)

    def test_markdown_preamble_captured(self):
        """Content before the first heading becomes a 'preamble' chunk — should PASS now."""
        preamble_body = "This is an introduction paragraph.\n"
        text = preamble_body + "\n## First Section\nSection body."
        chunks = chunk_document(text, "doc.md")
        titles = [c["section_title"] for c in chunks]
        self.assertIn("preamble", titles)

    def test_markdown_triple_hash_headings(self):
        """### headings split into named chunks — should PASS now.

        The regex is r'^(#{2,3} .+)$' so triple-hash headings must also produce
        named chunks.  This guards against regressions if the regex is changed.
        """
        body = "x" * 250
        text = f"### Alpha\n{body}\n\n### Beta\n{body}"
        with patch("chunker.CHUNK_MIN_CHARS", 200):
            chunks = chunk_document(text, "doc.md")
        titles = [c["section_title"] for c in chunks]
        self.assertIn("Alpha", titles)
        self.assertIn("Beta", titles)
        self.assertGreaterEqual(len(chunks), 2)

    def test_markdown_no_headings_falls_through_to_plaintext(self):
        """A doc with no ## / ### headings uses the plaintext strategy — should PASS now.

        With no Markdown headings the markdown strategy returns None and the
        plaintext strategy produces chunks with empty section_title strings.
        """
        text = "No headings here.\nJust regular lines of text."
        chunks = chunk_document(text, "doc.txt")
        # Plaintext strategy produces empty section_title strings
        for chunk in chunks:
            self.assertEqual(chunk["section_title"], "")


class TestJsonStrategy(unittest.TestCase):
    """Tests for the JSON top-level key split strategy."""

    def test_json_splits_by_key(self):
        """Multi-key JSON dict produces one chunk per key — should PASS now."""
        # Each value is large enough to exceed CHUNK_MIN_CHARS (200) so that
        # _merge_small_chunks does not collapse all keys into the first chunk.
        long_val = "v" * 250
        data = {"alpha": long_val, "beta": long_val, "gamma": long_val}
        text = json.dumps(data)
        with patch("chunker.CHUNK_MIN_CHARS", 200):
            chunks = chunk_document(text, "data.json")
        titles = [c["section_title"] for c in chunks]
        self.assertIn("alpha", titles)
        self.assertIn("beta", titles)
        self.assertIn("gamma", titles)
        self.assertEqual(len(chunks), 3)

    def test_json_array_falls_through_to_plaintext(self):
        """A JSON array (not a dict) falls through to plaintext strategy — should PASS now."""
        text = json.dumps([1, 2, 3, 4, 5])
        chunks = chunk_document(text, "data.json")
        # Plaintext chunks have empty section_title
        for chunk in chunks:
            self.assertEqual(chunk["section_title"], "")

    def test_json_single_key_falls_through_to_plaintext(self):
        """A single-key JSON dict falls through to plaintext strategy — should PASS now.

        Splitting a single-key object provides no benefit so the JSON strategy
        returns None and plaintext is used.
        """
        text = json.dumps({"only_key": "some value"})
        chunks = chunk_document(text, "data.json")
        # Plaintext chunks have empty section_title
        for chunk in chunks:
            self.assertEqual(chunk["section_title"], "")

    def test_json_invalid_json_falls_through_to_plaintext(self):
        """A malformed .json file falls through to plaintext strategy — should PASS now.

        _chunk_json catches json.JSONDecodeError and returns None, so the
        plaintext strategy handles the file instead.
        """
        text = "this is not valid json { at all }"
        chunks = chunk_document(text, "broken.json")
        # Plaintext chunks have empty section_title
        for chunk in chunks:
            self.assertEqual(chunk["section_title"], "")


class TestPlaintextStrategy(unittest.TestCase):
    """Tests for the plain-text line-boundary fallback strategy."""

    def test_plaintext_fallback_splits_at_max_chars(self):
        """Text > CHUNK_MAX_CHARS splits into multiple chunks, each <= CHUNK_MAX_CHARS — PASS now."""
        max_chars = 100
        min_chars = 10
        # Build a text clearly larger than max_chars with line breaks so chunk_text
        # can split it neatly.
        line = "A" * 60 + "\n"
        text = line * 5  # ~300 chars — well over max_chars=100
        with patch("chunker.CHUNK_MAX_CHARS", max_chars), \
             patch("chunker.CHUNK_MIN_CHARS", min_chars):
            chunks = chunk_document(text, "plain.txt")
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"]), max_chars)

    def test_empty_document_returns_one_chunk(self):
        """Whitespace-only input returns exactly 1 chunk — should PASS now."""
        text = "   \n\n   "
        chunks = chunk_document(text, "empty.txt")
        self.assertEqual(len(chunks), 1)


class TestPostProcessing(unittest.TestCase):
    """Tests for _merge_small_chunks and _split_large_chunks post-processing."""

    def test_merge_small_chunks_forward(self):
        """A tiny chunk that precedes a larger one is merged forward — should PASS now.

        Strategy: use ## headings to get predictable chunks, make the first heading
        body tiny (< CHUNK_MIN_CHARS) and the second heading body large enough to
        be above CHUNK_MIN_CHARS after merging.  The result should be fewer chunks
        than headings.
        """
        min_chars = 50
        max_chars = 2000
        # First section body: only 5 chars (well below min_chars=50)
        # Second section body: 200 chars (above min_chars)
        small_body = "tiny."
        large_body = "B" * 200
        text = f"## Small Section\n{small_body}\n\n## Large Section\n{large_body}"
        with patch("chunker.CHUNK_MIN_CHARS", min_chars), \
             patch("chunker.CHUNK_MAX_CHARS", max_chars):
            chunks = chunk_document(text, "doc.md")
        # The two sections should have merged into one chunk because the first
        # section body was below CHUNK_MIN_CHARS.
        self.assertEqual(len(chunks), 1)

    def test_merge_small_chunks_backward_tail(self):
        """A tiny FINAL chunk is merged backward into its predecessor.

        EXPECTED TO FAIL before Fix 3 — _merge_small_chunks only does forward
        merging; a small trailing chunk is left as a separate micro-MemCube.
        """
        min_chars = 50
        max_chars = 2000
        # First section: large (200 chars) — above min_chars
        # Second section: tiny (5 chars) — below min_chars, no successor to merge into
        large_body = "A" * 200
        small_body = "tiny."
        text = f"## Big Section\n{large_body}\n\n## Tiny Tail\n{small_body}"
        with patch("chunker.CHUNK_MIN_CHARS", min_chars), \
             patch("chunker.CHUNK_MAX_CHARS", max_chars):
            chunks = chunk_document(text, "doc.md")
        # After Fix 3: the tiny tail should be merged backward, leaving 1 chunk.
        self.assertEqual(len(chunks), 1)

    def test_split_large_chunks_adds_part_titles(self):
        """A chunk > CHUNK_MAX_CHARS is sub-split with '(part N)' appended to title — PASS now."""
        max_chars = 100
        min_chars = 10
        # Create a heading section whose body far exceeds max_chars.
        # Use newlines so chunk_text can split it.
        big_body = ("Z" * 60 + "\n") * 5  # ~300 chars
        text = f"## Big Heading\n{big_body}"
        with patch("chunker.CHUNK_MAX_CHARS", max_chars), \
             patch("chunker.CHUNK_MIN_CHARS", min_chars):
            chunks = chunk_document(text, "doc.md")
        # At least one sub-chunk title should contain "(part "
        part_titles = [c["section_title"] for c in chunks if "(part " in c["section_title"]]
        self.assertGreater(len(part_titles), 0)

    def test_single_chunk_document_has_index_zero(self):
        """A short document returns exactly 1 chunk with chunk_index == 0 — PASS now."""
        text = "Short document."
        chunks = chunk_document(text, "short.txt")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_index"], 0)


class TestEdgeCases(unittest.TestCase):
    """Edge-case tests for boundary conditions and regex behaviour introduced by Fix 4."""

    def test_shebang_line_not_matched_as_heading(self):
        """A shebang line (#!/usr/bin/env python) must NOT be treated as a heading.

        The regex r'^(#{1,3} .+)$' requires a SPACE after the hash(es).
        Shebang lines start with '#!' (no space after '#') so they must NOT match.
        """
        text = "#!/usr/bin/env python\nprint('hello')\n"
        # No heading → falls through to plaintext → section_title is ''
        chunks = chunk_document(text, "script.py")
        for chunk in chunks:
            self.assertEqual(chunk["section_title"], "",
                             "Shebang line must not be matched as a Markdown heading")

    def test_python_comment_with_space_matches_as_h1(self):
        """A Python '# comment' (single hash + space) IS matched as an H1 heading.

        This is a known side-effect of Fix 4 extending the regex from #{2,3} to
        #{1,3}.  The behaviour is documented here so future changes to the regex
        do not silently regress without a test catching it.

        In practice the impact is benign: small comment-split sub-chunks are
        collapsed back into one chunk by _merge_small_chunks at default settings.
        """
        # Use min_chars=1 so that merging does NOT collapse the chunks, allowing
        # us to observe the raw heading-split behaviour directly.
        text = "# Section A\nline one\n\n# Section B\nline two\n"
        with patch("chunker.CHUNK_MIN_CHARS", 1):
            chunks = chunk_document(text, "module.py")
        titles = [c["section_title"] for c in chunks]
        # With #{1,3} both single-hash lines are treated as headings.
        self.assertIn("Section A", titles)
        self.assertIn("Section B", titles)

    def test_h4_heading_not_matched(self):
        """An H4 heading ('#### heading') must NOT be matched — pattern is #{1,3}.

        Four or more consecutive hashes fall outside the {1,3} quantifier and
        must pass through to the plaintext strategy (section_title == '').
        """
        body = "x" * 300  # Large enough to avoid merging issues
        text = f"#### Level Four Heading\n{body}"
        chunks = chunk_document(text, "doc.md")
        for chunk in chunks:
            self.assertNotIn("Level Four Heading", chunk["section_title"],
                             "H4 heading must not be matched by #{1,3} regex")


class TestMergeSmallChunksUnit(unittest.TestCase):
    """Direct unit tests for _merge_small_chunks covering backward-pass guard conditions
    and title-preservation behaviour that are not exercised by the higher-level
    chunk_document integration tests."""

    def _make_chunk(self, text, index=0, title=""):
        return {"text": text, "chunk_index": index, "section_title": title}

    def test_single_tiny_chunk_backward_pass_does_not_trigger(self):
        """A list with exactly one tiny chunk must NOT have the backward pass fire.

        len(merged) >= 2 is False when merged has only one element, so the tail
        must be left untouched (not merged into a non-existent predecessor).
        """
        chunks = [self._make_chunk("hi", 0, "Only")]
        result = _merge_small_chunks(chunks, min_chars=50)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "hi")
        self.assertEqual(result[0]["section_title"], "Only")

    def test_two_tiny_chunks_forward_merges_leaves_one_element_backward_skipped(self):
        """Two tiny chunks: forward pass merges chunk[0] into chunk[1], producing
        one element in merged.  The backward pass guard (len(merged) >= 2) must
        be False and must NOT attempt to merge into a non-existent predecessor.
        """
        chunks = [
            self._make_chunk("ab", 0, "A"),
            self._make_chunk("cd", 1, "B"),
        ]
        result = _merge_small_chunks(chunks, min_chars=50)
        self.assertEqual(len(result), 1,
                         "Forward pass should have merged both tiny chunks into one")
        # Backward pass must not fire — if it did it would crash (no predecessor)
        # or double-mutate the text. Verify the combined text is correct.
        self.assertIn("ab", result[0]["text"])
        self.assertIn("cd", result[0]["text"])

    def test_backward_tail_merge_preserves_predecessor_section_title(self):
        """When a tiny tail is backward-merged into its predecessor, the predecessor's
        section_title must survive unchanged.  The merged text must contain both
        the predecessor body and the tail body.
        """
        large_body = "A" * 200
        small_body = "tiny."
        chunks = [
            self._make_chunk(large_body, 0, "Big Section"),
            self._make_chunk(small_body, 1, "Tiny Tail"),
        ]
        result = _merge_small_chunks(chunks, min_chars=50)
        self.assertEqual(len(result), 1,
                         "Tiny tail must be backward-merged into its predecessor")
        self.assertEqual(result[0]["section_title"], "Big Section",
                         "Predecessor section_title must be preserved after backward merge")
        self.assertIn(large_body, result[0]["text"])
        self.assertIn(small_body, result[0]["text"])

    def test_re_sequencing_after_backward_merge(self):
        """After a backward tail-merge, the surviving chunk must have chunk_index == 0."""
        large_body = "A" * 200
        chunks = [
            self._make_chunk(large_body, 0, "First"),
            self._make_chunk("short", 1, "Second"),
        ]
        result = _merge_small_chunks(chunks, min_chars=50)
        self.assertEqual(result[0]["chunk_index"], 0)

    def test_empty_list_returns_empty(self):
        """An empty input list must return an empty list without error."""
        result = _merge_small_chunks([], min_chars=50)
        self.assertEqual(result, [])
