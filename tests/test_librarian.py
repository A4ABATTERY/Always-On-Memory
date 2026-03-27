"""
Librarian tests for Always-On-Memory Agent.
"""

import unittest
from pathlib import Path
from librarian import chunk_text, is_binary_file

class TestLibrarian(unittest.TestCase):

    def test_chunk_text_basic(self):
        text = "Line 1\nLine 2\nLine 3"
        chunks = chunk_text(text, max_chars=10)
        # Should break on newlines or max_chars
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks).replace("\n", ""), text.replace("\n", ""))

    def test_chunk_text_no_split_needed(self):
        text = "Short text"
        chunks = chunk_text(text, max_chars=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_binary_detection(self):
        # Create a tiny mock binary file
        test_file = Path("test_binary.bin")
        test_file.write_bytes(bytes([0, 1, 2, 3, 255] * 100))
        try:
            self.assertTrue(is_binary_file(test_file))
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_text_detection(self):
        test_file = Path("test_text.txt")
        test_file.write_text("This is normal text.")
        try:
            self.assertFalse(is_binary_file(test_file))
        finally:
            if test_file.exists():
                test_file.unlink()

if __name__ == "__main__":
    unittest.main()
