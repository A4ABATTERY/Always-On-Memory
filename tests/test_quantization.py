"""
Quantization benchmark for Always-On-Memory v3.
"""

import unittest
import numpy as np
from utils import serialize_int8
from turboquant import get_turboquant

class TestQuantization(unittest.TestCase):
    
    def test_quantization_mse(self):
        """Benchmark MSE between original and int8 quantized embeddings.

        Uses dim=3072 to match the production vec_memories / vec_documents schema.
        A 1024-dim quantized vector (1024 bytes) cannot be inserted into the
        int8[3072] column, so the test must use the production dimension.
        """
        dim = 3072  # Must match vec0 column definition in database.py
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)

        q_vec_bytes = serialize_int8(vec.tolist())

        # Enforce the byte-length contract: 3072 int8 values = 3072 bytes
        self.assertEqual(len(q_vec_bytes), dim, f"Expected {dim} bytes, got {len(q_vec_bytes)}")

        q_vec = np.frombuffer(q_vec_bytes, dtype=np.int8).astype(np.float32)

        tq = get_turboquant(dim=dim)
        scale = np.sqrt(dim) * 64.0
        reconstructed_rotated = q_vec / scale

        rotated_vec = tq.transform(vec)

        mse = np.mean((rotated_vec - reconstructed_rotated)**2)
        print(f"\nTurboQuant Rotation MSE (dim={dim}): {mse:.8f}")

        self.assertLess(mse, 0.001, "Quantization error too high!")

    def test_quantization_range(self):
        """Verify that quantized values are within int8 range."""
        vec = [1.0, -1.0, 0.0, 0.5, -0.5]
        q_bytes = serialize_int8(vec)
        q_vec = np.frombuffer(q_bytes, dtype=np.int8)
        
        self.assertEqual(len(q_vec), len(vec))
        self.assertTrue(np.all(q_vec <= 127))
        self.assertTrue(np.all(q_vec >= -128))

if __name__ == "__main__":
    unittest.main()
