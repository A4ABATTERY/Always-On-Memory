"""
Quantization benchmark for Always-On-Memory v3.
"""

import unittest
import numpy as np
from utils import serialize_int8
from turboquant import get_turboquant

class TestQuantization(unittest.TestCase):
    
    def test_quantization_mse(self):
        """Benchmark MSE between original and int8 quantized embeddings."""
        # Generate random normalized float32 vector (simulating embedding)
        dim = 1024
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        
        # Quantize to int8 [-128, 127]
        # In AOM, we use unit normalization so float [-1, 1] maps to int8 [-127, 127]
        # Or similar scaling. 
        # Let's check how utils.serialize_int8 does it (assumed logic).
        
        q_vec_bytes = serialize_int8(vec.tolist())
        q_vec = np.frombuffer(q_vec_bytes, dtype=np.int8).astype(np.float32)
        
        # Rescale back to float (TurboQuant uses internal scale)
        tq = get_turboquant(dim=dim)
        scale = np.sqrt(dim) * 64.0
        reconstructed_rotated = q_vec / scale
        
        # Compare with the ROTATED original vector
        rotated_vec = tq.transform(vec)
        
        mse = np.mean((rotated_vec - reconstructed_rotated)**2)
        print(f"\nTurboQuant Rotation MSE: {mse:.8f}")
        
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
