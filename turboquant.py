"""
TurboQuant Module — High-performance vector quantization using random orthogonal rotations.
Implements the core logic from the TurboQuant paper (3.5-bit simulation).
"""

import numpy as np
from typing import Dict, List, Optional

class TurboQuant:
    """
    Implements TurboQuant-style vector optimization.
    Uses a persistent random orthogonal rotation to improve quantization fidelity.
    """
    def __init__(self, dim: int = 3072, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self._rotation_matrix = None
        self._qjl_matrix = None
        
        # Scalar Quantizer setup (6 levels ≈ 2.5 bits)
        self.levels = 6
        self.bins = np.linspace(-3.0, 3.0, self.levels - 1)
        self.centroids = np.linspace(-3.5, 3.5, self.levels)

    @property
    def rotation_matrix(self) -> np.ndarray:
        if self._rotation_matrix is None:
            # Use deterministic seed for persistence across restarts
            state = np.random.get_state()
            np.random.seed(self.seed)
            # Generate a random orthogonal matrix via QR decomposition
            H = np.random.randn(self.dim, self.dim)
            Q, R = np.linalg.qr(H)
            self._rotation_matrix = Q.astype(np.float32)
            np.random.set_state(state)
        return self._rotation_matrix

    @property
    def qjl_matrix(self) -> np.ndarray:
        if self._qjl_matrix is None:
            state = np.random.get_state()
            np.random.seed(self.seed + 1) # Different seed for QJL
            self._qjl_matrix = np.random.randn(self.dim, self.dim).astype(np.float32)
            np.random.set_state(state)
        return self._qjl_matrix

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """Applies random orthogonal rotation to the vector."""
        if vector.shape[0] != self.dim:
            # Handle dimension mismatch if necessary (e.g. padding/clipping)
            if vector.shape[0] < self.dim:
                padded = np.zeros(self.dim, dtype=np.float32)
                padded[:vector.shape[0]] = vector
                vector = padded
            else:
                vector = vector[:self.dim]
                
        # Normalize to unit length before rotation
        norm = np.linalg.norm(vector)
        if norm > 1e-9:
            vector = vector / norm
            
        return self.rotation_matrix @ vector

    def quantize_to_int8(self, vector: List[float]) -> bytes:
        """
        Full TurboQuant-inspired int8 quantization.
        1. Normalization
        2. Random Rotation
        3. Scalar Quantization to [-127, 127]
        """
        arr = np.array(vector, dtype=np.float32)
        rotated = self.transform(arr)
        # Scale to int8 range [-127, 127]
        # After rotation, values follow a bell curve centered at 0.
        # We clip to 3 standard deviations (approx) or use a fixed scale.
        # For unit-norm vectors in high dimensions, components are small.
        # Standard deviation is approx 1/sqrt(dim).
        scale = np.sqrt(self.dim) * 64.0 # Empirical scale factor
        quantized = np.clip(np.round(rotated * scale), -128, 127).astype(np.int8)
        return quantized.tobytes()

    def get_query_vector(self, vector: List[float]) -> List[float]:
        """Prepares a query vector by applying the same rotation."""
        arr = np.array(vector, dtype=np.float32)
        return self.transform(arr).tolist()

# Singleton instance for the default Gemini embedding dimension
_default_tq = None

def get_turboquant(dim: int = 3072) -> TurboQuant:
    global _default_tq
    if _default_tq is None or _default_tq.dim != dim:
        _default_tq = TurboQuant(dim=dim)
    return _default_tq
