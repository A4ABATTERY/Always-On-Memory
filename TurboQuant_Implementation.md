Based on the implementations provided (and the overarching architecture of the TurboQuant paper), the **3.5-bit TurboQuant** compression relies on a two-stage approach:

1. **Stage 1 (2.5 bits per dimension):** Extract the norm of the vector, apply a random orthogonal rotation matrix to "mix" the data into a predictable bell curve, and run a scalar quantization (grouping values into roughly 6 buckets, which takes ~2.5 bits).
2. **Stage 2 (1 bit per dimension):** Calculate the "leftover" error (the residual), project it through a random matrix, and extract just the positive/negative signs (1 bit). This prevents the AI from becoming biased when matching documents later.

Below is a complete, simplified Python algorithm demonstrating a **Document Ingestion Pipeline** for a Vector Search DB. It simulates generating an embedding, compressing it down using the 3.5-bit TurboQuant logic, and storing it.

### Python Algorithm for Document Ingestion

```python
import numpy as np
from typing import Dict

class TurboQuant35Bit:
    """
    A simplified software simulation of TurboQuant's 3.5-bit compression.
    """
    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        np.random.seed(seed)
        
        # 1. Random Rotation Matrix (Dense QR Decomposition to make it orthogonal)
        # This mixes the coordinates so they follow a predictable Normal/Beta distribution
        random_matrix = np.random.randn(dim, dim)
        Q, R_mat = np.linalg.qr(random_matrix)
        self.rotation_matrix = Q  
        
        # 2. Scalar Quantizer setup (2.5 bits ≈ 6 discrete levels)
        # In a real implementation, Lloyd-Max algorithms optimally place these centroids.
        self.levels = 6
        self.bins = np.linspace(-3.0, 3.0, self.levels - 1)
        self.centroids = np.linspace(-3.5, 3.5, self.levels)
        
        # 3. QJL (Quantized Johnson-Lindenstrauss) Random Projection Matrix
        # Used for the 1-bit residual correction
        self.qjl_matrix = np.random.randn(dim, dim)

    def compress(self, vector: np.ndarray) -> Dict:
        """Compresses a dense Float32 vector down to 3.5 bits per dimension."""
        # A. Extract and save the magnitude (norm)
        norm = np.linalg.norm(vector)
        normalized_vector = vector / (norm + 1e-9)
        
        # B. STAGE 1: Random Rotation
        rotated = self.rotation_matrix @ normalized_vector
        
        # C. STAGE 1: Scalar Quantization (2.5-bit)
        # Digitize assigns each value to one of the 6 buckets (0 to 5)
        quantized_indices = np.digitize(rotated, self.bins)
        
        # D. STAGE 2: 1-Bit QJL Residual Correction
        # Figure out the error introduced by quantization
        dequantized = self.centroids[quantized_indices]
        residual = rotated - dequantized
        
        # Project the error and extract just the sign bit (1 = positive, 0 = negative)
        qjl_projection = self.qjl_matrix @ residual
        qjl_signs = np.where(qjl_projection > 0, 1, 0)
        
        # Return the tiny memory footprint! 
        # (In C/C++/Zig, 'indices' and 'qjl_signs' are physically bit-packed together)
        return {
            "norm": norm,                              # FP16 (1 per vector)
            "indices": quantized_indices.astype(np.uint8), # 2.5 bits per dim
            "qjl_signs": qjl_signs.astype(np.bool_),       # 1 bit per dim
        }

class VectorSearchDB:
    """
    Mock Vector Database handling document ingestion.
    """
    def __init__(self, dim: int):
        self.compressor = TurboQuant35Bit(dim=dim)
        self.document_store = {}
        
    def ingest_document(self, doc_id: str, text: str, embedding: np.ndarray):
        """
        The Ingestion Pipeline: Text -> Embedding -> TurboQuant Compression -> DB Storage
        """
        print(f"Ingesting: {doc_id}...")
        
        # 1. Compress the embedding vector
        compressed_vector = self.compressor.compress(embedding)
        
        # 2. Store in the database
        self.document_store[doc_id] = {
            "text": text,
            "vector_data": compressed_vector
        }
        print(f"✅ Successfully compressed and stored '{doc_id}'.\n")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Typically, embeddings from OpenAI or BERT have sizes like 128, 768, or 1536
    DIMENSION = 128 
    
    # Initialize our DB
    db = VectorSearchDB(dim=DIMENSION)
    
    # 1. A new document arrives into the pipeline
    doc_id = "doc_001"
    text = "Large language models rely heavily on key-value cache compression."
    
    # 2. (Mocking an embedding model like SentenceTransformers converting text to a vector)
    mock_embedding = np.random.randn(DIMENSION) 
    
    # 3. Ingest!
    db.ingest_document(doc_id, text, mock_embedding)
    
    # Let's inspect what is actually saved in memory
    saved_data = db.document_store[doc_id]["vector_data"]
    
    print("--- Memory Footprint Snapshot ---")
    print(f"Norm saved: {saved_data['norm']:.4f} (Float)")
    print(f"Stage 1 Quantized Array: {saved_data['indices'][:10]}... (Stored as packed 2.5-bit ints)")
    print(f"Stage 2 Error Correction: {saved_data['qjl_signs'][:10]}... (Stored as packed 1-bit booleans)")
```

### How this algorithm fits into a larger system
1. **Embedding Model**: In a real app, `mock_embedding` would be created by a model like `text-embedding-3-small`.
2. **Bit-packing**: Python arrays natively take up a lot of space. In a production engine (like the C/Zig/CUDA repos you linked), `indices` and `qjl_signs` are shifted and masked into exact raw binary bytes so they physically take up exactly 3.5 bits per index in RAM.
3. **Retrieval**: When a user searches the DB, the DB applies the exact same Rotation and QJL matrices to the user's search query, allowing it to mathematically estimate the document match (Inner Product / Dot Product) entirely in the compressed state without ever unzipping the data.