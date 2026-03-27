# Data Model & Persistence

AOM v3 treats memory as a structured, hierarchical operating system. This document explains the core `MemCube` data model, the underlying SQLite schema, and the vector storage system.

## 📦 The MemCube

The `MemCube` is the atomic unit of memory in AOM. It is designed to be portable across different AI platforms while maintaining high semantic density.

### Sectors
Each `MemCube` is assigned to one of four sectors:
- **`semantic`**: Permanent facts, architectural rules, and system conventions.
- **`episodic`**: Events, specific task outcomes, and chronological history.
- **`procedural`**: "How-to" guides, workflows, and testing patterns.
- **`reflection`**: Lessons learned from failures, reasoning paths, and meta-strategy.

### Schema (Pydantic)
```python
class MemCube(BaseModel):
    cube_id: str          # UUID
    sector: str           # semantic | episodic | procedural | reflection
    source: str           # Origin (file name, API source)
    raw_text: str         # The full uncompressed memory content
    summary: str          # A 1-2 sentence high-level summary
    entities: List[str]   # Extracted people, companies, concepts
    topics: List[str]     # Semantic tags (e.g., "authentication", "refactor")
    connections: List     # Links to other MemCubes or Source Files
    embedding: List       # 3072-dimensional vector (Gemini)
    importance: float     # 0.0 to 1.0
    consolidated: bool    # True if merged into an Insight Cube
```

## 🗄️ SQLite Schema

AOM uses a unified SQLite database for both relational metadata and vector storage (via `sqlite-vec`).

### `memories` Table
Stores the relational and text-based metadata for each `MemCube`.
- `id` (INTEGER PRIMARY KEY)
- `cube_id` (TEXT)
- `sector` (TEXT)
- `summary` (TEXT)
- `raw_text` (TEXT)
- `connections` (JSON)
- `importance_score` (FLOAT)
- `consolidated` (BOOLEAN)
- `created_at` (DATETIME)

### `vec_memories` (Virtual Table)
An FTS-like virtual table provided by `sqlite-vec` for high-performance vector operations.
- `memory_id` (INTEGER references memories.id)
- `embedding` (VECTOR int8)

## 🏎️ TurboQuant (3.5-bit Quantization)

To ensure long-term scalability without exploding the database size, AOM implements **TurboQuant**.

### The Process:
1. **Random Orthogonal Rotation**: The incoming float32 vector is rotated into a dense space using a seeded rotation matrix.
2. **Scalar Quantization**: The rotated values are then quantized into fixed-width `int8` representations.
3. **Storage**: The resulting `int8` vector is stored in the `vec_memories` table.

### Result: 
- **90% Smaller Storage**: Vectors are significantly compressed.
- **High FIDELITY**: Semantic distance is preserved with minimal Mean Squared Error (MSE), ensuring search accuracy remains high.

## 📚 Librarian Documents

The **Librarian** also indexes project source code into chunks for "Structural Linkage".
- **`documents`**: Metadata for source code chunks (path, hash, content).
- **`vec_documents`**: Vector embeddings for source code chunks, allowing the system to detect "semantic drift" when a file is modified.
