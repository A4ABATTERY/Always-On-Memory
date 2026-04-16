# API Reference

AOM v3 exposes a simple, asynchronous HTTP API (via `aiohttp`) for interacting with the memory system. By default, the server runs on port `8888`.

## 🔍 Retrieval Endpoints

### `GET /query?q=<question>`
Performs a hybrid semantic search across memories and code chunks, then synthesizes an answer.
- **Example**: `GET /query?q=How does TurboQuant work?`
- **Response**:
  ```json
  {
    "question": "How does TurboQuant work?",
    "answer": "TurboQuant is a 3.5-bit quantization mechanism..."
  }
  ```

### `GET /search?q=<query>&k=5`
Returns raw semantic search results from the Librarian's code index.
- **Example**: `GET /search?q=database schema`
- **Response**:
  ```json
  {
    "results": [
      { "path": "database.py", "snippet": "create table memories...", "distance": 0.12 }
    ],
    "count": 1
  }
  ```

### `GET /links`
Returns all active structural links between memories and source files.

---

## 📥 Ingestion Endpoints

### `POST /ingest`
Manually ingest text into the memory system.
- **Payload**: `{ "text": "Project goal: Build a self-healing memory OS", "source": "manual" }`
- **Response**: `{ "status": "ingested", "response": "MemCube #123 created" }`

### `POST /import_cubes`
Import a list of portable `MemCube` objects (JSON) into the database.
- **Payload**: `{ "cubes": [...] }` or just a JSON array `[...]`.

---

## ⚙️ Management Endpoints

### `GET /status`
Returns high-level statistics about the memory system.
- **Response**:
  ```json
  {
    "total_memories": 150,
    "consolidated": 120,
    "unconsolidated": 30,
    "sectors": { "semantic": 40, "episodic": 110 }
  }
  ```

### `POST /consolidate`
Manually trigger a standard adversarial consolidation cycle.

### `POST /reconsolidate`
Manually trigger a "Deep" (AutoDream-style) reconsolidation cycle.

### `POST /improve`
Manually trigger a self-improvement audit to discover new skills or rules.

### `POST /clear`
Performs a full reset of the memory database and clears the `inbox/` directory. **Use with caution.**

### `POST /delete`
Delete a specific memory by ID.
- **Payload**: `{ "memory_id": 123 }`

---

## 📤 Export Endpoints

### `GET /export_cubes?ids=1,2,3`
Exports specified (or all) memories as a list of portable `MemCube` objects.
- **Example**: `GET /export_cubes?ids=45,67`
