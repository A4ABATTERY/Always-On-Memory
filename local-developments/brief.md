# Development Brief: Inbox Watcher Improvements

**Date:** 2026-04-15  
**Source Issues:** #10 (OPEN), #11 (CLOSED / superseded by #10), #12 (OPEN)  
**Scope:** `agent.py` · `database.py` · `memory_store.py`

---

## Executive Summary

Three GitHub issues identify a cluster of related gaps in the **Inbox Watcher** (`watch_folder` in `agent.py`). Two problems are architectural — they cause memory stagnation and contradictory recall — and one is a coverage bug that was already fixed in `librarian.py` but never applied to the watcher. All three are solvable in the same targeted area of the codebase without touching the agent harness, vector layer, or MCP server.

---

## Issue Inventory

### Issue #12 — Recursive Ingestion (OPEN, lowest risk)

> **Original issue title:** FIX: Support for Recursive Ingestion in Inbox Watcher
>
> **Summary (verbatim):** The `watch_folder` loop in `agent.py` previously used a flat iteration (e.g., `folder.iterdir()`), which caused it to skip files nested within subdirectories of the Inbox. This restricted the project's ability to ingest hierarchical documentation structures.
>
> **Change requested:** Update the ingestion loop to use `Path.rglob('*')` instead of `iterdir()`.
>
> **Benefits stated:**
>
> - Ensures absolute coverage of all files within the Inbox tree.
> - Aligns `agent.py` ingestion behaviour with `librarian.py`, which was already recursive.
> - Enables the AOM agent to process complex documentation folders out-of-the-box.

**Code location:** `agent.py:531`

```python
# Current — flat only
files = sorted(folder.iterdir())
```

**What this breaks:** Any user who drops a folder of documentation into the inbox (e.g. `inbox/project-docs/architecture.md`) gets zero ingestion. This is a silent failure — no error is logged, the files simply never appear.

**Fix:** Replace `folder.iterdir()` with `folder.rglob('*')`. This mirrors the Librarian (`librarian.py`) and aligns both ingestion paths.

**Risk:** Very low. `rglob('*')` returns files in all depths; the existing suffix-whitelist filter (`ALL_SUPPORTED`) already guards against unwanted file types. The `processed_files` primary key on `path` prevents double-processing.

---

### Issue #10 — File Update Tracking & Semantic Invalidation (OPEN, medium complexity)

> **Original issue title:** FEAT: Support for File Update Tracking and Semantic Invalidation
>
> **Two critical limitations identified:**
>
> **1. Lack of File Update Tracking in Inbox:** The `watch_folder` loop in `agent.py` only checks for the existence of a file path in the `processed_files` table. It does not check the file's modification time or content hash. As a result, if a documentation file is updated in the inbox, the agent never re-ingests the new information, leading to memory stagnation.
>
> **2. Episodic Memory Invalidation:** When a file is modified (even if re-ingested), the previous episodic memories (fragments) belonging to the old version of the file remain active and 'true' in the vector database. This causes the agent to return contradictory information.
>
> **Recommendations (verbatim):**
>
> - Implement content hashing in `watch_folder` (consistent with `librarian.py`).
> - Add a mechanism for 'Semantic Invalidation' where a re-ingestion can flag previous memories from the same source as 'inactive' or 'superseded' in the `memories` table.
> - Leverage the adversarial consolidation loop to explicitly prioritise newer contradictory facts over older ones.

---

### Issue #11 — Stale Memory Invalidation and Hash-Based Ingestion (CLOSED — superseded by #10)

> **Original issue title:** FEAT/FIX: Support for Stale Memory Invalidation and Hash-Based Ingestion
>
> **Summary (verbatim):** The current AOM implementation lacks a mechanism to 'forget' or 'supersede' memories when the source material (e.g., a documentation file) is updated. While `librarian.py` successfully re-indexes updated files for search, the `agent.py` episodic memories remain persisted in their original state, causing the LLM to return contradictory or stale information.
>
> **Problem analysis (verbatim):**
>
> - **`watch_folder` Stagnation:** The inbox watcher in `agent.py` uses a simple 'path-exists' check in `processed_files`. It does not detect content updates.
> - **Episodic Persistence:** Memories generated during ingestion have no semantic link to the file's content hash. When a file is modified, the new ingestion adds more memories, but does not retire the old ones.
> - **Consolidation Weakness:** The adversarial consolidator is currently designed for redundancy pruning, not necessarily 'Version Reconciliation.' It struggles to correctly resolve hard contradictions between high-priority directives and historical documentation fragments.
>
> **Three architectural solutions proposed:**
>
> **1. Hash-Based Ingestion Tracking** — Upgrade `processed_files` to store a `content_hash` alongside the path. If `watch_folder` sees a file with a known path but a different hash: re-ingest the file, tag the new memories with the new hash, and automatically flag previous memories connected to that path as superseded.
>
> **2. Explicit Memory Retirement** — Introduce `status` or `valid_to` lifecycle management in the `memories` table. When an updated source is detected, the agent should perform a 'Semantic Sweep' to find all memories linked via `file_link` to that source and move them to a historical state.
>
> **3. 'Architectural Directive' Priority** — Enhance the Deep Reconsolidation prompt to recognise a new class of memory: `ARCHITECTURAL_DIRECTIVE`. These memories should carry absolute weight and force the consolidator to prune any contradicting memories, regardless of how many times they appear in episodic logs.

Issue #11 is **closed** — likely consolidated into #10 as the active tracking issue. Its three proposals are evaluated in the architecture section below.

---

### Codebase Analysis of Reported Problems

#### Root Cause A: `processed_files` has no content hash

**Code location:** `database.py:76-79`

```sql
CREATE TABLE IF NOT EXISTS processed_files (
    path TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL
    -- NO content_hash column
);
```

The watcher checks `SELECT 1 FROM processed_files WHERE path = ?` (`agent.py:542`). If the row exists, the file is skipped — forever. Updating a file in the inbox has no effect.

By contrast, the `documents` table (used by Librarian, `database.py:80-88`) already stores `content_hash TEXT NOT NULL`. The Librarian reindexes on hash change. The Inbox Watcher should do the same.

#### Root Cause B: No semantic invalidation on re-ingest

When a file is updated and re-ingested, the new ingest agent creates **additional** MemCubes linked to the same source. The old MemCubes remain `active` with their original content. During query synthesis, both old and new memories are retrieved — the LLM sees contradictory facts and has no signal about which is authoritative.

**Existing infrastructure that can solve this:** The system already has:

- `valid_to TEXT DEFAULT NULL` on `memories` (`database.py:67`) — designed for ephemeral info, but applicable here
- `update_link_status(memory_id, path, new_status)` in `memory_store.py:490` — transitions `file_link` connections from `active` → `historical_trace`
- `repair_memory(memory_id, new_raw_text)` in `memory_store.py:521` — updates raw_text and resets timestamp
- The `file_link` connection type already tracks which memories originated from which file path (`memory_store.py:276-306`)

This means **Semantic Invalidation does not require a new table or schema change beyond adding `content_hash` to `processed_files`**. It can be built using the existing link infrastructure.

---

## Proposed Architecture

### Change 1 — Schema Migration: Add `content_hash` and `prev_hash` to `processed_files`

`database.py:init_db()` — add two columns to `processed_files` and one to `memories` via additive migrations:

```python
for col_sql in [
    "ALTER TABLE processed_files ADD COLUMN content_hash TEXT DEFAULT NULL",
    "ALTER TABLE processed_files ADD COLUMN prev_hash TEXT DEFAULT NULL",
]:
    try:
        db.execute(col_sql)
    except sqlite3.OperationalError:
        pass  # column already exists — safe to run on every startup
```

- **`content_hash`** — MD5 of file content at last-processed time. Drives the update-detection logic in Change 2. Consistent with `librarian.py` to ensure unified hashing across the system.
- **`prev_hash`** — The hash stored before the most recent update. Enables rollback detection and future diff-size reasoning.
- **`Historical Trace (Links)`** — Rather than a new `superseded_by` pointer column, the system leverages the existing `connections` infrastructure. Stale memories have their `file_link` and `memory_link` statuses updated to `historical_trace`. This maintains the "chain of belief" for audit queries without database schema bloat.

> [!NOTE]
> Existing rows in `processed_files` receive `content_hash = NULL`, which the watcher treats as "unknown — re-check on next poll." This is the safe conservative default for live deployments.

---

### Change 2 — Hash-Gated Watcher Logic in `watch_folder`

`agent.py:524-589` — the watcher loop needs two new behaviours:

**New file (no row in `processed_files`):** behaviour unchanged — ingest and record.

**Known file, same hash:** skip (current behaviour, now hash-verified).

**Known file, changed hash (update detected):**

1. Compute MD5 of new file content.
2. Calculate the **relative path** from the inbox root: `rel_path = str(f.relative_to(folder))`.
3. Call `memory_store.get_memories_by_source(rel_path)` to find all MemCubes linked to this file.
4. For each linked memory ID:
   - Call `update_memory_validity(memory_id, now_iso)` — sets `valid_to = now()`.
   - Call `update_link_status(memory_id, rel_path, "historical_trace")` — marks the `file_link` as historical.
5. Ingest the new content as fresh MemCubes: `await agent.ingest(text, source=rel_path, metadata={"ingest_source": "watcher_poll"})`.

> [!CAUTION]
> **Correctness vs. Availability Trade-off:** Step 4 expires memories _before_ the new ingestion completes. This creates a ~30s "Total Amnesia" window where the system has no data for that file. As an agentic tool, AOM prioritizes **correctness (no contradictions)** over transient availability.

6. On ingest **success**: `UPDATE processed_files SET prev_hash = content_hash, content_hash = ?, processed_at = ? WHERE path = ?`.
7. On ingest **failure**:
   - Delete new memories created by the failed poll: `DELETE FROM memories WHERE source = ? AND created_at >= ? AND metadata LIKE '%"ingest_source": "watcher_poll"%'`.
   - Restore original memories: clear `valid_to` and set link status back to `active`.
   - **Fix:** Remove the malformed `(FAILED)` string from the `processed_at` timestamp insertion; simply do not write a new hash record so the next poll retries.

**Partial-ingest race condition (reviewer-identified):** `agent.ingest()` is an async LLM call that writes new memories to the database incrementally before returning. If it raises an exception partway through, new memories may already exist in the DB while the old memories have had `valid_to` set. Simply un-expiring the old memories leaves both old and new coexisting — producing duplicates on every transient failure.

The cleanup query `DELETE FROM memories WHERE source = str(f) AND created_at >= expiry_ts` handles this by removing memories created since the expiry started. The `expiry_ts` timestamp must be recorded _before_ step 3 begins to create a clean boundary.

**Crash safety:** If the process crashes between steps 3 and 5, `valid_to` stamps remain in place but no hash is written. On restart, the watcher re-detects the hash difference and re-runs the sweep. Any orphaned partial-ingest memories from the previous crash will be caught by the cleanup query in the new except block — or will be treated as stale (expired) until the sweep completes.

**Performance:** Before the per-file loop, batch the `processed_files` lookup into a single query:

```python
rows = db.execute(
    f"SELECT path, content_hash FROM processed_files WHERE path IN ({','.join('?' * len(files))})",
    [str(f) for f in files]
).fetchall()
hash_map = {row['path']: row['content_hash'] for row in rows}
```

This avoids O(N) individual DB round-trips per 5-second poll cycle when the inbox has many files.

---

### Change 3 — Portability & Recursive Globbing

`agent.py:531` — switch to recursive scanning and relative pathing for the `source` field.

**Recursive logic:**

```python
# agent.py:531
files = sorted(f for f in folder.rglob('*') if f.is_file())
```

**Portability Fix:**
The current `source = f.name` creates collisions for files with the same name in different subdirectories. However, using the absolute path (`str(f)`) breaks the database if the workspace is moved.

**New Standard:** Always pass the **relative path** from the inbox root as `source`.

```python
# agent.py:552
rel_path = str(f.relative_to(folder))
await agent.ingest(text, source=rel_path, metadata={"ingest_source": "watcher_poll"})
```

This ensures `source` is globally unique within the inbox tree and remains valid even if the parent directory path changes.

---

### New Helper: `get_memories_by_source` in `memory_store.py`

```python
def get_memories_by_source(rel_path: str) -> list[int]:
    """Return memory IDs linked to the given relative file path."""
    # 1. Direct source match
    # 2. file_link path match (handles legacy memories stored as absolute paths)
```

The `file_link` branch serves as a migration fallback: if it doesn't find a match for the relative path, it should attempt to match the absolute path suffix to recover memories created before the portability fix.

---

## What This Does NOT Change

- The `documents` / `vec_documents` tables used by Librarian — they are already hash-gated and untouched.
- The adversarial consolidation harness, AutoDream, or Deep Reconsolidation — no changes to `agents_factory.py`.
- The MCP server or REST API — no new endpoints are needed. The `/links` endpoint (`GET /links`) already surfaces `file_link` status, so `historical_trace` entries will be visible there automatically.
- The vector embedding pipeline — superseded memories retain their embeddings in `vec_memories`. There is no standalone KNN/vector-search path over `vec_memories`, so no filter is needed there. Orphaned embeddings will be cleaned up naturally when AutoDream decays a superseded memory's `importance_score` to zero (triggering deletion of both the `memories` and `vec_memories` rows at `agent.py:469-470`).

---

## Change 4 — `valid_to` Filter on Consolidation Read Paths

> **Reviewer correction:** `read_all_memories` (`memory_store.py:92`) already filters on `valid_to`. The "Critical companion" change originally stated here was a false requirement — that filter was added in a prior sprint.

**The actual gap** is three functions used as agent tools that do **not** filter `valid_to`. A memory marked superseded will still be fed into AutoDream clustering and consolidation via these paths, potentially causing it to re-emerge in a synthesised Insight Cube:

| Function                              | Line | Used by                           |
| ------------------------------------- | ---- | --------------------------------- |
| `read_unconsolidated_memories`        | 132  | Consolidation agent (30-min loop) |
| `read_unconsolidated_with_embeddings` | 155  | AutoDream clustering              |
| `read_memory_partition`               | 229  | Self-improvement agent (EvoSkill) |

**Fix:** Add `AND (valid_to IS NULL OR valid_to > datetime('now'))` to the `WHERE` clause of each of these three functions.

Example for `read_unconsolidated_memories`:

```sql
-- Before
SELECT * FROM memories WHERE consolidated = 0 ORDER BY created_at LIMIT ?
-- After
SELECT * FROM memories WHERE consolidated = 0
  AND (valid_to IS NULL OR valid_to > datetime('now'))
ORDER BY created_at LIMIT ?
```

---

## Issue #11's Additional Proposals — Evaluation

Issue #11 proposed an `ARCHITECTURAL_DIRECTIVE` memory priority class for the Deep Reconsolidation prompt. This is **out of scope for this brief** — it is a significant agent-prompt change with unclear evaluation criteria, and the hash-gating + invalidation approach already solves the contradictions problem at the data layer before the consolidator even sees them. This can be revisited if stale-data contradictions persist after the above changes are deployed.

---

## Affected Files Summary

| File              | Change                                                                                                                                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `database.py`     | Add `content_hash`, `prev_hash` to `processed_files`; add `superseded_by` to `memories`; run additive migrations in `init_db()`                                                                             |
| `agent.py`        | `watch_folder`: switch to `rglob('*')`, batch DB lookup, hash-gated update detection, Semantic Sweep with rollback                                                                                          |
| `memory_store.py` | Add `get_memories_by_source(filename, full_path)`; add `valid_to` filter to `read_unconsolidated_memories` (line 132), `read_unconsolidated_with_embeddings` (line 155), `read_memory_partition` (line 229) |

**Functions to reuse (do not duplicate):**

- `update_memory_validity(memory_id, valid_to)` — `memory_store.py:467` — sets/clears `valid_to` on a memory row
- `update_link_status(memory_id, path, new_status)` — `memory_store.py:490` — sets `file_link` connection status
- `repair_memory(memory_id, new_raw_text)` — `memory_store.py:521` — updates `raw_text` and resets timestamp

**Required co-change — `update_memory_validity` type signature:** The current signature is `valid_to: str`. Change 2's rollback calls `update_memory_validity(memory_id, None)` to clear the field. Python won't raise at runtime (SQLite stores `None` as NULL correctly), but static type checkers (mypy, Pylance) will flag every rollback call site as a type error. Update the signature to `Optional[str]`:

```python
# memory_store.py:467
def update_memory_validity(memory_id: int, valid_to: Optional[str]) -> Dict[str, Any]:
```

**Edge case — failed ingest: remove malformed timestamp sentinel.** The current except block at `agent.py:573-578` writes a deliberately broken `processed_at` value:

```python
(str(f), datetime.now(timezone.utc).isoformat() + " (FAILED)")
```

This string is not a valid ISO timestamp. Any future code that parses `processed_at` as a datetime will raise. The fix should not just omit the hash — it should remove this entire `INSERT INTO processed_files` call from the except block. On failure, no row is written at all, so the next poll retries ingestion cleanly.

---

## Test Coverage

Existing test files to extend:

- `tests/test_core_unittest.py` — add test for hash-change detection and re-ingest
- `tests/test_edge_cases.py` — add test for file-in-subdirectory ingestion (recursive)
- `tests/test_consolidation_flow.py` — add test that superseded memories (`valid_to` set) do not appear in query results

New test scenario (integration):

1. Drop `inbox/docs/notes.txt` → confirm ingestion.
2. Overwrite `inbox/docs/notes.txt` with different content → confirm re-ingestion + old memory `valid_to` is set.
3. Query for content from the old version → confirm it does not appear (verify `valid_to` filter).
4. Verify `source` and `file_link` use relative, not absolute, paths.

---

## Research Synthesis: External Literature & Applied Insights

The following is a synthesis of 14 external sources read in full. Each section maps source findings directly to decisions in this brief or surfaces patterns worth carrying forward.

---

### Memory Lifecycle: What the Field Has Solved (and What It Hasn't)

**mem0** ([repo](https://github.com/mem0ai/mem0)) advertises "continuous learning over time" but its invalidation logic is purely LLM-based (probabilistic). It lacks the **deterministic content-hash triggers** required for source-synced documentation. mem0 is the cautionary example for deterministic systems: probabilistic invalidation is insufficient when ground truth is based on specific file versions.

**MemOS** ([repo](https://github.com/MemTensor/MemOS)) takes a replacement-based approach: memories can be "correcting, supplementing, or replacing" via natural-language feedback APIs and a deletion endpoint. This is human-in-the-loop — a user explicitly corrects a memory. AOM's inbox watcher scenario requires _automated_ detection of source-document changes, which MemOS does not address. However, MemOS's deletion-by-ID API is exactly what the Semantic Sweep step (Change 2, step 3) uses — the function already exists in `memory_store.py`.

**Key synthesis:** Neither mem0 nor MemOS implement automated hash-based invalidation. AOM's proposed design (content hash → automatic supersession sweep) is ahead of both.

---

### The "Dirty Flag" Pattern — SemaClaw

SemaClaw ([arXiv:2604.11548](https://arxiv.org/abs/2604.11548)) marks memory files as **"dirty"** when compaction occurs, signalling that surrounding content requires re-indexing before the next retrieval — preventing stale results during compaction windows.

**Applied to AOM — operation ordering matters:** When `watch_folder` detects a hash change, the linked MemCubes must be marked dirty (`valid_to = now()`) _before_ the re-ingest call fires. If re-ingest fails partway through, dirty memories are at least excluded from retrieval rather than leaving a half-replaced set visible. The sequence must be:

1. Mark old memories superseded (`valid_to = now()`)
2. Ingest new content
3. On ingest failure → roll back the `valid_to` stamps

SemaClaw also uses a **50-day FIFO rolling window** for daily logs, treating older entries as progressively lower-priority. AOM's AutoDream importance decay is structurally equivalent. The key distinction: AutoDream applies _soft_ decay; file-update detection should apply a _hard_ expiry (`valid_to`) — the two mechanisms serve different triggers.

SemaClaw's **source tracking** via a `source:` designation field mirrors AOM's `file_link` connection type. Both scope queries to curated vs. historical knowledge. AOM's `/links` endpoint already surfaces this.

---

### Importance Decay as Soft Invalidation — Awesome-Agent-Memory Survey

The [Awesome-Agent-Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory) survey catalogues **widemem-ai** (2025), which implements "batch conflict resolution with importance scoring and temporal decay." This is the soft-invalidation approach: rather than hard-expiring old memories on file update, decay their `importance_score` aggressively so they lose retrieval priority without disappearing.

**Applied to AOM — a two-tier invalidation model for future consideration:**

| Tier     | Trigger                        | Mechanism                                          | Effect                                         |
| -------- | ------------------------------ | -------------------------------------------------- | ---------------------------------------------- |
| **Soft** | Minor hash change (small diff) | Decay `importance_score` by 0.3 on linked memories | Old memories rank lower but remain retrievable |
| **Hard** | Significant hash change        | Set `valid_to = now()` on linked memories          | Old memories fully excluded from retrieval     |

AOM has no diff-size detection yet, so **hard invalidation for all hash changes** is the correct conservative default for this sprint. The soft tier can be introduced once diff-size detection is available.

The survey also references **HippoRAG** (2024), which distinguishes between memories "still active in episodic form" vs. "consolidated into semantic schema." The implication: `sector = 'episodic'` memories from an outdated file version should be retired first; `sector = 'semantic'` memories derived by multi-source consolidation are more durable and may survive a file update if they have been synthesised with other independent sources.

**Applied to AOM:** The Semantic Sweep in Change 2 should retire `sector = 'episodic'` memories unconditionally, then check `sector = 'semantic'` memories for independence from the updated source (via the `connections` field) before expiring them.

---

### Context Rot & The Ralph Loop — LangChain Harness Anatomy

LangChain's harness anatomy ([article](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness/)) identifies **Context Rot** — performance degradation as stale intermediate results accumulate in the context window — as the core failure mode in long-running agents. Their counter-measure is the **Ralph Loop**: each iteration starts with a _clean_ context window, but reads persisted state from the previous iteration's filesystem.

**Applied to AOM:** The `consolidation_loop` and `autodream_loop` already implement the Ralph Loop pattern — each cycle reads from the database (persistent state) and runs fresh agent instances. The missing piece is that stale inbox memories pollute the "persistent state" that these fresh cycles read. Hash-gated invalidation is the prerequisite for Context Rot not accumulating in the memory database itself.

The article also establishes **Git versioning as a memory primitive** — rollback, branching, history. For AOM, this suggests storing `prev_hash TEXT DEFAULT NULL` in `processed_files` so that a file restored to a previous version can be detected and the `valid_to` stamps reversed.

---

### Plan Compliance & Memory Refresh Rate — arXiv:2604.12147

"From Plan to Action" ([arXiv:2604.12147](https://arxiv.org/abs/2604.12147)) found that agent plan fidelity degrades as context accumulates — and that **periodic reinsertion ("Reminded Plan Setting") improved compliance by 2–4%**. The mechanism: stale high-frequency patterns overwhelm newer, lower-frequency instructions numerically.

**Applied to AOM:** Old inbox memories from a superseded file version will, by cosine similarity, rank highly for queries referencing the same topic. Because they have high access counts and established `importance_score` values, they numerically out-compete newer memories. The `valid_to` hard-expiry is the correct counter-measure — it removes them from the retrieval pool entirely rather than trying to out-score them.

This directly validates the "Open Design Question" in this brief: **`valid_to` filtering in `read_all_memories` is not optional.** Without it, superseded memories will re-surface in exactly the same pattern this paper describes.

---

### `superseded_by` Pointer vs. Hard Expiry — Memory in the Age of AI Agents

"Memory in the Age of AI Agents" ([arXiv:2512.13564](https://arxiv.org/abs/2512.13564)) proposes a full lifecycle model:

> **Stages:** encoding → storage → retrieval → utilization → forgetting/expiration

For invalidation, the paper surveys three strategies:

1. **TTL-based expiry** — a `valid_to` field (already exists in AOM's schema)
2. **Event-triggered invalidation** — a new document version cascades invalidation to all derived MemCubes (the Semantic Sweep in Change 2)
3. **Confidence decay** — `importance_score` decreases as a function of time since last corroboration (AOM's AutoDream already does this)

Critically, the paper recommends **storing both old and new versions rather than hard deletion.** Rationale: temporal reasoning — "what did the agent believe at time T?"

**Applied to AOM — Memory Links vs Column Pointers:**
The Audit and Review confirmed that "Memory Links can show historical overviews, while File Links describe current ground truths." Setting `valid_to = now()` on superseded memories correctly excludes them from live queries. Auditability is preserved by updating link statuses:

- **Live queries:** filter `WHERE valid_to IS NULL OR valid_to > datetime('now')` — stale memories excluded.
- **Audit queries:** join on `connections` (Memory Links) where status is `historical_trace` to reconstruct the agent's belief state.
- **Rollback:** if a re-ingest was erroneous, clear `valid_to` and restore link status to `active`.

This avoids adding a `superseded_by` pointer column, keeping the schema clean while fulfilling the "Temporal Audit" research requirement via the existing connection graph.

---

### Reward Hacking & The ARCHITECTURAL_DIRECTIVE Proposal — Anthropic Research

Anthropic's automated alignment research ([article](https://www.anthropic.com/research/automated-alignment-researchers)) documents **reward hacking**: agents finding correlative shortcuts that game evaluation metrics. In AOM's adversarial consolidation context, the equivalent risk is: if a deprecated file version's memories _outnumber_ the new version's memories (the file was just updated), the Generator-Evaluator harness may "vote" to preserve the old content as the majority view.

**This validates Issue #11's `ARCHITECTURAL_DIRECTIVE` proposal** — the concern is real. However, the recommended implementation order stands: hard invalidation (removing old memories from the pool _before_ consolidation runs) solves the problem at the data layer. `ARCHITECTURAL_DIRECTIVE` is a harness-layer safety net for partial-invalidation edge cases — Phase 2, not Phase 1.

The research also recommends **tamper-evident logging**: ground-truth records that agents cannot overwrite. For AOM, the `processed_files` hash record must be written by the deterministic watcher loop, not by agent inference. Agent tools should have no write access to `processed_files`.

---

### Meta-Harness: Full History Access & Additive-Safe Schema Changes

Meta-Harness ([arXiv:2603.28052](https://arxiv.org/abs/2603.28052)) emphasises that effective harness optimisation requires **full execution history** for causal diagnosis. The proposer identifies confounded design changes by comparing traces across prior runs.

**Applied to AOM:** The current `processed_files` schema stores only `path` and `processed_at`. Adding `content_hash` (Change 1) is the minimum. Consider also storing `prev_hash TEXT DEFAULT NULL` to enable:

- Rollback detection: file restored to previous version → reverse `valid_to` stamps on the matching old memories
- Future diff-size estimation: string distance between hashes as a soft/hard tier trigger
- Audit trail: debuggability when a memory was unexpectedly superseded

The Meta-Harness principle of **"safe additive improvements"** directly maps to the migration strategy: `ALTER TABLE processed_files ADD COLUMN` is additive (no data loss). Existing rows get `NULL` hash, which the watcher treats as "unknown — re-check on next poll." This is the correct conservative default for live deployments.

---

### Open SWE: Mid-Run Update Injection

Open SWE ([article](https://www.langchain.com/blog/open-swe-an-open-source-framework-for-internal-coding-agents/)) injects mid-execution updates (new Linear comments, Slack messages) via `check_message_queue_before_model` middleware without restarting the agent run.

**Applied to AOM — concurrency safety:** If a file is updated while an ingest cycle is in-flight for that file (rare but possible), the hash record in `processed_files` acts as a natural serialisation lock. The in-flight ingest commits the old hash; the _next_ 5-second poll detects the updated hash and triggers re-ingestion. No additional locking is required — the hash record itself serialises concurrent updates.

---

### Eval Strategy — LangChain Deep Agents Evals

LangChain's eval post ([article](https://www.langchain.com/blog/how-we-build-evals-for-deep-agents/)) defines memory eval across three axes: contextual recall, preference learning, and state persistence. The principle: **"More evals ≠ better agents. Build targeted evals that reflect desired production behaviours."**

The three-step integration test in this brief maps directly to their axes:

| Test Step                                  | LangChain Eval Axis                    |
| ------------------------------------------ | -------------------------------------- |
| New file dropped → memory created          | State persistence                      |
| File overwritten → old memories superseded | State persistence: correct retirement  |
| Query old-version content → not returned   | Contextual recall: stale data excluded |

This confirms the three-step test is sufficient for this sprint. A broad regression suite is not needed.

---

### Literature Validation Summary

| Design Choice in This Brief                                                            | Literature Support                                                                                                      |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Hard `valid_to` expiry (not soft decay) for file updates                               | SemaClaw dirty-flag ordering; arXiv:2604.12147 (stale memories numerically dominate by cosine similarity)               |
| Hash-gated watcher (not mtime)                                                         | Consistent with Librarian.py; Meta-Harness additive-safe approach                                                       |
| `valid_to` filter on consolidation read paths (not `read_all_memories` — already done) | arXiv:2604.12147; LangChain Context Rot — stale memories re-enter synthesis via AutoDream clustering                    |
| Retire `episodic` memories first, check `semantic` for multi-source independence       | HippoRAG episodic vs. semantic distinction (Awesome-Agent-Memory survey)                                                |
| `ARCHITECTURAL_DIRECTIVE` deferred to Phase 2                                          | Anthropic reward-hacking findings validate the concern; data-layer fix removes stale memories before consolidation runs |
| Store `prev_hash` alongside `content_hash` in `processed_files`                        | Meta-Harness full-history access principle; enables rollback and diff-size estimation                                   |
| Dirty-flag ordering: expire old → ingest new → rollback on failure                     | SemaClaw dirty-flag pattern                                                                                             |
| `superseded_by` pointer for temporal auditability                                      | arXiv:2512.13564 — store both versions, enable "what did agent believe at time T?" queries                              |
| Batch `processed_files` lookup before per-file loop                                    | Performance: avoids O(N) DB round-trips per 5-second poll                                                               |
| 3-step integration test is sufficient                                                  | LangChain evals: targeted over broad; maps to contextual recall + state persistence axes                                |
