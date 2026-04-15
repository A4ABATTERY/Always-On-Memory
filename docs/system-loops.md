# Autonomous System Loops

AOM v3 is designed to be "Always-On" and self-maintaining. It achieves this through several background loops that run concurrently to optimize, consolidate, and verify the memory system.

## 1. Adversarial Consolidation Loop
### The "Dreaming" Cycle
- **Interval**: Every 30 minutes (configurable).
- **Trigger**: When unconsolidated memories reach a threshold.
- **Process**:
    - Select a cluster of related `MemCubes`.
    - The **Generator Agent** synthesizes a parent "Insight Cube".
    - The **Evaluator Agent** performs an adversarial audit.
    - If the score is valid (e.g., > 0.85), the new Insight Cube is saved, and the sources are ARCHIVED.
- **Result**: Drastic reduction in query context size while preserving the most important facts.

## 2. AutoDream Optimization Loop
### The "Deeper Reflection" Cycle
- **Interval**: Triggered during system idle time (configured by `IDLE_THRESHOLD_MINUTES`).
- **Process**:
    - **Adversarial Reconsolidation**: Runs deeper, higher-quality consolidation using more powerful models (e.g., `gemini-3-flash-preview`).
    - **Redundancy Pruning**: Identifies and merges overlapping or redundant Insight Cubes.
    - **Topic Reorganization**: Regenerates the global topic map based on the new, higher-density memories.
- **Result**: A pristine, hierarchically organized knowledge base.

## 3. Sync Worker & Librarian Loop
### The "Self-Healing" Cycle
- **Interval**: Real-time (via file system watchers) and periodic scans.
- **Process**:
    - **Librarian Scan**: Watches the project code for any modifications.
    - **Drift Detection**: When a file changes, the Librarian calculates the new embedding and compares it to linked `MemCubes`.
    - **Sync Queueing**: If semantic drift exceeds the `DRIFT_THRESHOLD` (e.g., 0.18), a task is pushed to the **Sync Queue**.
    - **Sync Auditor Execution**: The Auditor analyzes the code change and updates the link status:
        - `ACTIVE`: Still valid.
        - `REPAIR`: System suggests an update to the memory.
        - `HISTORICAL`: The implementation has diverged completely from the original intent.
- **Result**: Memoirs and documentation never become "stale" or disconnected from reality.

## 4. Importance Decay Loop
### The "Forgetting" Cycle
- **Process**: Gradually decays the `importance_score` of memories that are not frequently recalled or referenced.
- **Result**: Ensures the context window is always prioritized for the most relevant and "fresh" insights.

## 5. Inbox Watcher & Semantic Invalidation Loop
### The "Ingestion & Update Tracking" Cycle
- **Interval**: Real-time poll (every 5s).
- **Process**:
    - **Recursive Scan**: Deep scans the `inbox/` for new or modified nested files.
    - **Update Tracking**: Compares MD5 content hashes against stored `processed_files`.
    - **Semantic Invalidation**: If a known file is modified:
        - Prevents contradictions by immediately marking old linked memories as superseded (`valid_to = now()`) and transitioning connections to `historical_trace`.
        - Safely re-ingests the updated content to produce fresh MemCubes.
        - Supports a robust transactional roll-back in case the new ingestion fails gracefully during API interactions.
- **Result**: Guarantees the system stays semantically aligned with the latest source documents without retaining contradiction-causing stale memory contexts.
