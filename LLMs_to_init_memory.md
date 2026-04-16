# Instruction: Always-On-Memory (AOM) Initialization

Use this instruction when starting up a fresh AOM instance or upgrading to `gemini-embedding-2-preview` to ensure the memory layer is correctly grounded and semantic drift is minimized.

## Phase 0: Environment & Model Pre-check
If you are upgrading from a previous embedding model (e.g., `text-embedding-004`) to `gemini-embedding-2-preview`:
1.  **Semantic Mismatch**: New embedding prefixes (`title: | text:`) create a semantic mismatch.
2.  **Mandatory Re-index**: You **must** delete your `memory.db` file to trigger a full re-index of the codebase and memories.
3.  **Client Singleton**: Ensure `utils.py` is configured with the `google-genai` SDK v0.3.0+.

## Phase 1: Codebase Grounding
AOM's Librarian automatically indexes source code, but high-level architectural insights must be explicitly generated and ingested.

1.  **Identify Watch Directories**: Read the `WATCH_DIRS` environment variable from the project's `.env` file.
2.  **Generate Core Summaries**: For each directory in `WATCH_DIRS`, perform a high-level scan (READMEs, entry points, architecture docs).
3.  **Populate Initial Memories**: Ingest the following via `POST /ingest` (Sector: `Semantic`):
    *   **Project Purpose**: A 2-3 sentence overview of what the project does.
    *   **Tech Stack**: Key languages, frameworks, and databases used.
    *   **Core Modules**: High-level mapping of the directory structure to functional components.
    *   **Style/Patterns**: Standard coding patterns or architectural rules found in the codebase.

## Phase 2: Conversation History Salvage
To provide continuity, scan the local system for previous conversation artifacts from other LLM tools and ingest summarized highlights.

### Search Locations (OS-Specific)

| Tool | Linux Path | macOS Path | Windows Path |
| :--- | :--- | :--- | :--- |
| **Antigravity** | `~/.gemini/antigravity/brain/` | `~/.gemini/antigravity/brain/` | `%USERPROFILE%\.gemini\antigravity\brain\` |
| **Claude Code** | `~/.claude/projects/` | `~/.claude/projects/` | `~\.claude\projects\` |
| **Cursor Chat** | `~/.config/Cursor/User/workspaceStorage/` | `~/Library/Application/Support/Cursor/User/workspaceStorage/` | `%APPDATA%\Cursor\User\workspaceStorage\` |
| **VS Code Copilot**| `~/.config/Code/User/workspaceStorage/` | `~/Library/Application/Support/Code/User/workspaceStorage/` | `%APPDATA%\Code\User\workspaceStorage\` |
| **Continue.dev**| `~/.continue/` | `~/.continue/` | `~\.continue\` |
| **Windsurf** | `~/.codeium/windsurf/cascade/` | `~/.codeium/windsurf/cascade/` | `~\.codeium\windsurf\cascade\` |
| **OpenCode** | `~/.local/share/opencode/` | `~/.local/share/opencode/` | `%USERPROFILE%\.local\share\opencode\` |

### Extraction & Ingestion Strategy
1.  **Iterate**: Walk the paths above looking for recent `.md`, `.json`, or `.jsonl` files.
2.  **Summarize**: For the 5-10 most recent sessions, extract:
    *   **Key Decisions**: "Switched to PydanticAI", "Enabled TurboQuant".
    *   **Resolved Issues**: "Fixed initialization deadlock in agent.py".
    *   **Current State**: "Working on V3.3 upgrade".
3.  **Ingest**: Save these as **Episodic** (for events) or **Reflection** (for lessons learned) memories.

## Phase 3: Final Verification
1.  **Check Status**: Call `GET /status`.
2.  **Verify**: Ensure `total_memories > 0` and `indexed_documents` reflects the codebase size.
3.  **Test Query**: Run a test query like `GET /query?q=Summarize the project architecture` to verify grounding. 
