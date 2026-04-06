# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for sqlite-vec native extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source (respects .dockerignore)
COPY . .

# Create default volume mount points
RUN mkdir -p /inbox /watchdirs

# REST API port and MCP server port
EXPOSE 8888 8765

# Environment variable documentation (all overridable at runtime)
ENV GOOGLE_API_KEY=""
ENV MODEL="gemini-3.1-flash-lite-preview"
ENV SMART_MODEL="gemini-3-flash-preview"
ENV MEMORY_DB="/app/memory.db"
ENV INBOX_DIR="/inbox"
ENV WATCH_DIRS="/watchdirs"
ENV MCP_PORT="8765"
ENV MCP_HOST="0.0.0.0"
ENV AOM_API_KEYS=""
# Set AOM_MCP_NO_AUTH=true only for localhost-only deployments.
# Leaving AOM_API_KEYS empty without this flag causes startup to fail with RuntimeError.
ENV AOM_MCP_NO_AUTH=""

ENTRYPOINT ["python", "agent.py"]
CMD ["--watch", "/inbox", "--port", "8888", "--mcp-port", "8765"]
