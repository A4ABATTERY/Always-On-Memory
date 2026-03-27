# Getting Started

This guide will help you set up and run the Always-On-Memory (AOM) v3 system for development and testing.

## 📋 Prerequisites

- **Python**: 3.11 or higher.
- **SQLite**: 3.35+ (for JSON support).
- **Google API Key**: A valid key for the Gemini models.
- **sqlite-vec**: Ensure the `sqlite-vec` extension is available (usually installed via pip).

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/A4ABATTERY/Always-On-Memory.git
   cd Always-On-Memory
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

Create a `.env` file in the root directory with the following variables:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
WATCH_DIRS=/path/to/your/project/src,/path/to/another/dir
IDLE_THRESHOLD_MINUTES=30
```

Refer to [Data Model & Persistence](data-model.md) for details on how `WATCH_DIRS` affects the Librarian indexer.

## 🏃 Running the Agent

Start the agent with the following command:

```bash
python agent.py --watch ./inbox --port 8000
```

This will start:
- The **Inbox Watcher** on `./inbox`.
- The **Librarian** on your `WATCH_DIRS`.
- The **HTTP API** on port `8000`.
- The **Background Loops** (Consolidation, AutoDream).

## 🧪 Documentation "Hello World"

To verify your setup, try ingesting a new memory via the API:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "AOM is now operational", "source": "setup"}'
```

Then, query the system:

```bash
curl "http://localhost:8000/query?q=what+is+the+status+of+AOM"
```

## 🧪 Running Tests

AOM uses a comprehensive test suite. Always run this before submitting a PR:

```bash
./.venv/bin/python -m unittest discover tests
```

---
*Next Step: Explore the [Architecture Overview](architecture.md) to understand how the components interact.*
