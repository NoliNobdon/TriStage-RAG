# Non-MCP TriStage Retrieval App (Chat-first, no LLM)
It is still in progress

A standalone, three-stage retrieval system you can run without an MCP client. It uses the same pipeline components under `src/` and supports both a CLI and a simple web UI for embedding documents and asking questions.

## Features
- Stage 1: fast candidate generation (embeddings + optional BM25)
- Stage 2: multi-vector rescoring (ColBERT-style)
- Stage 3: cross-encoder reranking
- Local model cache in `../models` (top-level repo `models/`)
- Index in `../faiss_index`
- Document persistence in `../data`
- Web UI: ChatGPT-style chat first. Upload files, Embed, then ask questions.
- No LLM generation: answers are composed from top retrieved snippets.
- Embed settings page: control chunk size/overlap, embed from repo `documents/`, and see status.
- JSON APIs for search, stats, clearing data, embedded manifest, and documents status.

## Requirements
- Python 3.10+
- Windows (cmd) instructions shown, but commands are platform-agnostic

Install dependencies:

```cmd
pip install -r non_mcp\requirements.txt
```

This includes: sentence-transformers, torch, faiss-cpu, transformers, Flask (for the web UI), pypdf, python-docx, etc.

Note: If you use gated models (e.g., `google/embeddinggemma-300m`), ensure your Hugging Face credentials are configured in your environment.

## Quick start (Web UI)
Run the simple web interface to embed files and ask questions:

```cmd
python non_mcp\webui\app.py
```

Open http://127.0.0.1:5051 (lands on the chat view)

- Click Embed (header) to open embed settings, or use the upload + Embed near the composer
- Embed files (txt/json/md/pdf/docx), optionally from the repo `documents/` folder
- Ask a question in chat; answers show top snippets only (no model generation)

Environment variables (optional):
- `NON_MCP_WEBUI_LOG_LEVEL` (default: INFO)
- `NON_MCP_WEBUI_LOG_FILE` (default: non_mcp_webui.log)
- `NON_MCP_DEVICE` (default: cpu)

## Quick start (CLI)
Interactive console app with document management and search:

```cmd
python non_mcp\main.py
```

Helpful flags:

```cmd
python non_mcp\main.py --help
```

Examples:

```cmd
:: Use repo paths by default and search a one-off query
python non_mcp\main.py --query "neural networks" --top-k 5

:: Load docs from a file or folder before querying
python non_mcp\main.py --load non_mcp\test_docs.json --query "attention mechanism" --top-k 5

:: Use YAML config (overrides device/paths/logging) and enable debug logs
python non_mcp\main.py --config non_mcp\pipeline_config.yaml --log-level DEBUG
```

Tiny helper: return only the top Stage-3 passage (no UI/CLI loop):

```cmd
python non_mcp\respond_stage3.py "your question"
:: optionally ingest a folder of .txt/.md files (Stage 1 index will be saved for reuse)
python non_mcp\respond_stage3.py "your question" documents
```

CLI options include:
- `--models-dir` (default: `../models`)
- `--data-dir` (default: `../data`)
- `--index-dir` (default: `../faiss_index`)
- `--device` (`cpu`/`cuda`)
- `--query`, `--top-k`, `--load`
- `--log-level`, `--log-file`, `--config`

## Paths and storage
- Models: `../models` (top-level repo `models/`), e.g., `models/embeddinggemma-300m`
- Index: `../faiss_index`
- Data: `../data` (documents persisted by the app)

You can change these via CLI flags or a config YAML (`non_mcp/pipeline_config.yaml`).

## Web API
- `GET /api/search?q=your+query&top_k=10` → JSON search
- `GET /api/stats` → system info and counters
- `POST /api/clear` → clear all documents/index and chat history
- `GET /api/embedded` → embedded manifest (tracked entries)
- `GET /api/documents-status` → status of files under `documents/` (embedded or not)

## Troubleshooting
- Gated models (e.g., `google/embeddinggemma-300m`) require Hugging Face auth. Configure HF token in your environment before first use.
- GPU vs CPU: set `--device cuda` (CLI) or `NON_MCP_DEVICE=cuda` (web UI). Falls back to CPU if CUDA isn’t available.
- Logs:
  - CLI: `--log-level DEBUG --log-file retrieval_pipeline.log`
  - Web UI: `NON_MCP_WEBUI_LOG_LEVEL=DEBUG` (file at `non_mcp_webui.log`)
- Index missing/empty: embed documents first (web UI `/chat` → upload + Embed, or CLI `--load`).

## Project structure (non_mcp)
```
non_mcp/
  main.py                 # CLI app
  pipeline_config.yaml    # Example config
  requirements.txt        # Non-MCP app deps
  test_docs.json          # Sample documents
  webui/
    app.py                # Flask web UI
    templates/
  chat.html           # Chat-like Q&A UI (default)
  embed.html          # Embed settings page
```

## License
This component is part of the TriStage-RAG repository and follows the repository’s LICENSE.
