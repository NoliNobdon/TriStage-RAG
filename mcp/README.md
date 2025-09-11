# MCP Retrieval Server

Run the TriStage retrieval pipeline as an MCP stdio server for IDEs and agents.

## Quick start

1) Install dependencies (from repo root):

```cmd
pip install -r requirements.txt
```

2) Configure `mcp/config.yaml` (models, index, logging). Set any secrets in `.env` at the repo root (e.g., `HUGGING_FACE_HUB_TOKEN`).

3) Run the config-driven server (recommended):

```cmd
python run_mcp_server_config.py
```

This loads the pipeline using `mcp/config.yaml` (single source of truth) and starts the stdio server.

Alternative (defaults from src, ignores YAML):

```cmd
python run_mcp_server.py
```

## Paths
- Models cache: `./models`
- FAISS index: `./faiss_index`
- Logs: defined in `mcp/config.yaml` (e.g., `retrieval_pipeline.log`)

## Notes
- Uses stdio transport; integrate with clients that speak MCP (e.g., Claude Code).
- Logging level/file are controlled by YAML; no forced overrides in the runner.
- If using gated models (e.g., `google/embeddinggemma-300m`), ensure your HF token is available via `.env`.
