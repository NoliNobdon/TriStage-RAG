#!/usr/bin/env python3
"""
Run the MCP retrieval server using settings from mcp/config.yaml without modifying src.
"""
import os
import sys
from pathlib import Path
import yaml
import logging

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Ensure project root on path and set CWD
PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env for HF tokens, etc.
if load_dotenv:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

from src.retrieval_pipeline import RetrievalPipeline, PipelineConfig
from src.mcp_retrieval_server import RetrievalMCPServer
import mcp.server.stdio
from mcp.server import NotificationOptions
from mcp.server.models import InitializationOptions


def _build_pipeline_config_from_yaml(cfg_path: Path) -> PipelineConfig:
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    p = data.get("pipeline", {})
    s1 = p.get("stage1", {})
    s2 = p.get("stage2", {})
    s3 = p.get("stage3", {})

    return PipelineConfig(
        # General
        device=p.get("device", "auto"),
        cache_dir=p.get("cache_dir", "./models"),
        index_dir=p.get("index_dir", "./faiss_index"),
        log_level=p.get("log_level", "INFO"),
        log_file=p.get("log_file", "retrieval_pipeline.log"),
        enable_timing=p.get("enable_timing", True),
        save_intermediate_results=p.get("save_intermediate_results", False),
        auto_cleanup=p.get("auto_cleanup", True),
        max_memory_usage_gb=p.get("max_memory_usage_gb", 4.0),

        # Stage 1
        stage1_model=s1.get("model", "google/embeddinggemma-300m"),
        stage1_top_k=s1.get("top_k", 500),
        stage1_batch_size=s1.get("batch_size", 32),
        stage1_enable_bm25=s1.get("enable_bm25", True),
        stage1_bm25_top_k=s1.get("bm25_top_k", 300),
        stage1_fusion_method=s1.get("fusion_method", "rrf"),
        stage1_use_fp16=s1.get("use_fp16", True),

        # Stage 2
        stage2_model=s2.get("model", "lightonai/GTE-ModernColBERT-v1"),
        stage2_top_k=s2.get("top_k", 100),
        stage2_batch_size=s2.get("batch_size", 16),
        stage2_max_seq_length=s2.get("max_seq_length", 192),
        stage2_use_fp16=s2.get("use_fp16", True),
        stage2_scoring_method=s2.get("scoring_method", "maxsim"),

        # Stage 3
        stage3_model=s3.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        stage3_top_k=s3.get("top_k", 20),
        stage3_batch_size=s3.get("batch_size", 32),
        stage3_max_length=s3.get("max_length", 256),
        stage3_use_fp16=s3.get("use_fp16", True),
    )


async def main():
    # Let the pipeline configure logging based on config.log_level/log_file

    cfg_path = PROJECT_ROOT / "mcp" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # Build pipeline with config.yaml
    pipeline_cfg = _build_pipeline_config_from_yaml(cfg_path)
    pipeline = RetrievalPipeline(config=pipeline_cfg)

    # Start server and inject configured pipeline
    server = RetrievalMCPServer()
    server.pipeline = pipeline

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="retrieval-mcp",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
