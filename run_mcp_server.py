#!/usr/bin/env python3
"""
Working MCP server runner for Claude Code
"""
import sys
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Set up the environment
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

# Load environment variables from .env at repo root (for HF tokens, etc.)
env_path = Path(project_root) / ".env"
if load_dotenv and env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")

# Import and run the server
from src.mcp_retrieval_server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())