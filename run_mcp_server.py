#!/usr/bin/env python3
"""
Working MCP server runner for Claude Code
"""
import sys
import os

# Set up the environment
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, project_root)

# Import and run the server
from src.mcp_retrieval_server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())