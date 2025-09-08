#!/usr/bin/env python3
"""
Test MCP server startup without stdio
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import logging
from src.mcp_retrieval_server import RetrievalMCPServer

async def test_init():
    try:
        print("Initializing RetrievalMCPServer...")
        server = RetrievalMCPServer()
        print("Server initialized successfully")
        print("Server object created:", server)
        return True
    except Exception as e:
        print("Server initialization failed:", e)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(test_init())
    print("Test result:", "PASS" if result else "FAIL")