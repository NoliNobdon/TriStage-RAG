#!/usr/bin/env python3
"""
Test MCP server stdio communication
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import logging
from src.mcp_retrieval_server import RetrievalMCPServer
import mcp.server.stdio

async def test_stdio():
    try:
        print("Creating MCP server...")
        server = RetrievalMCPServer()
        print("Server created successfully")
        
        print("Setting up stdio server...")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            print("Stdio server created successfully")
            print("Read stream:", read_stream)
            print("Write stream:", write_stream)
            print("Streams are valid")
            return True
    except Exception as e:
        print("Stdio setup failed:", e)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    result = asyncio.run(test_stdio())
    print("Test result:", "PASS" if result else "FAIL")