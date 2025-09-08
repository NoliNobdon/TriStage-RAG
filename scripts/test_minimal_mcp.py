#!/usr/bin/env python3
"""
Minimal MCP server test
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import logging
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from mcp.server import NotificationOptions

async def main():
    logging.basicConfig(level=logging.DEBUG)
    
    # Create a minimal server
    server = Server("test-mcp")
    
    # Add a simple tool
    @server.list_tools()
    async def list_tools():
        return []
    
    print("Starting minimal MCP server...")
    
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            print("Stdio server created, running...")
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="test-mcp",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        print("Error running server:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())