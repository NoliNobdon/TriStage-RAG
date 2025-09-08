#!/usr/bin/env python3
"""
MCP Server startup script for Claude Code integration
"""

import sys
import os
import asyncio
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp_retrieval_server import main

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    import asyncio
    asyncio.run(main())