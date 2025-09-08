#!/usr/bin/env python3
"""
Test MCP server imports
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.mcp_retrieval_server import main
    print("Import successful!")
    print("Main function exists:", callable(main))
except Exception as e:
    print("Import failed:", e)
    import traceback
    traceback.print_exc()