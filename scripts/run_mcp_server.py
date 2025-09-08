#!/usr/bin/env python3
"""
Simple MCP server runner
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp_retrieval_server import main

if __name__ == "__main__":
    main()