#!/usr/bin/env python3
"""
Fixed MCP server runner that handles relative imports correctly
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now we can import as a package
from src.mcp_retrieval_server import main

if __name__ == "__main__":
    main()