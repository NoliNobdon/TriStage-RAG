"""
MCP server module for 3-stage RAG pipeline
This module provides MCP server integration for the 3-stage retrieval pipeline
"""

def serve():
    """Serve the MCP server"""
    print("MCP server for 3-stage RAG pipeline starting...")
    return True

def register_tristage_tools(server):
    """Register 3-stage tools with MCP server"""
    if hasattr(server, 'add_tool'):
        # Add search tool
        server.add_tool({
            "name": "tristage_search",
            "description": "Search using 3-stage retrieval pipeline",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        })
        
        # Add documents tool
        server.add_tool({
            "name": "tristage_add_documents",
            "description": "Add documents to the pipeline",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["documents"]
            }
        })
        
        # Add pipeline info tool
        server.add_tool({
            "name": "tristage_get_pipeline_info",
            "description": "Get pipeline information",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        })
        
        # Add clear cache tool
        server.add_tool({
            "name": "tristage_clear_cache",
            "description": "Clear pipeline cache",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "clear_cache": {"type": "boolean", "default": True}
                },
                "required": []
            }
        })
    
    return True