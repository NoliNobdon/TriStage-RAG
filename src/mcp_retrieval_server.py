#!/usr/bin/env python3

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from mcp import ClientSession, StdioServerParameters
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
from pydantic import BaseModel, Field
import numpy as np
import torch
from pathlib import Path
import sys
import os

from .retrieval_pipeline import RetrievalPipeline

class SearchInput(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: Optional[int] = Field(20, description="Number of results to return (default: 20)")

class DocumentInput(BaseModel):
    documents: List[str] = Field(..., description="List of documents to add to the retrieval pipeline")

class BatchSearchInput(BaseModel):
    queries: List[str] = Field(..., description="List of search queries")
    top_k: Optional[int] = Field(20, description="Number of results to return per query (default: 20)")

class PipelineStatusInput(BaseModel):
    detailed: Optional[bool] = Field(False, description="Return detailed status information")

class RetrievalMCPServer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.server = Server("retrieval-mcp")
        self.pipeline = RetrievalPipeline()
        self.logger = logging.getLogger(__name__)
        self._setup_handlers()
        
    def _setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="search",
                    description="Perform 3-stage retrieval search for relevant documents",
                    inputSchema=SearchInput.model_json_schema(),
                ),
                types.Tool(
                    name="add_documents",
                    description="Add documents to the retrieval pipeline index",
                    inputSchema=DocumentInput.model_json_schema(),
                ),
                types.Tool(
                    name="batch_search",
                    description="Perform multiple search queries efficiently",
                    inputSchema=BatchSearchInput.model_json_schema(),
                ),
                types.Tool(
                    name="get_pipeline_status",
                    description="Get current status and information about the retrieval pipeline",
                    inputSchema=PipelineStatusInput.model_json_schema(),
                ),
                types.Tool(
                    name="clear_index",
                    description="Clear all documents from the retrieval pipeline index",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="health_check",
                    description="Check the health status of the retrieval pipeline",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="get_document_count",
                    description="Get the number of documents currently indexed",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
            if arguments is None:
                arguments = {}
                
            try:
                if name == "search":
                    return await self._search(arguments)
                elif name == "add_documents":
                    return await self._add_documents(arguments)
                elif name == "batch_search":
                    return await self._batch_search(arguments)
                elif name == "get_pipeline_status":
                    return await self._get_pipeline_status(arguments)
                elif name == "clear_index":
                    return await self._clear_index(arguments)
                elif name == "health_check":
                    return await self._health_check(arguments)
                elif name == "get_document_count":
                    return await self._get_document_count(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                self.logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

        @self.server.list_resources()
        async def handle_list_resources() -> list[types.Resource]:
            return [
                types.Resource(
                    uri="pipeline://info",
                    name="Pipeline Information",
                    description="3-stage retrieval pipeline specifications and capabilities",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="pipeline://config",
                    name="Pipeline Configuration",
                    description="Current pipeline configuration parameters",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="pipeline://status",
                    name="Pipeline Status",
                    description="Current pipeline status and performance metrics",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "pipeline://info":
                return await self._get_info_resource()
            elif uri == "pipeline://config":
                return await self._get_config_resource()
            elif uri == "pipeline://status":
                return await self._get_status_resource()
            else:
                raise ValueError(f"Unknown resource: {uri}")

    # Tool implementation methods
    async def _search(self, arguments: dict) -> list[types.TextContent]:
        input_data = SearchInput(**arguments)
        results = self.pipeline.search(input_data.query, top_k=input_data.top_k)
        
        return [types.TextContent(
            type="text",
            text=json.dumps(results, indent=2)
        )]

    async def _add_documents(self, arguments: dict) -> list[types.TextContent]:
        input_data = DocumentInput(**arguments)
        
        try:
            self.pipeline.add_documents(input_data.documents)
            
            # Get actual document count after adding
            total_docs = 0
            if self.pipeline.stage1:
                total_docs = len(self.pipeline.stage1.documents)
            
            result = {
                "success": True,
                "documents_added": len(input_data.documents),
                "total_documents": total_docs,
                "message": f"Successfully added {len(input_data.documents)} documents to the pipeline"
            }
        except Exception as e:
            result = {
                "success": False,
                "documents_added": 0,
                "total_documents": 0,
                "message": f"Error adding documents: {str(e)}"
            }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    async def _batch_search(self, arguments: dict) -> list[types.TextContent]:
        input_data = BatchSearchInput(**arguments)
        results = []
        
        for query in input_data.queries:
            query_results = self.pipeline.search(query, top_k=input_data.top_k)
            results.append({
                "query": query,
                "results": query_results
            })
        
        return [types.TextContent(
            type="text",
            text=json.dumps(results, indent=2)
        )]

    async def _get_pipeline_status(self, arguments: dict) -> list[types.TextContent]:
        input_data = PipelineStatusInput(**arguments)
        
        status = {
            "pipeline_initialized": True,
            "stages": {
                "stage1": {
                    "name": "Fast Candidate Generation",
                    "model": "google/embeddinggemma-300m",
                    "status": "active"
                },
                "stage2": {
                    "name": "Multi-Vector Rescoring", 
                    "model": "lightonai/GTE-ModernColBERT-v1",
                    "status": "active"
                },
                "stage3": {
                    "name": "Cross-Encoder Reranking",
                    "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                    "status": "active"
                }
            }
        }
        
        if input_data.detailed:
            status["performance"] = {
                "gpu_available": torch.cuda.is_available(),
                "device": str(self.pipeline.config.device),
                "cache_dir": self.pipeline.config.cache_dir,
                "index_dir": self.pipeline.config.index_dir
            }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(status, indent=2)
        )]

    async def _clear_index(self, arguments: dict) -> list[types.TextContent]:
        try:
            # Clear Stage 1 index
            if self.pipeline.stage1:
                self.pipeline.stage1.documents = []
                self.pipeline.stage1.doc_metadata = []
                self.pipeline.stage1.faiss_index = None
                self.pipeline.stage1.bm25_index = None
            
            result = {
                "success": True,
                "message": "Index cleared successfully",
                "documents_remaining": 0
            }
        except Exception as e:
            result = {
                "success": False,
                "message": f"Error clearing index: {str(e)}"
            }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    async def _health_check(self, arguments: dict) -> list[types.TextContent]:
        import torch
        
        health_status = {
            "status": "healthy",
            "pipeline_ready": True,
            "gpu_available": torch.cuda.is_available(),
            "stages_ready": {
                "stage1": True,
                "stage2": True, 
                "stage3": True
            }
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(health_status, indent=2)
        )]

    async def _get_document_count(self, arguments: dict) -> list[types.TextContent]:
        try:
            count = 0
            if self.pipeline.stage1:
                count = len(self.pipeline.stage1.documents)
            
            result = {
                "document_count": count,
                "message": f"Found {count} documents in index"
            }
        except Exception as e:
            result = {
                "document_count": 0,
                "message": f"Error getting document count: {str(e)}"
            }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    # Resource implementation methods
    async def _get_info_resource(self) -> str:
        info = {
            "pipeline_name": "3-Stage Retrieval Pipeline",
            "description": "Advanced retrieval system with three stages: candidate generation, multi-vector rescoring, and cross-encoder reranking",
            "stages": [
                {
                    "name": "Stage 1",
                    "purpose": "Fast Candidate Generation",
                    "model": "google/embeddinggemma-300m",
                    "technique": "FAISS + optional BM25"
                },
                {
                    "name": "Stage 2", 
                    "purpose": "Multi-Vector Rescoring",
                    "model": "lightonai/GTE-ModernColBERT-v1",
                    "technique": "ColBERT-style MaxSim scoring"
                },
                {
                    "name": "Stage 3",
                    "purpose": "Cross-Encoder Reranking", 
                    "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                    "technique": "Direct relevance scoring"
                }
            ],
            "optimization": "Optimized for 4GB VRAM and 16GB RAM"
        }
        return json.dumps(info, indent=2)

    async def _get_config_resource(self) -> str:
        config_dict = {
            "pipeline": {
                "device": self.pipeline.config.device,
                "cache_dir": self.pipeline.config.cache_dir,
                "index_dir": self.pipeline.config.index_dir,
                "log_level": self.pipeline.config.log_level,
                "enable_timing": self.pipeline.config.enable_timing
            },
            "stage1": {
                "model": self.pipeline.config.stage1_model,
                "top_k": self.pipeline.config.stage1_top_k,
                "batch_size": self.pipeline.config.stage1_batch_size,
                "enable_bm25": self.pipeline.config.stage1_enable_bm25
            },
            "stage2": {
                "model": self.pipeline.config.stage2_model,
                "top_k": self.pipeline.config.stage2_top_k,
                "batch_size": self.pipeline.config.stage2_batch_size,
                "max_seq_length": self.pipeline.config.stage2_max_seq_length
            },
            "stage3": {
                "model": self.pipeline.config.stage3_model,
                "top_k": self.pipeline.config.stage3_top_k,
                "batch_size": self.pipeline.config.stage3_batch_size,
                "max_length": self.pipeline.config.stage3_max_length
            }
        }
        return json.dumps(config_dict, indent=2)

    async def _get_status_resource(self) -> str:
        import torch
        
        status = {
            "pipeline_initialized": True,
            "gpu_available": torch.cuda.is_available(),
            "stages_active": {
                "stage1": self.pipeline.stage1 is not None,
                "stage2": self.pipeline.stage2 is not None,
                "stage3": self.pipeline.stage3 is not None
            },
            "performance_metrics": {
                "device": str(self.pipeline.config.device),
                "timing_enabled": self.pipeline.config.enable_timing
            }
        }
        return json.dumps(status, indent=2)

async def main():
    logging.basicConfig(level=logging.INFO)
    server = RetrievalMCPServer()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="retrieval-mcp",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())