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
from pathlib import Path
import sys
import os

from embedding_service import EmbeddingService, EmbeddingConfig

class EmbeddingInput(BaseModel):
    text: str = Field(..., description="Text to generate embeddings for")
    instruction: Optional[str] = Field(None, description="Optional instruction to guide embedding generation")

class DocumentInput(BaseModel):
    documents: List[str] = Field(..., description="List of documents to generate embeddings for")
    instructions: Optional[List[str]] = Field(None, description="Optional instructions for each document")

class SimilarityInput(BaseModel):
    query_embedding: List[float] = Field(..., description="Query embedding vector")
    document_embeddings: List[List[float]] = Field(..., description="Document embedding vectors")

class BatchInput(BaseModel):
    documents_list: List[List[str]] = Field(..., description="List of document batches to process")

class ModelSelectionInput(BaseModel):
    model_type: str = Field(..., description="Model type to use (default, code, document, multilingual)")

class ModelAnalysisInput(BaseModel):
    text: str = Field(..., description="Text to analyze for model selection")

class EmbeddingMCPServer:
    def __init__(self, config_path: str = "config.yaml"):
        self.server = Server("embedding-mcp")
        self.embedding_service = EmbeddingService(config_path)
        self.logger = logging.getLogger(__name__)
        self._setup_handlers()
        
    def _setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="encode_query",
                    description="Generate embeddings for a single text query",
                    inputSchema=EmbeddingInput.model_json_schema(),
                ),
                types.Tool(
                    name="encode_documents",
                    description="Generate embeddings for multiple documents",
                    inputSchema=DocumentInput.model_json_schema(),
                ),
                types.Tool(
                    name="compute_similarity",
                    description="Compute similarity between query and document embeddings",
                    inputSchema=SimilarityInput.model_json_schema(),
                ),
                types.Tool(
                    name="batch_encode",
                    description="Process multiple document batches efficiently",
                    inputSchema=BatchInput.model_json_schema(),
                ),
                types.Tool(
                    name="get_model_info",
                    description="Get information about the current embedding model",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="clear_cache",
                    description="Clear the embedding cache",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="health_check",
                    description="Check the health status of the embedding service",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="set_model",
                    description="Manually set the embedding model type",
                    inputSchema=ModelSelectionInput.model_json_schema(),
                ),
                types.Tool(
                    name="analyze_text",
                    description="Analyze text to determine which model would be selected",
                    inputSchema=ModelAnalysisInput.model_json_schema(),
                ),
                types.Tool(
                    name="list_models",
                    description="List all available models and their status",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
            if arguments is None:
                arguments = {}
                
            try:
                if name == "encode_query":
                    return await self._encode_query(arguments)
                elif name == "encode_documents":
                    return await self._encode_documents(arguments)
                elif name == "compute_similarity":
                    return await self._compute_similarity(arguments)
                elif name == "batch_encode":
                    return await self._batch_encode(arguments)
                elif name == "get_model_info":
                    return await self._get_model_info(arguments)
                elif name == "clear_cache":
                    return await self._clear_cache(arguments)
                elif name == "health_check":
                    return await self._health_check(arguments)
                elif name == "set_model":
                    return await self._set_model(arguments)
                elif name == "analyze_text":
                    return await self._analyze_text(arguments)
                elif name == "list_models":
                    return await self._list_models(arguments)
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
                    uri="model://info",
                    name="Model Information",
                    description="Current model specifications and capabilities",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="model://config",
                    name="Model Configuration",
                    description="Current model configuration parameters",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="model://status",
                    name="Model Status",
                    description="Current model status and performance metrics",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            try:
                if uri == "model://info":
                    return await self._get_model_info_resource()
                elif uri == "model://config":
                    return await self._get_config_resource()
                elif uri == "model://status":
                    return await self._get_status_resource()
                else:
                    raise ValueError(f"Unknown resource: {uri}")
            except Exception as e:
                self.logger.error(f"Error reading resource {uri}: {e}")
                return json.dumps({"error": str(e)})

    async def _encode_query(self, arguments: dict) -> list[types.TextContent]:
        input_data = EmbeddingInput(**arguments)
        embedding = self.embedding_service.encode_query(input_data.text)
        
        result = {
            "embedding": embedding.tolist(),
            "shape": list(embedding.shape),
            "text_length": len(input_data.text),
            "instruction": input_data.instruction
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    async def _encode_documents(self, arguments: dict) -> list[types.TextContent]:
        input_data = DocumentInput(**arguments)
        
        if input_data.instructions and len(input_data.instructions) != len(input_data.documents):
            raise ValueError("Number of instructions must match number of documents")
            
        embeddings = self.embedding_service.encode_document(input_data.documents)
        
        result = {
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape),
            "document_count": len(input_data.documents),
            "embedding_dimension": embeddings.shape[1]
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    async def _compute_similarity(self, arguments: dict) -> list[types.TextContent]:
        input_data = SimilarityInput(**arguments)
        
        query_embedding = np.array(input_data.query_embedding)
        document_embeddings = np.array(input_data.document_embeddings)
        
        similarities = self.embedding_service.similarity(query_embedding, document_embeddings)
        
        result = {
            "similarities": similarities.tolist(),
            "shape": list(similarities.shape),
            "document_count": len(input_data.document_embeddings),
            "most_similar_index": int(np.argmax(similarities)),
            "highest_similarity": float(np.max(similarities))
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    async def _batch_encode(self, arguments: dict) -> list[types.TextContent]:
        input_data = BatchInput(**arguments)
        results = self.embedding_service.batch_encode_documents(input_data.documents_list)
        
        result = {
            "batch_count": len(results),
            "results": [result.tolist() for result in results],
            "batch_shapes": [list(result.shape) for result in results]
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    async def _get_model_info(self, arguments: dict) -> list[types.TextContent]:
        info = self.embedding_service.get_model_info()
        
        return [types.TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]

    async def _clear_cache(self, arguments: dict) -> list[types.TextContent]:
        self.embedding_service.clear_cache()
        
        return [types.TextContent(
            type="text",
            text=json.dumps({"message": "Cache cleared successfully"}, indent=2)
        )]

    async def _health_check(self, arguments: dict) -> list[types.TextContent]:
        models_loaded = len(self.embedding_service._models)
        cache_size = len(self.embedding_service._cache) if self.embedding_service.config.enable_caching else 0
        current_model = self.embedding_service.get_current_model()
        
        health_status = {
            "status": "healthy" if models_loaded > 0 else "no_models_loaded",
            "models_loaded": models_loaded,
            "current_model": current_model,
            "cache_size": cache_size,
            "config": {
                "default_model": self.embedding_service.config.default_model,
                "device": self.embedding_service.config.device,
                "enable_caching": self.embedding_service.config.enable_caching,
                "batch_size": self.embedding_service.config.batch_size,
                "auto_selection_enabled": True
            }
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(health_status, indent=2)
        )]
    
    async def _set_model(self, arguments: dict) -> list[types.TextContent]:
        input_data = ModelSelectionInput(**arguments)
        success = self.embedding_service.set_model(input_data.model_type)
        
        result = {
            "success": success,
            "model_type": input_data.model_type,
            "current_model": self.embedding_service.get_current_model(),
            "message": f"Model set to {input_data.model_type}" if success else f"Failed to set model to {input_data.model_type}"
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    async def _analyze_text(self, arguments: dict) -> list[types.TextContent]:
        input_data = ModelAnalysisInput(**arguments)
        selected_model = self.embedding_service._select_model(input_data.text)
        
        result = {
            "text": input_data.text[:100] + "..." if len(input_data.text) > 100 else input_data.text,
            "selected_model": selected_model,
            "confidence": "high" if selected_model != "default" else "medium",
            "reasoning": {
                "code": "Code keywords detected" if selected_model == "code" else None,
                "multilingual": "Multiple languages detected" if selected_model == "multilingual" else None,
                "document": "General document content" if selected_model == "document" else None,
                "default": "Using default model" if selected_model == "default" else None
            }
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    async def _list_models(self, arguments: dict) -> list[types.TextContent]:
        available_models = self.embedding_service.get_available_models()
        
        result = {
            "available_models": available_models,
            "current_model": self.embedding_service.get_current_model(),
            "total_models": len(available_models),
            "loaded_models": sum(1 for model in available_models.values() if model.get("loaded", False))
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    async def _get_model_info_resource(self) -> str:
        info = self.embedding_service.get_model_info()
        return json.dumps(info, indent=2)

    async def _get_config_resource(self) -> str:
        config_dict = {
            "model_name": self.embedding_service.config.model_name,
            "device": self.embedding_service.config.device,
            "max_length": self.embedding_service.config.max_length,
            "batch_size": self.embedding_service.config.batch_size,
            "cache_dir": self.embedding_service.config.cache_dir,
            "enable_caching": self.embedding_service.config.enable_caching,
            "cache_size": self.embedding_service.config.cache_size,
            "parallel_processing": self.embedding_service.config.parallel_processing,
            "max_workers": self.embedding_service.config.max_workers,
            "max_text_length": self.embedding_service.config.max_text_length,
            "min_text_length": self.embedding_service.config.min_text_length,
            "allowed_languages": self.embedding_service.config.allowed_languages
        }
        return json.dumps(config_dict, indent=2)

    async def _get_status_resource(self) -> str:
        model_loaded = self.embedding_service._model is not None
        cache_size = len(self.embedding_service._cache) if self.embedding_service.config.enable_caching else 0
        
        status = {
            "model_loaded": model_loaded,
            "cache_size": cache_size,
            "service_initialized": True,
            "performance_metrics": {
                "cache_hit_rate": 0.0,  # Placeholder
                "avg_processing_time": 0.0,  # Placeholder
                "total_requests": 0  # Placeholder
            }
        }
        return json.dumps(status, indent=2)

async def main():
    logging.basicConfig(level=logging.INFO)
    server = EmbeddingMCPServer()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="embedding-mcp",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())