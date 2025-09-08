# 3-Stage Retrieval Pipeline MCP Server

## Overview

The MCP server provides a Model Context Protocol interface for the advanced 3-stage retrieval pipeline. It enables efficient document retrieval with GPU acceleration using three sophisticated stages:

1. **Stage 1**: Fast candidate generation with FAISS + optional BM25
2. **Stage 2**: Multi-vector rescoring with ColBERT-style MaxSim
3. **Stage 3**: Cross-encoder reranking for final ranking

## Available Tools

### 1. `search`
Perform 3-stage retrieval search for relevant documents.

**Input:**
```json
{
  "query": "What is machine learning?",
  "top_k": 20  // Optional, defaults to 20
}
```

**Output:**
```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "doc_id": 0,
      "document": "Machine learning is a subset of artificial intelligence...",
      "score": 0.032,
      "stage": "stage3",
      "stage2_score": 0.905,
      "stage3_score": 1.0
    }
  ],
  "timing": {
    "stage1_time": 0.248,
    "stage2_time": 0.191,
    "stage3_time": 0.058,
    "total_time": 0.499
  }
}
```

### 2. `add_documents`
Add documents to the retrieval pipeline index.

**Input:**
```json
{
  "documents": [
    "Document 1 text content...",
    "Document 2 text content...",
    "Document 3 text content..."
  ]
}
```

**Output:**
```json
{
  "success": true,
  "documents_added": 3,
  "total_documents": 3,
  "message": "Successfully added 3 documents to the pipeline"
}
```

### 3. `batch_search`
Perform multiple search queries efficiently.

**Input:**
```json
{
  "queries": ["What is quantum computing?", "Explain neural networks"],
  "top_k": 10  // Optional, defaults to 20
}
```

### 4. `get_pipeline_status`
Get current status and information about the retrieval pipeline.

**Input:**
```json
{
  "detailed": true  // Optional, defaults to false
}
```

### 5. `health_check`
Check the health status of the retrieval pipeline.

**Input:**
```json
{}
```

### 6. `get_document_count`
Get the number of documents currently indexed.

**Input:**
```json
{}
```

### 7. `clear_index`
Clear all documents from the retrieval pipeline index.

**Input:**
```json
{}
```

## Available Resources

### 1. `pipeline://info`
Pipeline specifications and capabilities.

### 2. `pipeline://config`
Current pipeline configuration parameters.

### 3. `pipeline://status`
Current pipeline status and performance metrics.

## Usage Examples

### Basic Search
```bash
# Call the search tool
{
  "name": "search",
  "arguments": {
    "query": "What is machine learning?",
    "top_k": 5
  }
}
```

### Add Documents
```bash
# Call the add_documents tool
{
  "name": "add_documents",
  "arguments": {
    "documents": [
      "Machine learning is a subset of artificial intelligence...",
      "Deep learning uses neural networks with multiple layers..."
    ]
  }
}
```

### Batch Search
```bash
# Call the batch_search tool
{
  "name": "batch_search",
  "arguments": {
    "queries": [
      "What is quantum computing?",
      "How do neural networks work?"
    ],
    "top_k": 3
  }
}
```

## Performance

The pipeline is optimized for:
- **GPU Memory**: Uses ~2.9-3.1GB of 4GB VRAM
- **System RAM**: Uses ~13.2GB of 16GB total
- **Search Speed**: Average 0.4-0.5 seconds per query
- **GPU Acceleration**: All stages use CUDA with FP16 precision

## Configuration

The pipeline uses the configuration from `config.yaml` with settings for:
- Model selection and parameters
- GPU/CPU device configuration
- Memory optimization
- Performance tuning

## Running the Server

```bash
# From the project directory
cd src
python mcp_retrieval_server.py
```

The server will initialize and load all three stages with GPU acceleration if available.