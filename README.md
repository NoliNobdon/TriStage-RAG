# TriStage-RAG

A state-of-the-art 3-stage retrieval system based on the theoretical foundations from "On the Theoretical Limitations of Embedding-Based Retrieval" - optimized for 4GB VRAM systems.

## 🎯 Project Overview

**TriStage-RAG** implements a theoretically-grounded 3-stage retrieval pipeline that addresses fundamental limitations of single-vector embedding models through progressive refinement:

```
Query → Stage 1 (Fast Candidate Generation) → Stage 2 (Multi-Vector Rescoring) → Stage 3 (Cross-Encoder Reranking) → Results
```

### Theoretical Foundation

Based on Weller et al. (2025) "On the Theoretical Limitations of Embedding-Based Retrieval":

> **"We demonstrate that we may encounter these theoretical limitations in realistic settings with extremely simple queries. We connect known results in learning theory, showing that the number of top-k subsets of documents capable of being returned as the result of some query is limited by the dimension of the embedding."**

**Key Findings:**
- **Embedding Dimension Barrier**: Fixed dimension `d` cannot represent all document combinations
- **Combinatorial Explosion**: Web-scale search requires computationally infeasible dimensions  
- **Real-World Impact**: State-of-the-art models fail on simple tasks testing these limits

### 3-Stage Solution

Our architecture addresses these limitations:

- **Stage 1**: Fast candidate generation using FAISS + optional BM25 (500-800 candidates)
- **Stage 2**: Multi-vector rescoring with ColBERT-style MaxSim (top 100)  
- **Stage 3**: Cross-encoder reranking with direct relevance scoring (top 20)

## 🏗️ System Architecture

```
┌─────────────────┐
│    Query Input  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│    Stage 1      │
│ Fast Candidate  │
│   Generation    │
│ (FAISS + BM25)  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│    Stage 2      │
│ Multi-Vector    │
│   Rescoring     │
│ (ColBERT MaxSim)│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│    Stage 3      │
│ Cross-Encoder   │
│   Reranking     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐
│   Results       │    │   MCP Server    │
│   (Top 20)      │◄───│   Interface     │
└─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
rag_mcp/
├── src/                        # Core implementation
│   ├── embedding_service.py    # Single-model embedding service  
│   ├── mcp_retrieval_server.py # MCP server with 7 tools
│   ├── retrieval_pipeline.py  # 3-stage pipeline orchestrator
│   ├── stage1_retriever.py     # Stage 1: FAISS + BM25
│   ├── stage2_rescorer.py      # Stage 2: ColBERT MaxSim
│   ├── stage3_reranker.py      # Stage 3: Cross-encoder
│   └── __init__.py            # Package initialization
├── config/
│   └── config.yaml           # Unified configuration
├── tests/
│   ├── validate_mcp_server.py # MCP server tests
│   └── test_pipeline.py       # Pipeline tests
├── models/                    # Downloaded models (~2-5GB)
├── faiss_index/               # FAISS index storage
├── docs/                      # Research papers
├── demo.py                    # Interactive demo
├── monitor_usage.py           # Usage monitoring
└── README.md                  # This file
```

## 🚀 System Components

### Stage 1: Fast Candidate Generation
- **Model**: `google/embeddinggemma-300m`
- **Technology**: FAISS index + optional BM25 hybrid search
- **Output**: ~500-800 candidates
- **Features**: Reciprocal rank fusion, configurable top-k

### Stage 2: Multi-Vector Rescoring  
- **Model**: `lightonai/GTE-ModernColBERT-v1`
- **Technology**: ColBERT-style MaxSim scoring
- **Optimization**: 192 token sequences for 4GB VRAM
- **Output**: Top 100 rescored candidates

### Stage 3: Cross-Encoder Reranking
- **Model**: `cross-encoder/ms-marco-MiniLM-L6-v2`
- **Technology**: Direct query-document relevance scoring
- **Optimization**: 256 token length, adaptive batching
- **Output**: Final top 20 ranked results

### MCP Server Interface
- **7 Tools**: search, add_documents, batch_search, get_pipeline_status, clear_index, health_check, get_document_count
- **3 Resources**: pipeline://info, pipeline://config, pipeline://status
- **Features**: Document tracking, index management, health monitoring

## 🎯 Use Cases

- **Enterprise Search**: High-accuracy document retrieval
- **RAG Applications**: Enhanced LLM responses with relevant context
- **Research**: Academic paper and literature search
- **Knowledge Management**: Organizational document retrieval
- **Customer Support**: Intelligent FAQ and support document search

## 🛠️ Key Features

### Theoretically Grounded
- **Addresses Embedding Limits**: Goes beyond single-vector constraints
- **Progressive Refinement**: Each stage overcomes different limitations
- **Multi-Vector Approach**: Captures relationships single vectors cannot represent

### Performance Optimized
- **4GB VRAM Ready**: All models configured for limited GPU memory
- **16GB RAM Compatible**: Efficient memory management
- **Fast Search**: 350-900ms end-to-end latency
- **Memory Management**: Automatic GPU cleanup and optimization

### Production Ready
- **Comprehensive Logging**: Detailed performance tracking
- **Error Handling**: Robust exception management with fallbacks
- **Configuration Management**: YAML-based flexible config
- **MCP Integration**: Standard protocol for AI applications

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd rag_mcp

# Install dependencies
pip install torch sentence-transformers faiss-cpu transformers pyyaml mcp numpy

# Configure environment
cp .env.example .env
# Edit .env with your Hugging Face token if needed
```

### Basic Usage

#### Interactive Demo
```bash
# Run the interactive demo
python demo.py
```

#### MCP Server
```bash
# Start the MCP retrieval server
python src/mcp_retrieval_server.py
```

#### Programmatic Pipeline Usage
```python
from src.retrieval_pipeline import RetrievalPipeline

# Initialize pipeline
pipeline = RetrievalPipeline('config/config.yaml')

# Add documents
documents = ["Your documents here..."]
pipeline.add_documents(documents)

# Search with 3-stage refinement
results = pipeline.search("Your query here")
for result in results['results']:
    print(f"Score: {result['stage3_score']:.4f}")
    print(f"Document: {result['document'][:100]}...")
```

#### MCP Server Tools
```python
# Available MCP tools:
- search: Perform 3-stage retrieval search
- add_documents: Add documents to index
- batch_search: Multiple search queries
- get_pipeline_status: Pipeline information
- clear_index: Clear all documents
- health_check: System health status
- get_document_count: Number of indexed documents
```

## 📊 Performance Characteristics

### Expected Performance (4GB VRAM System)
| Component | Time | VRAM Usage | Output |
|-----------|------|-----------|--------|
| **Stage 1** | 50-150ms | ~1GB | 500-800 candidates |
| **Stage 2** | 200-400ms | ~2GB | 100 candidates |
| **Stage 3** | 100-250ms | ~1GB | 20 results |
| **Total** | **350-800ms** | **~4GB** | **20 results** |

### Memory Requirements
- **VRAM**: 3-4GB peak (GPU recommended, CPU fallback available)
- **RAM**: 8-16GB depending on dataset size
- **Storage**: 2-5GB for model caches

## 🔧 Configuration

Unified YAML configuration (`config/config.yaml`):

```yaml
# 3-Stage Retrieval Pipeline Configuration
pipeline:
  device: "cuda"  # cuda, cpu, auto
  cache_dir: "./models"
  index_dir: "./faiss_index"
  
  # Stage 1: Fast Candidate Generation
  stage1:
    model: "google/embeddinggemma-300m"
    top_k: 500
    batch_size: 32
    enable_bm25: true
    bm25_top_k: 300
    fusion_method: "rrf"
    use_fp16: true
    
  # Stage 2: Multi-Vector Rescoring
  stage2:
    model: "lightonai/GTE-ModernColBERT-v1"
    top_k: 100
    batch_size: 16
    max_seq_length: 192
    use_fp16: true
    scoring_method: "maxsim"
    
  # Stage 3: Cross-Encoder Reranking
  stage3:
    model: "cross-encoder/ms-marco-MiniLM-L6-v2"
    top_k: 20
    batch_size: 32
    max_length: 256
    use_fp16: true
```

## 🧪 Testing

### Test Suite
```bash
# Test the 3-stage pipeline
python tests/test_pipeline.py

# Validate MCP server functionality  
python tests/validate_mcp_server.py

# Run interactive demo
python demo.py

# Monitor system usage
python monitor_usage.py
```

## 🎯 Why This Architecture?

### Beyond Single-Vector Limitations

Traditional RAG systems rely on single-vector embeddings, but research shows fundamental theoretical barriers:

1. **Dimension Barrier**: Fixed embedding dimension `d` limits representable document combinations
2. **Combinatorial Explosion**: Web-scale search requires infeasible dimensions  
3. **Expressive Limits**: Some document relationships cannot be captured

### Our 3-Stage Solution

1. **Stage 1**: Overcomes combinatorial explosion with efficient retrieval
2. **Stage 2**: Multi-vector scoring captures complex relationships
3. **Stage 3**: Direct relevance scoring avoids embedding constraints

**Theoretical Advantages:**
- **Progressive Refinement**: Each stage addresses different limitations
- **Beyond Single Vectors**: Multi-vector and cross-encoder approaches
- **Expressive Power**: Represents theoretically impossible combinations for single vectors

## 🔮 Future Roadmap

### Planned Enhancements
- **Additional Models**: Support for newer embedding and reranking models
- **Distributed Processing**: Multi-GPU and horizontal scaling
- **Real-time Updates**: Incremental index updates
- **Advanced Analytics**: Query performance insights
- **Caching Layer**: Intelligent result caching
- **Performance Optimization**: Further memory and speed improvements

### Model Alternatives
- **Stage 1**: `all-MiniLM-L6-v2`, `e5-small-v2`
- **Stage 2**: `ColBERTv2`, `rank_T5`
- **Stage 3**: `bge-reranker-base`, `MiniLM-L6-v2`

## 🤝 Contributing

This is an active research and development project. Contributions are welcome!

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes with tests
4. **Validate** with the test suite
5. **Submit** a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section in docs
2. Review test outputs and logs
3. Open an issue with detailed error information
4. Include system specs (GPU/RAM) and configuration

---

**Built with ❤️ at the intersection of information retrieval theory and practical AI systems**

*Inspired by "On the Theoretical Limitations of Embedding-Based Retrieval" by Weller et al. (2025)*