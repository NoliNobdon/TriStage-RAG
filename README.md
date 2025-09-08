# RAG MCP: Advanced Retrieval-Augmented Generation System

A state-of-the-art RAG system combining a production-ready Model Context Protocol (MCP) embedding server with an advanced 3-stage retrieval pipeline optimized for 4GB VRAM systems.

## 🎯 Project Overview

This project implements a complete RAG solution with two main components:

1. **MCP Embedding Server** - Production-grade text embedding generation with intelligent multi-model selection
2. **3-Stage Retrieval Pipeline** - State-of-the-art search architecture with progressive refinement

### Key Innovation: 3-Stage Retrieval Architecture

```
Query → Stage 1 (Fast Candidate Generation) → Stage 2 (Multi-Vector Rescoring) → Stage 3 (Cross-Encoder Reranking) → Results
```

- **Stage 1**: Retrieve 500-800 candidates using `google/embeddinggemma-300m` + FAISS + optional BM25
- **Stage 2**: Refine to top 100 using `lightonai/GTE-ModernColBERT-v1` with MaxSim scoring  
- **Stage 3**: Final ranking to top 20 using `cross-encoder/ms-marco-MiniLM-L6-v2`

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude App   │    │   Query Input   │    │   Document Store│
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────┬───────────┴──────────┬───────────┘
                     │                      │
                ┌────▼─────┐         ┌────▼─────┐
                │   MCP    │         │3-Stage   │
                │ Embedding│         │Retrieval  │
                │  Server  │         │ Pipeline  │
                └────┬─────┘         └────┬─────┘
                     │                      │
                ┌────▼─────┐         ┌────▼─────┐
                │   Multi  │         │  Stage 1  │
                │  Model   │         │Fast Cand. │
                │Selection │         └────┬─────┘
                └────┬─────┘              │
                     │              ┌────▼─────┐
                ┌────▼─────┐         │  Stage 2  │
                │ Sentence │         │Multi-Vec. │
                │Transformers│       │Rescoring  │
                └───────────┘         └────┬─────┘
                                        │
                                    ┌────▼─────┐
                                    │  Stage 3  │
                                    │Cross-Enc. │
                                    │Reranking  │
                                    └───────────┘
```

## 📁 Current Project Structure

```
rag_mcp/
├── src/                        # Core implementation
│   ├── embedding_service.py    # Multi-model embedding service
│   ├── mcp_embedding_server.py # MCP server with 10 tools
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
├── logs/                      # Application logs
├── demo.py                    # Interactive demo
└── README.md                  # This file
```

## 🚀 What This Project Does

### MCP Embedding Server
- **Multi-Model Intelligence**: Automatically selects best model based on content type
  - **Code**: `all-MiniLM-L6-v2` for technical content
  - **Documents**: `multi-qa-mpnet-base-dot-v1` for general text  
  - **Multilingual**: `paraphrase-multilingual-MiniLM-L12-v2` for multiple languages
  - **Default**: `google/embeddinggemma-300m` fallback
- **10 MCP Tools**: Complete embedding API (encode_query, encode_documents, similarity, etc.)
- **3 MCP Resources**: Model info, config, and status endpoints
- **Production Features**: Caching, batch processing, validation, logging

### 3-Stage Retrieval Pipeline
- **Stage 1 - Fast Candidate Generation**:
  - Model: `google/embeddinggemma-300m`
  - Technology: FAISS index + optional BM25
  - Output: ~500-800 candidates
  - Features: Reciprocal rank fusion, hybrid search

- **Stage 2 - Multi-Vector Rescoring**:
  - Model: `lightonai/GTE-ModernColBERT-v1`
  - Technology: ColBERT-style MaxSim scoring
  - Optimization: 192 token sequences for 4GB VRAM
  - Output: Top 100 rescored candidates

- **Stage 3 - Cross-Encoder Reranking**:
  - Model: `cross-encoder/ms-marco-MiniLM-L6-v2`
  - Technology: Direct query-document relevance scoring
  - Optimization: 256 token length, adaptive batching
  - Output: Final top 20 ranked results

## 🎯 Use Cases

This system is designed for:
- **Enterprise Search**: High-accuracy document retrieval
- **RAG Applications**: Enhanced LLM responses with relevant context
- **Research**: Academic paper and literature search
- **Knowledge Management**: Organizational document retrieval
- **Customer Support**: Intelligent FAQ and support document search
- **Content Recommendation**: Related content discovery

## 🛠️ Key Features

### Performance Optimized
- **4GB VRAM Ready**: All models configured for limited GPU memory
- **16GB RAM Compatible**: Efficient memory management
- **Fast Search**: 350-900ms end-to-end latency
- **Scalable**: Handles large document collections

### Production Ready
- **Comprehensive Logging**: Detailed performance tracking
- **Error Handling**: Robust exception management
- **Configuration Management**: YAML-based flexible config
- **Memory Management**: Automatic cleanup and optimization
- **Testing**: Complete test suite for both components

### Developer Friendly
- **Simple API**: Easy-to-use Python interface
- **MCP Integration**: Standard protocol for AI applications
- **Extensible**: Modular architecture for customization
- **Well Documented**: Comprehensive usage examples

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
# Edit .env with your Hugging Face token
```

### Basic Usage

#### 3-Stage Pipeline Demo
```bash
# Run the interactive demo
python demo.py
```

#### MCP Server
```bash
# Start the MCP server
python src/mcp_embedding_server.py
```

#### Programmatic Pipeline Usage
```python
from src.retrieval_pipeline import RetrievalPipeline

# Initialize pipeline
pipeline = RetrievalPipeline()

# Add documents
documents = ["Your documents here..."]
pipeline.add_documents(documents)

# Search with 3-stage refinement
results = pipeline.search("Your query here")
for result in results['results']:
    print(f"Score: {result['stage3_score']:.4f}")
    print(f"Document: {result['document'][:100]}...")
```

## 📊 Performance Characteristics

### Expected Performance (4GB VRAM System)
| Component | Time | VRAM Usage | Output |
|-----------|------|-----------|--------|
| **Stage 1** | 50-100ms | ~1GB | 500-800 candidates |
| **Stage 2** | 200-500ms | ~2GB | 100 candidates |
| **Stage 3** | 100-300ms | ~1GB | 20 results |
| **Total** | **350-900ms** | **~4GB** | **20 results** |

### Memory Requirements
- **VRAM**: 3-4GB peak (GPU recommended but CPU fallback available)
- **RAM**: 8-12GB depending on dataset size
- **Storage**: 2-5GB for model caches

## 🔧 Configuration

The system uses a unified YAML configuration (`config/config.yaml`) that controls both the MCP server and retrieval pipeline:

```yaml
# MCP Server Settings
embedding:
  model_mode: "multiple"  # Intelligent model selection
  default_model: "google/embeddinggemma-300m"

# 3-Stage Pipeline Settings  
pipeline:
  stage1:
    model: "google/embeddinggemma-300m"
    top_k: 500
    enable_bm25: true
    
  stage2:
    model: "lightonai/GTE-ModernColBERT-v1"
    max_seq_length: 192
    
  stage3:
    model: "cross-encoder/ms-marco-MiniLM-L6-v2"
    top_k: 20
```

## 🧪 Testing

### Comprehensive Test Suite
```bash
# Test the 3-stage pipeline
python tests/test_pipeline.py

# Validate MCP server functionality  
python tests/validate_mcp_server.py

# Run interactive demo
python demo.py
```

## 🎯 Why This Architecture?

### Theoretical Foundation: Beyond Single-Vector Limitations

Our 3-stage architecture is motivated by recent research showing fundamental limitations of single-vector embedding models. According to Weller et al. (2025) in "On the Theoretical Limitations of Embedding-Based Retrieval":

> **"We demonstrate that we may encounter these theoretical limitations in realistic settings with extremely simple queries. We connect known results in learning theory, showing that the number of top-k subsets of documents capable of being returned as the result of some query is limited by the dimension of the embedding."**

**Key Findings:**
- **Embedding Dimension Barrier**: For fixed dimension `d`, there exist document combinations that cannot be represented
- **Combinatorial Explosion**: Web-scale search requires dimensions that are computationally infeasible  
- **Real-World Impact**: Even state-of-the-art models fail on simple tasks that test these theoretical limits

### 3-Stage Retrieval: A Solution to Theoretical Limits

Our architecture addresses these limitations through:

1. **Stage 1 - Fast Candidate Generation**: Overcomes the "combinatorial explosion" problem by using efficient dense retrieval with optional BM25 hybrid search
2. **Stage 2 - Multi-Vector Rescoring**: Uses ColBERT-style MaxSim scoring to capture relationships that single vectors cannot represent
3. **Stage 3 - Cross-Encoder Reranking**: Applies direct query-document relevance scoring, avoiding embedding space constraints

**Theoretical Advantages:**
- **Progressive Refinement**: Each stage uses complementary techniques that overcome different limitations
- **Beyond Single Vectors**: Multi-vector and cross-encoder approaches avoid the sign-rank limitations of single-vector embeddings
- **Expressive Power**: Can represent document combinations that are theoretically impossible for single-vector models

### Multi-Model Intelligence
- **Content-Aware**: Automatic model selection based on document type
- **Specialized Models**: Each model optimized for specific content types  
- **Fallback Safety**: Graceful degradation to default models

### Empirical Validation

The research introduces the **LIMIT dataset** which demonstrates these theoretical limitations in practice:
- **Simple Task**: "Who likes Quokkas?" with documents containing preference information
- **State-of-the-Art Failure**: Even the best embedding models achieve <20% recall@100
- **Cross-Encoder Success**: Direct relevance models (like our Stage 3) can solve these tasks perfectly

Our 3-stage pipeline specifically addresses these challenges by combining approaches that work where single-vector embeddings fail.

## 🔮 Future Roadmap

### Planned Enhancements
- **Additional Models**: Support for newer embedding and reranking models
- **Distributed Processing**: Multi-GPU and horizontal scaling
- **Real-time Updates**: Incremental index updates
- **Advanced Analytics**: Query performance insights and optimization
- **Caching Layer**: Intelligent result caching
- **API Extensions**: REST API for pipeline access

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
1. Check the troubleshooting section below
2. Review test outputs and logs
3. Open an issue with detailed error information
4. Include system specs (GPU/RAM) and configuration

---

**Built with ❤️ at the intersection of modern information retrieval and practical AI systems**