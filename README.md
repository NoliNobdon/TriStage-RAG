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
├── benchmark/                 # MTEB evaluation suite
│   ├── run_mteb_evaluation.py  # MTEB benchmark runner
│   ├── tristage_mteb_model.py # MTEB-compatible 3-stage model
│   ├── limit_mteb_tasks.py    # LIMIT dataset tasks
│   ├── download_limit_dataset.py # Dataset downloader
│   ├── download_models.py     # Model downloader
│   └── limit_dataset/         # Auto-downloaded datasets
├── config/
│   └── config.yaml           # Unified configuration
├── models/                    # Downloaded models (~2-5GB)
├── faiss_index/               # FAISS index storage
├── logs/                      # Log files directory
├── demo.py                    # Interactive demo
├── run_benchmark.py           # **NEW**: Complete workflow automation
├── run_mcp_server.py          # MCP server runner
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── LICENSE                   # MIT License
└── README.md                 # This file
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
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Hugging Face token if needed (use HUGGING_FACE_HUB_TOKEN variable)
```

### One-Click Complete Workflow

**New: Single Script Automation**
```bash
# Run the complete benchmark workflow automatically
python run_benchmark.py
```

This single script handles everything:
- **Step 1**: Automatically downloads LIMIT dataset if not available
- **Step 2**: Automatically downloads all required models (including gated models)
- **Step 3**: Runs the complete MTEB benchmark evaluation

**Environment Variables for Automation:**
```bash
# Optional: Set in .env file for custom behavior
TRISTAGE_DEVICE=auto          # auto, cuda, cpu
TRISTAGE_LOW_MEMORY=false     # true for low-memory mode
LOG_LEVEL=INFO               # DEBUG, INFO, WARNING, ERROR
HUGGING_FACE_HUB_TOKEN=your_token  # For gated models
TRISTAGE_SAMPLE_SIZE=1000     # Number of documents to sample (null for full)
```

### Configuration File Customization

The benchmark uses `benchmark/config.yaml` for all settings. Key sections to edit:

**📁 File: `benchmark/config.yaml`**

```yaml
# Lines 4-7: Basic settings
benchmark:
  device: "auto"              # Line 5: Change to "cuda" or "cpu"
  low_memory_mode: false      # Line 7: Set to true for limited RAM systems
  
  # Lines 15-18: Dataset settings
  dataset:
    sample_size: null        # Line 18: Set to number (e.g., 100) for testing
    
  # Lines 26-34: Stage 1 model settings
  stage1:
    batch_size: 32           # Line 32: Adjust batch size (e.g., 64, 128, 256)
    top_k: 500               # Line 33: Number of candidates to retrieve
    
  # Lines 36-44: Stage 2 model settings  
  stage2:
    batch_size: 16           # Line 40: ColBERT batch size
    top_k: 100               # Line 41: Candidates to keep after reranking
    
  # Lines 46-54: Stage 3 model settings
  stage3:
    batch_size: 32           # Line 49: Cross-encoder batch size
    top_k: 20                # Line 50: Final results to return
    
  # Lines 65-67: MTEB evaluation settings
  evaluation:
    encode_kwargs:
      batch_size: 32         # Line 66: MTEB encoding batch size
```

### Code Customization Points

**📁 File: `run_benchmark.py`**

```python
# Line 110-120: Task selection - modify which tasks to run
tasks = []
for task_name in config.get_tasks():
    if task_name == "LIMITSmallRetrieval":  # Quick test
        tasks.append(LIMITSmallRetrieval())
    elif task_name == "LIMITRetrieval":     # Full evaluation
        tasks.append(LIMITRetrieval())

# Line 135-144: Evaluation parameters - modify MTEB behavior
encode_kwargs = config.get("benchmark.evaluation.encode_kwargs", {'batch_size': 32})
results = evaluation.run(
    model,
    output_folder=str(output_path),
    encode_kwargs=encode_kwargs,
    overwrite_results=True
)
```

**📁 File: `benchmark/config_loader.py`**

```python
# Line 88-95: Environment variable overrides - add custom env vars
def _apply_env_overrides(self, config: Dict[str, Any]):
    if os.getenv("TRISTAGE_DEVICE"):
        config["benchmark"]["device"] = os.getenv("TRISTAGE_DEVICE")
    # Add your custom environment variables here
```

**📁 File: `benchmark/tristage_mteb_model.py`**

```python
# Line 50-80: Model initialization - modify pipeline behavior
class TriStageMTEBModel:
    def __init__(self, device="auto", cache_dir="../models", 
                 index_dir="./faiss_index", pipeline_config=None):
        # Customize model loading and pipeline configuration
```

### Common Customization Scenarios

**🎯 Quick Testing (Small Dataset)**
```yaml
# In benchmark/config.yaml
dataset:
  sample_size: 100  # Only use 100 documents for testing
evaluation:
  tasks:
    - "LIMITSmallRetrieval"  # Only run quick evaluation
```

**⚡ Performance Optimization (Large Batch Size)**
```yaml
# In benchmark/config.yaml
stage1:
  batch_size: 256          # Larger batches for GPU
stage2:
  batch_size: 64           # Larger ColBERT batches
stage3:
  batch_size: 128          # Larger cross-encoder batches
evaluation:
  encode_kwargs:
    batch_size: 256        # Match stage1 batch size
```

**💾 Low Memory Mode (Limited RAM)**
```bash
# Set environment variable
export TRISTAGE_LOW_MEMORY=true
# Or edit config.yaml:
# low_memory_mode: true
```

**🔧 CPU-Only Mode**
```bash
# Set environment variable
export TRISTAGE_DEVICE=cpu
# Or edit config.yaml line 5:
# device: "cpu"
```

**📊 Custom Task Selection**
```yaml
# In benchmark/config.yaml
evaluation:
  tasks:
    - "LIMITSmallRetrieval"  # Quick test (46 docs, 1000 queries)
    # - "LIMITRetrieval"     # Comment out full evaluation
```

### Basic Usage

#### Interactive Demo
```bash
# Run the interactive demo
python demo.py
```

#### MCP Server
```bash
# Start the MCP retrieval server (run from repo root)
python run_mcp_server.py
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
| **Stage 1** | 50-150ms | ~1GB | 500 candidates |
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

### Complete Automated Testing
```bash
# Run the complete benchmark workflow (recommended)
python run_benchmark.py
```

### Individual Component Testing
```bash
# Run interactive demo
python demo.py

# Start MCP server for testing
python run_mcp_server.py

# Check model availability only
python benchmark/download_models.py --info

# Download models only
python benchmark/download_models.py --check-only

# Run MTEB benchmark manually
python benchmark/run_mteb_evaluation.py --tasks LIMITSmallRetrieval
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
1. Check the logs in the logs/ directory
2. Review demo.py output for basic functionality testing
3. Ensure all dependencies are installed via requirements.txt
4. Verify Hugging Face token is set in .env if needed
5. Check GPU/CPU compatibility with your system

---

**Built with ❤️ at the intersection of information retrieval theory and practical AI systems**

*Inspired by "On the Theoretical Limitations of Embedding-Based Retrieval" by Weller et al. (2025)*
