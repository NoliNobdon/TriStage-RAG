#!/usr/bin/env python3
"""
Simple demo of the 3-Stage Retrieval Pipeline
"""

import sys
import os

# Ensure repository root is on sys.path so 'src' is treated as a package
repo_root = os.path.dirname(__file__)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.retrieval_pipeline import RetrievalPipeline

def demo():
    print("3-Stage Retrieval Pipeline Demo")
    print("=" * 50)
    
    # Create sample documents
    documents = [
        "Python is a high-level programming language that supports multiple programming paradigms.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "Neural networks are computing systems inspired by biological neural networks in human brains.",
        "Natural language processing enables computers to understand and process human language.",
        "Quantum computing uses quantum mechanics to perform calculations much faster than classical computers.",
        "CRISPR gene editing allows scientists to make precise changes to DNA sequences.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "The human brain contains approximately 86 billion neurons and controls all bodily functions.",
        "Supply chain management involves coordinating all activities from sourcing to delivery.",
        "Cryptocurrency is a digital currency that uses cryptography for security."
    ]
    
    print(f"Created {len(documents)} sample documents")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = RetrievalPipeline()
    
    # Add documents
    print("Adding documents to pipeline...")
    pipeline.add_documents(documents)
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How does quantum computing work?",
        "Explain neural networks"
    ]
    
    print("\nTesting queries:")
    print("-" * 30)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: '{query}'")
        print("-" * 20)
        
        try:
            result = pipeline.search(query, top_k=3)
            
            print(f"Found {len(result['results'])} results:")
            for j, res in enumerate(result['results'], 1):
                score = res.get('stage3_score', 0)
                doc_preview = res['document'][:80] + "..."
                print(f"  {j}. Score: {score:.3f} - {doc_preview}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nDemo completed successfully!")
    print("\nPipeline Features:")
    print("  • Stage 1: Fast candidate generation with FAISS + optional BM25")
    print("  • Stage 2: Multi-vector rescoring with ColBERT-style MaxSim")
    print("  • Stage 3: Cross-encoder reranking for final ranking")
    print("  • Optimized for 4GB VRAM and 16GB RAM")
    print("  • Configurable for different use cases")

if __name__ == "__main__":
    demo()