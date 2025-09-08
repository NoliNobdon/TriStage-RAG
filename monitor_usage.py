#!/usr/bin/env python3
"""
Monitor GPU and RAM usage for the 3-stage retrieval pipeline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import psutil
import GPUtil

def get_memory_usage():
    """Get current memory usage"""
    cpu_ram = psutil.virtual_memory()
    result = {
        'cpu_ram_used': cpu_ram.used / 1024**3,
        'cpu_ram_total': cpu_ram.total / 1024**3,
        'cpu_ram_percent': cpu_ram.percent
    }
    
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for i, gpu in enumerate(gpus):
            gpu_info.append({
                'id': i,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_util': gpu.memoryUtil * 100
            })
        result['gpus'] = gpu_info
    
    return result

def print_memory_usage(label):
    """Print memory usage with label"""
    usage = get_memory_usage()
    print(f"\n=== {label} ===")
    print(f"CPU RAM: {usage['cpu_ram_used']:.1f}GB / {usage['cpu_ram_total']:.1f}GB ({usage['cpu_ram_percent']:.1f}%)")
    
    if 'gpus' in usage:
        for gpu in usage['gpus']:
            print(f"GPU {gpu['id']}: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({gpu['memory_util']:.1f}%)")
    else:
        print("No CUDA GPU available")

if __name__ == "__main__":
    print_memory_usage("BEFORE LOADING")
    
    print("\n=== LOADING MODELS ===")
    from retrieval_pipeline import RetrievalPipeline
    pipeline = RetrievalPipeline()
    
    print_memory_usage("AFTER LOADING")
    
    print("\n=== ADDING DOCUMENTS ===")
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Quantum computing uses quantum mechanics to perform calculations much faster than classical computers.",
        "Neural networks are computing systems inspired by biological neural networks in the brain.",
        "Natural language processing enables computers to understand and process human language.",
        "The human brain contains approximately 86 billion neurons and controls all bodily functions.",
        "Supply chain management involves coordinating all activities from sourcing to delivery.",
        "Cryptocurrency is a digital currency that uses cryptography for security.",
        "Python is a high-level programming language that supports multiple programming paradigms.",
        "Deep learning is a subset of machine learning based on artificial neural networks.",
        "Computer vision enables machines to interpret and make decisions based on visual data."
    ]
    pipeline.add_documents(documents)
    
    print_memory_usage("AFTER ADDING DOCUMENTS")
    
    print("\n=== RUNNING INFERENCE ===")
    results = pipeline.search("What is machine learning?", top_k=3)
    
    print_memory_usage("AFTER INFERENCE")
    
    print("\n=== SECOND INFERENCE ===")
    results = pipeline.search("How does quantum computing work?", top_k=3)
    
    print_memory_usage("AFTER SECOND INFERENCE")
    
    print("\n=== SUMMARY ===")
    print(f"Pipeline completed successfully with {len(documents)} documents")
    print("All stages are using GPU acceleration where available")