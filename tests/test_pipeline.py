#!/usr/bin/env python3
"""
Test script for the 3-Stage Retrieval Pipeline
Demonstrates complete pipeline functionality with sample data
"""

import sys
import os
import time
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from retrieval_pipeline import RetrievalPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_documents() -> List[str]:
    """Create sample documents for testing"""
    documents = [
        # Technology/Programming
        "Python is a high-level programming language that supports multiple programming paradigms including object-oriented, procedural, and functional programming. It was created by Guido van Rossum and first released in 1991.",
        
        "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance through experience.",
        
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through their connectionist structure.",
        
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        
        # Science/Research
        "Quantum computing is a type of computation that harnesses quantum mechanical phenomena like superposition and entanglement to perform calculations much faster than classical computers.",
        
        "CRISPR gene editing is a revolutionary technology that allows scientists to make precise changes to DNA sequences. It has enormous potential for treating genetic diseases.",
        
        "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, human activities have been the main driver of climate change since the 1800s.",
        
        "The human brain contains approximately 86 billion neurons and is responsible for controlling thought, memory, emotion, touch, motor skills, vision, breathing, temperature, hunger, and every process that regulates our body.",
        
        # Business/Economics
        "Supply chain management involves the coordination and management of all activities involved in sourcing, procurement, conversion, and logistics management. It's crucial for business success.",
        
        "Cryptocurrency is a digital or virtual currency that uses cryptography for security. Bitcoin, created in 2009, was the first decentralized cryptocurrency.",
        
        "E-commerce has transformed retail by allowing businesses to sell products and services online. The global e-commerce market continues to grow rapidly each year.",
        
        "Artificial intelligence in business applications includes customer service chatbots, predictive analytics, automated decision-making, and personalized marketing campaigns.",
        
        # Health/Medicine
        "Telemedicine allows healthcare providers to evaluate, diagnose, and treat patients remotely using telecommunications technology. It became especially important during the COVID-19 pandemic.",
        
        "Precision medicine is an approach to patient care that allows doctors to select treatments that are most likely to help patients based on a genetic understanding of their disease.",
        
        "Mental health awareness has increased significantly in recent years, with more people recognizing the importance of psychological well-being and seeking help when needed.",
        
        "Vaccines work by training the immune system to recognize and combat pathogens, either viruses or bacteria. They have been crucial in preventing numerous infectious diseases.",
        
        # Environment/Sustainability
        "Renewable energy sources like solar, wind, and hydroelectric power are becoming increasingly important as we transition away from fossil fuels to combat climate change.",
        
        "Sustainable agriculture focuses on farming practices that meet current food needs without compromising the ability of future generations to meet their own needs.",
        
        "Biodiversity conservation is essential for maintaining ecosystem services, supporting human well-being, and preserving the natural world for future generations.",
        
        "Carbon capture and storage technologies aim to capture carbon dioxide emissions from sources like power plants and store them underground to prevent atmospheric release."
    ]
    
    return documents

def create_sample_queries() -> List[str]:
    """Create sample queries for testing"""
    return [
        "What is machine learning and how does it work?",
        "How does quantum computing differ from classical computing?",
        "What are the applications of artificial intelligence in business?",
        "Explain the basics of CRISPR gene editing technology",
        "How does climate change affect global ecosystems?",
        "What is the importance of biodiversity conservation?",
        "How do vaccines work to prevent diseases?",
        "What are the main challenges in supply chain management?",
        "How has telemedicine changed healthcare delivery?",
        "What programming languages are commonly used in data science?"
    ]

def test_pipeline_basic():
    """Test basic pipeline functionality"""
    logger.info("=== Testing Basic Pipeline Functionality ===")
    
    # Create sample data
    documents = create_sample_documents()
    queries = create_sample_queries()
    
    logger.info(f"Created {len(documents)} sample documents")
    logger.info(f"Created {len(queries)} sample queries")
    
    try:
        # Initialize pipeline
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        pipeline = RetrievalPipeline(config_path=config_path)
        
        # Add documents to pipeline
        logger.info("Adding documents to pipeline...")
        pipeline.add_documents(documents)
        
        # Test single query
        logger.info("Testing single query search...")
        query = queries[0]
        start_time = time.time()
        result = pipeline.search(query)
        end_time = time.time()
        
        logger.info(f"Query: '{query}'")
        logger.info(f"Search completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Found {len(result['results'])} results")
        
        # Display top results
        for i, res in enumerate(result['results'][:3]):
            logger.info(f"  Result {i+1}: Score={res.get('stage3_score', 0):.4f}")
            logger.info(f"    Document: {res['document'][:100]}...")
        
        # Test batch search
        logger.info("Testing batch search...")
        batch_start = time.time()
        batch_results = pipeline.batch_search(queries[:3])
        batch_end = time.time()
        
        logger.info(f"Batch search completed in {batch_end - batch_start:.2f} seconds")
        logger.info(f"Processed {len(batch_results)} queries")
        
        # Get pipeline info
        pipeline_info = pipeline.get_pipeline_info()
        logger.info(f"Pipeline info: {pipeline_info['config']['stage1']['model']}")
        
        logger.info("✓ Basic pipeline test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic pipeline test failed: {e}")
        return False

def test_pipeline_performance():
    """Test pipeline performance with larger dataset"""
    logger.info("=== Testing Pipeline Performance ===")
    
    try:
        # Create larger dataset
        base_docs = create_sample_documents()
        expanded_docs = []
        
        # Create variations of base documents
        for i in range(10):  # Create 10x dataset
            for doc in base_docs:
                expanded_doc = f"{doc} (Variant {i+1}) - This is an expanded version for testing scalability and performance of the retrieval pipeline."
                expanded_docs.append(expanded_doc)
        
        logger.info(f"Created expanded dataset with {len(expanded_docs)} documents")
        
        # Initialize pipeline
        pipeline = RetrievalPipeline()
        
        # Add documents
        add_start = time.time()
        pipeline.add_documents(expanded_docs)
        add_end = time.time()
        
        logger.info(f"Document indexing completed in {add_end - add_start:.2f} seconds")
        
        # Test search performance
        test_queries = [
            "machine learning algorithms",
            "quantum computing applications",
            "artificial intelligence business",
            "climate change impact",
            "gene editing technology"
        ]
        
        search_times = []
        for query in test_queries:
            search_start = time.time()
            result = pipeline.search(query)
            search_end = time.time()
            search_times.append(search_end - search_start)
        
        avg_search_time = sum(search_times) / len(search_times)
        logger.info(f"Average search time: {avg_search_time:.2f} seconds")
        logger.info(f"Search times: {[f'{t:.2f}s' for t in search_times]}")
        
        # Get performance stats
        perf_stats = pipeline.performance_stats
        logger.info(f"Performance stats: {perf_stats}")
        
        logger.info("✓ Performance test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        return False

def test_pipeline_configuration():
    """Test pipeline with different configurations"""
    logger.info("=== Testing Pipeline Configuration ===")
    
    try:
        # Test without BM25
        logger.info("Testing pipeline without BM25...")
        from retrieval_pipeline import PipelineConfig
        
        config = PipelineConfig()
        config.stage1_enable_bm25 = False
        config.stage1_top_k = 300
        
        pipeline = RetrievalPipeline(config=config)
        documents = create_sample_documents()[:10]  # Smaller dataset
        pipeline.add_documents(documents)
        
        result = pipeline.search("What is artificial intelligence?")
        logger.info(f"Without BM25: Found {len(result['results'])} results")
        
        # Test with different models (simulate)
        logger.info("Testing configuration export...")
        pipeline.export_config("test_config.yaml")
        logger.info("Configuration exported successfully")
        
        # Clean up
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")
        
        logger.info("✓ Configuration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def test_error_handling():
    """Test pipeline error handling"""
    logger.info("=== Testing Error Handling ===")
    
    try:
        pipeline = RetrievalPipeline()
        
        # Test empty search
        try:
            result = pipeline.search("")
            logger.info("Empty query handled gracefully")
        except Exception as e:
            logger.warning(f"Empty query error: {e}")
        
        # Test with no documents
        try:
            result = pipeline.search("test query")
            logger.info("No documents case handled")
        except Exception as e:
            logger.warning(f"No documents error: {e}")
        
        # Test invalid config path
        try:
            pipeline = RetrievalPipeline(config_path="nonexistent_config.yaml")
            logger.info("Invalid config path handled")
        except Exception as e:
            logger.warning(f"Invalid config error: {e}")
        
        logger.info("✓ Error handling test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting 3-Stage Retrieval Pipeline Tests")
    logger.info("=" * 50)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_pipeline_basic),
        ("Performance", test_pipeline_performance),
        ("Configuration", test_pipeline_configuration),
        ("Error Handling", test_error_handling)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            test_results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            test_results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name:30} {status}")
        if success:
            passed += 1
    
    logger.info("-"*50)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Pipeline is ready for use.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)