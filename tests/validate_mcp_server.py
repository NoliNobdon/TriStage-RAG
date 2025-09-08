#!/usr/bin/env python3

import json
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def validate_mcp_server():
    """Validate MCP server structure and imports"""
    
    print("Validating MCP Server Implementation...")
    
    # Test 1: Check if MCP server file exists
    mcp_file = Path(__file__).parent.parent / "src" / "mcp_embedding_server.py"
    if not mcp_file.exists():
        print("FAIL: mcp_embedding_server.py not found")
        return False
    else:
        print("PASS: mcp_embedding_server.py exists")
    
    # Test 2: Check if embedding service exists
    service_file = Path(__file__).parent.parent / "src" / "embedding_service.py"
    if not service_file.exists():
        print("FAIL: embedding_service.py not found")
        return False
    else:
        print("PASS: embedding_service.py exists")
    
    # Test 3: Check if config file exists
    config_file = Path(__file__).parent.parent / "config" / "config.yaml"
    if not config_file.exists():
        print("FAIL: config.yaml not found")
        return False
    else:
        print("PASS: config.yaml exists")
    
    # Test 4: Try to import MCP server
    try:
        from mcp_embedding_server import EmbeddingMCPServer
        print("PASS: Successfully imported EmbeddingMCPServer")
    except ImportError as e:
        print(f"FAIL: Failed to import EmbeddingMCPServer: {e}")
        return False
    
    # Test 5: Try to import embedding service
    try:
        from embedding_service import EmbeddingService, EmbeddingConfig
        print("PASS: Successfully imported EmbeddingService and EmbeddingConfig")
    except ImportError as e:
        print(f"FAIL: Failed to import embedding service: {e}")
        return False
    
    # Test 6: Check MCP server class structure
    try:
        # Check if the class has the required methods
        server_class = EmbeddingMCPServer
        required_methods = ['_setup_handlers', '_encode_query', '_encode_documents', '_get_model_info_resource']
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(server_class, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"FAIL: EmbeddingMCPServer missing methods: {missing_methods}")
            return False
        else:
            print("PASS: EmbeddingMCPServer has all required methods")
    except Exception as e:
        print(f"FAIL: Error checking MCP server structure: {e}")
        return False
    
    # Test 7: Check Pydantic models
    try:
        from mcp_embedding_server import EmbeddingInput, DocumentInput, SimilarityInput, BatchInput
        print("PASS: All Pydantic models imported successfully")
    except ImportError as e:
        print(f"FAIL: Failed to import Pydantic models: {e}")
        return False
    
    # Test 8: Check config structure
    try:
        config = EmbeddingConfig()
        required_attrs = ['model_mode', 'default_model', 'device', 'max_length', 'batch_size', 'cache_dir', 
                         'enable_caching', 'cache_size', 'log_level', 'log_format', 'log_file', 'models']
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(config, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"FAIL: EmbeddingConfig missing attributes: {missing_attrs}")
            return False
        else:
            print("PASS: EmbeddingConfig has all required attributes")
    except Exception as e:
        print(f"FAIL: Error checking EmbeddingConfig: {e}")
        return False
    
    # Test 9: Validate MCP tools structure
    try:
        # Check if we can create input models
        embedding_input = EmbeddingInput(text="test")
        document_input = DocumentInput(documents=["test1", "test2"])
        similarity_input = SimilarityInput(query_embedding=[0.1, 0.2], document_embeddings=[[0.1, 0.2], [0.3, 0.4]])
        batch_input = BatchInput(documents_list=[["doc1"], ["doc2", "doc3"]])
        
        print("PASS: All input models can be instantiated")
    except Exception as e:
        print(f"FAIL: Error creating input models: {e}")
        return False
    
    # Test 10: Check if README.md exists and has MCP documentation
    readme_file = Path(__file__).parent.parent / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        if "MCP" in readme_content and "embedding" in readme_content:
            print("PASS: README.md exists and contains MCP documentation")
        else:
            print("FAIL: README.md doesn't contain MCP documentation")
            return False
    else:
        print("FAIL: README.md not found")
        return False
    
    print("\nMCP Server Validation Summary:")
    print("SUCCESS: All 10 validation tests passed!")
    print("\nMCP Server Features:")
    print("- 7 embedding tools (encode_query, encode_documents, compute_similarity, etc.)")
    print("- 3 model resources (info, config, status)")
    print("- Pydantic input validation")
    print("- Error handling and logging")
    print("- Integration with production EmbeddingService")
    print("- Configuration management")
    print("- Caching support")
    
    return True

if __name__ == "__main__":
    success = validate_mcp_server()
    sys.exit(0 if success else 1)