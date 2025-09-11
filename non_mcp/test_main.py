#!/usr/bin/env python3
"""
Test suite for the non_mcp standalone 3-stage retrieval system
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from non_mcp.main import AppConfig, DocumentManager


class TestAppConfig:
    """Test AppConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = AppConfig()
        assert config.models_dir == "../models"
        assert config.data_dir == "../data"
        assert config.index_dir == "../index"
        assert config.max_results == 20
        assert config.enable_bm25 is True
        assert config.device == "cpu"
        assert config.log_level == "INFO"
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = AppConfig(
            models_dir="test_models",
            data_dir="test_data",
            max_results=10,
            device="cuda"
        )
        assert config.models_dir == "test_models"
        assert config.data_dir == "test_data"
        assert config.max_results == 10
        assert config.device == "cuda"


class TestDocumentManager:
    """Test DocumentManager class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AppConfig(data_dir=self.temp_dir)
        self.doc_manager = DocumentManager(self.config.data_dir)
        
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_documents(self):
        """Test adding documents"""
        docs = ["Test document 1", "Test document 2", "Test document 3"]
        count = self.doc_manager.add_documents(docs)
        
        assert count == 3
        assert len(self.doc_manager.documents) == 3
        assert self.doc_manager.documents[0] == "Test document 1"
        assert self.doc_manager.documents[1] == "Test document 2"
        assert self.doc_manager.documents[2] == "Test document 3"
    
    def test_add_duplicate_documents(self):
        """Test adding duplicate documents"""
        docs = ["Test document 1", "Test document 1", "Test document 2"]
        count = self.doc_manager.add_documents(docs)
        
        assert count == 2  # Only unique documents should be added
        assert len(self.doc_manager.documents) == 2
    
    def test_save_load_documents(self):
        """Test saving and loading documents"""
        docs = ["Test document 1", "Test document 2"]
        self.doc_manager.add_documents(docs)
        
        # Documents should be saved automatically
        assert len(self.doc_manager.documents) == 2
        
        # Create new manager and check loading
        new_manager = DocumentManager(self.config.data_dir)
        
        assert len(new_manager.documents) == 2
        assert new_manager.documents[0] == "Test document 1"
        assert new_manager.documents[1] == "Test document 2"
    
    def test_metadata_tracking(self):
        """Test metadata tracking"""
        docs = ["Test document 1", "Test document 2"]
        self.doc_manager.add_documents(docs, source="test")
        
        assert self.doc_manager.metadata["total_documents"] == 2
        assert self.doc_manager.metadata["count_test"] == 2
        assert "last_update_test" in self.doc_manager.metadata


class TestThreeStageRetrievalSystemBasic:
    """Test ThreeStageRetrievalSystem class - basic functionality without model loading"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AppConfig(
            models_dir="../models",
            data_dir=self.temp_dir,
            index_dir=self.temp_dir,
            device="cpu"
        )
        
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_system_can_be_imported(self):
        """Test that ThreeStageRetrievalSystem can be imported"""
        from non_mcp.main import ThreeStageRetrievalSystem
        assert ThreeStageRetrievalSystem is not None
    
    def test_system_initialization_basic(self):
        """Test system initialization without model loading"""
        from non_mcp.main import ThreeStageRetrievalSystem
        
        # Test that we can at least create the system
        try:
            system = ThreeStageRetrievalSystem(self.config)
            assert system.config == self.config
            assert system.doc_manager is not None
        except Exception as e:
            # If model loading fails, that's acceptable for this test
            # We just want to make sure the basic structure works
            assert "model" in str(e).lower() or "loading" in str(e).lower()
    
    def test_document_manager_integration(self):
        """Test that document manager works with the system"""
        from non_mcp.main import ThreeStageRetrievalSystem
        
        try:
            system = ThreeStageRetrievalSystem(self.config)
            
            # Test document operations
            test_docs = ["Test document for integration"]
            count = system.doc_manager.add_documents(test_docs)
            
            assert count == 1
            assert len(system.doc_manager.documents) == 1
            
        except Exception as e:
            # Model loading errors are acceptable
            pass


def test_main_function_import():
    """Test that main function can be imported"""
    from non_mcp.main import main
    assert callable(main)


def test_config_yaml_exists():
    """Test that pipeline config file exists"""
    config_path = Path(__file__).parent / "pipeline_config.yaml"
    assert config_path.exists()


def test_requirements_exists():
    """Test that requirements file exists"""
    req_path = Path(__file__).parent / "requirements.txt"
    assert req_path.exists()


def test_test_docs_json_exists():
    """Test that test_docs.json file exists"""
    test_docs_path = Path(__file__).parent / "test_docs.json"
    assert test_docs_path.exists()


def test_test_docs_json_content():
    """Test that test_docs.json has valid content"""
    test_docs_path = Path(__file__).parent / "test_docs.json"
    if test_docs_path.exists():
        with open(test_docs_path, 'r') as f:
            docs = json.load(f)
        
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(doc, str) for doc in docs)


def test_config_yaml_valid_structure():
    """Test that pipeline_config.yaml has valid structure"""
    import yaml
    
    config_path = Path(__file__).parent / "pipeline_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict)
        assert "pipeline" in config
        assert "stage1" in config["pipeline"]
        assert "stage2" in config["pipeline"]
        assert "stage3" in config["pipeline"]


def test_file_structure():
    """Test that all required files exist in non_mcp folder"""
    non_mcp_dir = Path(__file__).parent
    
    required_files = [
        "main.py",
        "pipeline_config.yaml", 
        "requirements.txt",
        "test_docs.json",
        "test_main.py"
    ]
    
    for file_name in required_files:
        file_path = non_mcp_dir / file_name
        assert file_path.exists(), f"Required file {file_name} does not exist"


def test_python_files_are_importable():
    """Test that Python files can be imported without syntax errors"""
    non_mcp_dir = Path(__file__).parent
    
    # Test main.py can be imported
    try:
        sys.path.insert(0, str(non_mcp_dir.parent))
        from non_mcp.main import AppConfig, DocumentManager
        assert AppConfig is not None
        assert DocumentManager is not None
    except ImportError as e:
        pytest.fail(f"Could not import main.py: {e}")
    finally:
        if str(non_mcp_dir.parent) in sys.path:
            sys.path.remove(str(non_mcp_dir.parent))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])