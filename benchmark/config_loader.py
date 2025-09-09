#!/usr/bin/env python3
"""
Configuration loader for TriStage-RAG benchmark
Handles loading and validation of benchmark-specific configuration
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class BenchmarkConfig:
    """Benchmark configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize benchmark configuration"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Environment variable overrides disabled: config.yaml is the single source of truth
            # self._apply_env_overrides(config)
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Benchmark config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _apply_env_overrides(self, config: Dict[str, Any]):
        """Overrides via environment are intentionally disabled to enforce config.yaml authority."""
        return
    
    def _validate_config(self):
        """Validate configuration structure"""
        required_sections = ["benchmark"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate benchmark section
        benchmark = self.config["benchmark"]
        required_keys = ["device", "cache_dir", "index_dir", "output_dir"]
        for key in required_keys:
            if key not in benchmark:
                raise ValueError(f"Missing required benchmark configuration key: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_device(self) -> str:
        """Get device configuration"""
        return self.get("benchmark.device", "auto")
    
    def get_cache_dir(self) -> str:
        """Get absolute cache directory for models at repo top-level (rag_mcp/models)."""
        value = self.get("benchmark.cache_dir", "models")
        repo_root = Path(__file__).parent.parent
        path = Path(value)
        if path.is_absolute():
            return str(path)
        # Normalize common forms to repo_root/models
        normalized = str(path).replace('\\', '/').strip()
        if normalized in {"models", "./models", "../models"} or normalized.endswith("/models"):
            return str((repo_root / "models").resolve())
        # Fallback: resolve relative to repo root
        return str((repo_root / path).resolve())
    
    def get_index_dir(self) -> str:
        """Get absolute index directory; default to repo-level faiss_index."""
        value = self.get("benchmark.index_dir", "faiss_index")
        path = Path(value)
        repo_root = Path(__file__).parent.parent
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return str(path)
    
    def get_output_dir(self) -> str:
        """Get absolute output directory; default to benchmark/mteb_results to match repo layout."""
        value = self.get("benchmark.output_dir", "./mteb_results")
        path = Path(value)
        # Outputs live under benchmark folder by convention
        benchmark_dir = Path(__file__).parent
        if not path.is_absolute():
            path = (benchmark_dir / path).resolve()
        return str(path)
    
    def get_log_level(self) -> str:
        """Get log level"""
        return self.get("benchmark.log_level", "INFO")
    
    def is_low_memory_mode(self) -> bool:
        """Check if low memory mode is enabled"""
        return self.get("benchmark.low_memory_mode", False)
    
    def get_dataset_path(self) -> str:
        """Get dataset path (relative to benchmark folder)"""
        return self.get("benchmark.dataset.dataset_path", "./limit_dataset")
    
    def get_sample_size(self) -> Optional[int]:
        """Get sample size"""
        return self.get("benchmark.dataset.sample_size")
    
    def get_model_config(self, stage: str) -> Dict[str, Any]:
        """Get model configuration for specific stage"""
        return self.get(f"benchmark.models.{stage}", {})
    
    def get_tasks(self) -> list:
        """Get evaluation tasks"""
        return self.get("benchmark.evaluation.tasks", ["LIMITSmallRetrieval"])
    
    def get_low_memory_config(self) -> Dict[str, Any]:
        """Get low memory configuration"""
        return self.get("benchmark.low_memory_config", {})
    
    def get_pipeline_overrides(self) -> Dict[str, Any]:
        """Get pipeline configuration overrides"""
        if self.is_low_memory_mode():
            return self.get_low_memory_config()
        return {}
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"BenchmarkConfig(path={self.config_path}, device={self.get_device()}, low_memory={self.is_low_memory_mode()})"