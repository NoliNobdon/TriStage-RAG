import os
import yaml
import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Import our stage modules
from stage1_retriever import Stage1Retriever, Stage1Config
from stage2_rescorer import ColBERTScorer, Stage2Config  
from stage3_reranker import AdaptiveCrossEncoderReranker, Stage3Config

@dataclass
class PipelineConfig:
    """Configuration for the complete 3-stage pipeline"""
    
    # Stage 1 configuration
    stage1_model: str = "google/embeddinggemma-300m"
    stage1_top_k: int = 500
    stage1_batch_size: int = 32
    stage1_enable_bm25: bool = True
    stage1_bm25_top_k: int = 300
    stage1_fusion_method: str = "rrf"
    stage1_use_fp16: bool = True
    
    # Stage 2 configuration
    stage2_model: str = "lightonai/GTE-ModernColBERT-v1"
    stage2_top_k: int = 100
    stage2_batch_size: int = 16
    stage2_max_seq_length: int = 192
    stage2_use_fp16: bool = True
    stage2_scoring_method: str = "maxsim"
    
    # Stage 3 configuration
    stage3_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    stage3_top_k: int = 20
    stage3_batch_size: int = 32
    stage3_max_length: int = 256
    stage3_use_fp16: bool = True
    
    # General configuration
    device: str = "auto"
    cache_dir: str = "./models"
    index_dir: str = "./faiss_index"
    log_level: str = "INFO"
    log_file: str = "retrieval_pipeline.log"
    enable_timing: bool = True
    save_intermediate_results: bool = False
    
    # Memory optimization
    auto_cleanup: bool = True
    max_memory_usage_gb: float = 4.0

class RetrievalPipeline:
    """Complete 3-stage retrieval pipeline"""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[PipelineConfig] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = PipelineConfig()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize stages
        self.stage1 = None
        self.stage2 = None
        self.stage3 = None
        
        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "avg_stage1_time": 0.0,
            "avg_stage2_time": 0.0,
            "avg_stage3_time": 0.0,
            "avg_total_time": 0.0,
            "stage_time_history": []
        }
        
        self.logger.info("RetrievalPipeline initialized")
    
    def _load_config(self, config_path: str) -> PipelineConfig:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            pipeline_data = config_data.get('pipeline', {})
            
            return PipelineConfig(
                # Stage 1
                stage1_model=pipeline_data.get('stage1', {}).get('model', "google/embeddinggemma-300m"),
                stage1_top_k=pipeline_data.get('stage1', {}).get('top_k', 500),
                stage1_batch_size=pipeline_data.get('stage1', {}).get('batch_size', 32),
                stage1_enable_bm25=pipeline_data.get('stage1', {}).get('enable_bm25', True),
                stage1_bm25_top_k=pipeline_data.get('stage1', {}).get('bm25_top_k', 300),
                stage1_fusion_method=pipeline_data.get('stage1', {}).get('fusion_method', "rrf"),
                stage1_use_fp16=pipeline_data.get('stage1', {}).get('use_fp16', True),
                
                # Stage 2
                stage2_model=pipeline_data.get('stage2', {}).get('model', "lightonai/GTE-ModernColBERT-v1"),
                stage2_top_k=pipeline_data.get('stage2', {}).get('top_k', 100),
                stage2_batch_size=pipeline_data.get('stage2', {}).get('batch_size', 16),
                stage2_max_seq_length=pipeline_data.get('stage2', {}).get('max_seq_length', 192),
                stage2_use_fp16=pipeline_data.get('stage2', {}).get('use_fp16', True),
                stage2_scoring_method=pipeline_data.get('stage2', {}).get('scoring_method', "maxsim"),
                
                # Stage 3
                stage3_model=pipeline_data.get('stage3', {}).get('model', "cross-encoder/ms-marco-MiniLM-L6-v2"),
                stage3_top_k=pipeline_data.get('stage3', {}).get('top_k', 20),
                stage3_batch_size=pipeline_data.get('stage3', {}).get('batch_size', 32),
                stage3_max_length=pipeline_data.get('stage3', {}).get('max_length', 256),
                stage3_use_fp16=pipeline_data.get('stage3', {}).get('use_fp16', True),
                
                # General
                device=pipeline_data.get('device', "auto"),
                cache_dir=pipeline_data.get('cache_dir', "./models"),
                index_dir=pipeline_data.get('index_dir', "./faiss_index"),
                log_level=pipeline_data.get('log_level', "INFO"),
                log_file=pipeline_data.get('log_file', "retrieval_pipeline.log"),
                enable_timing=pipeline_data.get('enable_timing', True),
                save_intermediate_results=pipeline_data.get('save_intermediate_results', False),
                auto_cleanup=pipeline_data.get('auto_cleanup', True),
                max_memory_usage_gb=pipeline_data.get('max_memory_usage_gb', 4.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return PipelineConfig()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
    
    def initialize_stages(self):
        """Initialize all three stages of the pipeline"""
        self.logger.info("Initializing pipeline stages...")
        
        try:
            # Stage 1: Fast Candidate Generation
            stage1_config = Stage1Config(
                model_name=self.config.stage1_model,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
                index_dir=self.config.index_dir,
                top_k_candidates=self.config.stage1_top_k,
                batch_size=self.config.stage1_batch_size,
                enable_bm25=self.config.stage1_enable_bm25,
                bm25_top_k=self.config.stage1_bm25_top_k,
                fusion_method=self.config.stage1_fusion_method,
                use_fp16=self.config.stage1_use_fp16
            )
            self.stage1 = Stage1Retriever(stage1_config)
            self.logger.info("Stage 1 initialized")
            
            # Stage 2: Multi-Vector Rescoring
            stage2_config = Stage2Config(
                model_name=self.config.stage2_model,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
                max_seq_length=self.config.stage2_max_seq_length,
                batch_size=self.config.stage2_batch_size,
                top_k_candidates=self.config.stage2_top_k,
                use_fp16=self.config.stage2_use_fp16,
                scoring_method=self.config.stage2_scoring_method
            )
            self.stage2 = ColBERTScorer(stage2_config)
            self.logger.info("Stage 2 initialized")
            
            # Stage 3: Cross-Encoder Reranking
            stage3_config = Stage3Config(
                model_name=self.config.stage3_model,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
                max_length=self.config.stage3_max_length,
                batch_size=self.config.stage3_batch_size,
                top_k_final=self.config.stage3_top_k,
                use_fp16=self.config.stage3_use_fp16
            )
            self.stage3 = AdaptiveCrossEncoderReranker(stage3_config)
            self.logger.info("Stage 3 initialized")
            
            self.logger.info("All pipeline stages initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline stages: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the pipeline index"""
        if not self.stage1:
            self.initialize_stages()
        
        self.logger.info(f"Adding {len(documents)} documents to pipeline")
        
        try:
            # Stage 1 handles document indexing
            self.stage1.add_documents(documents, metadata)
            self.logger.info("Documents added successfully")
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
    
    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Execute complete 3-stage search pipeline"""
        if not self.stage1 or not self.stage2 or not self.stage3:
            self.initialize_stages()
        
        top_k = top_k or self.config.stage3_top_k
        
        start_time = time.time() if self.config.enable_timing else None
        
        try:
            self.logger.info(f"Starting 3-stage search for query: '{query[:100]}...'")
            
            # Stage 1: Fast Candidate Generation
            stage1_start = time.time() if self.config.enable_timing else None
            stage1_results = self.stage1.search(query, self.config.stage1_top_k)
            stage1_time = time.time() - stage1_start if self.config.enable_timing else None
            
            self.logger.info(f"Stage 1 completed: {len(stage1_results)} candidates")
            
            if not stage1_results:
                return {
                    "query": query,
                    "results": [],
                    "stage1_results": [],
                    "stage2_results": [],
                    "timing": self._calculate_timing(start_time, stage1_time, None, None),
                    "performance_stats": self.performance_stats
                }
            
            # Stage 2: Multi-Vector Rescoring
            stage2_start = time.time() if self.config.enable_timing else None
            stage2_results = self.stage2.rescore_candidates(query, stage1_results)
            stage2_time = time.time() - stage2_start if self.config.enable_timing else None
            
            self.logger.info(f"Stage 2 completed: {len(stage2_results)} rescored candidates")
            
            if not stage2_results:
                return {
                    "query": query,
                    "results": [],
                    "stage1_results": stage1_results,
                    "stage2_results": [],
                    "timing": self._calculate_timing(start_time, stage1_time, stage2_time, None),
                    "performance_stats": self.performance_stats
                }
            
            # Stage 3: Cross-Encoder Reranking
            stage3_start = time.time() if self.config.enable_timing else None
            final_results = self.stage3.rerank(query, stage2_results)
            stage3_time = time.time() - stage3_start if self.config.enable_timing else None
            
            # Apply final top-k filter
            final_results = final_results[:top_k]
            
            total_time = time.time() - start_time if self.config.enable_timing else None
            
            self.logger.info(f"Stage 3 completed: {len(final_results)} final results")
            
            # Update performance stats
            if self.config.enable_timing:
                self._update_performance_stats(stage1_time, stage2_time, stage3_time, total_time)
            
            # Prepare result
            result = {
                "query": query,
                "results": final_results,
                "stage1_results": stage1_results if self.config.save_intermediate_results else [],
                "stage2_results": stage2_results if self.config.save_intermediate_results else [],
                "timing": self._calculate_timing(start_time, stage1_time, stage2_time, stage3_time),
                "performance_stats": self.performance_stats.copy()
            }
            
            # Auto-cleanup if enabled
            if self.config.auto_cleanup:
                self._cleanup_memory()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            raise
    
    def batch_search(self, queries: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute batch search for multiple queries"""
        results = []
        for query in queries:
            result = self.search(query, top_k)
            results.append(result)
        return results
    
    def save_index(self, index_path: Optional[str] = None):
        """Save pipeline index to disk"""
        if not self.stage1:
            raise ValueError("Pipeline not initialized")
        
        if index_path is None:
            index_path = os.path.join(self.config.index_dir, "pipeline_index.pkl")
        
        # Stage 1 handles the main index
        self.stage1.save_index(index_path)
        
        self.logger.info(f"Pipeline index saved to {index_path}")
    
    def load_index(self, index_path: Optional[str] = None):
        """Load pipeline index from disk"""
        if not self.stage1:
            self.initialize_stages()
        
        if index_path is None:
            index_path = os.path.join(self.config.index_dir, "pipeline_index.pkl")
        
        self.stage1.load_index(index_path)
        
        self.logger.info(f"Pipeline index loaded from {index_path}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the pipeline"""
        info = {
            "config": asdict(self.config),
            "stages_initialized": {
                "stage1": self.stage1 is not None,
                "stage2": self.stage2 is not None,
                "stage3": self.stage3 is not None
            },
            "performance_stats": self.performance_stats
        }
        
        # Add stage-specific info
        if self.stage1:
            info["stage1_stats"] = self.stage1.get_stats()
        
        if self.stage2:
            info["stage2_info"] = self.stage2.get_model_info()
        
        if self.stage3:
            info["stage3_info"] = self.stage3.get_model_info()
        
        return info
    
    def _calculate_timing(self, total_start: Optional[float], stage1_time: Optional[float], 
                         stage2_time: Optional[float], stage3_time: Optional[float]) -> Dict[str, float]:
        """Calculate timing information"""
        if not self.config.enable_timing:
            return {}
        
        total_time = time.time() - total_start if total_start else None
        
        return {
            "stage1_time": stage1_time or 0.0,
            "stage2_time": stage2_time or 0.0,
            "stage3_time": stage3_time or 0.0,
            "total_time": total_time or 0.0
        }
    
    def _update_performance_stats(self, stage1_time: float, stage2_time: float, stage3_time: float, total_time: float):
        """Update performance statistics"""
        self.performance_stats["total_queries"] += 1
        
        # Update moving averages
        alpha = 1.0 / self.performance_stats["total_queries"]
        
        self.performance_stats["avg_stage1_time"] = (
            (1 - alpha) * self.performance_stats["avg_stage1_time"] + alpha * stage1_time
        )
        self.performance_stats["avg_stage2_time"] = (
            (1 - alpha) * self.performance_stats["avg_stage2_time"] + alpha * stage2_time
        )
        self.performance_stats["avg_stage3_time"] = (
            (1 - alpha) * self.performance_stats["avg_stage3_time"] + alpha * stage3_time
        )
        self.performance_stats["avg_total_time"] = (
            (1 - alpha) * self.performance_stats["avg_total_time"] + alpha * total_time
        )
        
        # Store timing history
        self.performance_stats["stage_time_history"].append({
            "stage1": stage1_time,
            "stage2": stage2_time,
            "stage3": stage3_time,
            "total": total_time
        })
        
        # Keep only last 100 timing records
        if len(self.performance_stats["stage_time_history"]) > 100:
            self.performance_stats["stage_time_history"] = self.performance_stats["stage_time_history"][-100:]
    
    def _cleanup_memory(self):
        """Clean up memory between queries"""
        try:
            if self.stage2:
                self.stage2.clear_gpu_memory()
            if self.stage3:
                self.stage3.clear_gpu_memory()
        except Exception as e:
            self.logger.warning(f"Error during memory cleanup: {e}")
    
    def export_config(self, config_path: str):
        """Export current configuration to YAML file"""
        config_dict = {
            "pipeline": asdict(self.config)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.logger.info(f"Configuration exported to {config_path}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self._cleanup_memory()
        except:
            pass