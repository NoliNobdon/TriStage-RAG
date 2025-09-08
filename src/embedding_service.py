import os
import yaml
import logging
import hashlib
import threading
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

@dataclass
class EmbeddingConfig:
    model_name: str = "google/embeddinggemma-300m"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 32
    cache_dir: str = "./models"
    enable_caching: bool = True
    cache_size: int = 1000
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "embedding_service.log"
    max_text_length: int = 10000
    min_text_length: int = 1

class EmbeddingService:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = "config.yaml"):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.config = self._load_config(config_path)
            self._setup_logging()
            self._model = None
            self._model_lock = threading.Lock()
            self._cache = {}
            self.logger.info("EmbeddingService initialized")
    
    def _load_config(self, config_path: str) -> EmbeddingConfig:
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Get embedding config from pipeline config
            pipeline_data = config_data.get('pipeline', {})
            stage1_data = pipeline_data.get('stage1', {})
            
            device = pipeline_data.get('device', 'cpu')
            if device == 'auto':
                device = 'cpu'
            
            config_dict = {
                'model_name': stage1_data.get('model', 'google/embeddinggemma-300m'),
                'device': device,
                'max_length': stage1_data.get('max_text_length', 512),
                'batch_size': stage1_data.get('batch_size', 32),
                'cache_dir': pipeline_data.get('cache_dir', './models'),
                'enable_caching': True,
                'cache_size': 1000,
                'log_level': pipeline_data.get('log_level', 'INFO'),
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_file': pipeline_data.get('log_file', 'embedding_service.log'),
                'max_text_length': 10000,
                'min_text_length': 1
            }
            
            return EmbeddingConfig(**config_dict)
        except Exception as e:
            print(f"Error loading config: {e}")
            return EmbeddingConfig()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format,
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_model(self) -> SentenceTransformer:
        """Get or load the embedding model"""
        if self._model is not None:
            return self._model
        
        with self._model_lock:
            if self._model is None:
                try:
                    self.logger.info(f"Loading model: {self.config.model_name}")
                    self._model = SentenceTransformer(
                        self.config.model_name,
                        device=self.config.device,
                        cache_folder=self.config.cache_dir
                    )
                    self.logger.info(f"Model {self.config.model_name} loaded successfully")
                except Exception as e:
                    self.logger.error(f"Error loading model {self.config.model_name}: {e}")
                    raise
            
            return self._model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        model = self._get_model()
        
        return {
            "model_name": self.config.model_name,
            "device": model.device,
            "max_seq_length": model.max_seq_length,
            "embedding_dimension": model.get_sentence_embedding_dimension(),
            "cache_size": len(self._cache) if self.config.enable_caching else 0,
            "enable_caching": self.config.enable_caching
        }
    
    def _validate_text(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        if len(text) < self.config.min_text_length:
            return False
        if len(text) > self.config.max_text_length:
            return False
        return True
    
    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        if self.config.enable_caching:
            text_hash = self._get_text_hash(text)
            if len(self._cache) >= self.config.cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[text_hash] = embedding
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        if self.config.enable_caching:
            text_hash = self._get_text_hash(text)
            return self._cache.get(text_hash)
        return None
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query text"""
        if not self._validate_text(query):
            raise ValueError(f"Invalid query text: must be between {self.config.min_text_length} and {self.config.max_text_length} characters")
        
        cached_embedding = self._get_cached_embedding(query)
        if cached_embedding is not None:
            self.logger.debug("Using cached embedding for query")
            return cached_embedding
        
        model = self._get_model()
        try:
            embedding = model.encode(query, convert_to_numpy=True)
            self._cache_embedding(query, embedding)
            self.logger.debug(f"Encoded query using model: {self.config.model_name}")
            return embedding
        except Exception as e:
            self.logger.error(f"Error encoding query: {e}")
            raise
    
    def encode_document(self, documents: List[str]) -> np.ndarray:
        """Encode a list of documents"""
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        for doc in documents:
            if not self._validate_text(doc):
                raise ValueError(f"Invalid document text: must be between {self.config.min_text_length} and {self.config.max_text_length} characters")
        
        # Check cache first
        cached_embeddings = []
        uncached_docs = []
        uncached_indices = []
        
        for i, doc in enumerate(documents):
            cached_embedding = self._get_cached_embedding(doc)
            if cached_embedding is not None:
                cached_embeddings.append((i, cached_embedding))
            else:
                uncached_docs.append(doc)
                uncached_indices.append(i)
        
        # Encode uncached documents
        if uncached_docs:
            model = self._get_model()
            try:
                new_embeddings = model.encode(
                    uncached_docs,
                    batch_size=self.config.batch_size,
                    convert_to_numpy=True
                )
                
                # Cache new embeddings
                for doc, embedding in zip(uncached_docs, new_embeddings):
                    self._cache_embedding(doc, embedding)
                
                # Add to results
                for i, embedding in zip(uncached_indices, new_embeddings):
                    cached_embeddings.append((i, embedding))
                    
            except Exception as e:
                self.logger.error(f"Error encoding documents: {e}")
                raise
        
        # Sort by original index and create result array
        cached_embeddings.sort(key=lambda x: x[0])
        
        if cached_embeddings:
            result = np.array([embedding for _, embedding in cached_embeddings])
        else:
            # Fallback dimension
            result = np.zeros((len(documents), 768))
        
        self.logger.debug(f"Encoded {len(documents)} documents using model: {self.config.model_name}")
        return result
    
    def similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents"""
        try:
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            similarities = np.dot(doc_norms, query_norm)
            return similarities.reshape(1, -1)
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            raise
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    def __del__(self):
        pass