import os
import yaml
import logging
import hashlib
import threading
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

@dataclass
class ModelConfig:
    model_name: str
    max_length: int
    description: str
    keywords: List[str]

@dataclass
class EmbeddingConfig:
    model_mode: str = "multiple"  # 'single' or 'multiple'
    default_model: str = "google/embeddinggemma-300m"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 32
    cache_dir: str = "./models"
    enable_caching: bool = True
    cache_size: int = 1000
    parallel_processing: bool = True
    max_workers: int = 4
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "embedding_service.log"
    max_text_length: int = 10000
    min_text_length: int = 1
    allowed_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de"])
    models: Dict[str, ModelConfig] = field(default_factory=dict)

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
            self._models = {}  # Multiple models cache
            self._current_model_name = None
            self._model_lock = threading.Lock()
            self._executor = None
            self._cache = {}
            self.logger.info("EmbeddingService initialized")
    
    def _load_config(self, config_path: str) -> EmbeddingConfig:
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            embedding_data = config_data.get('embedding', {})
            
            # Extract base config
            base_config = {
                'model_mode': embedding_data.get('model_mode', 'multiple'),
                'default_model': embedding_data.get('default_model', 'google/embeddinggemma-300m'),
                'device': embedding_data.get('device', 'auto'),
                'max_length': embedding_data.get('max_length', 512),
                'batch_size': embedding_data.get('batch_size', 32),
                'cache_dir': embedding_data.get('cache_dir', './models'),
                'enable_caching': embedding_data.get('performance', {}).get('enable_caching', True),
                'cache_size': embedding_data.get('performance', {}).get('cache_size', 1000),
                'parallel_processing': embedding_data.get('performance', {}).get('parallel_processing', True),
                'max_workers': embedding_data.get('performance', {}).get('max_workers', 4),
                'log_level': embedding_data.get('logging', {}).get('level', 'INFO'),
                'log_format': embedding_data.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                'log_file': embedding_data.get('logging', {}).get('file', 'embedding_service.log'),
                'max_text_length': embedding_data.get('validation', {}).get('max_text_length', 10000),
                'min_text_length': embedding_data.get('validation', {}).get('min_text_length', 1),
                'allowed_languages': embedding_data.get('validation', {}).get('allowed_languages', ['en', 'es', 'fr', 'de'])
            }
            
            # Extract model configurations
            models_config = {}
            models_data = embedding_data.get('models', {})
            for model_name, model_info in models_data.items():
                models_config[model_name] = ModelConfig(
                    model_name=model_info.get('model_name', model_name),
                    max_length=model_info.get('max_length', 512),
                    description=model_info.get('description', ''),
                    keywords=model_info.get('keywords', [])
                )
            
            base_config['models'] = models_config
            return EmbeddingConfig(**base_config)
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
    
    def _select_model(self, text: str) -> str:
        """Intelligently select the best model based on content analysis"""
        # If single model mode, always use default
        if self.config.model_mode == 'single':
            return 'default'
        
        # Multiple model mode - analyze content
        text_lower = text.lower()
        
        # Check for code keywords
        code_keywords = set(self.config.models.get('code', ModelConfig('', 512, '', [])).keywords)
        if code_keywords and any(keyword in text_lower for keyword in code_keywords):
            self.logger.debug("Selected code model based on keyword detection")
            return 'code'
        
        # Check for multilingual content
        multilingual_keywords = set(self.config.models.get('multilingual', ModelConfig('', 512, '', [])).keywords)
        if multilingual_keywords and any(keyword in text_lower for keyword in multilingual_keywords):
            self.logger.debug("Selected multilingual model based on keyword detection")
            return 'multilingual'
        
        # Default to document model or default model
        if 'document' in self.config.models:
            return 'document'
        
        return 'default'
    
    def _get_model_for_text(self, text: str) -> SentenceTransformer:
        """Get or load the appropriate model for the given text"""
        model_type = self._select_model(text)
        
        # Determine the actual model name
        if model_type == 'default' or model_type not in self.config.models:
            model_name = self.config.default_model
        else:
            model_name = self.config.models[model_type].model_name
        
        # Check if model is already loaded
        if model_name in self._models:
            self._current_model_name = model_name
            return self._models[model_name]
        
        # Load the model
        with self._model_lock:
            if model_name not in self._models:
                try:
                    self.logger.info(f"Loading model: {model_name} (type: {model_type})")
                    model = SentenceTransformer(
                        model_name,
                        device=self.config.device,
                        cache_folder=self.config.cache_dir
                    )
                    self._models[model_name] = model
                    self.logger.info(f"Model {model_name} loaded successfully")
                except Exception as e:
                    self.logger.error(f"Error loading model {model_name}: {e}")
                    # Fallback to default model
                    if model_name != self.config.default_model:
                        self.logger.info(f"Falling back to default model: {self.config.default_model}")
                        return self._get_model_for_text(text)  # Recursive call for default
                    raise
            
            self._current_model_name = model_name
            return self._models[model_name]
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models"""
        models_info = {}
        
        # Add default model
        models_info['default'] = {
            'model_name': self.config.default_model,
            'description': 'Default model',
            'loaded': self.config.default_model in self._models
        }
        
        # Add specialized models
        for model_type, model_config in self.config.models.items():
            models_info[model_type] = {
                'model_name': model_config.model_name,
                'description': model_config.description,
                'loaded': model_config.model_name in self._models
            }
        
        return models_info
    
    def get_current_model(self) -> str:
        """Get the currently active model name"""
        return self._current_model_name or self.config.default_model
    
    def set_model(self, model_type: str) -> bool:
        """Manually set the model type"""
        if model_type == 'default' or model_type in self.config.models:
            # Force load the model to verify it works
            try:
                if model_type == 'default':
                    model_name = self.config.default_model
                else:
                    model_name = self.config.models[model_type].model_name
                
                if model_name not in self._models:
                    self._get_model_for_text("test")  # Load the model
                
                self._current_model_name = model_name
                self.logger.info(f"Model set to: {model_name} ({model_type})")
                return True
            except Exception as e:
                self.logger.error(f"Error setting model {model_type}: {e}")
                return False
        return False
    
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
        if not self._validate_text(query):
            raise ValueError(f"Invalid query text: must be between {self.config.min_text_length} and {self.config.max_text_length} characters")
        
        cached_embedding = self._get_cached_embedding(query)
        if cached_embedding is not None:
            self.logger.debug("Using cached embedding for query")
            return cached_embedding
        
        model = self._get_model_for_text(query)
        try:
            embedding = model.encode(query, convert_to_numpy=True)
            self._cache_embedding(query, embedding)
            self.logger.debug(f"Encoded query using model: {self.get_current_model()}")
            return embedding
        except Exception as e:
            self.logger.error(f"Error encoding query: {e}")
            raise
    
    def encode_document(self, documents: List[str]) -> np.ndarray:
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        for doc in documents:
            if not self._validate_text(doc):
                raise ValueError(f"Invalid document text: must be between {self.config.min_text_length} and {self.config.max_text_length} characters")
        
        # Group documents by model type for batch processing
        docs_by_model = {}
        cached_embeddings = []
        
        for i, doc in enumerate(documents):
            cached_embedding = self._get_cached_embedding(doc)
            if cached_embedding is not None:
                cached_embeddings.append((i, cached_embedding))
            else:
                model_type = self._select_model(doc)
                if model_type not in docs_by_model:
                    docs_by_model[model_type] = []
                docs_by_model[model_type].append((i, doc))
        
        # Process each group with its appropriate model
        all_embeddings = []
        
        for model_type, doc_pairs in docs_by_model.items():
            indices, uncached_docs = zip(*doc_pairs)
            
            # Get a sample document to select the model
            sample_doc = uncached_docs[0]
            model = self._get_model_for_text(sample_doc)
            
            try:
                new_embeddings = model.encode(
                    uncached_docs,
                    batch_size=self.config.batch_size,
                    convert_to_numpy=True
                )
                
                for doc, embedding in zip(uncached_docs, new_embeddings):
                    self._cache_embedding(doc, embedding)
                
                for i, embedding in zip(indices, new_embeddings):
                    all_embeddings.append((i, embedding))
                    
            except Exception as e:
                self.logger.error(f"Error encoding documents with model {model_type}: {e}")
                raise
        
        # Combine cached and new embeddings
        all_embeddings.extend(cached_embeddings)
        
        # Sort by original index and create result array
        all_embeddings.sort(key=lambda x: x[0])
        
        if all_embeddings:
            result = np.array([embedding for _, embedding in all_embeddings])
        else:
            # Fallback dimension
            result = np.zeros((len(documents), 768))
        
        self.logger.debug(f"Encoded {len(documents)} documents using multiple models")
        return result
    
    def similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        try:
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            similarities = np.dot(doc_norms, query_norm)
            return similarities.reshape(1, -1)
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            raise
    
    def batch_encode_documents(self, documents_list: List[List[str]]) -> List[np.ndarray]:
        if not documents_list:
            return []
        
        if self.config.parallel_processing:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(self.encode_document, docs) for docs in documents_list]
                results = []
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        self.logger.error(f"Error in batch processing: {e}")
                        results.append(np.array([]))
                return results
        else:
            return [self.encode_document(docs) for docs in documents_list]
    
    def get_model_info(self) -> Dict[str, Any]:
        current_model_name = self.get_current_model()
        if current_model_name not in self._models:
            # Load default model if none loaded
            self._get_model_for_text("test")
        
        current_model = self._models.get(current_model_name)
        
        return {
            "current_model": current_model_name,
            "device": current_model.device if current_model else "unknown",
            "max_seq_length": current_model.max_seq_length if current_model else 512,
            "embedding_dimension": current_model.get_sentence_embedding_dimension() if current_model else 768,
            "cache_size": len(self._cache) if self.config.enable_caching else 0,
            "available_models": self.get_available_models(),
            "model_mode": self.config.model_mode,
            "auto_selection_enabled": self.config.model_mode == "multiple"
        }
    
    def clear_cache(self):
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    def __del__(self):
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)