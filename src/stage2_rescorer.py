import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import math
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Stage2Config:
    model_name: str = "lightonai/GTE-ModernColBERT-v1"
    device: str = "auto"
    cache_dir: str = "./models"
    max_seq_length: int = 192  # Optimized for 4GB VRAM
    batch_size: int = 16
    top_k_candidates: int = 100
    use_fp16: bool = True
    pooling_method: str = "cls"  # "cls", "mean", or "max"
    normalize_embeddings: bool = True
    scoring_method: str = "maxsim"  # "maxsim" or "colbert"
    use_gpu_if_available: bool = True

class ColBERTScorer:
    """ColBERT-style MaxSim scoring implementation for multi-vector retrieval"""
    
    def __init__(self, config: Stage2Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
        # Load model
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the best device for inference"""
        if self.config.device == "auto":
            if torch.cuda.is_available() and self.config.use_gpu_if_available:
                return "cuda"  # Try GPU first
            else:
                return "cpu"
        else:
            return self.config.device
    
    def _load_model(self):
        """Load the ColBERT-style model"""
        try:
            self.logger.info(f"Loading Stage 2 model: {self.config.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Try to load on GPU, fallback to CPU on OOM
            original_device = self.device
            try:
                self.model.to(self.device)
                self.model.eval()
                self.logger.info(f"Model loaded successfully on {self.device}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device == "cuda":
                    self.logger.warning(f"CUDA OOM: {e}. Falling back to CPU.")
                    self.device = "cpu"
                    self.model.to(self.device)
                    self.model.eval()
                    self.logger.info(f"Model loaded successfully on {self.device} (fallback)")
                else:
                    raise
            
            # Set mixed precision if enabled
            self.use_amp = self.config.use_fp16 and self.device == "cuda"
            
            self.logger.info(f"Using FP16: {self.use_amp}")
            
        except Exception as e:
            self.logger.error(f"Error loading Stage 2 model: {e}")
            raise
    
    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts with proper truncation and padding"""
        # Handle empty texts
        texts = [text if text and text.strip() else "empty" for text in texts]
        
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def _pool_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings based on the configured method"""
        if self.config.pooling_method == "cls":
            # Use CLS token embedding
            return embeddings[:, 0, :]
        elif self.config.pooling_method == "mean":
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.config.pooling_method == "max":
            # Max pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(embeddings, 1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.config.pooling_method}")
    
    def _encode_single_text(self, text: str) -> torch.Tensor:
        """Encode a single text to get token embeddings"""
        # Handle empty text
        if not text or not text.strip():
            text = "empty"
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
            padding=False  # No padding for single text
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)
        
        # Get token embeddings (last hidden state)
        token_embeddings = outputs.last_hidden_state
        
        # Get actual sequence length (excluding padding)
        attention_mask = encoded["attention_mask"]
        seq_length = attention_mask.sum().item()
        
        # Return only non-padded tokens
        return token_embeddings[:, :seq_length, :]
    
    def _maxsim_score(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute MaxSim score between query and document token embeddings"""
        # query_embeddings: [1, query_len, embedding_dim]
        # doc_embeddings: [1, doc_len, embedding_dim]
        
        # Compute cosine similarity matrix
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        doc_norm = F.normalize(doc_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix: [query_len, doc_len]
        sim_matrix = torch.matmul(query_norm.squeeze(0), doc_norm.squeeze(0).T)
        
        # MaxSim: take maximum similarity for each query token
        max_sim_scores = torch.max(sim_matrix, dim=-1)[0]
        
        # Final score: mean of max similarities
        return torch.mean(max_sim_scores)
    
    def _colbert_score(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute ColBERT score (more sophisticated than MaxSim)"""
        # Normalize embeddings
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        doc_norm = F.normalize(doc_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(query_norm.squeeze(0), doc_norm.squeeze(0).T)
        
        # ColBERT scoring: sum of max similarities
        max_sim_scores = torch.max(sim_matrix, dim=-1)[0]
        
        # Apply softmax to query tokens and sum
        query_weights = F.softmax(max_sim_scores, dim=0)
        final_score = torch.sum(max_sim_scores * query_weights)
        
        return final_score
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query to get token embeddings"""
        return self._encode_single_text(query)
    
    def encode_documents_batch(self, documents: List[str]) -> List[torch.Tensor]:
        """Encode a batch of documents to get token embeddings"""
        all_embeddings = []
        
        # Process in batches to manage memory
        for i in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[i:i + self.config.batch_size]
            
            encoded = self._tokenize_batch(batch_docs)
            
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**encoded)
                else:
                    outputs = self.model(**encoded)
            
            # Get token embeddings for each document in batch
            for j in range(len(batch_docs)):
                attention_mask = encoded["attention_mask"][j]
                seq_length = attention_mask.sum().item()
                
                # Get non-padded token embeddings
                token_embeddings = outputs.last_hidden_state[j, :seq_length, :]
                all_embeddings.append(token_embeddings)
            
            # Clear GPU memory if needed
            if self.device == "cuda" and i % (self.config.batch_size * 2) == 0:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except AttributeError:
                    # torch.cuda not available
                    pass
        
        return all_embeddings
    
    def rescore_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rescore candidates using multi-vector similarity"""
        if not candidates:
            return []
        
        self.logger.info(f"Rescoring {len(candidates)} candidates with Stage 2")
        
        # Encode query once
        query_embeddings = self.encode_query(query)
        
        # Extract documents from candidates
        documents = [candidate["document"] for candidate in candidates]
        
        # Encode all documents
        try:
            doc_embeddings_list = self.encode_documents_batch(documents)
        except Exception as e:
            self.logger.error(f"Error encoding documents: {e}")
            # Fallback: return original candidates with default scores
            return candidates
        
        # Compute scores for each candidate
        scored_candidates = []
        
        for i, (candidate, doc_embeddings) in enumerate(zip(candidates, doc_embeddings_list)):
            try:
                if self.config.scoring_method == "maxsim":
                    score = self._maxsim_score(query_embeddings, doc_embeddings)
                else:  # colbert
                    score = self._colbert_score(query_embeddings, doc_embeddings)
                
                # Convert to Python float
                score_float = float(score.item())
                
                # Update candidate with new score
                updated_candidate = candidate.copy()
                updated_candidate["stage2_score"] = score_float
                updated_candidate["stage"] = "stage2"
                
                scored_candidates.append(updated_candidate)
                
            except Exception as e:
                self.logger.warning(f"Error scoring candidate {i}: {e}")
                # Keep original candidate with default score
                updated_candidate = candidate.copy()
                updated_candidate["stage2_score"] = 0.0
                updated_candidate["stage"] = "stage2"
                scored_candidates.append(updated_candidate)
        
        # Sort by Stage 2 score (descending)
        scored_candidates.sort(key=lambda x: x["stage2_score"], reverse=True)
        
        # Keep top-k candidates
        top_candidates = scored_candidates[:self.config.top_k_candidates]
        
        self.logger.info(f"Stage 2 rescoring completed. Top score: {top_candidates[0]['stage2_score'] if top_candidates else 0:.4f}")
        
        return top_candidates
    
    def encode_single_document(self, document: str) -> torch.Tensor:
        """Encode a single document (useful for indexing)"""
        return self._encode_single_text(document)
    
    def compute_similarity_matrix(self, query: str, documents: List[str]) -> np.ndarray:
        """Compute similarity matrix between query and documents"""
        query_embeddings = self.encode_query(query)
        doc_embeddings_list = self.encode_documents_batch(documents)
        
        similarities = []
        for doc_embeddings in doc_embeddings_list:
            if self.config.scoring_method == "maxsim":
                score = self._maxsim_score(query_embeddings, doc_embeddings)
            else:
                score = self._colbert_score(query_embeddings, doc_embeddings)
            similarities.append(score.item())
        
        return np.array(similarities)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_seq_length": self.config.max_seq_length,
            "use_fp16": self.use_amp,
            "pooling_method": self.config.pooling_method,
            "scoring_method": self.config.scoring_method,
            "batch_size": self.config.batch_size,
            "embedding_dim": self.model.config.hidden_size if self.model else None
        }
    
    def clear_gpu_memory(self):
        """Clear GPU memory if using CUDA"""
        if self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, AttributeError):
                # torch or torch.cuda not available during cleanup
                pass
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.clear_gpu_memory()
        except:
            # Silently ignore cleanup errors during object destruction
            pass