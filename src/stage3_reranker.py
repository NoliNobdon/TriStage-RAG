import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import math
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Stage3Config:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device: str = "auto"
    cache_dir: str = "./models"
    max_length: int = 256  # Optimized for 4GB VRAM
    batch_size: int = 32  # Safe batch size for 4GB VRAM
    top_k_final: int = 20
    use_fp16: bool = True
    use_gpu_if_available: bool = True
    activation_fxn: str = "sigmoid"  # "sigmoid" or "softmax"
    normalize_scores: bool = True

class CrossEncoderReranker:
    """Stage 3: Cross-Encoder Reranker for final document ranking"""
    
    def __init__(self, config: Stage3Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
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
        """Load the cross-encoder model"""
        try:
            self.logger.info(f"Loading Stage 3 model: {self.config.model_name}")
            
            # Try to load as CrossEncoder first (preferred)
            try:
                self.model = CrossEncoder(
                    self.config.model_name,
                    device=self.device,
                    max_length=self.config.max_length,
                    cache_folder=self.config.cache_dir
                )
                self.logger.info("Loaded as SentenceTransformers CrossEncoder")
                self.use_sentence_transformers = True
            except Exception:
                self.logger.info("Falling back to HuggingFace AutoModel")
                self.use_sentence_transformers = False
                
                # Load tokenizer and model separately
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir
                )
                
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir
                )
                
                # Try to load on GPU, fallback to CPU on OOM
                try:
                    self.model.to(self.device)
                    self.model.eval()
                    self.logger.info(f"HuggingFace model loaded successfully on {self.device}")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and self.device == "cuda":
                        self.logger.warning(f"CUDA OOM: {e}. Falling back to CPU.")
                        self.device = "cpu"
                        self.model.to(self.device)
                        self.model.eval()
                        self.logger.info(f"HuggingFace model loaded successfully on {self.device} (fallback)")
                    else:
                        raise
            
            # Set mixed precision if enabled
            self.use_amp = self.config.use_fp16 and self.device == "cuda"
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            self.logger.info(f"Using FP16: {self.use_amp}")
            
        except Exception as e:
            self.logger.error(f"Error loading Stage 3 model: {e}")
            raise
    
    def _prepare_input_pairs(self, query: str, documents: List[str]) -> List[Tuple[str, str]]:
        """Prepare query-document pairs for cross-encoder"""
        pairs = []
        for doc in documents:
            pairs.append((query, doc))
        return pairs
    
    def _predict_with_sentence_transformers(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict using SentenceTransformers CrossEncoder"""
        try:
            # Convert to list of lists format expected by CrossEncoder
            sentence_pairs = [[query, doc] for query, doc in pairs]
            
            # Predict scores
            scores = self.model.predict(
                sentence_pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
            
            return scores.tolist()
            
        except Exception as e:
            self.logger.error(f"Error with SentenceTransformers CrossEncoder: {e}")
            raise
    
    def _predict_with_huggingface(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict using HuggingFace AutoModel"""
        all_scores = []
        
        # Process in batches
        for i in range(0, len(pairs), self.config.batch_size):
            batch_pairs = pairs[i:i + self.config.batch_size]
            
            # Tokenize the batch
            queries = [pair[0] for pair in batch_pairs]
            documents = [pair[1] for pair in batch_pairs]
            
            encoded = self.tokenizer(
                queries,
                documents,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Predict scores
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**encoded)
                else:
                    outputs = self.model(**encoded)
            
                # Get logits and apply activation
                logits = outputs.logits
                
                if self.config.activation_fxn == "sigmoid":
                    scores = torch.sigmoid(logits).squeeze(-1)
                else:  # softmax
                    scores = F.softmax(logits, dim=-1)[:, 1]  # Probability of positive class
                
                batch_scores = scores.cpu().tolist()
                all_scores.extend(batch_scores)
            
            # Clear GPU memory if needed
            if self.device == "cuda" and i % (self.config.batch_size * 2) == 0:
                torch.cuda.empty_cache()
        
        return all_scores
    
    def predict(self, query: str, documents: List[str]) -> List[float]:
        """Predict relevance scores for query-document pairs"""
        if not documents:
            return []
        
        # Prepare input pairs
        pairs = self._prepare_input_pairs(query, documents)
        
        # Predict based on model type
        if self.use_sentence_transformers:
            scores = self._predict_with_sentence_transformers(pairs)
        else:
            scores = self._predict_with_huggingface(pairs)
        
        # Normalize scores if requested
        if self.config.normalize_scores:
            scores = self._normalize_scores(scores)
        
        return scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return scores
        
        scores_array = np.array(scores)
        
        # Min-max normalization
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score > min_score:
            normalized = (scores_array - min_score) / (max_score - min_score)
        else:
            normalized = np.zeros_like(scores_array)
        
        return normalized.tolist()
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder"""
        if not candidates:
            return []
        
        self.logger.info(f"Reranking {len(candidates)} candidates with Stage 3")
        
        # Extract documents from candidates
        documents = [candidate["document"] for candidate in candidates]
        
        # Get relevance scores
        try:
            scores = self.predict(query, documents)
        except Exception as e:
            self.logger.error(f"Error reranking: {e}")
            # Fallback: return original candidates sorted by previous scores
            return candidates
        
        # Update candidates with new scores
        reranked_candidates = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            updated_candidate = candidate.copy()
            updated_candidate["stage3_score"] = score
            updated_candidate["stage"] = "stage3"
            reranked_candidates.append(updated_candidate)
        
        # Sort by Stage 3 score (descending)
        reranked_candidates.sort(key=lambda x: x["stage3_score"], reverse=True)
        
        # Keep top-k final results
        final_results = reranked_candidates[:self.config.top_k_final]
        
        self.logger.info(f"Stage 3 reranking completed. Top score: {final_results[0]['stage3_score'] if final_results else 0:.4f}")
        
        return final_results
    
    def batch_rerank(self, queries: List[str], candidates_list: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Rerank multiple query-candidate pairs in batch"""
        if not queries or not candidates_list:
            return []
        
        if len(queries) != len(candidates_list):
            raise ValueError("Number of queries must match number of candidate lists")
        
        results = []
        for query, candidates in zip(queries, candidates_list):
            reranked = self.rerank(query, candidates)
            results.append(reranked)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        info = {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "use_fp16": self.use_amp,
            "activation_function": self.config.activation_fxn,
            "normalize_scores": self.config.normalize_scores,
            "top_k_final": self.config.top_k_final
        }
        
        if self.use_sentence_transformers:
            info["model_type"] = "SentenceTransformers CrossEncoder"
        else:
            info["model_type"] = "HuggingFace AutoModel"
            info["num_labels"] = self.model.num_labels if self.model else None
        
        return info
    
    def clear_gpu_memory(self):
        """Clear GPU memory if using CUDA"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.clear_gpu_memory()
        except:
            pass

class AdaptiveCrossEncoderReranker(CrossEncoderReranker):
    """Adaptive reranker that adjusts batch size based on input length"""
    
    def __init__(self, config: Stage3Config):
        super().__init__(config)
        self.max_text_length = config.max_length // 2  # Reserve space for both query and document
    
    def _adaptive_batch_size(self, texts: List[str]) -> int:
        """Determine optimal batch size based on text lengths"""
        if not texts:
            return self.config.batch_size
        
        # Calculate average text length
        avg_length = sum(len(text.split()) for text in texts) / len(texts)
        
        # Adjust batch size based on average length
        if avg_length > 200:
            return max(4, self.config.batch_size // 4)
        elif avg_length > 100:
            return max(8, self.config.batch_size // 2)
        elif avg_length > 50:
            return max(16, self.config.batch_size // 1)
        else:
            return self.config.batch_size
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank with adaptive batch sizing"""
        if not candidates:
            return []
        
        # Extract documents
        documents = [candidate["document"] for candidate in candidates]
        
        # Calculate adaptive batch size
        adaptive_batch_size = self._adaptive_batch_size([query] + documents)
        
        # Temporarily adjust batch size
        original_batch_size = self.config.batch_size
        self.config.batch_size = adaptive_batch_size
        
        try:
            result = super().rerank(query, candidates)
        finally:
            # Restore original batch size
            self.config.batch_size = original_batch_size
        
        return result