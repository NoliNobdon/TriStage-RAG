import os
import json
import pickle
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor
import math

@dataclass
class Stage1Config:
    model_name: str = "google/embeddinggemma-300m"
    device: str = "auto"
    cache_dir: str = "./models"
    index_dir: str = "./faiss_index"
    top_k_candidates: int = 500
    batch_size: int = 32
    max_text_length: int = 512
    enable_bm25: bool = True
    bm25_top_k: int = 300
    fusion_method: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "weighted"
    rrf_k: int = 60
    dense_weight: float = 0.7
    bm25_weight: float = 0.3
    use_fp16: bool = True
    nlist: int = 100  # FAISS IVF clusters
    nprobe: int = 10  # FAISS search probes

class BM25Index:
    """Simple BM25 index implementation for document retrieval"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        self.avg_doc_len = 0
        self.corpus_size = 0
        self.vocabulary = set()
        self.documents = []
        
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def fit(self, documents: List[str]):
        """Fit BM25 index on documents"""
        self.documents = documents
        self.corpus_size = len(documents)
        
        # Tokenize all documents and calculate document frequencies
        tokenized_docs = []
        for doc in documents:
            tokens = self.tokenize(doc)
            tokenized_docs.append(tokens)
            self.vocabulary.update(tokens)
            
            # Calculate term frequencies for this document
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            
            self.doc_freqs.append(term_freq)
            self.doc_lens.append(len(tokens))
        
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Calculate IDF scores
        for token in self.vocabulary:
            df = sum(1 for doc_freq in self.doc_freqs if token in doc_freq)
            self.idf[token] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for query against document"""
        if doc_idx >= len(self.doc_freqs):
            return 0.0
            
        query_tokens = self.tokenize(query)
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token in doc_freq and token in self.idf:
                tf = doc_freq[token]
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for top-k documents using BM25"""
        scores = []
        for doc_idx in range(len(self.documents)):
            score = self.score(query, doc_idx)
            scores.append((doc_idx, score))
        
        # Sort by score (descending) and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class Stage1Retriever:
    """Stage 1: Fast Candidate Generation with Dense Embeddings + FAISS + Optional BM25"""
    
    def __init__(self, config: Stage1Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.embedding_dim = None
        
        # Initialize indexes
        self.faiss_index = None
        self.bm25_index = None
        self.documents = []
        self.doc_metadata = []
        
        # Ensure directories exist
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.index_dir, exist_ok=True)
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            self.logger.info(f"Loading Stage 1 model: {self.config.model_name}")
            
            # Handle device selection
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = SentenceTransformer(
                self.config.model_name,
                device=device,
                cache_folder=self.config.cache_dir
            )
            
            # Get embedding dimension
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                # Fallback: encode a sample to get dimension
                sample_embedding = self.model.encode("sample text", convert_to_numpy=True)
                self.embedding_dim = sample_embedding.shape[0]
            
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"Error loading Stage 1 model: {e}")
            raise
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts"""
        try:
            # Use FP16 if enabled and supported
            if self.config.use_fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.config.batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error encoding batch: {e}")
            raise
    
    def _create_faiss_index(self, embeddings: np.ndarray):
        """Create FAISS index for fast similarity search"""
        try:
            d = embeddings.shape[1]  # embedding dimension
            
            # Use IVF index for better performance on large datasets
            if len(embeddings) > 1000:
                quantizer = faiss.IndexFlatIP(d)  # Inner product (cosine similarity)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, d, self.config.nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Train the index
                self.faiss_index.train(embeddings)
                
                # Add vectors to index
                self.faiss_index.add(embeddings)
                
                # Set nprobe for search
                self.faiss_index.nprobe = self.config.nprobe
            else:
                # For smaller datasets, use flat index
                self.faiss_index = faiss.IndexFlatIP(d)
                self.faiss_index.add(embeddings)
            
            self.logger.info(f"FAISS index created with {len(embeddings)} vectors")
            
        except Exception as e:
            self.logger.error(f"Error creating FAISS index: {e}")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the index"""
        if not documents:
            return
        
        self.logger.info(f"Adding {len(documents)} documents to Stage 1 index")
        
        # Store documents and metadata
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        if metadata is None:
            metadata = [{}] * len(documents)
        self.doc_metadata.extend(metadata)
        
        # Encode documents
        embeddings = self._encode_batch(documents)
        embeddings = self._normalize_embeddings(embeddings)
        
        # Create or update FAISS index
        if self.faiss_index is None:
            self._create_faiss_index(embeddings)
        else:
            self.faiss_index.add(embeddings)
        
        # Create or update BM25 index if enabled
        if self.config.enable_bm25:
            if self.bm25_index is None:
                self.bm25_index = BM25Index()
                self.bm25_index.fit(self.documents)
            else:
                # For simplicity, recreate BM25 index
                self.bm25_index.fit(self.documents)
        
        self.logger.info(f"Documents added successfully. Total documents: {len(self.documents)}")
    
    def _reciprocal_rank_fusion(self, dense_results: List[Tuple[int, float]], 
                               bm25_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Combine results using Reciprocal Rank Fusion"""
        scores = defaultdict(float)
        
        # Add dense scores
        for rank, (doc_idx, score) in enumerate(dense_results):
            scores[doc_idx] += 1.0 / (self.config.rrf_k + rank + 1)
        
        # Add BM25 scores
        for rank, (doc_idx, score) in enumerate(bm25_results):
            scores[doc_idx] += 1.0 / (self.config.rrf_k + rank + 1)
        
        # Sort by combined score
        fused_results = [(doc_idx, score) for doc_idx, score in scores.items()]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def _weighted_fusion(self, dense_results: List[Tuple[int, float]], 
                        bm25_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Combine results using weighted scores"""
        scores = defaultdict(float)
        
        # Normalize and add dense scores
        if dense_results:
            max_dense_score = max(score for _, score in dense_results)
            for doc_idx, score in dense_results:
                scores[doc_idx] += self.config.dense_weight * (score / max_dense_score)
        
        # Normalize and add BM25 scores
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            for doc_idx, score in bm25_results:
                scores[doc_idx] += self.config.bm25_weight * (score / max_bm25_score)
        
        # Sort by combined score
        fused_results = [(doc_idx, score) for doc_idx, score in scores.items()]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for documents using Stage 1 retrieval"""
        if self.faiss_index is None:
            raise ValueError("No documents indexed. Call add_documents() first.")
        
        top_k = top_k or self.config.top_k_candidates
        
        # Encode query
        query_embedding = self._encode_batch([query])
        query_embedding = self._normalize_embeddings(query_embedding)
        
        # Dense search with FAISS
        dense_scores, dense_indices = self.faiss_index.search(query_embedding, top_k)
        
        # Convert to list of (doc_idx, score) tuples
        dense_results = [(int(idx), float(score)) for idx, score in zip(dense_indices[0], dense_scores[0]) if idx >= 0]
        
        # BM25 search if enabled
        bm25_results = []
        if self.config.enable_bm25 and self.bm25_index is not None:
            bm25_results = self.bm25_index.search(query, self.config.bm25_top_k)
        
        # Combine results
        if self.config.enable_bm25 and bm25_results:
            if self.config.fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(dense_results, bm25_results)
            else:  # weighted
                fused_results = self._weighted_fusion(dense_results, bm25_results)
            
            # Take top-k from fused results
            final_results = fused_results[:top_k]
        else:
            final_results = dense_results[:top_k]
        
        # Format results with metadata
        results = []
        for doc_idx, score in final_results:
            if doc_idx < len(self.documents):
                result = {
                    "doc_id": doc_idx,
                    "document": self.documents[doc_idx],
                    "score": score,
                    "metadata": self.doc_metadata[doc_idx],
                    "stage": "stage1"
                }
                results.append(result)
        
        self.logger.info(f"Stage 1 search completed. Found {len(results)} candidates")
        return results
    
    def save_index(self, index_path: Optional[str] = None):
        """Save the index to disk"""
        if index_path is None:
            index_path = os.path.join(self.config.index_dir, "stage1_index.pkl")
        
        index_data = {
            "documents": self.documents,
            "doc_metadata": self.doc_metadata,
            "config": self.config.__dict__,
            "bm25_index": self.bm25_index
        }
        
        # Save FAISS index separately
        faiss_path = os.path.join(self.config.index_dir, "stage1_faiss.index")
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, faiss_path)
        
        # Save other data
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        self.logger.info(f"Stage 1 index saved to {index_path}")
    
    def load_index(self, index_path: Optional[str] = None):
        """Load the index from disk"""
        if index_path is None:
            index_path = os.path.join(self.config.index_dir, "stage1_index.pkl")
        
        if not os.path.exists(index_path):
            self.logger.warning(f"Index file not found: {index_path}")
            return
        
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.documents = index_data["documents"]
        self.doc_metadata = index_data["doc_metadata"]
        self.bm25_index = index_data.get("bm25_index")
        
        # Load FAISS index
        faiss_path = os.path.join(self.config.index_dir, "stage1_faiss.index")
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
        
        self.logger.info(f"Stage 1 index loaded from {index_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dim,
            "faiss_index_type": type(self.faiss_index).__name__ if self.faiss_index else None,
            "bm25_enabled": self.config.enable_bm25,
            "bm25_vocabulary_size": len(self.bm25_index.vocabulary) if self.bm25_index else 0,
            "config": self.config.__dict__
        }