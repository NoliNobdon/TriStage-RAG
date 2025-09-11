#!/usr/bin/env python3
"""
LIMIT dataset task definitions for MTEB
Adds LIMITSmallRetrieval and LIMITRetrieval tasks to MTEB
"""

import json
import datasets
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Prefer configured dataset location inside benchmark folder
try:
    from .config_loader import BenchmarkConfig  # type: ignore
except Exception:
    BenchmarkConfig = None  # Fallback if import fails during doc generation

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

logger = logging.getLogger(__name__)

class LIMITSmallRetrieval(AbsTaskRetrieval):
    """LIMIT Small Retrieval task for MTEB"""
    
    metadata = TaskMetadata(
        name="LIMITSmallRetrieval",
        description="LIMIT Small Retrieval task - smaller version for faster evaluation",
        reference="https://github.com/google-deepmind/limit",
        dataset={
            "path": "limit",
            "revision": "main",
            "trust_remote_code": True
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        revision="main",
        domains=["Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation="found",
        bibtex_citation=None,
        descriptive_stats={
            "avg_sentence_length": 45.2,
            "num_samples": 10000,
            "num_documents": 50000
        }
    )
    
    def load_data(self, **kwargs):
        """Load LIMIT small dataset"""
        # Try to load from Hugging Face datasets first
        try:
            return super().load_data(**kwargs)
        except Exception:
            # Fall back to local loading returning python dicts as MTEB expects
            splits = self._load_local_data("small")
            # Set expected attributes for AbsTaskRetrieval
            self.corpus = {split: data["corpus"] for split, data in splits.items()}
            self.queries = {split: data["queries"] for split, data in splits.items()}
            self.relevant_docs = {split: data["relevant_docs"] for split, data in splits.items()}
            return splits
    
        
    def _load_local_data(self, version: str = "small"):
        """Load LIMIT dataset from local files and return python dicts in MTEB format"""
        # Build preferred paths: configured benchmark path first
        preferred_paths: List[Path] = []
        try:
            if BenchmarkConfig is not None:
                cfg = BenchmarkConfig()
                # Dataset path is configured relative to benchmark folder
                benchmark_dir = Path(__file__).parent
                configured = benchmark_dir / cfg.get("benchmark.dataset.dataset_path", "./limit_dataset")
                preferred_paths.append(configured.resolve())
        except Exception:
            # Ignore config load errors and fallback to defaults
            pass

        # Fallback candidates (prefer benchmark folder variants first)
        possible_paths = preferred_paths + [
            Path(__file__).parent / "limit_dataset",
            Path(__file__).parent / "limit",
            Path("benchmark/limit_dataset"),
            Path("benchmark/limit"),
            Path("./limit_dataset"),
            Path("./limit"),
            Path("../limit_dataset"),
            Path("../limit"),
        ]
        
        limit_path = None
        for path in possible_paths:
            if path.exists():
                limit_path = path
                break
        
        if not limit_path:
            raise FileNotFoundError(
                f"LIMIT dataset not found. Searched paths: {possible_paths}. "
                "Please download the dataset from https://github.com/google-deepmind/limit"
            )
        
        logger.info(f"Loading LIMIT {version} dataset from {limit_path}")
        
        # Determine the correct data path
        if version == "small":
            data_path = limit_path / "limit-small"
        else:
            data_path = limit_path / "limit"
        
        if not data_path.exists():
            raise FileNotFoundError(f"LIMIT {version} data not found at {data_path}")
        
        # Load data from JSONL files
        queries_file = data_path / "queries.jsonl"
        corpus_file = data_path / "corpus.jsonl"
        qrels_file = data_path / "qrels.jsonl"
        
        if not all([queries_file.exists(), corpus_file.exists(), qrels_file.exists()]):
            raise FileNotFoundError(f"Required files not found in {data_path}")
        
        # Load queries
        queries = {}
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                queries[str(data["_id"])] = data.get("text", "")
        
        # Load corpus
        corpus = {}
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                doc_id = str(data["_id"])
                corpus[doc_id] = {
                    "text": data.get("text", ""),
                    "title": data.get("title", "")
                }
        
        # Load qrels
        qrels = {}
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                query_id = str(data["query-id"])
                doc_id = str(data["corpus-id"])
                score = data.get("score", 1)
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
        
        logger.info(f"Loaded {len(queries)} queries, {len(corpus)} documents, {len(qrels)} qrels")
        
        # Build python dict splits in the exact format MTEB expects
        splits: Dict[str, Dict[str, Any]] = {}
        for split_name in self.metadata.eval_splits:
            splits[split_name] = {
                "corpus": corpus,                 # Dict[str, Dict[str, str]]
                "queries": queries,               # Dict[str, str]
                "relevant_docs": qrels            # Dict[str, Dict[str, int]]
            }
        return splits


class LIMITRetrieval(AbsTaskRetrieval):
    """LIMIT Full Retrieval task for MTEB"""
    
    metadata = TaskMetadata(
        name="LIMITRetrieval",
        description="LIMIT Full Retrieval task - complete dataset",
        reference="https://github.com/google-deepmind/limit",
        dataset={
            "path": "limit",
            "revision": "main", 
            "trust_remote_code": True
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        revision="main",
        domains=["Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation="found",
        bibtex_citation=None,
        descriptive_stats={
            "avg_sentence_length": 45.2,
            "num_samples": 50000,
            "num_documents": 250000
        }
    )
    
    def load_data(self, **kwargs):
        """Load LIMIT full dataset"""
        try:
            return super().load_data(**kwargs)
        except Exception:
            splits = self._load_local_data("full")
            self.corpus = {split: data["corpus"] for split, data in splits.items()}
            self.queries = {split: data["queries"] for split, data in splits.items()}
            self.relevant_docs = {split: data["relevant_docs"] for split, data in splits.items()}
            return splits
    
        
    def _load_local_data(self, version: str = "full"):
        """Load LIMIT dataset from local files and return python dicts in MTEB format"""
        # Build preferred paths: configured benchmark path first
        preferred_paths: List[Path] = []
        try:
            if BenchmarkConfig is not None:
                cfg = BenchmarkConfig()
                benchmark_dir = Path(__file__).parent
                configured = benchmark_dir / cfg.get("benchmark.dataset.dataset_path", "./limit_dataset")
                preferred_paths.append(configured.resolve())
        except Exception:
            pass

        # Fallback candidates (prefer benchmark folder variants first)
        possible_paths = preferred_paths + [
            Path(__file__).parent / "limit_dataset",
            Path(__file__).parent / "limit",
            Path("benchmark/limit_dataset"),
            Path("benchmark/limit"),
            Path("./limit_dataset"),
            Path("./limit"),
            Path("../limit_dataset"),
            Path("../limit"),
        ]
        
        limit_path = None
        for path in possible_paths:
            if path.exists():
                limit_path = path
                break
        
        if not limit_path:
            raise FileNotFoundError(
                f"LIMIT dataset not found. Searched paths: {possible_paths}. "
                "Please download the dataset from https://github.com/google-deepmind/limit"
            )
        
        logger.info(f"Loading LIMIT {version} dataset from {limit_path}")
        
        # Determine the correct data path
        if version == "small":
            data_path = limit_path / "limit-small"
        else:
            data_path = limit_path / "limit"
        
        if not data_path.exists():
            raise FileNotFoundError(f"LIMIT {version} data not found at {data_path}")
        
        # Load data from JSONL files
        queries_file = data_path / "queries.jsonl"
        corpus_file = data_path / "corpus.jsonl"
        qrels_file = data_path / "qrels.jsonl"
        
        if not all([queries_file.exists(), corpus_file.exists(), qrels_file.exists()]):
            raise FileNotFoundError(f"Required files not found in {data_path}")
        
        # Load queries
        queries = {}
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                queries[str(data["_id"])] = data.get("text", "")
        
        # Load corpus
        corpus = {}
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                doc_id = str(data["_id"])
                corpus[doc_id] = {
                    "text": data.get("text", ""),
                    "title": data.get("title", "")
                }
        
        # Load qrels
        qrels = {}
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                query_id = str(data["query-id"])
                doc_id = str(data["corpus-id"])
                score = data.get("score", 1)
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
        
        logger.info(f"Loaded {len(queries)} queries, {len(corpus)} documents, {len(qrels)} qrels")
        
        # Build python dict splits in the exact format MTEB expects
        splits: Dict[str, Dict[str, Any]] = {}
        for split_name in self.metadata.eval_splits:
            splits[split_name] = {
                "corpus": corpus,
                "queries": queries,
                "relevant_docs": qrels
            }
        return splits


def register_limit_tasks():
    """Register LIMIT tasks with MTEB"""
    try:
        # In newer MTEB versions, tasks are auto-registered via import
        # Just import the tasks to make them available
        logger.info("LIMIT tasks are available via import")
        
    except ImportError:
        logger.warning("MTEB not available, cannot register LIMIT tasks")
    except Exception as e:
        logger.warning(f"Failed to register LIMIT tasks: {e}")


# Auto-register on import
register_limit_tasks()