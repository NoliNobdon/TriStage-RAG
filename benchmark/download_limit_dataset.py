#!/usr/bin/env python3
"""
Download and prepare LIMIT dataset from Google DeepMind repository
https://github.com/google-deepmind/limit/tree/main/data

This script downloads the LIMIT dataset and converts it to the format
expected by the MTEB benchmark evaluation.
"""

import os
import json
import requests
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LIMITDatasetDownloader:
    """Downloads and prepares LIMIT dataset for benchmarking"""
    
    def __init__(self, base_dir: str = "./limit"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # URLs for LIMIT dataset files
        self.data_urls = {
            "limit-small": {
                "queries": "https://raw.githubusercontent.com/google-deepmind/limit/main/data/limit-small/queries.jsonl",
                "corpus": "https://raw.githubusercontent.com/google-deepmind/limit/main/data/limit-small/corpus.jsonl",
                "qrels": "https://raw.githubusercontent.com/google-deepmind/limit/main/data/limit-small/qrels.jsonl"
            },
            "limit": {
                "queries": "https://raw.githubusercontent.com/google-deepmind/limit/main/data/limit/queries.jsonl",
                "corpus": "https://raw.githubusercontent.com/google-deepmind/limit/main/data/limit/corpus.jsonl", 
                "qrels": "https://raw.githubusercontent.com/google-deepmind/limit/main/data/limit/qrels.jsonl"
            }
        }
    
    def download_file(self, url: str, destination: Path) -> bool:
        """Download a file from URL to destination"""
        try:
            logger.info(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def download_dataset(self, dataset_name: str = "limit-small") -> bool:
        """Download a specific LIMIT dataset"""
        if dataset_name not in self.data_urls:
            logger.error(f"Unknown dataset: {dataset_name}. Available: {list(self.data_urls.keys())}")
            return False
        
        dataset_dir = self.base_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        logger.info(f"Downloading LIMIT dataset: {dataset_name}")
        
        success = True
        for file_type, url in self.data_urls[dataset_name].items():
            destination = dataset_dir / f"{file_type}.jsonl"
            
            if destination.exists():
                logger.info(f"File already exists: {destination}")
                continue
            
            if not self.download_file(url, destination):
                success = False
        
        if success:
            logger.info(f"Successfully downloaded {dataset_name} dataset to {dataset_dir}")
        else:
            logger.error(f"Failed to download some files for {dataset_name}")
        
        return success
    
    def validate_dataset(self, dataset_name: str = "limit-small") -> bool:
        """Validate that the dataset files are properly formatted"""
        dataset_dir = self.base_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False
        
        required_files = ["queries.jsonl", "corpus.jsonl", "qrels.jsonl"]
        
        for file_name in required_files:
            file_path = dataset_dir / file_name
            
            if not file_path.exists():
                logger.error(f"Missing file: {file_path}")
                return False
            
            # Validate JSONL format
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = 0
                    for line in f:
                        line = line.strip()
                        if line:
                            json.loads(line)
                            line_count += 1
                
                logger.info(f"Validated {file_name}: {line_count} entries")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSONL in {file_name}: {e}")
                return False
            except Exception as e:
                logger.error(f"Error reading {file_name}: {e}")
                return False
        
        logger.info(f"Dataset {dataset_name} validation passed")
        return True
    
    def get_dataset_info(self, dataset_name: str = "limit-small") -> Dict[str, Any]:
        """Get information about the downloaded dataset"""
        dataset_dir = self.base_dir / dataset_name
        
        if not dataset_dir.exists():
            return {"error": "Dataset not found"}
        
        info = {
            "dataset_name": dataset_name,
            "path": str(dataset_dir),
            "files": {},
            "total_size_mb": 0
        }
        
        for file_name in ["queries.jsonl", "corpus.jsonl", "qrels.jsonl"]:
            file_path = dataset_dir / file_name
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                info["files"][file_name] = {
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                }
                info["total_size_mb"] += info["files"][file_name]["size_mb"]
        
        info["total_size_mb"] = round(info["total_size_mb"], 2)
        
        return info
    
    def prepare_for_benchmark(self, dataset_name: str = "limit-small") -> str:
        """Prepare dataset for benchmarking and return the path"""
        if not self.validate_dataset(dataset_name):
            logger.error(f"Dataset validation failed for {dataset_name}")
            return None
        
        dataset_path = self.base_dir / dataset_name
        logger.info(f"Dataset ready for benchmarking: {dataset_path}")
        
        return str(dataset_path)


def main():
    """Main function to download LIMIT dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download LIMIT dataset for benchmarking")
    parser.add_argument("--dataset", choices=["limit-small", "limit"], default="limit-small",
                       help="Which LIMIT dataset to download")
    parser.add_argument("--output-dir", default="./limit",
                       help="Directory to save the dataset")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing dataset, don't download")
    parser.add_argument("--info", action="store_true",
                       help="Show information about downloaded dataset")
    
    args = parser.parse_args()
    
    downloader = LIMITDatasetDownloader(args.output_dir)
    
    if args.info:
        info = downloader.get_dataset_info(args.dataset)
        print(json.dumps(info, indent=2))
        return
    
    if args.validate_only:
        if downloader.validate_dataset(args.dataset):
            print(f"✅ Dataset {args.dataset} is valid")
            dataset_path = downloader.prepare_for_benchmark(args.dataset)
            print(f"📁 Dataset path: {dataset_path}")
        else:
            print(f"❌ Dataset {args.dataset} validation failed")
        return
    
    # Download the dataset
    if downloader.download_dataset(args.dataset):
        print(f"✅ Successfully downloaded {args.dataset} dataset")
        
        # Validate and prepare for benchmarking
        dataset_path = downloader.prepare_for_benchmark(args.dataset)
        if dataset_path:
            print(f"🎯 Dataset ready for benchmarking!")
            print(f"   Use this path in benchmark: --limit-path {dataset_path}")
            
            # Show dataset info
            info = downloader.get_dataset_info(args.dataset)
            print(f"📊 Dataset size: {info['total_size_mb']} MB")
            for file_name, file_info in info['files'].items():
                print(f"   {file_name}: {file_info['size_mb']} MB")
    else:
        print(f"❌ Failed to download {args.dataset} dataset")


if __name__ == "__main__":
    main()