#!/usr/bin/env python3
"""
Download and manage models for TriStage-RAG benchmark
Handles automatic downloading of required models if they don't exist
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from huggingface_hub import snapshot_download, hf_hub_download, login
from huggingface_hub.utils import GatedRepoError
from dotenv import load_dotenv
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Downloads and manages models for TriStage-RAG benchmark"""
    
    def __init__(self, models_dir: str = "../models", hf_token: Optional[str] = None):
        # Normalize and ensure directory exists (including parents)
        self.models_dir = Path(models_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load .env file if it exists
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        
        # Get token from multiple sources (priority: argument > env var > .env file)
        self.hf_token = (
            hf_token or 
            os.getenv("HF_TOKEN") or 
            os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        
        # Required models for the 3-stage pipeline
        self.required_models = {
            "stage1": {
                "name": "google/embeddinggemma-300m",
                "files": ["model.safetensors", "tokenizer.model", "config.json"],
                "description": "Stage 1: Dense embedding model",
                "gated": True
            },
            "stage2": {
                "name": "lightonai/GTE-ModernColBERT-v1", 
                "files": ["model.safetensors", "config.json"],
                "description": "Stage 2: ColBERT reranking model",
                "gated": False
            },
            "stage3": {
                "name": "cross-encoder/ms-marco-MiniLM-L6-v2",
                "files": ["model.safetensors", "config.json", "tokenizer_config.json"],
                "description": "Stage 3: Cross-encoder reranking model",
                "gated": False
            }
        }
        
        # Low-memory alternatives
        self.low_memory_models = {
            "stage1": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "files": ["model.safetensors", "config.json", "tokenizer.json"],
                "description": "Stage 1: Lightweight embedding model (low-mem)",
                "gated": False
            },
            "stage2": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "files": ["model.safetensors", "config.json", "tokenizer.json"],
                "description": "Stage 2: Lightweight reranking model (low-mem)",
                "gated": False
            },
            "stage3": {
                "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "files": ["model.safetensors", "config.json"],
                "description": "Stage 3: Lightweight cross-encoder (low-mem)",
                "gated": False
            }
        }

        # Migrate any legacy nested directories (e.g., models/google/embeddinggemma-300m -> models/embeddinggemma-300m)
        try:
            self._migrate_legacy_structure()
        except Exception as e:
            logger.warning(f"Legacy model dir migration skipped: {e}")

    def _local_dir_for(self, model_name: str) -> Path:
        """Return the flattened local directory path for a repo id.
        Example: 'google/embeddinggemma-300m' -> '<models_dir>/embeddinggemma-300m'"""
        return self.models_dir / Path(model_name).name

    def _legacy_dir_for(self, model_name: str) -> Path:
        """Return the legacy nested directory path (may exist from older runs)."""
        return self.models_dir / model_name

    def _migrate_legacy_structure(self):
        """Move legacy nested model folders into flattened structure if needed."""
        for meta in [*self.required_models.values(), *self.low_memory_models.values()]:
            repo_id = meta["name"]
            legacy = self._legacy_dir_for(repo_id)
            flat = self._local_dir_for(repo_id)
            if legacy.exists() and legacy.is_dir() and not flat.exists():
                try:
                    flat.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.move(str(legacy), str(flat))
                    logger.info(f"Migrated legacy model dir {legacy} -> {flat}")
                except Exception as e:
                    logger.warning(f"Failed to migrate {legacy} -> {flat}: {e}")
    
    def check_model_exists(self, model_name: str, required_files: List[str]) -> bool:
        """Check if a model exists and has required files"""
        model_dir = self._local_dir_for(model_name)
        legacy_dir = self._legacy_dir_for(model_name)
        
        # Accept legacy dir as existing (will be migrated on next ensure)
        if not model_dir.exists() and not legacy_dir.exists():
            return False
        if not model_dir.exists() and legacy_dir.exists():
            # migrate silently
            try:
                import shutil
                flat = model_dir
                shutil.move(str(legacy_dir), str(flat))
                model_dir = flat
            except Exception:
                model_dir = legacy_dir
        
        # Check for required files
        for file_name in required_files:
            file_path = model_dir / file_name
            if not file_path.exists():
                logger.warning(f"Missing file: {file_path}")
                return False
        
        logger.info(f"Model {model_name} is complete")
        return True
    
    def download_model(self, model_name: str, description: str = "", gated: bool = False) -> bool:
        """Download a specific model"""
        try:
            logger.info(f"Downloading {description or model_name}...")
            
            # Check if this is a gated model and we have a token
            if gated and not self.hf_token:
                logger.warning(f"Model {model_name} is gated but no Hugging Face token provided")
                logger.warning("Please set one of the following:")
                logger.warning("  1. HF_TOKEN environment variable")
                logger.warning("  2. HUGGING_FACE_HUB_TOKEN environment variable") 
                logger.warning("  3. Create a .env file with HUGGING_FACE_HUB_TOKEN=your_token")
                logger.warning("  4. Use --hf-token argument")
                logger.warning("You can get a token from https://huggingface.co/settings/tokens")
                return False
            
            # Login if token is provided
            if self.hf_token:
                try:
                    login(token=self.hf_token)
                    logger.info("Successfully authenticated with Hugging Face")
                except Exception as e:
                    logger.error(f"Failed to authenticate with Hugging Face: {e}")
                    return False
            
            # Download model snapshot
            model_dir = self._local_dir_for(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Use snapshot_download to get the entire model
            snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                cache_dir=self.models_dir / ".cache",
                token=self.hf_token if gated else None
            )
            
            logger.info(f"Successfully downloaded {model_name}")
            return True
            
        except GatedRepoError as e:
            logger.error(f"Model {model_name} is gated and requires authentication")
            logger.error("Please set one of the following:")
            logger.error("  1. HF_TOKEN environment variable")
            logger.error("  2. HUGGING_FACE_HUB_TOKEN environment variable")
            logger.error("  3. Create a .env file with HUGGING_FACE_HUB_TOKEN=your_token")
            logger.error("  4. Use --hf-token argument")
            logger.error("You can get a token from https://huggingface.co/settings/tokens")
            logger.error(f"Also make sure you have requested access to {model_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def download_all_models(self, low_memory: bool = False) -> bool:
        """Download all required models"""
        models_to_use = self.low_memory_models if low_memory else self.required_models
        
        logger.info(f"Downloading {'low-memory' if low_memory else 'full'} model set...")
        
        success = True
        for stage, model_info in models_to_use.items():
            model_name = model_info["name"]
            description = model_info["description"]
            required_files = model_info["files"]
            gated = model_info.get("gated", False)
            
            # Check if model already exists
            if self.check_model_exists(model_name, required_files):
                logger.info(f"Model {model_name} already exists, skipping download")
                continue
            
            # Download the model
            if not self.download_model(model_name, description, gated):
                success = False
                logger.error(f"Failed to download {model_name}")
        
        if success:
            logger.info("All models downloaded successfully!")
        else:
            logger.error("Some models failed to download")
        
        return success
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        info = {
            "models_dir": str(self.models_dir),
            "models": {},
            "total_size_mb": 0,
            "all_models_complete": True,
            "hf_token_available": bool(self.hf_token),
            "token_source": self._get_token_source()
        }
        
        models_to_check = self.required_models
        
        for stage, model_info in models_to_check.items():
            model_name = model_info["name"]
            model_dir = self._local_dir_for(model_name)
            
            stage_info = {
                "stage": stage,
                "name": model_name,
                "description": model_info["description"],
                "gated": model_info.get("gated", False),
                "exists": model_dir.exists(),
                "complete": False,
                "size_mb": 0,
                "missing_files": []
            }
            
            if model_dir.exists():
                # Check required files
                required_files = model_info["files"]
                missing_files = []
                
                for file_name in required_files:
                    file_path = model_dir / file_name
                    if not file_path.exists():
                        missing_files.append(file_name)
                
                stage_info["missing_files"] = missing_files
                stage_info["complete"] = len(missing_files) == 0
                
                # Calculate size
                if model_dir.exists():
                    stage_info["size_mb"] = self._get_dir_size_mb(model_dir)
                
                if not stage_info["complete"]:
                    info["all_models_complete"] = False
            
            info["models"][stage] = stage_info
            info["total_size_mb"] += stage_info["size_mb"]
        
        info["total_size_mb"] = round(info["total_size_mb"], 2)
        
        return info
    
    def _get_token_source(self) -> str:
        """Determine where the token came from"""
        if hasattr(self, '_token_source'):
            return self._token_source
        
        sources = []
        if self.hf_token:
            if os.getenv("HF_TOKEN"):
                sources.append("HF_TOKEN env var")
            if os.getenv("HUGGING_FACE_HUB_TOKEN"):
                sources.append("HUGGING_FACE_HUB_TOKEN env var")
            
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                sources.append(".env file")
            
            if not sources:
                sources.append("argument")
        
        self._token_source = ", ".join(sources) if sources else "none"
        return self._token_source
    
    def _get_dir_size_mb(self, directory: Path) -> float:
        """Calculate directory size in MB"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        except (OSError, IOError):
            pass
        
        return round(total_size / (1024 * 1024), 2)
    
    def ensure_models_available(self, low_memory: bool = False) -> bool:
        """Ensure all required models are available, download if needed"""
        models_to_use = self.low_memory_models if low_memory else self.required_models
        
        # Check what models are missing
        missing_models = []
        for stage, model_info in models_to_use.items():
            model_name = model_info["name"]
            required_files = model_info["files"]
            
            if not self.check_model_exists(model_name, required_files):
                missing_models.append((stage, model_info))
        
        if not missing_models:
            logger.info("All required models are available")
            return True
        
        # Download missing models
        logger.info(f"Found {len(missing_models)} missing models, downloading...")
        
        success = True
        for stage, model_info in missing_models:
            model_name = model_info["name"]
            description = model_info["description"]
            gated = model_info.get("gated", False)
            
            if not self.download_model(model_name, description, gated):
                success = False
                logger.error(f"Failed to download {model_name}")
        
        return success
    
    def clean_models(self):
        """Clean up all downloaded models"""
        try:
            logger.info("Cleaning up all models...")
            
            models_to_clean = list(self.required_models.keys())
            for stage in models_to_clean:
                model_name = self.required_models[stage]["name"]
                model_dir = self._local_dir_for(model_name)
                
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    logger.info(f"Removed {model_dir}")
            
            # Also clean low-memory models if they exist
            for stage in self.low_memory_models.keys():
                model_name = self.low_memory_models[stage]["name"]
                model_dir = self._local_dir_for(model_name)
                
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    logger.info(f"Removed {model_dir}")
            
            logger.info("Model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main function to manage models"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for TriStage-RAG benchmark")
    parser.add_argument("--models-dir", default="../models",
                       help="Directory to save models")
    parser.add_argument("--low-memory", action="store_true",
                       help="Download low-memory model variants")
    parser.add_argument("--info", action="store_true",
                       help="Show information about available models")
    parser.add_argument("--clean", action="store_true",
                       help="Clean up all downloaded models")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check if models are available, don't download")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="Hugging Face token for gated models (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir, args.hf_token)
    
    if args.info:
        info = downloader.get_model_info()
        print(json.dumps(info, indent=2))
        return
    
    if args.clean:
        downloader.clean_models()
        return
    
    if args.check_only:
        if downloader.ensure_models_available(args.low_memory):
            print("✅ All models are available")
        else:
            print("❌ Some models are missing")
        return
    
    # Download models
    if downloader.download_all_models(args.low_memory):
        print("✅ All models downloaded successfully!")
        
        # Show final info
        info = downloader.get_model_info()
        print(f"📊 Total size: {info['total_size_mb']} MB")
        for stage, model_info in info['models'].items():
            print(f"   {stage}: {model_info['name']} ({model_info['size_mb']} MB)")
    else:
        print("❌ Failed to download some models")


if __name__ == "__main__":
    main()