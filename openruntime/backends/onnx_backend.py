"""
ONNX Backend for embeddings and inference
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    
from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class ONNXBackend(BaseBackend):
    """ONNX Runtime backend for embeddings and inference"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        self.sessions = {}
        self.model_info = {}
        
    async def initialize(self) -> None:
        """Initialize ONNX backend"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not installed. Run: pip install onnxruntime")
            
        # Set ONNX Runtime options
        self.providers = self._get_providers()
        
        # Load available models
        await self._load_models()
        
        self.initialized = True
        logger.info(f"ONNX backend initialized with providers: {self.providers}")
        
    def _get_providers(self) -> List[str]:
        """Get available ONNX Runtime providers"""
        providers = []
        
        if ONNX_AVAILABLE:
            available = ort.get_available_providers()
            
            # Prefer CoreML for Apple Silicon
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
                
            # Use CUDA if available
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
                
            # Always include CPU as fallback
            providers.append("CPUExecutionProvider")
            
        return providers
        
    async def _load_models(self) -> None:
        """Load ONNX models"""
        models_dir = self.config.models_dir / "onnx"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define available embedding models
        embedding_models = {
            "all-MiniLM-L6-v2": {
                "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "max_length": 256
            },
            "all-mpnet-base-v2": {
                "url": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768,
                "max_length": 384
            },
            "multi-qa-MiniLM-L6-cos-v1": {
                "url": "https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                "dimensions": 384,
                "max_length": 512
            }
        }
        
        for model_name, info in embedding_models.items():
            self.model_info[model_name] = info
            model_path = models_dir / f"{model_name}.onnx"
            
            # Check if model exists locally
            if model_path.exists():
                try:
                    await self._load_model(model_name, model_path)
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    
    async def _load_model(self, name: str, path: Path) -> None:
        """Load an ONNX model"""
        try:
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create inference session
            session = ort.InferenceSession(
                str(path),
                sess_options,
                providers=self.providers
            )
            
            self.sessions[name] = session
            logger.info(f"Loaded ONNX model: {name}")
            
        except Exception as e:
            logger.error(f"Error loading model {name}: {e}")
            raise
            
    async def embed(self, payload: Dict[str, Any]) -> Any:
        """Generate embeddings using ONNX models"""
        try:
            texts = payload.get("texts", [])
            model = payload.get("model", "all-MiniLM-L6-v2")
            normalize = payload.get("normalize", True)
            
            if isinstance(texts, str):
                texts = [texts]
                
            if model not in self.sessions:
                # Try to download and load model
                await self._download_and_load_model(model)
                
            if model not in self.sessions:
                raise ValueError(f"Model {model} not available")
                
            session = self.sessions[model]
            model_info = self.model_info.get(model, {})
            
            # Tokenize texts (simplified - in production use proper tokenizer)
            embeddings = []
            
            for text in texts:
                # Prepare input (this is simplified, real implementation needs proper tokenization)
                input_ids = self._simple_tokenize(text, model_info.get("max_length", 256))
                attention_mask = np.ones_like(input_ids)
                
                # Run inference
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
                
                outputs = session.run(None, inputs)
                embedding = outputs[0][0]  # Get first output, first batch
                
                # Normalize if requested
                if normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                        
                embeddings.append(embedding.tolist())
                
            return {
                "embeddings": embeddings,
                "model": model,
                "dimensions": model_info.get("dimensions", len(embeddings[0]) if embeddings else 0)
            }
            
        except Exception as e:
            logger.error(f"ONNX embedding error: {e}")
            raise
            
    def _simple_tokenize(self, text: str, max_length: int) -> np.ndarray:
        """Simple tokenization (placeholder - use proper tokenizer in production)"""
        # This is a simplified tokenization
        # In production, use transformers tokenizer or sentencepiece
        tokens = text.lower().split()[:max_length]
        
        # Convert to simple integer representation
        token_ids = []
        for token in tokens:
            # Simple hash-based token ID (not proper tokenization!)
            token_id = hash(token) % 30000 + 1
            token_ids.append(token_id)
            
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(0)
            
        return np.array([token_ids], dtype=np.int64)
        
    async def _download_and_load_model(self, model_name: str) -> None:
        """Download and load an ONNX model"""
        # In production, implement proper model downloading from HuggingFace
        # For now, log that model needs to be downloaded
        logger.info(f"Model {model_name} needs to be downloaded")
        
    async def inference(self, payload: Dict[str, Any]) -> Any:
        """Run inference using ONNX model"""
        try:
            model = payload.get("model")
            inputs = payload.get("inputs", {})
            
            if model not in self.sessions:
                raise ValueError(f"Model {model} not loaded")
                
            session = self.sessions[model]
            
            # Prepare inputs
            onnx_inputs = {}
            for input_name, input_data in inputs.items():
                if not isinstance(input_data, np.ndarray):
                    input_data = np.array(input_data)
                onnx_inputs[input_name] = input_data
                
            # Run inference
            outputs = session.run(None, onnx_inputs)
            
            # Format outputs
            output_names = [o.name for o in session.get_outputs()]
            results = {}
            for name, output in zip(output_names, outputs):
                results[name] = output.tolist() if isinstance(output, np.ndarray) else output
                
            return {
                "model": model,
                "outputs": results
            }
            
        except Exception as e:
            logger.error(f"ONNX inference error: {e}")
            raise
            
    async def shutdown(self) -> None:
        """Shutdown ONNX backend"""
        self.sessions.clear()
        self.initialized = False
        logger.info("ONNX backend shutdown")
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend metrics"""
        return {
            "initialized": self.initialized,
            "backend": "ONNX",
            "providers": self.providers,
            "loaded_models": list(self.sessions.keys()),
            "available_models": list(self.model_info.keys())
        }