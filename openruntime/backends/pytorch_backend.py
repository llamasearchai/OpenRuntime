"""
PyTorch Backend for GPU/CPU inference
"""

import asyncio
import logging
import platform
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class PyTorchBackend(BaseBackend):
    """PyTorch backend for neural network inference"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        self.device = None
        self.models = {}
        
    async def initialize(self) -> None:
        """Initialize PyTorch backend"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Run: pip install torch")
            
        # Detect and set device
        self._setup_device()
        
        # Load models
        await self._load_models()
        
        self.initialized = True
        logger.info(f"PyTorch backend initialized on device: {self.device}")
        
    def _setup_device(self) -> None:
        """Setup PyTorch device"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Metal Performance Shaders (MPS) available")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for PyTorch")
            
    async def _load_models(self) -> None:
        """Load PyTorch models"""
        models_dir = self.config.models_dir / "pytorch"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # List available model files
        for model_file in models_dir.glob("*.pt"):
            try:
                model_name = model_file.stem
                model = torch.load(model_file, map_location=self.device)
                self.models[model_name] = model
                logger.info(f"Loaded PyTorch model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
                
    async def inference(self, payload: Dict[str, Any]) -> Any:
        """Run inference using PyTorch"""
        try:
            model_name = payload.get("model")
            inputs = payload.get("inputs")
            
            # Convert inputs to tensor
            if isinstance(inputs, (list, np.ndarray)):
                tensor_input = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            elif isinstance(inputs, dict):
                tensor_input = {
                    k: torch.tensor(v, dtype=torch.float32).to(self.device)
                    if isinstance(v, (list, np.ndarray)) else v
                    for k, v in inputs.items()
                }
            else:
                tensor_input = inputs
                
            # Run inference
            with torch.no_grad():
                if model_name and model_name in self.models:
                    model = self.models[model_name]
                    model.eval()
                    output = model(tensor_input)
                else:
                    # Simple computation as fallback
                    if isinstance(tensor_input, torch.Tensor):
                        # Example: simple neural network forward pass
                        linear = nn.Linear(tensor_input.shape[-1], 128).to(self.device)
                        output = torch.relu(linear(tensor_input))
                    else:
                        output = tensor_input
                        
            # Convert output to list
            if isinstance(output, torch.Tensor):
                result = output.cpu().numpy().tolist()
            else:
                result = output
                
            return {
                "output": result,
                "model": model_name,
                "device": str(self.device),
                "backend": "PyTorch"
            }
            
        except Exception as e:
            logger.error(f"PyTorch inference error: {e}")
            raise
            
    async def embed(self, payload: Dict[str, Any]) -> Any:
        """Generate embeddings using PyTorch"""
        try:
            texts = payload.get("texts", [])
            model_name = payload.get("model", "sentence-transformer")
            
            if isinstance(texts, str):
                texts = [texts]
                
            embeddings = []
            
            for text in texts:
                # Simple embedding generation (placeholder)
                # In production, use proper sentence transformer
                
                # Convert text to tensor representation
                text_bytes = text.encode('utf-8')[:512]
                text_tensor = torch.tensor(
                    [float(b) for b in text_bytes],
                    dtype=torch.float32
                ).to(self.device)
                
                # Apply transformations
                if text_tensor.shape[0] < 384:
                    # Pad to minimum size
                    padding = torch.zeros(384 - text_tensor.shape[0]).to(self.device)
                    text_tensor = torch.cat([text_tensor, padding])
                elif text_tensor.shape[0] > 384:
                    # Truncate
                    text_tensor = text_tensor[:384]
                    
                # Simple embedding model
                with torch.no_grad():
                    # Layer normalization
                    embedding = torch.nn.functional.layer_norm(
                        text_tensor,
                        text_tensor.shape
                    )
                    
                    # Normalize
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                    
                embeddings.append(embedding.cpu().numpy().tolist())
                
            return {
                "embeddings": embeddings,
                "model": model_name,
                "dimensions": 384,
                "device": str(self.device),
                "backend": "PyTorch"
            }
            
        except Exception as e:
            logger.error(f"PyTorch embedding error: {e}")
            raise
            
    async def shutdown(self) -> None:
        """Shutdown PyTorch backend"""
        # Clear models
        self.models.clear()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.initialized = False
        logger.info("PyTorch backend shutdown")
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend metrics"""
        metrics = {
            "initialized": self.initialized,
            "backend": "PyTorch",
            "device": str(self.device) if self.device else "None",
            "models": list(self.models.keys()),
            "torch_version": torch.__version__ if TORCH_AVAILABLE else "N/A"
        }
        
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                metrics["cuda"] = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0)
                }
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                metrics["mps"] = {
                    "available": True,
                    "built": torch.backends.mps.is_built()
                }
                
        return metrics