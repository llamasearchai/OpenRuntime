"""
MLX Backend for Apple Silicon GPU acceleration
"""

import asyncio
import logging
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    
from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class MLXBackend(BaseBackend):
    """MLX backend for Apple Silicon Metal acceleration"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        self.models = {}
        self.device = None
        
    async def initialize(self) -> None:
        """Initialize MLX backend"""
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available. This backend requires Apple Silicon.")
            
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            raise RuntimeError("MLX backend only works on Apple Silicon Macs")
            
        # Set default device
        self.device = mx.gpu if mx.metal.is_available() else mx.cpu
        
        # Configure MLX
        mx.set_default_device(self.device)
        
        # Load available models
        await self._load_models()
        
        self.initialized = True
        logger.info(f"MLX backend initialized on device: {self.device}")
        
    async def _load_models(self) -> None:
        """Load MLX models"""
        models_dir = self.config.models_dir / "mlx"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # List of supported MLX models
        supported_models = [
            "mistral-7b",
            "llama-2-7b",
            "phi-2",
            "stablelm-2-1.6b",
            "gemma-2b"
        ]
        
        for model_name in supported_models:
            model_path = models_dir / model_name
            if model_path.exists():
                try:
                    await self._load_model(model_name, model_path)
                except Exception as e:
                    logger.warning(f"Failed to load MLX model {model_name}: {e}")
                    
    async def _load_model(self, name: str, path: Path) -> None:
        """Load an MLX model"""
        # Model loading would be implemented here
        # For now, store model info
        self.models[name] = {
            "path": path,
            "loaded": False
        }
        logger.info(f"Registered MLX model: {name}")
        
    async def inference(self, payload: Dict[str, Any]) -> Any:
        """Run inference using MLX"""
        try:
            model_name = payload.get("model", "phi-2")
            inputs = payload.get("inputs")
            
            # Convert inputs to MLX arrays
            if isinstance(inputs, (list, np.ndarray)):
                inputs = mx.array(inputs)
            elif isinstance(inputs, dict):
                inputs = {k: mx.array(v) if isinstance(v, (list, np.ndarray)) else v 
                         for k, v in inputs.items()}
                         
            # Perform computation (simplified example)
            # In real implementation, load and run actual model
            if isinstance(inputs, mx.array):
                # Example: matrix multiplication
                weights = mx.random.normal((inputs.shape[-1], 512))
                output = inputs @ weights
                
                # Apply activation
                output = mx.maximum(output, 0)  # ReLU
                
                result = output.tolist()
            else:
                result = {"error": "Invalid input format"}
                
            return {
                "model": model_name,
                "output": result,
                "device": str(self.device),
                "backend": "MLX"
            }
            
        except Exception as e:
            logger.error(f"MLX inference error: {e}")
            raise
            
    async def complete(self, payload: Dict[str, Any]) -> Any:
        """Generate text completion using MLX models"""
        try:
            prompt = payload.get("prompt", "")
            model = payload.get("model", "phi-2")
            max_tokens = payload.get("max_tokens", 100)
            temperature = payload.get("temperature", 0.7)
            
            # Check if model is available
            if model not in self.models:
                # Try to use llm-mlx if available
                return await self._complete_via_llm_mlx(prompt, model, max_tokens, temperature)
                
            # Generate completion (simplified)
            # In real implementation, use actual MLX model
            response = f"MLX completion for '{prompt[:50]}...' using {model}"
            
            return {
                "content": response,
                "model": model,
                "backend": "MLX",
                "device": str(self.device)
            }
            
        except Exception as e:
            logger.error(f"MLX completion error: {e}")
            raise
            
    async def _complete_via_llm_mlx(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Use llm-mlx plugin for completion"""
        try:
            import subprocess
            
            cmd = [
                "llm", "prompt",
                "-m", f"mlx-{model}",
                "-o", f"temperature={temperature}",
                "-o", f"max_tokens={max_tokens}",
                prompt
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {
                    "content": result.stdout,
                    "model": model,
                    "backend": "MLX (via llm-mlx)"
                }
            else:
                raise RuntimeError(f"llm-mlx error: {result.stderr}")
                
        except Exception as e:
            logger.error(f"llm-mlx error: {e}")
            raise
            
    async def embed(self, payload: Dict[str, Any]) -> Any:
        """Generate embeddings using MLX"""
        try:
            texts = payload.get("texts", [])
            model = payload.get("model", "sentence-transformer")
            
            if isinstance(texts, str):
                texts = [texts]
                
            embeddings = []
            
            for text in texts:
                # Simple embedding generation (placeholder)
                # In production, use proper sentence transformer model
                
                # Convert text to simple vector representation
                text_bytes = text.encode('utf-8')
                text_array = mx.array([float(b) for b in text_bytes[:512]])
                
                # Apply transformations
                embedding = mx.nn.gelu(text_array)
                embedding = mx.nn.layer_norm(embedding, embedding.shape)
                
                # Reduce dimensionality
                weights = mx.random.normal((len(text_array), 384))
                embedding = text_array @ weights
                
                # Normalize
                norm = mx.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    
                embeddings.append(embedding.tolist())
                
            return {
                "embeddings": embeddings,
                "model": model,
                "dimensions": 384,
                "backend": "MLX",
                "device": str(self.device)
            }
            
        except Exception as e:
            logger.error(f"MLX embedding error: {e}")
            raise
            
    async def shutdown(self) -> None:
        """Shutdown MLX backend"""
        self.models.clear()
        self.initialized = False
        logger.info("MLX backend shutdown")
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend metrics"""
        metrics = {
            "initialized": self.initialized,
            "backend": "MLX",
            "device": str(self.device) if self.device else "None",
            "metal_available": mx.metal.is_available() if MLX_AVAILABLE else False,
            "models": list(self.models.keys())
        }
        
        if MLX_AVAILABLE and mx.metal.is_available():
            # Get Metal device info
            try:
                metrics["metal_info"] = {
                    "device_name": mx.metal.device_info().get("name", "Unknown"),
                    "memory": mx.metal.device_info().get("memory", 0)
                }
            except:
                pass
                
        return metrics