"""
CPU Backend - Fallback backend for basic operations
"""

import asyncio
import hashlib
import logging
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np

from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class CPUBackend(BaseBackend):
    """CPU-only fallback backend"""

    def __init__(self, config: Any):
        super().__init__(config)

    async def initialize(self) -> None:
        """Initialize CPU backend"""
        self.initialized = True
        logger.info("CPU backend initialized")

    async def inference(self, payload: Dict[str, Any]) -> Any:
        """Run basic CPU inference"""
        try:
            inputs = payload.get("inputs")
            operation = payload.get("operation", "passthrough")

            if isinstance(inputs, (list, np.ndarray)):
                array_input = np.array(inputs)
            else:
                array_input = inputs

            # Perform basic operations
            if operation == "mean":
                result = float(np.mean(array_input))
            elif operation == "sum":
                result = float(np.sum(array_input))
            elif operation == "std":
                result = float(np.std(array_input))
            elif operation == "matmul" and "weights" in payload:
                weights = np.array(payload["weights"])
                result = (array_input @ weights).tolist()
            else:
                # Passthrough
                result = (
                    array_input.tolist() if isinstance(array_input, np.ndarray) else array_input
                )

            return {"output": result, "operation": operation, "backend": "CPU"}

        except Exception as e:
            logger.error(f"CPU inference error: {e}")
            raise

    async def embed(self, payload: Dict[str, Any]) -> Any:
        """Generate simple embeddings using CPU"""
        try:
            texts = payload.get("texts", [])
            dimensions = payload.get("dimensions", 384)

            if isinstance(texts, str):
                texts = [texts]

            embeddings = []

            for text in texts:
                # Simple hash-based embedding (not suitable for production!)
                # This is just a placeholder for testing

                # Create deterministic embedding from text
                text_hash = hashlib.sha256(text.encode()).digest()

                # Convert hash to float array
                embedding = []
                for i in range(0, min(len(text_hash), dimensions // 8), 1):
                    byte_val = text_hash[i]
                    # Generate 8 floats from each byte
                    for j in range(8):
                        bit = (byte_val >> j) & 1
                        val = (bit * 2 - 1) * 0.1 + np.random.randn() * 0.01
                        embedding.append(val)

                # Pad or truncate to desired dimensions
                if len(embedding) < dimensions:
                    embedding.extend([0.0] * (dimensions - len(embedding)))
                else:
                    embedding = embedding[:dimensions]

                # Normalize
                embedding = np.array(embedding)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                embeddings.append(embedding.tolist())

            return {
                "embeddings": embeddings,
                "dimensions": dimensions,
                "backend": "CPU",
                "method": "hash-based",
            }

        except Exception as e:
            logger.error(f"CPU embedding error: {e}")
            raise

    async def run_command(self, payload: Dict[str, Any]) -> Any:
        """Execute system commands"""
        try:
            command = payload.get("command")
            args = payload.get("args", [])
            timeout = payload.get("timeout", 30)

            if not command:
                raise ValueError("No command specified")

            # Build command list
            cmd = [command] + args

            # Run command
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

                return {
                    "returncode": process.returncode,
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else "",
                    "command": " ".join(cmd),
                }

            except asyncio.TimeoutError:
                process.kill()
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout} seconds",
                    "command": " ".join(cmd),
                }

        except Exception as e:
            logger.error(f"CPU command error: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown CPU backend"""
        self.initialized = False
        logger.info("CPU backend shutdown")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend metrics"""
        import psutil

        return {
            "initialized": self.initialized,
            "backend": "CPU",
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "numpy_version": np.__version__,
        }
