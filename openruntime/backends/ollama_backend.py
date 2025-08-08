"""
Ollama Backend for local LLM inference
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class OllamaBackend(BaseBackend):
    """Ollama backend for local model inference"""

    def __init__(self, config: Any):
        super().__init__(config)
        self.base_url = "http://localhost:11434"
        self.client = None
        self.available_models = []

    async def initialize(self) -> None:
        """Initialize Ollama backend"""
        self.client = httpx.AsyncClient(timeout=30.0)

        # Check Ollama availability
        if not await self._check_ollama():
            raise RuntimeError("Ollama not running. Start with: ollama serve")

        # Get available models
        await self._fetch_models()

        self.initialized = True
        logger.info(f"Ollama backend initialized with {len(self.available_models)} models")

    async def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def _fetch_models(self) -> None:
        """Fetch available Ollama models"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Error fetching Ollama models: {e}")

    async def complete(self, payload: Dict[str, Any]) -> Any:
        """Generate completion using Ollama"""
        try:
            prompt = payload.get("prompt", "")
            model = payload.get("model", "llama2")
            temperature = payload.get("temperature", 0.7)
            max_tokens = payload.get("max_tokens", 2000)
            stream = payload.get("stream", False)
            system = payload.get("system", "")

            request_data = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }

            if system:
                request_data["system"] = system

            if stream:
                return await self._stream_completion(request_data)
            else:
                response = await self.client.post(
                    f"{self.base_url}/api/generate", json=request_data
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "content": result.get("response", ""),
                        "model": model,
                        "total_duration": result.get("total_duration", 0),
                        "eval_count": result.get("eval_count", 0),
                    }
                else:
                    raise RuntimeError(f"Ollama error: {response.text}")

        except Exception as e:
            logger.error(f"Ollama completion error: {e}")
            raise

    async def _stream_completion(self, request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream completion from Ollama"""
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/api/generate", json=request_data
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

    async def embed(self, payload: Dict[str, Any]) -> Any:
        """Generate embeddings using Ollama"""
        try:
            texts = payload.get("texts", [])
            model = payload.get("model", "llama2")

            if isinstance(texts, str):
                texts = [texts]

            embeddings = []

            for text in texts:
                request_data = {"model": model, "prompt": text}

                response = await self.client.post(
                    f"{self.base_url}/api/embeddings", json=request_data
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])
                    embeddings.append(embedding)
                else:
                    embeddings.append([])

            return {
                "embeddings": embeddings,
                "model": model,
                "dimensions": len(embeddings[0]) if embeddings and embeddings[0] else 0,
            }

        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            raise

    async def run_agent(self, payload: Dict[str, Any]) -> Any:
        """Run agent-like conversation with Ollama"""
        try:
            messages = payload.get("messages", [])
            model = payload.get("model", "llama2")
            temperature = payload.get("temperature", 0.7)

            # Convert messages to Ollama format
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    prompt = f"System: {content}\n\n" + prompt
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"

            prompt += "Assistant: "

            # Generate response
            response = await self.complete(
                {"prompt": prompt, "model": model, "temperature": temperature}
            )

            return {"response": response.get("content", ""), "model": model, "backend": "Ollama"}

        except Exception as e:
            logger.error(f"Ollama agent error: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown Ollama backend"""
        if self.client:
            await self.client.aclose()
        self.initialized = False
        logger.info("Ollama backend shutdown")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend metrics"""
        metrics = {
            "initialized": self.initialized,
            "backend": "Ollama",
            "base_url": self.base_url,
            "available_models": self.available_models,
        }

        # Try to get Ollama version
        try:
            if self.client:
                response = await self.client.get(f"{self.base_url}/api/version")
                if response.status_code == 200:
                    metrics["version"] = response.json().get("version", "unknown")
        except:
            pass

        return metrics
