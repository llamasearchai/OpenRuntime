"""
OpenRuntime Engine - Complete Runtime System with LLM Integration
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator, Callable

import httpx
import numpy as np
from fastapi import WebSocket
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RuntimeBackend(str, Enum):
    """Available runtime backends"""
    MLX = "mlx"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    OLLAMA = "ollama"
    OPENAI = "openai"
    LLM_CLI = "llm_cli"
    CPU = "cpu"


class TaskType(str, Enum):
    """Task types supported by runtime"""
    INFERENCE = "inference"
    EMBEDDING = "embedding"
    COMPLETION = "completion"
    AGENT = "agent"
    TOOL_USE = "tool_use"
    WORKFLOW = "workflow"
    COMMAND = "command"


@dataclass
class RuntimeConfig:
    """Runtime configuration"""
    backend: RuntimeBackend = RuntimeBackend.CPU
    device: str = "auto"
    max_memory: Optional[int] = None
    batch_size: int = 1
    num_threads: int = 4
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".openruntime" / "cache")
    models_dir: Path = field(default_factory=lambda: Path.home() / ".openruntime" / "models")
    enable_monitoring: bool = True
    enable_caching: bool = True
    websocket_enabled: bool = True
    agent_memory_enabled: bool = True
    tool_use_enabled: bool = True


@dataclass
class TaskResult:
    """Result from task execution"""
    task_id: str
    task_type: TaskType
    status: str
    result: Any
    metrics: Dict[str, Any]
    error: Optional[str] = None
    duration_ms: float = 0.0
    backend_used: Optional[RuntimeBackend] = None


class RuntimeEngine:
    """Core runtime engine with complete LLM integration"""
    
    def __init__(self, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.backends: Dict[RuntimeBackend, Any] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.metrics: Dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_inference_time": 0.0,
            "backend_usage": {}
        }
        self._initialized = False
        self._websockets: List[WebSocket] = []
        
    async def initialize(self) -> None:
        """Initialize runtime engine and backends"""
        if self._initialized:
            return
            
        logger.info("Initializing OpenRuntime Engine...")
        
        # Create directories
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backends
        await self._initialize_backends()
        
        # Start background tasks
        asyncio.create_task(self._task_processor())
        asyncio.create_task(self._metrics_collector())
        
        self._initialized = True
        logger.info("OpenRuntime Engine initialized successfully")
        
    async def _initialize_backends(self) -> None:
        """Initialize available backends"""
        system_info = self._get_system_info()
        
        # Initialize MLX for Apple Silicon
        if system_info["is_apple_silicon"]:
            try:
                import mlx
                from .backends.mlx_backend import MLXBackend
                self.backends[RuntimeBackend.MLX] = MLXBackend(self.config)
                await self.backends[RuntimeBackend.MLX].initialize()
                logger.info("MLX backend initialized")
            except ImportError:
                logger.warning("MLX not available on this system")
                
        # Initialize PyTorch
        try:
            import torch
            from .backends.pytorch_backend import PyTorchBackend
            self.backends[RuntimeBackend.PYTORCH] = PyTorchBackend(self.config)
            await self.backends[RuntimeBackend.PYTORCH].initialize()
            logger.info("PyTorch backend initialized")
        except ImportError:
            logger.warning("PyTorch not available")
            
        # Initialize ONNX Runtime
        try:
            import onnxruntime
            from .backends.onnx_backend import ONNXBackend
            self.backends[RuntimeBackend.ONNX] = ONNXBackend(self.config)
            await self.backends[RuntimeBackend.ONNX].initialize()
            logger.info("ONNX backend initialized")
        except ImportError:
            logger.warning("ONNX Runtime not available")
            
        # Initialize LLM CLI backend
        if self._check_llm_cli():
            from .backends.llm_cli_backend import LLMCLIBackend
            self.backends[RuntimeBackend.LLM_CLI] = LLMCLIBackend(self.config)
            await self.backends[RuntimeBackend.LLM_CLI].initialize()
            logger.info("LLM CLI backend initialized")
            
        # Initialize Ollama backend
        if await self._check_ollama():
            from .backends.ollama_backend import OllamaBackend
            self.backends[RuntimeBackend.OLLAMA] = OllamaBackend(self.config)
            await self.backends[RuntimeBackend.OLLAMA].initialize()
            logger.info("Ollama backend initialized")
            
        # Initialize OpenAI backend
        if os.getenv("OPENAI_API_KEY"):
            from .backends.openai_backend import OpenAIBackend
            self.backends[RuntimeBackend.OPENAI] = OpenAIBackend(self.config)
            await self.backends[RuntimeBackend.OPENAI].initialize()
            logger.info("OpenAI backend initialized")
            
        # CPU backend is always available
        from .backends.cpu_backend import CPUBackend
        self.backends[RuntimeBackend.CPU] = CPUBackend(self.config)
        await self.backends[RuntimeBackend.CPU].initialize()
        logger.info("CPU backend initialized")
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "is_apple_silicon": platform.machine() == "arm64" and platform.system() == "Darwin"
        }
        
    def _check_llm_cli(self) -> bool:
        """Check if LLM CLI is installed"""
        try:
            result = subprocess.run(
                ["llm", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    async def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/api/tags", timeout=2.0)
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
            
    async def submit_task(
        self,
        task_type: TaskType,
        payload: Dict[str, Any],
        backend: Optional[RuntimeBackend] = None
    ) -> str:
        """Submit a task for execution"""
        task_id = f"{task_type.value}_{int(time.time() * 1000)}"
        
        task_data = {
            "id": task_id,
            "type": task_type,
            "payload": payload,
            "backend": backend,
            "submitted_at": time.time()
        }
        
        await self.task_queue.put(task_data)
        logger.info(f"Task {task_id} submitted")
        
        return task_id
        
    async def _task_processor(self) -> None:
        """Process tasks from queue"""
        while True:
            try:
                task_data = await self.task_queue.get()
                task_id = task_data["id"]
                
                # Create task coroutine
                task_coro = self._execute_task(task_data)
                task = asyncio.create_task(task_coro)
                
                # Track active task
                self.active_tasks[task_id] = task
                
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                
    async def _execute_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute a single task"""
        task_id = task_data["id"]
        task_type = task_data["type"]
        payload = task_data["payload"]
        backend_pref = task_data.get("backend")
        
        start_time = time.time()
        
        try:
            # Select backend
            backend = self._select_backend(task_type, backend_pref)
            if not backend:
                raise RuntimeError(f"No backend available for task type {task_type}")
                
            # Execute based on task type
            if task_type == TaskType.INFERENCE:
                result = await backend.inference(payload)
            elif task_type == TaskType.EMBEDDING:
                result = await backend.embed(payload)
            elif task_type == TaskType.COMPLETION:
                result = await backend.complete(payload)
            elif task_type == TaskType.AGENT:
                result = await backend.run_agent(payload)
            elif task_type == TaskType.TOOL_USE:
                result = await backend.use_tool(payload)
            elif task_type == TaskType.WORKFLOW:
                result = await backend.run_workflow(payload)
            elif task_type == TaskType.COMMAND:
                result = await backend.run_command(payload)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
            duration_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics["tasks_completed"] += 1
            self.metrics["total_inference_time"] += duration_ms
            backend_name = backend.__class__.__name__
            self.metrics["backend_usage"][backend_name] = \
                self.metrics["backend_usage"].get(backend_name, 0) + 1
                
            task_result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                status="completed",
                result=result,
                metrics={
                    "duration_ms": duration_ms,
                    "backend": backend_name
                },
                duration_ms=duration_ms,
                backend_used=backend_pref
            )
            
            # Broadcast to websockets
            await self._broadcast_result(task_result)
            
            return task_result
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.metrics["tasks_failed"] += 1
            
            task_result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                status="failed",
                result=None,
                metrics={},
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
            
            await self._broadcast_result(task_result)
            return task_result
            
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task_id, None)
            
    def _select_backend(
        self,
        task_type: TaskType,
        preferred: Optional[RuntimeBackend] = None
    ) -> Any:
        """Select appropriate backend for task"""
        # Use preferred if available
        if preferred and preferred in self.backends:
            backend = self.backends[preferred]
            if self._backend_supports_task(backend, task_type):
                return backend
                
        # Auto-select based on task type and availability
        priority_order = self._get_backend_priority(task_type)
        
        for backend_type in priority_order:
            if backend_type in self.backends:
                backend = self.backends[backend_type]
                if self._backend_supports_task(backend, task_type):
                    return backend
                    
        # Fallback to CPU
        return self.backends.get(RuntimeBackend.CPU)
        
    def _get_backend_priority(self, task_type: TaskType) -> List[RuntimeBackend]:
        """Get backend priority for task type"""
        if task_type == TaskType.INFERENCE:
            return [RuntimeBackend.MLX, RuntimeBackend.PYTORCH, RuntimeBackend.ONNX, RuntimeBackend.CPU]
        elif task_type == TaskType.EMBEDDING:
            return [RuntimeBackend.ONNX, RuntimeBackend.MLX, RuntimeBackend.PYTORCH, RuntimeBackend.CPU]
        elif task_type == TaskType.COMPLETION:
            return [RuntimeBackend.OPENAI, RuntimeBackend.OLLAMA, RuntimeBackend.LLM_CLI]
        elif task_type == TaskType.AGENT:
            return [RuntimeBackend.OPENAI, RuntimeBackend.OLLAMA]
        elif task_type == TaskType.TOOL_USE:
            return [RuntimeBackend.OPENAI, RuntimeBackend.LLM_CLI]
        elif task_type == TaskType.COMMAND:
            return [RuntimeBackend.LLM_CLI, RuntimeBackend.CPU]
        else:
            return [RuntimeBackend.CPU]
            
    def _backend_supports_task(self, backend: Any, task_type: TaskType) -> bool:
        """Check if backend supports task type"""
        method_map = {
            TaskType.INFERENCE: "inference",
            TaskType.EMBEDDING: "embed",
            TaskType.COMPLETION: "complete",
            TaskType.AGENT: "run_agent",
            TaskType.TOOL_USE: "use_tool",
            TaskType.WORKFLOW: "run_workflow",
            TaskType.COMMAND: "run_command"
        }
        
        method_name = method_map.get(task_type)
        return hasattr(backend, method_name) if method_name else False
        
    async def _broadcast_result(self, result: TaskResult) -> None:
        """Broadcast result to connected websockets"""
        if not self._websockets:
            return
            
        message = {
            "type": "task_result",
            "data": {
                "task_id": result.task_id,
                "status": result.status,
                "result": result.result,
                "metrics": result.metrics,
                "error": result.error
            }
        }
        
        disconnected = []
        for ws in self._websockets:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
                
        # Remove disconnected websockets
        for ws in disconnected:
            self._websockets.remove(ws)
            
    async def _metrics_collector(self) -> None:
        """Collect runtime metrics periodically"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                # Collect backend metrics
                for backend_type, backend in self.backends.items():
                    if hasattr(backend, "get_metrics"):
                        backend_metrics = await backend.get_metrics()
                        self.metrics[f"backend_{backend_type.value}"] = backend_metrics
                        
                # Log metrics
                logger.info(f"Runtime metrics: {self.metrics}")
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                
    async def register_websocket(self, websocket: WebSocket) -> None:
        """Register a websocket for updates"""
        self._websockets.append(websocket)
        logger.info(f"WebSocket registered. Total: {len(self._websockets)}")
        
    async def unregister_websocket(self, websocket: WebSocket) -> None:
        """Unregister a websocket"""
        if websocket in self._websockets:
            self._websockets.remove(websocket)
            logger.info(f"WebSocket unregistered. Total: {len(self._websockets)}")
            
    async def get_status(self) -> Dict[str, Any]:
        """Get runtime status"""
        return {
            "initialized": self._initialized,
            "backends": list(self.backends.keys()),
            "active_tasks": len(self.active_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "metrics": self.metrics,
            "websockets": len(self._websockets),
            "config": {
                "backend": self.config.backend.value,
                "device": self.config.device,
                "cache_enabled": self.config.enable_caching,
                "monitoring_enabled": self.config.enable_monitoring
            }
        }
        
    async def shutdown(self) -> None:
        """Shutdown runtime engine"""
        logger.info("Shutting down OpenRuntime Engine...")
        
        # Cancel active tasks
        for task in self.active_tasks.values():
            task.cancel()
            
        # Shutdown backends
        for backend in self.backends.values():
            if hasattr(backend, "shutdown"):
                await backend.shutdown()
                
        # Close websockets
        for ws in self._websockets:
            try:
                await ws.close()
            except Exception:
                pass
                
        self._initialized = False
        logger.info("OpenRuntime Engine shutdown complete")