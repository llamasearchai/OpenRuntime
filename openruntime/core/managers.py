import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .models import (
    AIAgent,
    AITaskRequest,
    AgentRole,
    AIProviderType,
    DeviceType,
    GPUDevice,
    RuntimeMetrics,
    TaskRequest,
    TaskResponse,
    TaskStatus,
    WorkflowType,
)

# Dependencies for managers
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch
    import torch.mps as mps
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLXRuntimeManager:
    """Manages MLX Metal operations and computations"""

    def __init__(self):
        self.devices = {}
        self._initialize_mlx()

    def _initialize_mlx(self):
        if not MLX_AVAILABLE:
            logger.warning("MLX not available - Metal operations will be simulated")
            return
        try:
            # MLX doesn't have device_info(), but we can check if it's available
            # by trying to create a small array
            test_array = mx.zeros((1, 1))
            _ = test_array.item()  # Force evaluation
            
            mlx_device = GPUDevice(
                id="mlx_metal_0",
                name="Apple Metal GPU (MLX)",
                type=DeviceType.MLX,
                memory_total=16 * 1024 * 1024 * 1024,
                memory_available=12 * 1024 * 1024 * 1024,
                compute_units=32,
                is_available=True,
                capabilities=["matrix_multiply", "neural_networks", "fft", "convolution"],
                driver_version="MLX 0.0.6",
                compute_capability="Metal 3.0",
            )
            self.devices["mlx_metal_0"] = mlx_device
            logger.info(f"Initialized MLX Metal device: {mlx_device.name}")
        except Exception as e:
            logger.warning(f"MLX initialization skipped: {e}")
            # Don't fail completely if MLX has issues

    async def matrix_multiply_mlx(self, size: int, dtype: str = "float32") -> Dict[str, Any]:
        if not MLX_AVAILABLE:
            return await self._simulate_matrix_multiply(size)
        try:
            start_time = time.time()
            a = mx.random.normal((size, size), dtype=getattr(mx, dtype))
            b = mx.random.normal((size, size), dtype=getattr(mx, dtype))
            c = mx.matmul(a, b)
            mx.eval(c)
            execution_time = time.time() - start_time
            gflops = (2 * size**3) / (execution_time * 1e9)
            return {
                "operation": "matrix_multiply_mlx",
                "size": size,
                "dtype": dtype,
                "execution_time": execution_time,
                "gflops": gflops,
                "result_shape": c.shape,
                "device": "mlx_metal_0",
            }
        except Exception as e:
            logger.error(f"MLX matrix multiplication failed: {e}")
            return await self._simulate_matrix_multiply(size)

    async def _simulate_matrix_multiply(self, size: int) -> Dict[str, Any]:
        start_time = time.time()
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        c = np.dot(a, b)
        execution_time = time.time() - start_time
        gflops = (2 * size**3) / (execution_time * 1e9)
        return {
            "operation": "matrix_multiply_simulated",
            "size": size,
            "execution_time": execution_time,
            "gflops": gflops,
            "result_shape": c.shape,
            "device": "cpu_0",
        }


class AIAgentManager:
    """Manages AI agents and workflows"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.openai_client = None
        self.agents: Dict[str, AIAgent] = {}
        self._initialize_openai()
        self._initialize_agents()

    def _initialize_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not OPENAI_AVAILABLE:
            logger.warning("OpenAI API key not found or library not installed. AI features will be mocked.")
            return
        try:
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def _initialize_agents(self):
        agents_config = [
            {"id": "perf_optimizer", "name": "Performance Optimizer", "role": AgentRole.PERFORMANCE_OPTIMIZER, "system_prompt": "You are a GPU performance optimization expert..."},
            {"id": "system_analyst", "name": "System Analyst", "role": AgentRole.SYSTEM_ANALYST, "system_prompt": "You are a system analysis expert..."},
            {"id": "code_generator", "name": "Code Generator", "role": AgentRole.CODE_GENERATOR, "system_prompt": "You are an expert programmer..."},
            {"id": "shell_executor", "name": "Shell Executor", "role": AgentRole.SHELL_EXECUTOR, "system_prompt": "You are a shell command expert..."},
        ]
        for cfg in agents_config:
            agent = AIAgent(id=cfg["id"], name=cfg["name"], role=cfg["role"], provider=AIProviderType.OPENAI, model="gpt-4o-mini", system_prompt=cfg["system_prompt"])
            self.agents[agent.id] = agent
        logger.info(f"Initialized {len(self.agents)} AI agents")

    async def execute_ai_task(self, task: AITaskRequest) -> Dict[str, Any]:
        if not self.openai_client:
            return {"error": "OpenAI client not available", "status": "failed"}
        try:
            agent = self._select_agent(task.agent_role, task.workflow_type)
            if not agent:
                raise ValueError("No agent found")
            messages = [{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": task.prompt}]
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=agent.model, messages=messages,
                temperature=task.temperature, max_tokens=task.max_tokens
            )
            return {"result": response.choices[0].message.content}
        except Exception as e:
            return {"error": str(e)}

    def _select_agent(self, preferred_role: Optional[AgentRole], workflow_type: WorkflowType) -> Optional[AIAgent]:
        role_map = {
            WorkflowType.COMPUTE_OPTIMIZATION: AgentRole.PERFORMANCE_OPTIMIZER,
            WorkflowType.SYSTEM_ANALYSIS: AgentRole.SYSTEM_ANALYST,
            WorkflowType.CODE_GENERATION: AgentRole.CODE_GENERATOR,
            WorkflowType.SHELL_AUTOMATION: AgentRole.SHELL_EXECUTOR,
        }
        role = preferred_role or role_map.get(workflow_type)
        return next((agent for agent in self.agents.values() if agent.role == role), None)


class GPURuntimeManager:
    def __init__(self, ai_config: Dict[str, Any] = None):
        self.devices: Dict[str, GPUDevice] = {}
        self.metrics_history: List[RuntimeMetrics] = []
        self.active_tasks: Dict[str, Dict] = {}
        self.completed_tasks: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.mlx_manager = MLXRuntimeManager()
        self.ai_manager = AIAgentManager(ai_config)
        self.ai_insights: List[Dict[str, Any]] = []
        self._initialize_devices()
        self._start_monitoring()

    def _initialize_devices(self):
        """Initialize and discover available GPU devices using system_profiler."""
        logger.info("Initializing devices...")
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, check=True
            )
            data = json.loads(result.stdout)
            gpu_data = data.get("SPDisplaysDataType", [])

            if not gpu_data:
                logger.warning("No GPU data found from system_profiler.")
            
            for i, gpu in enumerate(gpu_data):
                model = gpu.get("sppci_model", f"Unknown GPU {i}")
                vendor = gpu.get("spdisplays_vendor", "Unknown Vendor")
                vram_str = gpu.get("spdisplays_vram", "0 MB")
                
                try:
                    vram_mb = int(vram_str.split(" ")[0])
                    vram_bytes = vram_mb * 1024 * 1024
                except (ValueError, IndexError):
                    vram_bytes = 0

                device_id = f"metal_{i}"
                device = GPUDevice(
                    id=device_id,
                    name=f"{vendor} {model}",
                    type=DeviceType.METAL,
                    memory_total=vram_bytes,
                    memory_available=int(vram_bytes * 0.8),
                    compute_units=gpu.get("spdisplays_core_count", 0),
                    is_available=True,
                    capabilities=["metal", "pytorch_ops"] if TORCH_AVAILABLE else ["metal"],
                    driver_version=gpu.get("spdisplays_driver_version", "N/A")
                )
                self.devices[device_id] = device
                logger.info(f"Initialized device: {device.name}")

        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not query system_profiler for GPU info: {e}. Using fallback devices.")
            if MLX_AVAILABLE:
                self.devices.update(self.mlx_manager.devices)
            if TORCH_AVAILABLE and mps.is_available():
                self.devices["torch_metal_0"] = GPUDevice(id="torch_metal_0", name="Apple Metal GPU (PyTorch)", type=DeviceType.METAL, memory_total=16*1024**3, memory_available=12*1024**3, compute_units=32)
        
        self.devices["cpu_0"] = GPUDevice(id="cpu_0", name="CPU", type=DeviceType.CPU, memory_total=32*1024**3, memory_available=24*1024**3, compute_units=os.cpu_count() or 8)
        logger.info(f"Initialized CPU device.")

    def _start_monitoring(self):
        def monitor_loop():
            while True:
                try:
                    for device in self.devices.values():
                        self.metrics_history.append(self._collect_device_metrics(device))
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
        threading.Thread(target=monitor_loop, daemon=True).start()

    def _collect_device_metrics(self, device: GPUDevice) -> RuntimeMetrics:
        """Collect real-time metrics from a device."""
        # Get actual system metrics using platform-specific methods
        memory_usage = 0.0
        gpu_utilization = 0.0
        temperature = 0.0
        power_usage = 0.0
        
        try:
            # Memory usage calculation
            if device.memory_total > 0:
                memory_usage = (device.memory_total - device.memory_available) / device.memory_total * 100
            
            # Try to get real GPU utilization (platform-specific)
            if sys.platform == "darwin":
                # On macOS with Metal, we can estimate utilization based on active tasks
                active_task_count = len([t for t in self.active_tasks.values() if t['status'] == TaskStatus.RUNNING])
                gpu_utilization = min(100, active_task_count * 25)  # Estimate 25% per active task
                
                # For Apple Silicon, try to read power metrics
                try:
                    import subprocess
                    # Try powermetrics command (requires sudo)
                    result = subprocess.run(['sudo', '-n', 'powermetrics', '-n', '1', '-i', '1'], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        # Parse output for GPU power/temperature info
                        for line in result.stdout.split('\n'):
                            if 'GPU Power' in line:
                                power_match = re.search(r'(\d+\.?\d*)\s*mW', line)
                                if power_match:
                                    power_usage = float(power_match.group(1)) / 1000  # Convert to W
                except Exception as e:
                    # Fallback to estimates
                    logger.debug(f"Power metrics collection failed: {e}")
                    power_usage = 15 + gpu_utilization * 0.3
                    
                # Temperature estimate (Apple Silicon runs cool)
                temperature = 35 + gpu_utilization * 0.3
            else:
                # For other platforms, use estimates
                gpu_utilization = min(100, active_task_count * 30)
                temperature = 45 + gpu_utilization * 0.5
                power_usage = 20 + gpu_utilization * 0.5
                
        except Exception as e:
            logger.debug(f"Error collecting metrics: {e}")
            # Fallback to reasonable estimates
            gpu_utilization = 30
            temperature = 45
            power_usage = 20
        
        return RuntimeMetrics(
            device_id=device.id, 
            timestamp=datetime.now(),
            memory_usage=memory_usage,
            gpu_utilization=gpu_utilization, 
            temperature=temperature,
            power_usage=power_usage, 
            throughput=gpu_utilization * 100,  # Estimated throughput
            active_kernels=len(self.active_tasks)
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current runtime metrics."""
        active_task_count = len([t for t in self.active_tasks.values() if t['status'] == TaskStatus.RUNNING])
        
        # Calculate aggregate metrics from all devices
        total_memory = sum(d.memory_total for d in self.devices.values())
        available_memory = sum(d.memory_available for d in self.devices.values())
        memory_usage = ((total_memory - available_memory) / total_memory * 100) if total_memory > 0 else 0
        
        # Get latest device metrics if available
        gpu_utilization = 0
        if self.monitoring_data:
            latest_metrics = list(self.monitoring_data.values())
            if latest_metrics:
                gpu_utilization = sum(m.gpu_utilization for m in latest_metrics) / len(latest_metrics)
        
        return {
            "gpu_utilization": round(gpu_utilization, 1),
            "memory_usage": round(memory_usage, 1),
            "active_tasks": active_task_count,
            "ai_tasks_processed": getattr(self, 'ai_tasks_processed', 0),
            "total_devices": len(self.devices),
            "available_devices": len([d for d in self.devices.values() if d.is_available])
        }

    async def execute_task(self, task: TaskRequest) -> TaskResponse:
        start_time = time.time()
        try:
            device = self._select_device(task.device_preference)
            if not device:
                raise Exception("No suitable device available")
            self.active_tasks[task.task_id] = {"status": TaskStatus.RUNNING, "start_time": start_time}
            
            if task.operation == "compute":
                result = await self._execute_compute(task, device)
            else:
                raise ValueError("Unknown operation")

            execution_time = time.time() - start_time
            return TaskResponse(task_id=task.task_id, status=TaskStatus.COMPLETED, result=result, execution_time=execution_time, device_used=device.id)
        except Exception as e:
            return TaskResponse(task_id=task.task_id, status=TaskStatus.FAILED, error=str(e))

    def _select_device(self, preference: Optional[DeviceType]) -> Optional[GPUDevice]:
        # Simplified device selection
        return self.devices.get("mlx_metal_0") or self.devices.get("torch_metal_0") or self.devices.get("cpu_0")

    async def _execute_compute(self, task: TaskRequest, device: GPUDevice) -> Dict[str, Any]:
        size = task.data.get("size", 1024)
        if device.type == DeviceType.MLX:
            return await self.mlx_manager.matrix_multiply_mlx(size)
        else: # Fallback to CPU
            return await self.mlx_manager._simulate_matrix_multiply(size)