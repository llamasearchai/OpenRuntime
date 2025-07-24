#!/usr/bin/env python3
"""
OpenRuntime Enhanced: Advanced GPU Runtime System with AI Integration
A comprehensive GPU computing and ML inference platform for macOS

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Core dependencies
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# AI Integration
import openai

# GPU and ML dependencies (mock implementations for compatibility)
try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
    print("MLX Metal Performance framework available")
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available - using CPU fallback")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - using NumPy fallback")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OpenRuntime")

# Version information
__version__ = "2.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

# =============================================================================
# Core Data Models
# =============================================================================


class DeviceType(str, Enum):
    CPU = "cpu"
    MLX = "mlx"
    METAL = "metal"
    CUDA = "cuda"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AIProviderType(str, Enum):
    OPENAI = "openai"
    LOCAL = "local"


class WorkflowType(str, Enum):
    COMPUTE_OPTIMIZATION = "compute_optimization"
    MODEL_INFERENCE = "model_inference"
    SYSTEM_ANALYSIS = "system_analysis"
    CODE_GENERATION = "code_generation"
    SHELL_AUTOMATION = "shell_automation"


class AgentRole(str, Enum):
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    SYSTEM_ANALYST = "system_analyst"
    CODE_GENERATOR = "code_generator"
    SHELL_EXECUTOR = "shell_executor"
    WORKFLOW_COORDINATOR = "workflow_coordinator"


@dataclass
class GPUDevice:
    id: str
    name: str
    type: DeviceType
    memory_total: int
    memory_available: int
    compute_units: int
    is_available: bool = True


@dataclass
class RuntimeMetrics:
    device_id: str
    timestamp: datetime
    memory_usage: float
    gpu_utilization: float
    temperature: float
    power_usage: float
    throughput: float


@dataclass
class AIAgent:
    id: str
    name: str
    role: AgentRole
    provider: AIProviderType
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    is_active: bool = True
    capabilities: List[str] = None
    system_prompt: str = ""


class TaskRequest(BaseModel):
    task_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = Field(..., description="Operation type: compute, inference, benchmark")
    data: Dict[str, Any] = Field(default_factory=dict)
    device_preference: Optional[DeviceType] = None
    priority: int = Field(default=1, ge=1, le=10)


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class AITaskRequest(BaseModel):
    task_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_type: WorkflowType
    prompt: str
    context: Dict[str, Any] = Field(default_factory=dict)
    agent_role: Optional[AgentRole] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    stream: bool = False


class ShellCommandRequest(BaseModel):
    command: str
    use_ai: bool = True
    context: str = ""
    safety_check: bool = True
    timeout: int = 30


class CodeGenerationRequest(BaseModel):
    language: str
    description: str
    context: Dict[str, Any] = Field(default_factory=dict)
    optimization_target: str = "performance"
    include_tests: bool = True


# =============================================================================
# MLX Metal Performance Integration
# =============================================================================


class MLXRuntimeManager:
    """MLX Metal Performance Shaders integration for Apple Silicon"""

    def __init__(self):
        self.available = MLX_AVAILABLE
        self.device_info = self._get_device_info()

    def _get_device_info(self) -> Dict[str, Any]:
        """Get MLX device information"""
        if not self.available:
            return {"available": False, "reason": "MLX not installed"}

        try:
            # Get device capabilities
            return {
                "available": True,
                "device_count": 1,  # Apple Silicon typically has unified memory
                "memory_limit": mx.metal.get_memory_limit() if hasattr(mx, "metal") else 16 * 1024**3,
                "active_memory": mx.metal.get_active_memory() if hasattr(mx, "metal") else 0,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def matrix_multiply_mlx(self, size: int) -> Dict[str, Any]:
        """Perform matrix multiplication using MLX"""
        if not self.available:
            raise RuntimeError("MLX not available")

        start_time = time.time()

        # Create random matrices using MLX
        a = mx.random.normal((size, size))
        b = mx.random.normal((size, size))

        # Perform matrix multiplication on Metal
        c = mx.matmul(a, b)

        # Force evaluation
        mx.eval(c)

        duration = time.time() - start_time
        gflops = (2 * size**3) / (duration * 1e9)

        return {"gflops": gflops, "duration": duration, "device": "mlx_metal", "memory_used": c.nbytes}

    async def neural_network_inference_mlx(self, input_size: int, hidden_size: int, output_size: int) -> Dict[str, Any]:
        """Run neural network inference using MLX"""
        if not self.available:
            raise RuntimeError("MLX not available")

        start_time = time.time()

        # Create a simple neural network
        x = mx.random.normal((1, input_size))
        w1 = mx.random.normal((input_size, hidden_size))
        w2 = mx.random.normal((hidden_size, output_size))

        # Forward pass
        h = mx.maximum(mx.matmul(x, w1), 0)  # ReLU activation
        output = mx.matmul(h, w2)

        # Force evaluation
        mx.eval(output)

        duration = time.time() - start_time

        return {"inference_time": duration, "throughput": 1.0 / duration, "device": "mlx_metal", "output_shape": output.shape}


# =============================================================================
# AI Agent Manager
# =============================================================================


class AIAgentManager:
    """Manages AI agents and workflows using OpenAI directly"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.openai_client = None
        self.agents: Dict[str, AIAgent] = {}
        self._initialize_openai()
        self._initialize_agents()

    def _initialize_openai(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY") or self.config.get("openai_api_key")
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        else:
            logger.warning("OpenAI API key not provided - AI features will be limited")

    def _initialize_agents(self) -> None:
        """Initialize AI agents"""
        agents_config = [
            {
                "id": "perf_optimizer",
                "name": "Performance Optimizer",
                "role": AgentRole.PERFORMANCE_OPTIMIZER,
                "system_prompt": "You are a GPU performance optimization expert. Analyze system metrics and provide optimization recommendations for GPU workloads. Focus on memory usage, compute efficiency, and throughput optimization.",
                "capabilities": ["performance_analysis", "optimization_recommendations", "benchmarking"],
            },
            {
                "id": "system_analyst",
                "name": "System Analyst",
                "role": AgentRole.SYSTEM_ANALYST,
                "system_prompt": "You are a system analysis expert specializing in GPU computing systems. Analyze system behavior, identify bottlenecks, and provide diagnostic insights.",
                "capabilities": ["system_diagnostics", "bottleneck_analysis", "resource_monitoring"],
            },
            {
                "id": "code_generator",
                "name": "Code Generator",
                "role": AgentRole.CODE_GENERATOR,
                "system_prompt": "You are an expert programmer specializing in GPU computing and parallel algorithms. Generate optimized code for various programming languages with focus on performance.",
                "capabilities": ["code_generation", "optimization", "testing", "documentation"],
            },
            {
                "id": "shell_executor",
                "name": "Shell Executor",
                "role": AgentRole.SHELL_EXECUTOR,
                "system_prompt": "You are a shell command expert. Generate safe and efficient shell commands for system administration and automation tasks. Always prioritize security.",
                "capabilities": ["shell_commands", "automation", "system_administration"],
            },
        ]

        for agent_config in agents_config:
            agent = AIAgent(
                id=agent_config["id"],
                name=agent_config["name"],
                role=agent_config["role"],
                provider=AIProviderType.OPENAI,
                model="gpt-4o-mini",
                system_prompt=agent_config["system_prompt"],
                capabilities=agent_config["capabilities"],
            )
            self.agents[agent.id] = agent

        logger.info(f"Initialized {len(self.agents)} AI agents")

    async def execute_ai_task(self, task: AITaskRequest) -> Dict[str, Any]:
        """Execute an AI task using the appropriate agent"""
        start_time = time.time()

        if not self.openai_client:
            return {"error": "OpenAI client not available - check API key configuration", "task_id": task.task_id}

        try:
            # Select appropriate agent
            agent = self._select_agent(task.agent_role, task.workflow_type)
            if not agent:
                return {"error": "No suitable agent found for task", "task_id": task.task_id}

            # Prepare messages
            messages = [{"role": "system", "content": agent.system_prompt}]

            # Add context if provided
            if task.context:
                context_str = json.dumps(task.context, indent=2)
                messages.append({"role": "user", "content": f"Context: {context_str}\n\nTask: {task.prompt}"})
            else:
                messages.append({"role": "user", "content": task.prompt})

            # Execute with OpenAI
            response = await self._execute_openai_request(
                messages=messages, temperature=task.temperature, max_tokens=task.max_tokens, stream=task.stream
            )

            execution_time = time.time() - start_time

            return {
                "task_id": task.task_id,
                "agent_id": agent.id,
                "agent_name": agent.name,
                "workflow_type": task.workflow_type.value,
                "result": response,
                "execution_time": execution_time,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"AI task execution failed: {e}")
            return {"error": str(e), "task_id": task.task_id, "execution_time": time.time() - start_time, "status": "failed"}

    def _select_agent(self, preferred_role: Optional[AgentRole], workflow_type: WorkflowType) -> Optional[AIAgent]:
        """Select the most appropriate agent for a task"""
        if preferred_role:
            for agent in self.agents.values():
                if agent.role == preferred_role and agent.is_active:
                    return agent

        # Default agent selection based on workflow type
        role_mapping = {
            WorkflowType.COMPUTE_OPTIMIZATION: AgentRole.PERFORMANCE_OPTIMIZER,
            WorkflowType.SYSTEM_ANALYSIS: AgentRole.SYSTEM_ANALYST,
            WorkflowType.CODE_GENERATION: AgentRole.CODE_GENERATOR,
            WorkflowType.SHELL_AUTOMATION: AgentRole.SHELL_EXECUTOR,
            WorkflowType.MODEL_INFERENCE: AgentRole.PERFORMANCE_OPTIMIZER,
        }

        target_role = role_mapping.get(workflow_type, AgentRole.SYSTEM_ANALYST)
        for agent in self.agents.values():
            if agent.role == target_role and agent.is_active:
                return agent

        # Return any active agent as fallback
        for agent in self.agents.values():
            if agent.is_active:
                return agent

        return None

    async def _execute_openai_request(
        self, messages: List[Dict], temperature: float, max_tokens: int, stream: bool
    ) -> Dict[str, Any]:
        """Execute OpenAI API request"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=temperature, max_tokens=max_tokens, stream=stream
            )

            if stream:
                # Handle streaming response
                content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                return {"response": content, "type": "stream"}
            else:
                return {
                    "response": response.choices[0].message.content,
                    "type": "completion",
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                }
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    def _is_safe_command(self, command: str) -> bool:
        """Check if a shell command is safe to execute"""
        dangerous_patterns = [
            "rm -rf",
            "dd if=",
            "mkfs",
            "fdisk",
            "format",
            "sudo rm",
            "sudo su",
            "su -",
            "> /dev/",
            "chmod 777",
            "wget http",
            "curl http",
            "nc -l",
            "python -c",
            "eval",
            "init 0",
            "poweroff",
            "reboot",
            "killall",
            "pkill -9",
            "rm /",
            "mv /",
            "cp /",
            "> /etc",
            ">> /etc",
            "| bash",
            "| sh",
            "| zsh",
            "| fish",
            "&& rm",
            "; rm",
        ]

        command_lower = command.lower().strip()
        return not any(pattern in command_lower for pattern in dangerous_patterns)


# =============================================================================
# Enhanced GPU Runtime Manager
# =============================================================================


class EnhancedGPURuntimeManager:
    """Enhanced GPU runtime with MLX and AI integration"""

    def __init__(self, ai_config: Dict[str, Any] = None):
        # Initialize base GPU manager from openruntime.py
        from openruntime import GPURuntimeManager

        self.gpu_manager = GPURuntimeManager()

        # Initialize MLX manager
        self.mlx_manager = MLXRuntimeManager()

        # Initialize AI manager
        self.ai_manager = AIAgentManager(ai_config)

        # Enhanced metrics and insights
        self.ai_insights: List[Dict[str, Any]] = []

        logger.info("Enhanced GPU Runtime Manager initialized")

    async def execute_ai_enhanced_task(self, task_request: TaskRequest, ai_task: Optional[AITaskRequest] = None):
        """Execute a task with optional AI enhancement"""
        start_time = time.time()

        # Execute the base GPU task
        gpu_result = await self.gpu_manager.execute_task(task_request)

        # If AI task is provided, execute it as well
        ai_result = None
        if ai_task:
            ai_result = await self.ai_manager.execute_ai_task(ai_task)

        # Combine results
        combined_result = {
            "task_id": task_request.task_id,
            "gpu_result": asdict(gpu_result) if hasattr(gpu_result, "__dict__") else gpu_result,
            "ai_result": ai_result,
            "execution_time": time.time() - start_time,
            "enhanced": ai_result is not None,
        }

        # Store insights
        if ai_result and "error" not in ai_result:
            self.ai_insights.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_request.operation,
                    "ai_workflow": ai_task.workflow_type.value if ai_task else None,
                    "insight": ai_result.get("result", {}).get("response", ""),
                    "performance_metrics": gpu_result.metrics if hasattr(gpu_result, "metrics") else {},
                }
            )

        return combined_result

    async def get_ai_performance_insights(self) -> Dict[str, Any]:
        """Get AI-generated performance insights"""
        if not self.ai_insights:
            return {"message": "No AI insights available yet"}

        # Get recent metrics
        recent_metrics = self.gpu_manager.metrics_history[-10:] if hasattr(self.gpu_manager, "metrics_history") else []

        # Create analysis task
        analysis_task = AITaskRequest(
            workflow_type=WorkflowType.SYSTEM_ANALYSIS,
            prompt=f"""Analyze the following GPU performance data and provide optimization recommendations:

Recent Performance Metrics:
{json.dumps([asdict(m) for m in recent_metrics], indent=2, default=str)}

AI Insights History:
{json.dumps(self.ai_insights[-5:], indent=2)}

Provide specific recommendations for:
1. Performance optimization
2. Resource utilization improvements
3. Potential bottlenecks
4. Configuration suggestions
""",
            agent_role=AgentRole.PERFORMANCE_OPTIMIZER,
        )

        analysis_result = await self.ai_manager.execute_ai_task(analysis_task)

        return {
            "analysis": analysis_result,
            "metrics_analyzed": len(recent_metrics),
            "insights_count": len(self.ai_insights),
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Initialize Global Runtime
# =============================================================================

# Global runtime instance
enhanced_runtime = EnhancedGPURuntimeManager()

# =============================================================================
# WebSocket Manager
# =============================================================================


class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)

            for conn in disconnected:
                self.disconnect(conn)


websocket_manager = WebSocketManager()

# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    logger.info("OpenRuntime Enhanced starting up...")
    logger.info(f"MLX Available: {MLX_AVAILABLE}")
    logger.info(f"AI Agents: {len(enhanced_runtime.ai_manager.agents)}")
    logger.info("System ready")
    yield
    logger.info("OpenRuntime Enhanced shutting down...")


app = FastAPI(
    title="OpenRuntime Enhanced",
    description="Advanced GPU Runtime System with AI Integration",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "name": "OpenRuntime Enhanced",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "status": "running",
        "mlx_available": MLX_AVAILABLE,
        "ai_agents": len(enhanced_runtime.ai_manager.agents),
        "devices": len(enhanced_runtime.gpu_manager.devices),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "gpu_manager": "operational",
            "ai_manager": "operational" if enhanced_runtime.ai_manager.openai_client else "limited",
            "mlx_manager": "operational" if MLX_AVAILABLE else "unavailable",
        },
    }


@app.post("/ai/tasks", response_model=dict)
async def execute_ai_task(task: AITaskRequest):
    """Execute AI-powered task"""
    logger.info(f"Executing AI task: {task.workflow_type.value}")
    result = await enhanced_runtime.ai_manager.execute_ai_task(task)
    return result


@app.post("/ai/shell", response_model=dict)
async def execute_shell_with_ai(request: ShellCommandRequest):
    """Execute shell command with AI assistance"""
    if request.use_ai:
        ai_task = AITaskRequest(
            workflow_type=WorkflowType.SHELL_AUTOMATION,
            prompt=f"Generate and explain a safe shell command for: {request.command}",
            context={"original_command": request.command, "context": request.context},
        )
        result = await enhanced_runtime.ai_manager.execute_ai_task(ai_task)
        return result
    else:
        # Direct shell execution (with safety checks)
        if request.safety_check and not enhanced_runtime.ai_manager._is_safe_command(request.command):
            return {"error": "Command failed safety check", "command": request.command}

        try:
            result = subprocess.run(request.command, shell=True, capture_output=True, text=True, timeout=request.timeout)
            return {
                "command": request.command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            return {"error": str(e), "command": request.command}


@app.post("/ai/code", response_model=dict)
async def generate_code_with_ai(request: CodeGenerationRequest):
    """Generate code with AI assistance"""
    ai_task = AITaskRequest(
        workflow_type=WorkflowType.CODE_GENERATION,
        prompt=f"""Generate {request.language} code for: {request.description}

Requirements:
- Language: {request.language}
- Optimization target: {request.optimization_target}
- Include tests: {request.include_tests}
- Context: {json.dumps(request.context)}

Provide complete, working code with explanations.""",
        agent_role=AgentRole.CODE_GENERATOR,
    )

    result = await enhanced_runtime.ai_manager.execute_ai_task(ai_task)
    return result


@app.get("/ai/agents")
async def list_ai_agents():
    """List all available AI agents"""
    agents_info = []
    for agent in enhanced_runtime.ai_manager.agents.values():
        agents_info.append(asdict(agent))
    return {"agents": agents_info}


@app.get("/ai/insights")
async def get_ai_insights():
    """Get AI performance insights"""
    return await enhanced_runtime.get_ai_performance_insights()


@app.post("/tasks/enhanced", response_model=dict)
async def create_enhanced_task(gpu_task: dict, ai_task: Optional[AITaskRequest] = None):
    """Create enhanced task with GPU and AI components"""
    # Convert dict to TaskRequest
    task_request = TaskRequest(**gpu_task)
    result = await enhanced_runtime.execute_ai_enhanced_task(task_request, ai_task)
    return result


# Import remaining endpoints from base openruntime
@app.get("/devices")
async def list_devices():
    """List available devices"""
    devices = []
    for device in enhanced_runtime.gpu_manager.devices.values():
        device_dict = asdict(device)
        devices.append(device_dict)

    # Add MLX device info
    if MLX_AVAILABLE:
        mlx_info = enhanced_runtime.mlx_manager.device_info
        devices.append(
            {
                "id": "mlx_0",
                "name": "MLX Metal Device",
                "type": "mlx",
                "memory_total": mlx_info.get("memory_limit", 0),
                "memory_available": mlx_info.get("memory_limit", 0) - mlx_info.get("active_memory", 0),
                "compute_units": 1,
                "is_available": mlx_info.get("available", False),
            }
        )

    return {"devices": devices}


@app.get("/metrics")
async def get_system_metrics():
    """Get system metrics"""
    return {
        "gpu_metrics": (
            enhanced_runtime.gpu_manager.metrics_history[-10:]
            if hasattr(enhanced_runtime.gpu_manager, "metrics_history")
            else []
        ),
        "mlx_info": enhanced_runtime.mlx_manager.device_info,
        "ai_insights_count": len(enhanced_runtime.ai_insights),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/devices/{device_id}/metrics")
async def get_device_metrics(device_id: str):
    """Get metrics for specific device"""
    if device_id == "mlx_0":
        return {"device_id": device_id, "metrics": enhanced_runtime.mlx_manager.device_info, "type": "mlx"}
    else:
        # Delegate to base GPU manager
        metrics = [asdict(m) for m in enhanced_runtime.gpu_manager.metrics_history if m.device_id == device_id][-10:]
        return {"device_id": device_id, "metrics": metrics, "count": len(metrics)}


@app.post("/tasks", response_model=dict)
async def create_task(task_request: dict):
    """Create and execute GPU task"""
    task = TaskRequest(**task_request)
    result = await enhanced_runtime.gpu_manager.execute_task(task)

    # Convert result to dict if needed
    if hasattr(result, "__dict__"):
        return asdict(result)
    return result


@app.get("/tasks")
async def list_tasks():
    """List active tasks"""
    return {"active_tasks": list(enhanced_runtime.gpu_manager.active_tasks.keys())}


@app.post("/benchmark")
async def run_benchmark(benchmark_type: str = "compute"):
    """Run performance benchmark"""
    if benchmark_type == "mlx" and MLX_AVAILABLE:
        # MLX-specific benchmark
        try:
            matrix_result = await enhanced_runtime.mlx_manager.matrix_multiply_mlx(1024)
            nn_result = await enhanced_runtime.mlx_manager.neural_network_inference_mlx(512, 256, 10)

            return {
                "benchmark_type": "mlx",
                "results": {"matrix_multiply": matrix_result, "neural_network": nn_result},
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": f"MLX benchmark failed: {e}"}
    else:
        # Standard GPU benchmark
        task_request = TaskRequest(operation="benchmark", data={"type": benchmark_type})
        result = await enhanced_runtime.gpu_manager.execute_task(task_request)
        return asdict(result) if hasattr(result, "__dict__") else result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        await websocket.send_json({"type": "connection_established", "timestamp": datetime.now().isoformat()})

        while True:
            await asyncio.sleep(5)

            # Send system status
            await websocket.send_json(
                {
                    "type": "system_status",
                    "data": {
                        "active_devices": len(enhanced_runtime.gpu_manager.devices),
                        "ai_agents": len(enhanced_runtime.ai_manager.agents),
                        "mlx_available": MLX_AVAILABLE,
                        "insights_count": len(enhanced_runtime.ai_insights),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenRuntime Enhanced")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    print(f"OpenRuntime Enhanced v{__version__}")
    print(f"Author: {__author__} <{__email__}>")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "openruntime_enhanced:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
