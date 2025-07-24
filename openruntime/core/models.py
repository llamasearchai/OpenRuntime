from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    CPU = "cpu"
    MLX = "mlx"
    METAL = "metal"
    CUDA = "cuda"
    VULKAN = "vulkan"


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
    capabilities: List[str] = field(default_factory=list)
    driver_version: str = ""
    compute_capability: str = ""


@dataclass
class RuntimeMetrics:
    device_id: str
    timestamp: datetime
    memory_usage: float
    gpu_utilization: float
    temperature: float
    power_usage: float
    throughput: float
    active_kernels: int = 0


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
    capabilities: List[str] = field(default_factory=list)
    system_prompt: str = ""


class TaskRequest(BaseModel):
    task_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = Field(..., description="Operation type: compute, inference, benchmark, mlx_compute")
    data: Dict[str, Any] = Field(default_factory=dict)
    device_preference: Optional[DeviceType] = None
    priority: int = Field(default=1, ge=1, le=10)
    timeout: int = Field(default=300, ge=1, le=3600)


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    device_used: Optional[str] = None


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


class ComputeKernel(BaseModel):
    name: str
    source_code: str
    entry_point: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    target_device: DeviceType = DeviceType.MLX


class MLXModel(BaseModel):
    name: str
    model_type: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: Dict[str, Any] = Field(default_factory=dict)
