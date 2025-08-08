"""
OpenRuntime - High-performance GPU runtime with AI integration.

This package provides a comprehensive runtime system for GPU-accelerated computing
with integrated AI capabilities.
"""

__version__ = "2.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

from .core.api import app
from .core.managers import AIAgentManager, GPURuntimeManager, MLXRuntimeManager
from .core.models import (
    AgentRole,
    AIAgent,
    AITaskRequest,
    CodeGenerationRequest,
    ComputeKernel,
    DeviceType,
    GPUDevice,
    MLXModel,
    RuntimeMetrics,
    ShellCommandRequest,
    TaskRequest,
    TaskResponse,
    TaskStatus,
    WorkflowType,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Models
    "GPUDevice",
    "TaskRequest",
    "TaskResponse",
    "TaskStatus",
    "RuntimeMetrics",
    "WorkflowType",
    "AITaskRequest",
    "ShellCommandRequest",
    "CodeGenerationRequest",
    "DeviceType",
    "AgentRole",
    "AIAgent",
    "ComputeKernel",
    "MLXModel",
    # Managers
    "GPURuntimeManager",
    "MLXRuntimeManager",
    "AIAgentManager",
    # API
    "app",
]
