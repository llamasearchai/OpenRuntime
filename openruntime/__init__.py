"""
OpenRuntime - High-performance GPU runtime with AI integration.

This package provides a comprehensive runtime system for GPU-accelerated computing
with integrated AI capabilities.
"""

__version__ = "2.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

from .core.models import (
    GPUDevice,
    TaskRequest,
    TaskResponse,
    TaskStatus,
    RuntimeMetrics,
    WorkflowType,
    AITaskRequest,
    ShellCommandRequest,
    CodeGenerationRequest,
    DeviceType,
    AgentRole,
    AIAgent,
    ComputeKernel,
    MLXModel
)

from .core.managers import (
    GPURuntimeManager,
    MLXRuntimeManager,
    AIAgentManager
)

from .core.api import app

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
    "app"
]