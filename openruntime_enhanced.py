#!/usr/bin/env python3
"""
OpenRuntime Enhanced: Advanced GPU Runtime System with AI Integration
A comprehensive GPU computing and ML inference platform with OpenAI, LangChain, and shell-gpt integration

Features:
- Multi-GPU runtime management with AI orchestration
- OpenAI Agents SDK integration
- LangChain workflow automation
- Shell-GPT command execution
- Real-time AI-driven performance optimization
- Advanced ML model inference pipeline
- RESTful API endpoints with AI capabilities
- WebSocket streaming with AI insights
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Core dependencies
import numpy as np
import uvicorn
import yaml
from fastapi import (BackgroundTasks, Depends, FastAPI, HTTPException,
                     Security, WebSocket, status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

# AI Integration dependencies
try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available - AI features disabled")

try:
    from langchain.chains import LLMChain
    from langchain_community.tools import ShellTool
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import Tool
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available - workflow automation disabled")

try:
    import shell_gpt
    from shell_gpt.app import main as sgpt_main

    SHELLGPT_AVAILABLE = True
except ImportError:
    SHELLGPT_AVAILABLE = False
    print("shell-gpt not available - shell AI features disabled")

# GPU and ML dependencies
try:
    import metal_performance_shaders as mps

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger("OpenRuntimeEnhanced")

# =============================================================================
# Enhanced Data Models
# =============================================================================


class AIProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


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


@dataclass
class WorkflowTask:
    id: str
    workflow_type: WorkflowType
    agent_id: str
    input_data: Dict[str, Any]
    expected_output: str
    priority: int = 1
    timeout: int = 300
    dependencies: List[str] = None


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
    """
    Request model for code generation tasks.
    """


# =============================================================================
# AI Agent Manager
# =============================================================================


class AIAgentManager:
    """
    Manages AI agents for various tasks such as optimization, analysis, and code generation.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agents: Dict[str, AIAgent] = {}
        self.openai_client = None
        self.langchain_agents = {}
        # Note: ConversationBufferMemory is deprecated in LangChain v0.2+
        # Modern LangChain uses state management in LangGraph
        self.conversation_memory = None
        self._initialize_ai_providers()
        self._initialize_agents()

    def _initialize_ai_providers(self):
        """Initialize AI provider connections"""
        if OPENAI_AVAILABLE:
            api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not found")

    def _initialize_agents(self) -> None:
        """
        Initialize AI agents with specific roles and capabilities.
        """
        agent_configs = [
            {
                "id": "perf_optimizer",
                "name": "Performance Optimizer",
                "role": AgentRole.PERFORMANCE_OPTIMIZER,
                "provider": AIProviderType.OPENAI,
                "model": "gpt-4o-mini",
                "system_prompt": """You are a GPU performance optimization expert. 
                Analyze system metrics and provide optimization recommendations for GPU workloads.
                Focus on memory usage, compute efficiency, and throughput optimization.""",
                "capabilities": ["performance_analysis", "optimization_recommendations", "benchmarking"],
            },
            {
                "id": "system_analyst",
                "name": "System Analyst",
                "role": AgentRole.SYSTEM_ANALYST,
                "provider": AIProviderType.OPENAI,
                "model": "gpt-4o-mini",
                "system_prompt": """You are a system analysis expert specializing in GPU computing systems.
                Analyze system behavior, identify bottlenecks, and provide diagnostic insights.""",
                "capabilities": ["system_diagnostics", "bottleneck_analysis", "resource_monitoring"],
            },
            {
                "id": "code_generator",
                "name": "Code Generator",
                "role": AgentRole.CODE_GENERATOR,
                "provider": AIProviderType.OPENAI,
                "model": "gpt-4o-mini",
                "system_prompt": """You are an expert programmer specializing in GPU computing and parallel algorithms.
                Generate optimized code for various programming languages with focus on performance.""",
                "capabilities": ["code_generation", "optimization", "testing", "documentation"],
            },
            {
                "id": "shell_executor",
                "name": "Shell Executor",
                "role": AgentRole.SHELL_EXECUTOR,
                "provider": AIProviderType.OPENAI,
                "model": "gpt-4o-mini",
                "system_prompt": """You are a shell command expert. Generate safe and efficient shell commands
                for system administration and automation tasks. Always prioritize security.""",
                "capabilities": ["shell_commands", "automation", "system_administration"],
            },
        ]

        for config in agent_configs:
            agent = AIAgent(**config)
            self.agents[agent.id] = agent
            logger.info(f"Initialized agent: {agent.name}")

        if LANGCHAIN_AVAILABLE and self.openai_client:
            self._initialize_langchain_agents()

    def _initialize_langchain_agents(self):
        """Initialize LangChain agents for complex workflows"""
        try:
            llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                openai_api_key=self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
            )

            # Define tools for agents
            tools = [
                Tool(name="SystemMetrics", description="Get current system and GPU metrics", func=self._get_system_metrics),
                Tool(name="ExecuteShell", description="Execute shell commands safely", func=self._execute_shell_command),
                Tool(name="OptimizeWorkload", description="Optimize GPU workload parameters", func=self._optimize_workload),
            ]

            # Note: initialize_agent is deprecated in LangChain v0.2+
            # Modern approach uses LangGraph for agent workflows
            # For now, we use direct tool calling
            self.langchain_agents["coordinator"] = {"llm": llm, "tools": tools, "type": "tool_calling_agent"}

            logger.info("LangChain agents initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain agents: {e}")

    async def execute_ai_task(self, task: AITaskRequest) -> Dict[str, Any]:
        """
        Execute AI task with the appropriate agent.

        Args:
            task (AITaskRequest): The AI task request containing task details.

        Returns:
            Dict[str, Any]: The result of the AI task execution.
        """
        start_time = time.time()

        try:
            # Select appropriate agent
            agent = self._select_agent(task.agent_role, task.workflow_type)
            if not agent:
                raise ValueError("No suitable agent available")

            # Execute based on workflow type
            if task.workflow_type == WorkflowType.COMPUTE_OPTIMIZATION:
                result = await self._execute_optimization_task(task, agent)
            elif task.workflow_type == WorkflowType.SYSTEM_ANALYSIS:
                result = await self._execute_analysis_task(task, agent)
            elif task.workflow_type == WorkflowType.CODE_GENERATION:
                result = await self._execute_code_generation_task(task, agent)
            elif task.workflow_type == WorkflowType.SHELL_AUTOMATION:
                result = await self._execute_shell_automation_task(task, agent)
            else:
                result = await self._execute_general_task(task, agent)

            execution_time = time.time() - start_time

            return {
                "task_id": task.task_id,
                "agent_id": agent.id,
                "agent_name": agent.name,
                "result": result,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"AI task execution failed: {e}")
            return {"task_id": task.task_id, "error": str(e), "execution_time": time.time() - start_time}

    def _select_agent(self, preferred_role: Optional[AgentRole], workflow_type: WorkflowType) -> Optional[AIAgent]:
        """Select the most appropriate agent for the task"""
        if preferred_role:
            agent = next((a for a in self.agents.values() if a.role == preferred_role and a.is_active), None)
            if agent:
                return agent

        # Fallback based on workflow type
        role_mapping = {
            WorkflowType.COMPUTE_OPTIMIZATION: AgentRole.PERFORMANCE_OPTIMIZER,
            WorkflowType.SYSTEM_ANALYSIS: AgentRole.SYSTEM_ANALYST,
            WorkflowType.CODE_GENERATION: AgentRole.CODE_GENERATOR,
            WorkflowType.SHELL_AUTOMATION: AgentRole.SHELL_EXECUTOR,
        }

        target_role = role_mapping.get(workflow_type)
        if target_role:
            return next((a for a in self.agents.values() if a.role == target_role and a.is_active), None)

        # Return any active agent
        return next((a for a in self.agents.values() if a.is_active), None)

    async def _execute_optimization_task(self, task: AITaskRequest, agent: AIAgent) -> Dict[str, Any]:
        """
        Execute performance optimization task using AI agent.

        Args:
            task (AITaskRequest): The AI task request containing task details.
            agent (AIAgent): The AI agent to execute the task.

        Returns:
            Dict[str, Any]: The result of the optimization task.
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not available")

        system_context = task.context.get("system_metrics", {})

        prompt = f"""
        Analyze the following GPU system metrics and provide optimization recommendations:
        
        System Context: {json.dumps(system_context, indent=2)}
        
        User Request: {task.prompt}
        
        Please provide:
        1. Performance bottleneck analysis
        2. Specific optimization recommendations
        3. Expected performance improvements
        4. Implementation steps
        """

        response = await self.openai_client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": prompt}],
            temperature=task.temperature,
            max_tokens=task.max_tokens,
        )

        return {
            "type": "optimization_recommendations",
            "analysis": response.choices[0].message.content,
            "model_used": agent.model,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }

    async def _execute_analysis_task(self, task: AITaskRequest, agent: AIAgent) -> Dict[str, Any]:
        """Execute system analysis task"""
        if not self.openai_client:
            raise ValueError("OpenAI client not available")

        system_context = task.context.get("system_metrics", {})

        prompt = f"""
        Analyze the following system information and provide diagnostic insights:
        
        System Context: {json.dumps(system_context, indent=2)}
        
        User Request: {task.prompt}
        
        Please provide:
        1. System health assessment
        2. Bottleneck identification
        3. Resource utilization analysis
        4. Recommendations for improvement
        """

        response = await self.openai_client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": prompt}],
            temperature=task.temperature,
            max_tokens=task.max_tokens,
        )

        return {
            "type": "system_analysis",
            "analysis": response.choices[0].message.content,
            "model_used": agent.model,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }

    async def _execute_code_generation_task(self, task: AITaskRequest, agent: AIAgent) -> Dict[str, Any]:
        """
        Execute code generation task using AI agent.

        Args:
            task (AITaskRequest): The AI task request containing task details.
            agent (AIAgent): The AI agent to execute the task.

        Returns:
            Dict[str, Any]: The generated code and related information.
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not available")

        context = task.context

        prompt = f"""
        Generate optimized code based on the following request:
        
        Request: {task.prompt}
        Context: {json.dumps(context, indent=2)}
        
        Please provide:
        1. Complete, working code
        2. Performance optimizations
        3. Documentation
        4. Test cases (if applicable)
        """

        response = await self.openai_client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": prompt}],
            temperature=task.temperature,
            max_tokens=task.max_tokens,
        )

        return {
            "type": "code_generation",
            "code": response.choices[0].message.content,
            "model_used": agent.model,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }

    async def _execute_shell_automation_task(self, task: AITaskRequest, agent: AIAgent) -> Dict[str, Any]:
        """Execute shell automation task"""
        if SHELLGPT_AVAILABLE:
            return await self._execute_with_shell_gpt(task, agent)
        else:
            return await self._execute_shell_fallback(task, agent)

    async def _execute_with_shell_gpt(self, task: AITaskRequest, agent: AIAgent) -> Dict[str, Any]:
        """Execute shell task using shell-gpt"""
        try:
            # Use shell-gpt for command generation
            command_prompt = f"Generate shell command for: {task.prompt}"

            # Execute shell-gpt in subprocess
            result = subprocess.run(["sgpt", "--shell", command_prompt], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                generated_command = result.stdout.strip()

                # Safety check
                if self._is_safe_command(generated_command):
                    return {
                        "type": "shell_automation",
                        "generated_command": generated_command,
                        "status": "safe_to_execute",
                        "agent_used": "shell-gpt",
                    }
                else:
                    return {
                        "type": "shell_automation",
                        "generated_command": generated_command,
                        "status": "requires_manual_review",
                        "warning": "Command may be unsafe",
                        "agent_used": "shell-gpt",
                    }
            else:
                raise Exception(f"shell-gpt failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Shell-GPT execution failed: {e}")
            return await self._execute_shell_fallback(task, agent)

    async def _execute_shell_fallback(self, task: AITaskRequest, agent: AIAgent) -> Dict[str, Any]:
        """Fallback shell execution using OpenAI"""
        if not self.openai_client:
            raise ValueError("OpenAI client not available")

        prompt = f"""
        Generate a safe shell command for the following task:
        
        Task: {task.prompt}
        Context: {json.dumps(task.context, indent=2)}
        
        Requirements:
        - Command must be safe to execute
        - Include explanation of what the command does
        - Provide any necessary warnings
        """

        response = await self.openai_client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": prompt}],
            temperature=task.temperature,
            max_tokens=task.max_tokens,
        )

        return {
            "type": "shell_automation",
            "response": response.choices[0].message.content,
            "model_used": agent.model,
            "agent_used": "openai_fallback",
        }

    async def _execute_general_task(self, task: AITaskRequest, agent: AIAgent) -> Dict[str, Any]:
        """Execute general AI task"""
        if not self.openai_client:
            raise ValueError("OpenAI client not available")

        response = await self.openai_client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": task.prompt}],
            temperature=task.temperature,
            max_tokens=task.max_tokens,
        )

        return {
            "type": "general_task",
            "response": response.choices[0].message.content,
            "model_used": agent.model,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }

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

    def _get_system_metrics(self) -> str:
        """Get current system metrics"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
            }
            return json.dumps(metrics, indent=2)
        except Exception as e:
            return f"Error collecting metrics: {e}"

    def _execute_shell_command(self, command: str) -> str:
        """Execute shell command safely"""
        if not self._is_safe_command(command):
            return "Command rejected: potentially unsafe"

        try:
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=30)
            return f"Exit code: {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"
        except Exception as e:
            return f"Execution failed: {e}"

    def _optimize_workload(self, parameters: str) -> str:
        """Optimize workload parameters"""
        # Placeholder for workload optimization logic
        return f"Optimization recommendations for: {parameters}"


# Enhanced GPU Runtime Manager with AI Integration
class EnhancedGPURuntimeManager:
    def __init__(self, ai_config: Dict[str, Any] = None):
        # Initialize base GPU manager
        from openruntime import GPURuntimeManager

        self.gpu_manager = GPURuntimeManager()

        # Initialize AI agent manager
        self.ai_manager = AIAgentManager(ai_config)

        # Enhanced monitoring
        self.ai_insights: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []

        logger.info("Enhanced GPU Runtime Manager initialized with AI capabilities")

    async def execute_ai_enhanced_task(self, task_request, ai_task: Optional[AITaskRequest] = None):
        """Execute task with AI enhancement"""
        start_time = time.time()

        # Execute base GPU task
        gpu_result = await self.gpu_manager.execute_task(task_request)

        # If AI task provided, execute AI analysis
        if ai_task:
            ai_result = await self.ai_manager.execute_ai_task(ai_task)

            # Combine results
            enhanced_result = {
                "gpu_task": gpu_result.dict(),
                "ai_analysis": ai_result,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }

            # Store AI insights
            self.ai_insights.append(enhanced_result)

            return enhanced_result

        return gpu_result.dict()

    async def get_ai_performance_insights(self) -> Dict[str, Any]:
        """Get AI-driven performance insights"""
        if not self.ai_insights:
            return {"message": "No AI insights available"}

        # Analyze recent performance data
        recent_insights = self.ai_insights[-10:]  # Last 10 insights

        # Generate AI-driven recommendations
        ai_task = AITaskRequest(
            workflow_type=WorkflowType.COMPUTE_OPTIMIZATION,
            prompt="Analyze recent performance data and provide optimization recommendations",
            context={"recent_insights": recent_insights},
        )

        recommendations = await self.ai_manager.execute_ai_task(ai_task)

        return {
            "insights_count": len(self.ai_insights),
            "recent_performance": recent_insights,
            "ai_recommendations": recommendations,
        }


# =============================================================================
# Enhanced FastAPI Application
# =============================================================================

# Initialize enhanced components
enhanced_runtime = EnhancedGPURuntimeManager()
security = HTTPBearer()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("OpenRuntime Enhanced starting up...")
    logger.info(f"AI Agents: {len(enhanced_runtime.ai_manager.agents)}")
    logger.info(f"GPU Devices: {len(enhanced_runtime.gpu_manager.devices)}")
    logger.info(f"OpenAI: {'Available' if OPENAI_AVAILABLE else 'Not Available'}")
    logger.info(f"LangChain: {'Available' if LANGCHAIN_AVAILABLE else 'Not Available'}")
    logger.info(f"Shell-GPT: {'Available' if SHELLGPT_AVAILABLE else 'Not Available'}")
    logger.info("System ready for AI-enhanced GPU computing")

    yield

    # Shutdown
    logger.info("OpenRuntime Enhanced shutting down...")
    enhanced_runtime.gpu_manager.executor.shutdown(wait=True)


# Create enhanced FastAPI app
app = FastAPI(
    title="OpenRuntime Enhanced",
    description="Advanced GPU Runtime System with AI Integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Enhanced API Endpoints
# =============================================================================


@app.get("/")
async def root():
    """Enhanced root endpoint with AI status"""
    return {
        "name": "OpenRuntime Enhanced",
        "version": "2.0.0",
        "status": "running",
        "ai_enabled": OPENAI_AVAILABLE,
        "langchain_enabled": LANGCHAIN_AVAILABLE,
        "shell_gpt_enabled": SHELLGPT_AVAILABLE,
        "agents": len(enhanced_runtime.ai_manager.agents),
        "devices": len(enhanced_runtime.gpu_manager.devices),
        "ai_insights": len(enhanced_runtime.ai_insights),
        "websocket_available": True,
        "websocket_endpoint": "/ws",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/ai/tasks", response_model=dict)
async def execute_ai_task(task: AITaskRequest):
    """Execute AI task"""
    try:
        result = await enhanced_runtime.ai_manager.execute_ai_task(task)
        return result
    except asyncio.TimeoutError as e:
        return {"task_id": task.task_id, "error": f"Task timed out: {str(e)}", "status": "timeout", "execution_time": 0}
    except Exception as e:
        logger.error(f"AI task execution failed: {e}")
        return {"task_id": task.task_id, "error": str(e), "status": "failed", "execution_time": 0}


@app.post("/ai/shell", response_model=dict)
async def execute_shell_with_ai(request: ShellCommandRequest):
    """Execute shell command with AI assistance"""
    ai_task = AITaskRequest(
        workflow_type=WorkflowType.SHELL_AUTOMATION,
        prompt=request.command,
        context={"original_context": request.context, "safety_check": request.safety_check},
    )

    result = await enhanced_runtime.ai_manager.execute_ai_task(ai_task)
    return result


@app.post("/ai/code", response_model=dict)
async def generate_code_with_ai(request: CodeGenerationRequest):
    """Generate code with AI assistance"""
    ai_task = AITaskRequest(
        workflow_type=WorkflowType.CODE_GENERATION,
        prompt=f"Generate {request.language} code: {request.description}",
        context={
            "language": request.language,
            "optimization_target": request.optimization_target,
            "include_tests": request.include_tests,
            **request.context,
        },
    )

    result = await enhanced_runtime.ai_manager.execute_ai_task(ai_task)
    return result


@app.get("/ai/agents")
async def list_ai_agents():
    """List all AI agents"""
    agents = []
    for agent in enhanced_runtime.ai_manager.agents.values():
        agents.append(asdict(agent))
    return {"agents": agents}


@app.get("/ai/insights")
async def get_ai_insights():
    """Get AI performance insights"""
    return await enhanced_runtime.get_ai_performance_insights()


@app.post("/tasks/enhanced", response_model=dict)
async def create_enhanced_task(gpu_task: dict, ai_task: Optional[AITaskRequest] = None):
    """Create enhanced task with AI analysis"""
    from openruntime import TaskRequest

    # Convert dict to TaskRequest
    task_request = TaskRequest(**gpu_task)

    result = await enhanced_runtime.execute_ai_enhanced_task(task_request, ai_task)
    return result


# =============================================================================
# Original OpenRuntime Endpoints (for compatibility)
# =============================================================================


@app.get("/devices")
async def list_devices():
    """List available GPU devices"""
    devices = []
    for device in enhanced_runtime.gpu_manager.devices.values():
        devices.append(
            {
                "id": device.id,
                "name": device.name,
                "type": device.type,
                "memory_total": device.memory_total,
                "memory_available": device.memory_available,
                "compute_units": device.compute_units,
                "is_available": device.is_available,
            }
        )
    return {"devices": devices}


@app.get("/metrics")
async def get_system_metrics():
    """Get system metrics"""
    return {
        "system": enhanced_runtime.ai_manager._get_system_metrics(),
        "devices": len(enhanced_runtime.gpu_manager.devices),
        "ai_agents": len(enhanced_runtime.ai_manager.agents),
        "ai_insights": len(enhanced_runtime.ai_insights),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/devices/{device_id}/metrics")
async def get_device_metrics(device_id: str):
    """Get metrics for specific device"""
    for device in enhanced_runtime.gpu_manager.devices.values():
        if device.id == device_id:
            return {
                "device": {
                    "id": device.id,
                    "name": device.name,
                    "type": device.type,
                    "memory_total": device.memory_total,
                    "memory_available": device.memory_available,
                    "compute_units": device.compute_units,
                    "is_available": device.is_available,
                },
                "timestamp": datetime.now().isoformat(),
            }
    raise HTTPException(status_code=404, detail="Device not found")


@app.post("/tasks", response_model=dict)
async def create_task(task_request: dict):
    """Create GPU task (original OpenRuntime compatibility)"""
    from openruntime import TaskRequest

    task = TaskRequest(**task_request)
    result = await enhanced_runtime.gpu_manager.execute_task(task)
    return {
        "task_id": result.task_id,
        "status": result.status,
        "result": result.result,
        "metrics": result.metrics,
        "error": result.error,
        "execution_time": result.execution_time,
    }


@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return {"tasks": [], "message": "Task history not implemented in enhanced version"}


@app.post("/benchmark")
async def run_benchmark(benchmark_type: str = "compute"):
    """Run performance benchmark"""
    benchmark_task = {"operation": "benchmark", "data": {"type": benchmark_type}, "priority": 1}

    from openruntime import TaskRequest

    task = TaskRequest(**benchmark_task)
    result = await enhanced_runtime.gpu_manager.execute_task(task)
    return {
        "task_id": result.task_id,
        "status": result.status,
        "result": result.result,
        "metrics": result.metrics,
        "error": result.error,
        "execution_time": result.execution_time,
    }


# =============================================================================
# WebSocket Support
# =============================================================================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send periodic updates
            update = {
                "type": "system_update",
                "timestamp": datetime.now().isoformat(),
                "ai_agents": len(enhanced_runtime.ai_manager.agents),
                "devices": len(enhanced_runtime.gpu_manager.devices),
                "ai_insights": len(enhanced_runtime.ai_insights),
            }
            await websocket.send_json(update)
            await asyncio.sleep(5)  # Send updates every 5 seconds
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenRuntime Enhanced with AI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])

    args = parser.parse_args()
    print("OpenRuntime Enhanced: AI-Powered GPU Runtime System")
    print("=" * 60)
    print(f"AI Agents: {len(enhanced_runtime.ai_manager.agents)}")
    print(f"GPU Devices: {len(enhanced_runtime.gpu_manager.devices)}")
    print(f"OpenAI: {'Available' if OPENAI_AVAILABLE else 'Not Available'}")
    print(f"LangChain: {'Available' if LANGCHAIN_AVAILABLE else 'Not Available'}")
    print(f"Shell-GPT: {'Available' if SHELLGPT_AVAILABLE else 'Not Available'}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    uvicorn.run(
        "openruntime_enhanced:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
