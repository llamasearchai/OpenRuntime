import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .managers import GPURuntimeManager
from .models import (
    AITaskRequest,
    TaskRequest,
    TaskResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    app.state.runtime_manager = GPURuntimeManager()
    print("OpenRuntime starting up...")
    yield
    print("OpenRuntime shutting down...")
    app.state.runtime_manager.executor.shutdown(wait=True)


app = FastAPI(
    title="OpenRuntime",
    description="Advanced GPU Runtime System for macOS",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


websocket_manager = WebSocketManager()


@app.get("/")
async def root():
    return {"name": "OpenRuntime", "version": "2.0.0", "status": "running"}


@app.post("/tasks", response_model=TaskResponse)
async def create_task(task: TaskRequest, background_tasks: BackgroundTasks):
    result = await app.state.runtime_manager.execute_task(task)

    # Handle both Pydantic v1 and v2 models for broadcasting
    try:
        # Try Pydantic v2 method first
        result_data = result.model_dump()
    except AttributeError:
        try:
            # Fall back to Pydantic v1 method
            result_data = result.dict()
        except AttributeError:
            # Fall back to dataclass conversion
            result_data = (
                asdict(result) if hasattr(result, "__dataclass_fields__") else result.__dict__
            )

    await websocket_manager.broadcast({"type": "task_update", "data": result_data})
    return result


@app.get("/devices")
async def get_devices():
    return {"devices": [asdict(d) for d in app.state.runtime_manager.devices.values()]}


@app.post("/ai/tasks")
async def create_ai_task(task: AITaskRequest):
    return await app.state.runtime_manager.ai_manager.execute_ai_task(task)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if runtime manager is available
        if not hasattr(app.state, "runtime_manager") or not app.state.runtime_manager:
            return {"status": "unhealthy", "error": "Runtime manager not initialized"}

        # Check device availability
        devices = list(app.state.runtime_manager.devices.values())
        device_count = len(devices)

        # Get current metrics
        metrics = app.state.runtime_manager.get_metrics()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "OpenRuntime",
            "version": "2.0.0",
            "devices": device_count,
            "metrics": metrics,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/metrics")
async def get_metrics():
    """Get runtime metrics."""
    return app.state.runtime_manager.get_metrics()


@app.get("/ai/agents")
async def list_ai_agents():
    """List available AI agents and their capabilities."""
    agents = []
    role_map = {
        "code_generator": "DEVELOPER",
        "system_analyst": "ANALYST",
        "performance_optimizer": "OPTIMIZER",
        "shell_executor": "DEBUGGER",
    }
    for _, agent in app.state.runtime_manager.ai_manager.agents.items():
        # agent is a dataclass AIAgent
        role_label = role_map.get(agent.role.value, agent.role.name)
        agents.append(
            {
                "role": role_label,
                "name": agent.name,
                "description": agent.description or (agent.system_prompt[:120] if agent.system_prompt else ""),
                "capabilities": agent.capabilities,
            }
        )
    return {"agents": agents}


@app.post("/ai/insights")
async def generate_insights(request: dict):
    """Generate AI insights for system performance."""
    # Extract data from request
    metric_type = request.get("metric_type", "performance")
    timeframe = request.get("timeframe", "1h")

    # Get current metrics
    metrics = app.state.runtime_manager.get_metrics()

    # Generate insights using AI
    insights_prompt = f"""
    Analyze the following system metrics for {metric_type} over {timeframe}:
    
    GPU Utilization: {metrics.get('gpu_utilization', 0)}%
    Memory Usage: {metrics.get('memory_usage', 0)}%
    Active Tasks: {metrics.get('active_tasks', 0)}
    AI Tasks Processed: {metrics.get('ai_tasks_processed', 0)}
    
    Provide actionable insights and recommendations.
    """

    ai_request = AITaskRequest(workflow_type="system_analysis", prompt=insights_prompt)

    response = await app.state.runtime_manager.ai_manager.execute_ai_task(ai_request)

    return {
        "insights": response.get("result", "Unable to generate insights"),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/tasks/enhanced")
async def get_enhanced_tasks():
    """Get enhanced task information with detailed metrics."""
    # Get task history from runtime manager
    task_history = getattr(app.state.runtime_manager, "task_history", [])

    enhanced_tasks = []
    for task in task_history[-100:]:  # Last 100 tasks
        enhanced_task = {
            **task,
            "performance_metrics": {
                "execution_time_ms": task.get("execution_time", 0) * 1000,
                "memory_used_mb": task.get("memory_used", 0) / 1024 / 1024,
                "gpu_utilization": task.get("gpu_utilization", 0),
            },
        }
        enhanced_tasks.append(enhanced_task)

    return {
        "tasks": enhanced_tasks,
        "total_count": len(task_history),
        "timestamp": datetime.now().isoformat(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        # Send initial connection message
        await websocket.send_json(
            {
                "type": "connected",
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to OpenRuntime WebSocket",
            }
        )

        while True:
            # Send heartbeat
            await asyncio.sleep(5)

            # Include real metrics in heartbeat
            metrics = app.state.runtime_manager.get_metrics()
            await websocket.send_json(
                {"type": "heartbeat", "timestamp": datetime.now().isoformat(), "metrics": metrics}
            )
    except Exception:
        websocket_manager.disconnect(websocket)
