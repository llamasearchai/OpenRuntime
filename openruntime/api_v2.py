"""
OpenRuntime API v2 - Enhanced API with complete runtime integration
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from .runtime_engine import RuntimeBackend, RuntimeConfig, RuntimeEngine, TaskType

logger = logging.getLogger(__name__)

# Load environment variables from a local .env if present
load_dotenv()

# Metrics
request_counter = Counter("openruntime_requests_total", "Total requests", ["endpoint", "method"])
request_duration = Histogram(
    "openruntime_request_duration_seconds", "Request duration", ["endpoint"]
)
active_connections = Gauge("openruntime_websocket_connections", "Active WebSocket connections")
task_counter = Counter("openruntime_tasks_total", "Total tasks", ["type", "status"])


# Request models
class InferenceRequest(BaseModel):
    model: Optional[str] = None
    inputs: Any
    backend: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class CompletionRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    model: Optional[str] = "gpt-4-turbo-preview"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False
    backend: Optional[str] = None


class EmbeddingRequest(BaseModel):
    texts: Union[str, List[str]]
    model: Optional[str] = "text-embedding-3-small"
    backend: Optional[str] = None
    normalize: Optional[bool] = True


class AgentRequest(BaseModel):
    agent_type: Optional[str] = "developer"
    task: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    backend: Optional[str] = "openai"


class WorkflowRequest(BaseModel):
    workflow_type: str
    data: Dict[str, Any]
    backend: Optional[str] = "openai"


class CommandRequest(BaseModel):
    command_type: str
    prompt: Optional[str] = None
    command: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    backend: Optional[str] = "llm_cli"


# Global runtime engine
runtime_engine: Optional[RuntimeEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global runtime_engine

    # Startup
    logger.info("Starting OpenRuntime API v2...")

    # Initialize runtime engine
    config = RuntimeConfig(
        backend=RuntimeBackend.MLX if os.uname().machine == "arm64" else RuntimeBackend.CPU,
        enable_monitoring=True,
        enable_caching=True,
        websocket_enabled=True,
    )

    runtime_engine = RuntimeEngine(config)
    await runtime_engine.initialize()

    logger.info("OpenRuntime API v2 started successfully")

    yield

    # Shutdown
    logger.info("Shutting down OpenRuntime API v2...")

    if runtime_engine:
        await runtime_engine.shutdown()

    logger.info("OpenRuntime API v2 shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="OpenRuntime API v2",
    description="Advanced GPU Runtime System with Complete LLM Integration",
    version="2.0.0",
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


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "OpenRuntime API v2",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Multi-backend support (MLX, PyTorch, ONNX, Ollama, OpenAI, LLM CLI)",
            "OpenAI Agents SDK integration",
            "Real-time WebSocket streaming",
            "Embedding generation",
            "Command execution",
            "Workflow orchestration",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not runtime_engine or not runtime_engine._initialized:
        raise HTTPException(status_code=503, detail="Runtime not initialized")

    status = await runtime_engine.get_status()

    return {"status": "healthy", "runtime": status}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(generate_latest(), media_type="text/plain")


@app.post("/v2/inference")
async def inference(request: InferenceRequest):
    """Run inference"""
    request_counter.labels(endpoint="/v2/inference", method="POST").inc()

    backend = RuntimeBackend(request.backend) if request.backend else None

    task_id = await runtime_engine.submit_task(
        TaskType.INFERENCE,
        {"model": request.model, "inputs": request.inputs, "options": request.options or {}},
        backend,
    )

    # Wait for result (simplified - in production use proper async handling)
    await asyncio.sleep(0.1)

    return {"task_id": task_id, "status": "submitted"}


@app.post("/v2/completions")
async def completions(request: CompletionRequest):
    """Generate completions"""
    request_counter.labels(endpoint="/v2/completions", method="POST").inc()

    backend = RuntimeBackend(request.backend) if request.backend else RuntimeBackend.OPENAI

    payload = {
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }

    if request.messages:
        payload["messages"] = request.messages
    elif request.prompt:
        payload["prompt"] = request.prompt
    else:
        raise HTTPException(status_code=400, detail="Either prompt or messages required")

    if request.stream:

        async def stream_response():
            task_id = await runtime_engine.submit_task(TaskType.COMPLETION, payload, backend)

            # Stream results (simplified)
            for i in range(10):
                await asyncio.sleep(0.1)
                yield f"data: {json.dumps({'content': f'chunk_{i}'})}\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        task_id = await runtime_engine.submit_task(TaskType.COMPLETION, payload, backend)

        return {"task_id": task_id, "status": "submitted"}


@app.post("/v2/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Generate embeddings"""
    request_counter.labels(endpoint="/v2/embeddings", method="POST").inc()

    backend = RuntimeBackend(request.backend) if request.backend else RuntimeBackend.ONNX

    task_id = await runtime_engine.submit_task(
        TaskType.EMBEDDING,
        {"texts": request.texts, "model": request.model, "normalize": request.normalize},
        backend,
    )

    return {"task_id": task_id, "status": "submitted"}


@app.post("/v2/agents")
async def run_agent(request: AgentRequest):
    """Run agent workflow"""
    request_counter.labels(endpoint="/v2/agents", method="POST").inc()

    backend = RuntimeBackend(request.backend) if request.backend else RuntimeBackend.OPENAI

    task_id = await runtime_engine.submit_task(
        TaskType.AGENT,
        {
            "agent_type": request.agent_type,
            "task": request.task,
            "context": request.context,
            "session_id": request.session_id,
        },
        backend,
    )

    return {"task_id": task_id, "status": "submitted"}


@app.post("/v2/workflows")
async def run_workflow(request: WorkflowRequest):
    """Run multi-agent workflow"""
    request_counter.labels(endpoint="/v2/workflows", method="POST").inc()

    backend = RuntimeBackend(request.backend) if request.backend else RuntimeBackend.OPENAI

    task_id = await runtime_engine.submit_task(
        TaskType.WORKFLOW, {"workflow_type": request.workflow_type, "data": request.data}, backend
    )

    return {"task_id": task_id, "status": "submitted"}


@app.post("/v2/commands")
async def run_command(request: CommandRequest):
    """Execute commands"""
    request_counter.labels(endpoint="/v2/commands", method="POST").inc()

    backend = RuntimeBackend(request.backend) if request.backend else RuntimeBackend.LLM_CLI

    payload = {"command_type": request.command_type}

    if request.prompt:
        payload["prompt"] = request.prompt
    if request.command:
        payload["command"] = request.command
    if request.args:
        payload["args"] = request.args

    task_id = await runtime_engine.submit_task(TaskType.COMMAND, payload, backend)

    return {"task_id": task_id, "status": "submitted"}


@app.get("/v2/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    if task_id in runtime_engine.active_tasks:
        return {"task_id": task_id, "status": "running"}
    else:
        return {"task_id": task_id, "status": "completed"}


@app.get("/v2/backends")
async def list_backends():
    """List available backends"""
    backends = []

    for backend_type, backend in runtime_engine.backends.items():
        metrics = await backend.get_metrics()
        backends.append(
            {"type": backend_type.value, "initialized": backend.initialized, "metrics": metrics}
        )

    return {"backends": backends}


@app.websocket("/v2/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.inc()

    try:
        # Register websocket with runtime
        await runtime_engine.register_websocket(websocket)

        # Send initial status
        status = await runtime_engine.get_status()
        await websocket.send_json({"type": "status", "data": status})

        # Keep connection alive and handle messages
        while True:
            try:
                # Receive message
                data = await websocket.receive_json()

                # Handle different message types
                if data.get("type") == "submit_task":
                    task_type = TaskType(data.get("task_type"))
                    payload = data.get("payload", {})
                    backend = data.get("backend")

                    task_id = await runtime_engine.submit_task(
                        task_type, payload, RuntimeBackend(backend) if backend else None
                    )

                    await websocket.send_json({"type": "task_submitted", "task_id": task_id})

                elif data.get("type") == "get_status":
                    status = await runtime_engine.get_status()
                    await websocket.send_json({"type": "status", "data": status})

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

    finally:
        # Unregister websocket
        await runtime_engine.unregister_websocket(websocket)
        active_connections.dec()


@app.get("/v2/models")
async def list_models():
    """List available models across all backends"""
    models = {}

    for backend_type, backend in runtime_engine.backends.items():
        backend_name = backend_type.value

        if hasattr(backend, "available_models"):
            models[backend_name] = backend.available_models
        elif hasattr(backend, "models"):
            models[backend_name] = list(backend.models.keys())
        else:
            models[backend_name] = []

    return {"models": models}


@app.post("/v2/tools")
async def use_tool(tool_name: str, args: Dict[str, Any]):
    """Use a specific tool"""
    request_counter.labels(endpoint="/v2/tools", method="POST").inc()

    task_id = await runtime_engine.submit_task(
        TaskType.TOOL_USE, {"tool": tool_name, "args": args}, RuntimeBackend.OPENAI
    )

    return {"task_id": task_id, "status": "submitted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
