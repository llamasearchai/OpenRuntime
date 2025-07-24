# OpenRuntime Architecture

This document describes the architecture and design of OpenRuntime.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [API Design](#api-design)
6. [Security Architecture](#security-architecture)
7. [Performance Considerations](#performance-considerations)
8. [Deployment Architecture](#deployment-architecture)

## Overview

OpenRuntime is built as a modular, scalable system for GPU-accelerated computing with integrated AI capabilities. The architecture follows these key principles:

- **Modularity**: Clear separation between compute runtime, AI integration, and API layers
- **Scalability**: Designed to handle multiple concurrent tasks and scale horizontally
- **Extensibility**: Plugin architecture for adding new compute backends and AI models
- **Performance**: Optimized for low latency and high throughput
- **Reliability**: Comprehensive error handling and recovery mechanisms

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client Layer                            │
├─────────────────┬────────────────┬──────────────────────────────┤
│   CLI Tools     │   REST API     │      WebSocket API           │
├─────────────────┴────────────────┴──────────────────────────────┤
│                        API Gateway                               │
│                    (FastAPI + CORS)                              │
├──────────────────────────────────────────────────────────────────┤
│                      Service Layer                               │
├────────────────┬─────────────────┬──────────────────────────────┤
│ Task Manager   │  Device Manager │    AI Agent Manager          │
├────────────────┴─────────────────┴──────────────────────────────┤
│                     Runtime Layer                                │
├────────────────┬─────────────────┬──────────────────────────────┤
│ GPU Runtime    │  MLX Runtime    │    CPU Runtime               │
├────────────────┴─────────────────┴──────────────────────────────┤
│                    Hardware Abstraction Layer                    │
├────────────────┬─────────────────┬──────────────────────────────┤
│     Metal      │      CUDA       │       OpenCL                 │
└────────────────┴─────────────────┴──────────────────────────────┘
```

## Core Components

### 1. API Gateway

The API Gateway is built with FastAPI and provides:

- RESTful API endpoints
- WebSocket support for real-time updates
- Request validation and serialization
- CORS middleware for cross-origin requests
- Automatic API documentation (OpenAPI/Swagger)

**Key Files:**
- `openruntime/core/api.py` - Main API definition
- `openruntime_enhanced.py` - Enhanced API with additional features

### 2. Task Manager

Manages task lifecycle from submission to completion:

- Task queuing and prioritization
- Resource allocation
- Task execution monitoring
- Result collection and caching

**Key Classes:**
- `TaskManager` - Core task management
- `TaskQueue` - Priority-based task queue
- `TaskExecutor` - Task execution engine

### 3. Device Manager

Handles compute device discovery and management:

- Device detection (GPU, CPU, TPU)
- Capability querying
- Resource monitoring
- Device allocation strategies

**Key Classes:**
- `DeviceManager` - Device lifecycle management
- `GPUDevice` - GPU-specific functionality
- `DeviceAllocator` - Resource allocation algorithms

### 4. Runtime Managers

Platform-specific runtime implementations:

#### GPU Runtime Manager
- Metal support for macOS
- CUDA support for NVIDIA GPUs
- OpenCL fallback
- Memory management
- Kernel compilation and caching

#### MLX Runtime Manager
- Apple MLX framework integration
- Optimized for Apple Silicon
- Unified memory architecture support

#### AI Agent Manager
- Multiple AI agent roles (Developer, Analyst, Optimizer, Debugger)
- OpenAI API integration
- Local model support (future)
- Context management
- Token optimization

### 5. Models and Data Structures

**Core Models:**

```python
# Device Information
@dataclass
class DeviceInfo:
    id: str
    name: str
    type: str
    capabilities: List[str]
    memory_total: int
    memory_available: int
    status: str

# Task Request/Response
class TaskRequest(BaseModel):
    operation: str
    parameters: Dict[str, Any]
    device_preference: Optional[str]
    priority: int = 5

class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time: float
    device_used: Optional[str]

# AI Models
class AIRequest(BaseModel):
    workflow: WorkflowType
    prompt: str
    parameters: Optional[Dict[str, Any]]
    max_tokens: int = 2000
    temperature: float = 0.7
```

## Data Flow

### 1. Task Submission Flow

```
Client Request → API Gateway → Validation → Task Manager
                                              ↓
                                         Task Queue
                                              ↓
                                      Device Allocator
                                              ↓
                                       Runtime Manager
                                              ↓
                                     Hardware Execution
                                              ↓
                                      Result Collection
                                              ↓
                                    Response → Client
```

### 2. Real-time Updates Flow

```
Task Update → Event Bus → WebSocket Manager → Connected Clients
     ↑                           ↓
Runtime Events            Broadcast Updates
```

### 3. AI Workflow

```
AI Request → AI Agent Manager → Agent Selection
                                      ↓
                              Context Building
                                      ↓
                               Model Invocation
                                      ↓
                              Response Processing
                                      ↓
                                Result Caching
```

## API Design

### RESTful Principles

- Resource-based URLs (`/devices`, `/tasks`, `/ai/agents`)
- HTTP methods for actions (GET, POST, PUT, DELETE)
- Stateless communication
- JSON request/response format
- Consistent error responses

### Endpoint Categories

1. **System Endpoints**
   - `/` - Service info
   - `/health` - Health check
   - `/metrics` - System metrics

2. **Resource Endpoints**
   - `/devices` - Device management
   - `/tasks` - Task operations
   - `/ai/*` - AI operations

3. **WebSocket Endpoints**
   - `/ws` - Real-time updates

### API Versioning

Future versions will support:
- URL versioning: `/v2/tasks`
- Header versioning: `Accept: application/vnd.openruntime.v2+json`

## Security Architecture

### Current Implementation

1. **Local Deployment Security**
   - No authentication required
   - Localhost-only binding by default
   - CORS enabled for development

2. **Data Security**
   - Input validation on all endpoints
   - SQL injection prevention
   - XSS protection in responses

### Production Security (Planned)

1. **Authentication**
   - JWT token-based auth
   - API key authentication
   - OAuth2 integration

2. **Authorization**
   - Role-based access control (RBAC)
   - Resource-level permissions
   - Rate limiting per user/role

3. **Network Security**
   - TLS/SSL encryption
   - Certificate pinning
   - IP whitelisting

## Performance Considerations

### 1. Concurrency Model

- Async/await for I/O operations
- Thread pool for CPU-bound tasks
- Process pool for parallel execution
- GPU stream management

### 2. Caching Strategy

- Result caching with TTL
- Compiled kernel caching
- AI response caching
- Device capability caching

### 3. Resource Management

- Memory pooling for GPU allocations
- Connection pooling for external services
- Task queue with priority scheduling
- Automatic resource cleanup

### 4. Optimization Techniques

- Lazy loading of heavy dependencies
- Just-in-time compilation for kernels
- Batch processing for similar tasks
- Pipeline parallelism

## Deployment Architecture

### 1. Single Node Deployment

```
┌─────────────────┐
│   Nginx/Proxy   │
└────────┬────────┘
         │
┌────────┴────────┐
│   OpenRuntime   │
│   Application   │
├─────────────────┤
│   GPU/Compute   │
│   Resources     │
└─────────────────┘
```

### 2. Distributed Deployment

```
┌──────────────┐     ┌──────────────┐
│ Load Balancer│────▶│ Load Balancer│
└──────┬───────┘     └──────┬───────┘
       │                     │
┌──────┴───────┐     ┌──────┴───────┐
│ OpenRuntime  │     │ OpenRuntime  │
│   Node 1     │     │   Node 2     │
└──────┬───────┘     └──────┬───────┘
       │                     │
┌──────┴───────┐     ┌──────┴───────┐
│   GPU Pool   │     │   GPU Pool   │
└──────────────┘     └──────────────┘
       │                     │
       └──────┬──────────────┘
              │
       ┌──────┴───────┐
       │Shared Storage│
       └──────────────┘
```

### 3. Container Architecture

```yaml
services:
  openruntime:
    image: openruntime:latest
    deploy:
      replicas: 3
    volumes:
      - gpu_drivers:/usr/local/cuda
    devices:
      - /dev/dri:/dev/dri  # GPU access
  
  redis:
    image: redis:alpine
    # Task queue and caching
  
  prometheus:
    image: prometheus:latest
    # Metrics collection
  
  grafana:
    image: grafana:latest
    # Monitoring dashboard
```

## Extension Points

### 1. Compute Backend Plugins

```python
class ComputeBackend(ABC):
    @abstractmethod
    async def execute(self, operation: str, params: Dict) -> Result:
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        pass
```

### 2. AI Model Adapters

```python
class AIModelAdapter(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        pass
```

### 3. Monitoring Integrations

```python
class MetricsExporter(ABC):
    @abstractmethod
    def export_metrics(self, metrics: Dict) -> None:
        pass
```

## External Integrations

- **OpenAI**: Integrates with OpenAI's API for AI-powered tasks.
  - *Reference*: [OpenAI API Documentation](https://platform.openai.com/docs/)
- **LangChain**: Utilized for building advanced AI workflows.
  - *Reference*: [LangChain Documentation](https://python.langchain.com/)
- **MLX**: Apple's machine learning framework for Metal.
  - *Reference*: [MLX GitHub Repository](https://github.com/ml-explore/mlx)
- **PyTorch**: Popular open-source machine learning framework.
  - *Reference*: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Repository and Documentation

- **GitHub Repository**: [llamasearchai/OpenRuntime](https://github.com/llamasearchai/OpenRuntime)
- **Documentation Source**: `docs/` directory within the repository
- **CI/CD for Docs**: Automated deployment via GitHub Actions (see `.github/workflows/docs.yml`)

## Troubleshooting

### Common Issues

- **GPU Not Detected**: Ensure Metal is properly installed and drivers are up to date.
  - *Solution*: Check system compatibility, reinstall Xcode Command Line Tools.
- **AI Agent Errors**: Verify OpenAI API key and network connectivity.
  - *Solution*: Double-check API key, ensure outbound access to OpenAI endpoints.
- **Performance Degradation**: Review monitoring dashboards for bottlenecks.
  - *Solution*: Adjust task concurrency, optimize batch sizes.

### Debugging Architecture Components

- **API Layer**: Use FastAPI's debug mode and Uvicorn's `--log-level debug`.
- **Runtime Managers**: Enable verbose logging for detailed execution traces.
- **AI Agents**: Monitor agent interactions and prompt/response flows.

## Future Enhancements

### Scalability Improvements

- **Distributed Task Queue**: Implement a distributed message queue for enhanced task distribution (e.g., Kafka, RabbitMQ).
- **Kubernetes Integration**: Refine Kubernetes deployment for auto-scaling and high availability.
- **Multi-Cloud Support**: Extend deployment options to AWS, GCP, Azure.

### AI Enhancements

- **Custom Agent Training**: Allow users to train custom AI agents based on their data.
- **Model Fine-tuning**: Support for fine-tuning pre-trained MLX and PyTorch models.
- **Advanced AI Workflows**: More complex, multi-step AI-powered automation.

### GPU Feature Expansion

- **Vulkan Backend**: Support for Vulkan API on compatible GPUs.
- **OpenCL Backend**: Broaden compatibility with OpenCL-enabled devices.
- **Ray Integration**: Distributed computing with Ray for larger workloads.

## Conclusion

The OpenRuntime architecture is designed for high performance, modularity, and extensibility, providing a robust foundation for GPU computing and AI integration on macOS.