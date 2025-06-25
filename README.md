# OpenRuntime 

Advanced GPU Runtime System for macOS with AI Integration

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Metal](https://img.shields.io/badge/Metal-Performance-orange.svg)](https://developer.apple.com/metal/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-purple.svg)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

OpenRuntime  is a comprehensive GPU computing and ML inference platform designed specifically for macOS systems with Apple Silicon. It combines high-performance GPU acceleration with advanced AI capabilities through OpenAI integration, LangChain workflows, and shell-gpt automation.

### Key Features

- **Multi-GPU Runtime Management**: Advanced GPU device discovery and task scheduling
- **Metal Performance Shaders**: Native Apple Silicon GPU acceleration
- **AI Agent Integration**: OpenAI GPT-4o-mini agents for system optimization and analysis
- **LangChain Workflows**: Complex AI workflow automation and orchestration
- **Shell-GPT Integration**: AI-powered command line automation with safety checks
- **Real-time Monitoring**: Comprehensive metrics collection and WebSocket streaming
- **RESTful API**: Complete REST API with FastAPI and automatic documentation
- **High-Performance Rust Client**: Optional Rust client library for maximum performance
- **Docker Support**: Full containerization with monitoring stack
- **Production Ready**: Comprehensive logging, metrics, and deployment configurations

## Quick Start

### Prerequisites

- macOS 12.0+ (for Metal support)
- Python 3.11+
- OpenAI API key
- Optional: Rust 1.70+ (for high-performance client)
- Optional: Docker & Docker Compose

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/openruntime-enhanced.git
cd openruntime-enhanced
```

2. **Set up environment**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

3. **Quick setup**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

4. **Start the server**
```bash
python openruntime_enhanced.py
```

5. **Access the dashboard**
- API Documentation: http://localhost:8000/docs
- WebSocket Demo: ws://localhost:8000/ws
- Health Check: http://localhost:8000/health

### Docker Deployment

```bash
# Start full monitoring stack
docker-compose up -d

# Access services
# OpenRuntime: http://localhost:8000
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## Usage Examples

### Basic GPU Computation

```python
import requests

# Matrix multiplication on GPU
response = requests.post("http://localhost:8000/tasks", json={
    "operation": "compute",
    "data": {
        "type": "matrix_multiply",
        "size": 2048
    },
    "device_preference": "metal"
})

result = response.json()
print(f"GFLOPS: {result['result']['throughput']}")
```

### AI-Powered System Analysis

```python
# Get AI analysis of system performance
response = requests.post("http://localhost:8000/ai/tasks", json={
    "workflow_type": "system_analysis",
    "prompt": "Analyze current GPU performance and suggest optimizations",
    "context": {
        "system_metrics": {
            "gpu_utilization": 75,
            "memory_usage": 8.5,
            "temperature": 65
        }
    }
})

analysis = response.json()
print(analysis['result']['analysis'])
```

### ML Model Inference

```python
# Run ML inference with automatic GPU acceleration
response = requests.post("http://localhost:8000/tasks", json={
    "operation": "inference",
    "data": {
        "model": "resnet50",
        "batch_size": 32
    }
})

result = response.json()
print(f"Inference FPS: {result['result']['throughput_fps']}")
```

### Shell Automation with AI

```python
# Execute shell commands with AI assistance
response = requests.post("http://localhost:8000/shell/execute", json={
    "command": "Find all Python processes using more than 1GB memory",
    "use_ai": True,
    "safety_check": True
})

result = response.json()
print(f"Command: {result['command']}")
print(f"Output: {result['output']}")
```

### High-Performance Rust Client

```rust
use openai_runtime_rs::{OpenAIClient, ChatRequest, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIClient::new("your-api-key")?;
    
    let request = ChatRequest::new("gpt-4")
        .add_message(Message::user("Explain GPU computing"));
    
    let response = client.chat_completion(request).await?;
    println!("{}", response.content().unwrap_or("No response"));
    
    // Get client metrics
    let metrics = client.metrics();
    println!("Requests: {}", metrics.overall.total_requests);
    
    Ok(())
}
```

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  AI Agent Mgr   │    │ GPU Runtime Mgr │
│                 │◄──►│                 │◄──►│                 │
│ • REST API      │    │ • OpenAI Agents │    │ • Metal/CUDA    │
│ • WebSocket     │    │ • LangChain     │    │ • Device Mgmt   │
│ • Streaming     │    │ • Shell-GPT     │    │ • Task Sched    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Metrics Sys    │    │  WebSocket Mgr  │    │   Cache Layer   │
│                 │    │                 │    │                 │
│ • Prometheus    │    │ • Real-time     │    │ • LRU Cache     │
│ • Grafana       │    │ • Broadcasting  │    │ • TTL Support   │
│ • Custom        │    │ • Client Mgmt   │    │ • Invalidation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### GPU Pipeline

```
Input Data → Device Selection → Task Queuing → GPU Execution → Result Caching → Response
     │              │               │              │              │             │
     ▼              ▼               ▼              ▼              ▼             ▼
   JSON         Metal/CUDA      Priority       Kernel Exec    LRU Cache     FastAPI
  Validation     Detection       Queue         MPS/CUDA      TTL Expire    Response
```

## Configuration

### Main Configuration (`config/openruntime.yaml`)

```yaml
# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

# AI providers
ai_providers:
  openai:
    enabled: true
    api_key: "${OPENAI_API_KEY}"
    models:
      chat: "gpt-4"
      embeddings: "text-embedding-ada-002"

# GPU settings
gpu:
  enabled: true
  preferred_device: "metal"
  memory_limit: "80%"
  fallback_to_cpu: true

# Performance tuning
performance:
  max_concurrent_tasks: 10
  task_timeout: 300
  cache:
    enabled: true
    max_size: 1000
    ttl: 300
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export ANTHROPIC_API_KEY="..."
export LOG_LEVEL="info"
export GPU_MEMORY_LIMIT="80%"
export MAX_CONCURRENT_TASKS="10"
```

## Monitoring & Observability

### Metrics Available

- **Request Metrics**: Rate, latency, success rate by endpoint
- **GPU Metrics**: Utilization, memory usage, temperature, power
- **AI Metrics**: Token usage, model performance, agent execution times
- **System Metrics**: CPU, memory, disk, network usage

### Grafana Dashboards

- **System Overview**: High-level system health and performance
- **GPU Performance**: Detailed GPU utilization and benchmarks
- **AI Operations**: Agent performance and token usage analytics
- **Request Analytics**: API usage patterns and performance trends

### Log Analysis

```bash
# View real-time logs
tail -f logs/openruntime.log

# Search for errors
grep "ERROR" logs/openruntime.log

# Analyze performance patterns
grep "execution_time" logs/openruntime.log | jq '.execution_time'
```

## Testing & Benchmarking

### Run Comprehensive Tests

```bash
# Unit tests
python -m pytest tests/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Performance benchmarks
./scripts/benchmark.sh

# Stress testing
python scripts/stress_test.py --concurrent 20 --total 500
```

### GPU Benchmarks

```bash
# Metal performance test
curl -X POST "http://localhost:8000/benchmark?benchmark_type=compute"

# ML inference benchmark
curl -X POST "http://localhost:8000/benchmark?benchmark_type=ml"

# Comprehensive system benchmark
curl -X POST "http://localhost:8000/benchmark?benchmark_type=comprehensive"
```

## Production Deployment

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openruntime-enhanced
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openruntime-enhanced
  template:
    metadata:
      labels:
        app: openruntime-enhanced
    spec:
      containers:
      - name: openruntime
        image: openruntime-enhanced:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
```

### Load Balancer Configuration

```nginx
upstream openruntime_backend {
    least_conn;
    server openruntime-1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server openruntime-2:8000 weight=1 max_fails=3 fail_timeout=30s;
    server openruntime-3:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name api.openruntime.example.com;
    
    ssl_certificate /etc/ssl/certs/openruntime.crt;
    ssl_certificate_key /etc/ssl/private/openruntime.key;
    
    location / {
        proxy_pass http://openruntime_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

## Security

### API Security

- **Rate Limiting**: Built-in request rate limiting per client
- **Input Validation**: Comprehensive request validation with Pydantic
- **Safe Shell Execution**: AI-powered command safety analysis
- **Environment Isolation**: Containerized execution environments
- **Secrets Management**: Environment variable based secret handling

### Shell Command Safety

```python
# Commands are analyzed for safety before execution
safe_commands = [
    "ls -la",
    "ps aux | grep python",
    "nvidia-smi",
    "df -h"
]

dangerous_patterns = [
    "rm -rf",
    "dd if=",
    "sudo su",
    "wget.*|.*bash"
]
```

### Production Security Checklist

- [ ] Set strong JWT secrets
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up API key authentication
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Monitor for suspicious activity

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/your-org/openruntime-enhanced.git
cd openruntime-enhanced

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest

# Start development server
python openruntime_enhanced.py --reload
```

### Code Style

- **Python**: Black formatter, isort imports, flake8 linting
- **Rust**: Standard rustfmt formatting
- **Documentation**: Google-style docstrings
- **Type Hints**: Required for all Python functions

## API Documentation

### Core Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | System status and health |
| `/devices` | GET | List available GPU devices |
| `/tasks` | POST | Execute computational tasks |
| `/ai/tasks` | POST | Execute AI-powered tasks |
| `/benchmark` | POST | Run performance benchmarks |
| `/metrics` | GET | System metrics |
| `/ws` | WebSocket | Real-time updates |

### Response Formats

All responses follow this structure:

```json
{
    "status": "success|error",
    "data": {},
    "message": "Optional message",
    "timestamp": "2024-01-01T00:00:00Z",
    "execution_time": 0.123
}
```

## Troubleshooting

### Common Issues

**Metal Not Available**
```bash
# Check Metal support
python -c "import metal_performance_shaders; print('Metal available')"

# Fallback to CPU
export GPU_FALLBACK_TO_CPU=true
```

**OpenAI API Errors**
```bash
# Verify API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Check rate limits
grep "rate_limit" logs/openruntime.log
```

**High Memory Usage**
```bash
# Reduce cache size
export CACHE_MAX_SIZE=100

# Lower concurrent tasks
export MAX_CONCURRENT_TASKS=5
```

### Debug Mode

```bash
# Enable debug logging
python openruntime_enhanced.py --log-level debug

# Profile performance
python -m cProfile openruntime_enhanced.py
```

## Performance Tuning

### GPU Optimization

```yaml
# config/openruntime.yaml
gpu:
  memory_limit: "90%"  # Use more GPU memory
  optimization_level: "performance"  # vs "memory" or "balanced"
  batch_size: 64  # Larger batches for better throughput
```

### Concurrency Tuning

```yaml
performance:
  max_concurrent_tasks: 20  # Increase for more parallelism
  task_timeout: 600  # Longer timeout for complex tasks
  thread_pool_size: 16  # More worker threads
```

### Caching Strategy

```yaml
cache:
  enabled: true
  max_size: 5000  # Larger cache
  ttl: 1800  # Longer cache lifetime
  cleanup_interval: 300  # More frequent cleanup
```

## Advanced Features

### Custom AI Agents

```python
# Create custom AI agent
agent = AIAgent(
    id="custom_optimizer",
    name="Custom Performance Optimizer",
    role=AgentRole.PERFORMANCE_OPTIMIZER,
    provider=AIProviderType.OPENAI,
    model="gpt-4",
    system_prompt="""You are a specialized GPU performance optimizer.
    Focus on memory bandwidth optimization and compute unit utilization.""",
    capabilities=["memory_optimization", "bandwidth_analysis"]
)

agent_manager.register_agent(agent)
```

### Workflow Automation

```python
# Define complex workflow
workflow = WorkflowTask(
    id="auto_optimization",
    workflow_type=WorkflowType.COMPUTE_OPTIMIZATION,
    agent_id="perf_optimizer",
    input_data={
        "metrics": current_metrics,
        "workload_type": "ml_inference",
        "optimization_target": "throughput"
    },
    dependencies=["system_analysis", "baseline_benchmark"]
)

result = await workflow_manager.execute(workflow)
```

### Custom Compute Kernels

```python
# Compile custom Metal kernel
kernel_source = """
#include <metal_stdlib>
using namespace metal;

kernel void custom_matrix_multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Custom optimized matrix multiplication
    // ...
}
"""

kernel = ComputeKernel(
    name="optimized_matmul",
    source_code=kernel_source,
    entry_point="custom_matrix_multiply",
    parameters={"block_size": 16}
)

compiled = await runtime_manager.compile_kernel(kernel)
```

## Roadmap

###
- [ ] Support for additional GPU backends (Vulkan, OpenCL)
- [ ] Distributed computing across multiple machines
- [ ] Advanced AI model fine-tuning capabilities
- [ ] Real-time collaborative workflows
- [ ] Integration with Hugging Face Transformers
- [ ] Custom neural network architectures
- [ ] Advanced memory management and optimization
- [ ] Multi-tenant isolation and resource quotas
- [ ] Native support for other platforms (Linux, Windows)
- [ ] Edge computing deployment options
- [ ] Advanced security and compliance features
- [ ] Enterprise management console

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Apple** for Metal Performance Shaders framework
- **OpenAI** for GPT models and API
- **FastAPI** for the excellent web framework
- **LangChain** for AI workflow orchestration
- **Rust Community** for the high-performance client library

## Support
- **Email**: nikjois@llamasearch.ai

---

