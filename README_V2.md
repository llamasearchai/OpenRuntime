# OpenRuntime v2 - Advanced GPU Runtime System

## Overview

OpenRuntime v2 is a production-ready, multi-backend runtime system that provides unified access to various LLM providers, embedding models, and GPU acceleration frameworks. It seamlessly integrates OpenAI's Agents SDK, Simon Willison's LLM ecosystem, and native Apple Silicon optimization.

## Key Features

### Multi-Backend Architecture

- **MLX**: Native Apple Silicon GPU acceleration
- **PyTorch**: Cross-platform GPU/CPU support with MPS for Apple Silicon
- **ONNX Runtime**: High-performance inference with CoreML support
- **OpenAI**: Complete OpenAI API and Agents SDK integration
- **Ollama**: Local LLM inference with multiple model support
- **LLM CLI**: Integration with llm, llm-mlx, llm-ollama, llm-embed-onnx
- **CPU Fallback**: Always-available fallback for basic operations

### Advanced Capabilities

- **Agent Orchestration**: Multi-agent workflows with OpenAI Agents SDK
- **Tool Use**: Extensible tool system for agent capabilities
- **Embeddings**: Multiple embedding models via ONNX and OpenAI
- **WebSocket Streaming**: Real-time task updates and streaming responses
- **Command Execution**: LLM-powered command generation and execution
- **Workflow Management**: Complex multi-step task orchestration

### Production Features

- **Async Architecture**: High-performance async/await throughout
- **Task Queue**: Efficient task scheduling and management
- **Prometheus Metrics**: Built-in monitoring and observability
- **Session Management**: Stateful agent conversations
- **Auto Backend Selection**: Intelligent backend selection based on task type
- **Error Recovery**: Graceful error handling and fallback mechanisms

## Installation

### Prerequisites

- Python 3.9+
- macOS (Apple Silicon recommended) or Linux
- Optional: CUDA-capable GPU for PyTorch acceleration

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/nemesis-collective/OpenRuntime.git
cd OpenRuntime

# Run setup script
./setup_runtime.sh

# Set environment variables (optional)
export OPENAI_API_KEY="your-api-key"
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e ".[dev]"

# Install LLM CLI and plugins
pip install llm
llm install llm-ollama
llm install llm-embed-onnx

# For Apple Silicon
pip install mlx
llm install llm-mlx
```

## Usage

### Starting the Server

```bash
# Development mode with auto-reload
python -m openruntime.main_v2 --reload

# Production mode
python -m openruntime.main_v2 --workers 4

# Custom configuration
python -m openruntime.main_v2 --host 0.0.0.0 --port 8080
```

### Using the CLI

```bash
# Check system health
openruntime-cli health

# List available backends
openruntime-cli backends

# List available models
openruntime-cli models

# Generate completion
openruntime-cli complete "Explain quantum computing" --model gpt-4o-mini --backend openai

# Generate embeddings
openruntime-cli embed "Sample text" --backend onnx

# Run agent
openruntime-cli agent "Analyze this codebase and suggest improvements" --type developer

# Run workflow
openruntime-cli workflow analysis --data '{"target": "system"}'
```

### API Examples

#### Completions

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post("http://localhost:8000/v2/completions", json={
        "prompt": "Write a Python function to calculate fibonacci",
        "model": "gpt-4-turbo-preview",
        "backend": "openai"
    })
    print(response.json())
```

#### Embeddings

```python
response = await client.post("http://localhost:8000/v2/embeddings", json={
    "texts": ["Hello world", "OpenRuntime v2"],
    "model": "text-embedding-3-small",
    "backend": "onnx"
})
```

#### Agent Execution

```python
response = await client.post("http://localhost:8000/v2/agents", json={
    "agent_type": "developer",
    "task": "Review this code for security issues",
    "context": {"code": "..."},
    "backend": "openai"
})
```

#### WebSocket Streaming

```python
import websockets
import json

async with websockets.connect("ws://localhost:8000/v2/ws") as ws:
    # Submit task via WebSocket
    await ws.send(json.dumps({
        "type": "submit_task",
        "task_type": "completion",
        "payload": {"prompt": "Hello"},
        "backend": "ollama"
    }))

    # Receive updates
    async for message in ws:
        data = json.loads(message)
        print(f"Update: {data}")
```

## Architecture

### System Components

```
OpenRuntime v2
├── Runtime Engine (Core)
│   ├── Task Queue
│   ├── Backend Manager
│   ├── WebSocket Handler
│   └── Metrics Collector
├── Backends
│   ├── MLX Backend (Apple Silicon)
│   ├── PyTorch Backend (GPU/CPU)
│   ├── ONNX Backend (Embeddings)
│   ├── OpenAI Backend (Agents SDK)
│   ├── Ollama Backend (Local LLMs)
│   ├── LLM CLI Backend (Tool Integration)
│   └── CPU Backend (Fallback)
└── API Layer
    ├── REST Endpoints
    ├── WebSocket Server
    └── Prometheus Metrics
```

### Task Flow

1. **Request Reception**: API receives task request
2. **Backend Selection**: Automatic or manual backend selection
3. **Task Queuing**: Task added to async queue
4. **Execution**: Backend processes task
5. **Result Broadcasting**: WebSocket updates and response
6. **Metrics Collection**: Performance tracking

## Configuration

### Runtime Configuration

```python
from openruntime.runtime_engine import RuntimeConfig, RuntimeBackend

config = RuntimeConfig(
    backend=RuntimeBackend.MLX,  # Default backend
    device="auto",                # Device selection
    max_memory=8192,             # Memory limit (MB)
    batch_size=4,                # Batch processing size
    num_threads=8,               # Thread count
    enable_monitoring=True,      # Prometheus metrics
    enable_caching=True,         # Response caching
    websocket_enabled=True,      # WebSocket support
)
```

### Environment Variables

```bash
# OpenAI Configuration
export OPENAI_API_KEY="sk-..."

# Ollama Configuration
export OLLAMA_HOST="http://localhost:11434"

# Runtime Configuration
export OPENRUNTIME_CACHE_DIR="~/.openruntime/cache"
export OPENRUNTIME_MODELS_DIR="~/.openruntime/models"
```

## Backend Details

### MLX (Apple Silicon)

- Optimized for M1/M2/M3 chips
- Metal Performance Shaders acceleration
- Unified memory architecture benefits
- Models: Mistral, Llama 2, Phi-2, Gemma

### PyTorch

- CUDA support for NVIDIA GPUs
- MPS support for Apple Silicon
- CPU fallback
- Wide model compatibility

### ONNX Runtime

- High-performance inference
- CoreML provider for macOS
- Embedding models from HuggingFace
- Quantization support

### OpenAI

- Complete API integration
- Agents SDK with multi-agent support
- Tool use capabilities
- GPT-4, GPT-3.5, embeddings

### Ollama

- Local model execution
- No internet required
- Models: Llama 2, Mistral, CodeLlama
- Multi-modal support

### LLM CLI

- Simon Willison's LLM tool integration
- Plugin ecosystem
- Command generation
- Cross-model compatibility

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_runtime_engine.py -v

# Run with coverage
python -m pytest tests/ --cov=openruntime --cov-report=term-missing
```

## Monitoring

### Prometheus Metrics

- Request counts and latency
- Task execution metrics
- Backend performance
- WebSocket connections

Access metrics at: `http://localhost:8000/v2/metrics`

### Health Check

```bash
curl http://localhost:8000/v2/health
```

## Performance

### Benchmarks (Apple M2 Max)

- MLX Inference: ~100ms for 1K tokens
- ONNX Embeddings: ~5ms per text
- OpenAI Completion: ~2s (network dependent)
- Ollama Local: ~500ms for small models

### Optimization Tips

- Use MLX on Apple Silicon for best performance
- Enable caching for repeated queries
- Batch embedding requests
- Use WebSocket for streaming responses

## Troubleshooting

### Common Issues

1. **MLX not available**

   - Ensure you're on Apple Silicon Mac
   - Install with: `pip install mlx`

2. **Ollama connection failed**

   - Start Ollama: `ollama serve`
   - Check port 11434 is accessible

3. **ONNX models missing**

   - Models download on first use
   - Check ~/.openruntime/models/onnx/

4. **OpenAI API errors**
   - Verify OPENAI_API_KEY is set
   - Check API quota and limits

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenAI for the Agents SDK and API
- Simon Willison for the LLM ecosystem
- Apple for MLX framework
- The open-source community

## Support

- Issues: [GitHub Issues](https://github.com/nemesis-collective/OpenRuntime/issues)
- Documentation: [API Docs](http://localhost:8000/docs)
- Community: [Discord Server](https://discord.gg/openruntime)
