# Getting Started with OpenRuntime

This guide will help you get OpenRuntime up and running on your system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Starting the Service](#starting-the-service)
5. [Verifying Installation](#verifying-installation)
6. [First Steps](#first-steps)
7. [Next Steps](#next-steps)

## Prerequisites

### System Requirements

- **Operating System**: 
  - macOS 12.0+ (Monterey or later)
  - Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+)
- **Hardware**:
  - Apple Silicon Mac (M1/M2/M3) or Intel Mac with discrete GPU
  - Linux with NVIDIA GPU (CUDA 11.0+)
- **Software**:
  - Python 3.9 or higher
  - pip package manager
  - Git

### Checking Prerequisites

```bash
# Check Python version
python --version

# Check pip
pip --version

# Check Git
git --version

# macOS only - check for Metal support
system_profiler SPDisplaysDataType | grep Metal

# Linux only - check for NVIDIA GPU
nvidia-smi
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/openruntime.git
cd openruntime
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 4. Install OpenRuntime

```bash
# Install in development mode
pip install -e .

# Or install normally
pip install .
```

## Configuration

### Default Configuration

OpenRuntime works out of the box with sensible defaults. The default configuration file is located at `config/openruntime.yml`.

### Custom Configuration

Create a custom configuration file:

```yaml
# config/custom.yml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4

runtime:
  max_concurrent_tasks: 10
  task_timeout: 300
  enable_monitoring: true

ai:
  enable_openai: true
  openai_api_key: ${OPENAI_API_KEY}
  max_tokens: 2000

logging:
  level: INFO
  file: logs/openruntime.log
```

### Environment Variables

Set up environment variables:

```bash
# Create .env file
cp .env.example .env

# Edit .env with your settings
export OPENAI_API_KEY="your-api-key"
export OPENRUNTIME_CONFIG="config/custom.yml"
export OPENRUNTIME_LOG_LEVEL="INFO"
```

## Starting the Service

### Using the Python Script

```bash
# Start standard service
python openruntime_enhanced.py

# Specify custom host/port
python openruntime_enhanced.py --host 0.0.0.0 --port 8080

# Enable debug mode
python openruntime_enhanced.py --debug
```

### Using the CLI

```bash
# Start with default settings
openruntime server start

# Start with custom configuration
openruntime server start --config config/custom.yml

# Start in background
openruntime server start --daemon
```

### Using Docker

```bash
# Build Docker image
docker build -t openruntime .

# Run container
docker run -p 8000:8000 openruntime

# Run with GPU support (NVIDIA)
docker run --gpus all -p 8000:8000 openruntime
```

## Verifying Installation

### 1. Check Service Status

```bash
# Using CLI
openruntime status

# Using curl
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "service": "OpenRuntime",
  "version": "2.0.0",
  "devices": 1,
  "metrics": {
    "gpu_utilization": 0,
    "memory_usage": 0,
    "active_tasks": 0
  }
}
```

### 2. List Available Devices

```bash
# Using CLI
openruntime devices list

# Using API
curl http://localhost:8000/devices
```

### 3. Run a Test Task

```bash
# Simple compute test
openruntime execute matrix_multiply --size 100

# AI test
openruntime ai code_generation "Write a hello world function"
```

## First Steps

### 1. Run a Benchmark

Get a baseline performance measurement:

```bash
openruntime benchmark
```

### 2. Execute Your First Task

Python example:

```python
import requests

# Submit a task
response = requests.post('http://localhost:8000/tasks', json={
    'operation': 'matrix_multiply',
    'parameters': {'size': 1000}
})

result = response.json()
print(f"Task {result['task_id']} completed in {result['execution_time']}s")
```

### 3. Monitor Real-time Metrics

Connect to WebSocket for real-time updates:

```python
import asyncio
import websockets
import json

async def monitor():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Metrics: {data}")

asyncio.run(monitor())
```

## Troubleshooting Common Issues

### Service Won't Start

1. Check port availability:
   ```bash
   lsof -i :8000
   ```

2. Check logs:
   ```bash
   tail -f logs/openruntime.log
   ```

3. Verify Python path:
   ```bash
   which python
   python -c "import openruntime; print(openruntime.__version__)"
   ```

### GPU Not Detected

1. macOS: Ensure you have a supported GPU
   ```bash
   system_profiler SPDisplaysDataType
   ```

2. Linux: Check NVIDIA drivers
   ```bash
   nvidia-smi
   modinfo nvidia
   ```

### API Connection Refused

1. Check service is running:
   ```bash
   ps aux | grep openruntime
   ```

2. Check firewall settings:
   ```bash
   # macOS
   sudo pfctl -s rules

   # Linux
   sudo iptables -L
   ```

## Next Steps

Now that you have OpenRuntime running:

1. **Read the [CLI Guide](cli-guide.md)** to learn about all available commands
2. **Explore the [API Reference](api-reference.md)** for programmatic access
3. **Follow the [Tutorials](tutorials/index.md)** for practical examples
4. **Review the [Architecture](architecture.md)** to understand the system design

## Getting Help

- Check the [FAQ](faq.md)
- Search [existing issues](https://github.com/your-org/openruntime/issues)
- Join our [community forum](https://github.com/your-org/openruntime/discussions)
- Contact support: support@openruntime.io