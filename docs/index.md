# OpenRuntime Documentation

Welcome to OpenRuntime - a high-performance GPU runtime system with integrated AI capabilities for macOS and beyond.

## Overview

OpenRuntime is a comprehensive runtime system designed to maximize GPU utilization and provide seamless AI integration for compute-intensive applications. Built with modern Python and leveraging Metal Performance Shaders on macOS, OpenRuntime offers both a powerful API and command-line interface for managing GPU workloads.

## Key Features

- **High-Performance GPU Computing**: Optimized for Apple Silicon and NVIDIA GPUs
- **AI Integration**: Built-in AI agents for code generation, system analysis, and optimization
- **Real-time Monitoring**: WebSocket-based real-time metrics and task updates
- **Flexible API**: RESTful API with comprehensive endpoints
- **CLI Tools**: Rich command-line interface for all operations
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## Quick Links

- [Getting Started](getting-started.md) - Installation and basic usage
- [API Reference](api-reference.md) - Complete API documentation
- [CLI Guide](cli-guide.md) - Command-line interface documentation
- [Architecture](architecture.md) - System design and components
- [Tutorials](tutorials/index.md) - Step-by-step guides
- [Development](development.md) - Contributing and development setup

## System Requirements

- **macOS**: 12.0+ (Monterey or later) with Apple Silicon or Intel
- **Linux**: Ubuntu 20.04+ with NVIDIA GPU (CUDA 11.0+)
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space

## Installation

```bash
# Clone the repository
git clone https://github.com/llamasearchai/OpenRuntime.git
cd OpenRuntime

# Install dependencies
pip install -r requirements.txt

# Install OpenRuntime
pip install -e .
```

## Quick Start

### Starting the Service

```bash
# Start the standard OpenRuntime service
python openruntime_enhanced.py

# Or use the CLI
openruntime server start
```

### Basic Usage

```python
import asyncio
from openruntime import GPURuntimeManager

async def main():
    manager = GPURuntimeManager()
    
    # Execute a compute task
    result = await manager.execute_task({
        "operation": "matrix_multiply",
        "parameters": {"size": 1000}
    })
    
    print(f"Task completed in {result.execution_time}s")

asyncio.run(main())
```

### CLI Examples

```bash
# Check service status
openruntime status

# Run a benchmark
openruntime benchmark

# Execute an AI task
openruntime ai optimization "Optimize my neural network training loop"
```

## Documentation Structure

This documentation is organized into the following sections:

### For Users
- **Getting Started**: Installation, configuration, and first steps
- **CLI Guide**: Complete command-line interface reference
- **Tutorials**: Practical examples and use cases

### For Developers  
- **API Reference**: Detailed API endpoint documentation
- **Architecture**: System design, components, and data flow
- **Development**: Contributing guidelines and development setup

### For Operations
- **Deployment**: Production deployment guides
- **Monitoring**: Metrics, logging, and alerting setup
- **Troubleshooting**: Common issues and solutions

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/llamasearchai/OpenRuntime/issues)
- **Discussions**: [Community forum](https://github.com/llamasearchai/OpenRuntime/discussions)
- **Email**: support@openruntime.io

## License

OpenRuntime is licensed under the MIT License. See [LICENSE](https://github.com/llamasearchai/OpenRuntime/blob/main/LICENSE) for details.

---

*Last updated: 2024*