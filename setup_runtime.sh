#!/bin/bash

# OpenRuntime v2 Setup Script

echo "==================================="
echo "OpenRuntime v2 Setup"
echo "==================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install base requirements
echo "Installing base requirements..."
pip install -e ".[dev]"

# Install LLM CLI and plugins
echo "Installing LLM CLI and plugins..."
pip install llm

# Install LLM plugins
echo "Installing LLM plugins..."
llm install llm-ollama || true
llm install llm-embed-onnx || true

# Install MLX for Apple Silicon
if [[ $(uname -m) == 'arm64' ]]; then
    echo "Installing MLX for Apple Silicon..."
    pip install mlx || true
    llm install llm-mlx || true
fi

# Install OpenAI Agents SDK (when available)
echo "Checking for OpenAI Agents SDK..."
pip install openai-agents || echo "OpenAI Agents SDK not yet available, using openai package"

# Download ONNX models for embeddings
echo "Setting up ONNX models directory..."
mkdir -p ~/.openruntime/models/onnx

# Check Ollama
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "Ollama is installed"
    ollama list || echo "Ollama service not running. Start with: ollama serve"
else
    echo "Ollama not installed. Install from: https://ollama.ai"
fi

# Create directories
echo "Creating runtime directories..."
mkdir -p ~/.openruntime/cache
mkdir -p ~/.openruntime/models/pytorch
mkdir -p ~/.openruntime/models/mlx
mkdir -p logs

# Run tests
echo "Running tests..."
python -m pytest tests/test_runtime_engine.py -v || echo "Some tests failed - this is expected without all backends configured"

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "To start the server:"
echo "  python -m openruntime.main_v2"
echo ""
echo "To use the CLI:"
echo "  python openruntime_cli_v2.py --help"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Note: Set OPENAI_API_KEY environment variable for OpenAI features"