# OpenRuntime CLI Guide

Comprehensive guide for the OpenRuntime command-line interface.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Commands Reference](#commands-reference)
4. [Configuration](#configuration)
5. [Examples](#examples)
6. [Advanced Usage](#advanced-usage)

## Installation

The OpenRuntime CLI is installed automatically when you install the main package:

```bash
pip install openruntime

# Verify installation
python -m openruntime_cli --version
```

### Available CLIs

OpenRuntime provides multiple CLI tools:

1. **openruntime** - Main CLI with full features
2. **openruntime-simple** - Simplified CLI for basic operations
3. **openruntime-benchmark** - Dedicated benchmarking tool

## Basic Usage

### Command Structure

```bash
openruntime [OPTIONS] COMMAND [ARGS]...
```

### Global Options

- `--url URL` - OpenRuntime service URL (default: http://localhost:8000)
- `--config FILE` - Configuration file path
- `--debug` - Enable debug output
- `--json` - Output in JSON format
- `--help` - Show help message

### Getting Help

```bash
# General help
openruntime --help

# Command-specific help
openruntime status --help
openruntime execute --help
```

## Commands Reference

### Server Management

#### `server start`

Start the OpenRuntime service.

```bash
python -m openruntime.main --host 0.0.0.0 --port 8000
```

Examples:
```bash
# Start with defaults
python -m openruntime.main --host 0.0.0.0 --port 8000

# Start on custom port
openruntime server start --port 8080

# Start in background
openruntime server start --daemon
```

#### `server stop`

Stop the OpenRuntime service.

```bash
openruntime server stop
```

#### `server restart`

Restart the OpenRuntime service.

```bash
openruntime server restart
```

### System Information

#### `status`

Get service status and metrics.

```bash
openruntime status [OPTIONS]

Options:
  --detailed  Show detailed metrics
  --watch     Continuously update (like top)
```

Output example:
```
OpenRuntime Status
═══════════════════

Service: Running
Version: 2.0.0
Uptime: 2h 15m

System Metrics
──────────────
GPU Utilization: 75.5%
Memory Usage: 45.2%
Active Tasks: 3
AI Tasks Processed: 12

Available Devices
─────────────────
ID       Type    Capabilities             Status
device_0 gpu     metal, mlx, unified_mem  available
```

#### `devices`

List available compute devices.

```bash
openruntime devices [OPTIONS]

Options:
  --format TEXT  Output format (table|json|csv) [default: table]
```

### Task Execution

#### `execute`

Execute a compute task.

```bash
openruntime execute TASK_TYPE [OPTIONS]

Arguments:
  TASK_TYPE  Type of task (compute|matrix_multiply|vector_operations|memory_bandwidth)

Options:
  --size INTEGER       Size parameter for the task
  --iterations INTEGER Number of iterations
  --device TEXT        Device preference (auto|gpu|cpu|device_id)
  --priority INTEGER   Task priority (1-10)
  --async              Don't wait for completion
```

Examples:
```bash
# Matrix multiplication
openruntime execute matrix_multiply --size 2000

# Vector operations with iterations
openruntime execute vector_operations --size 10000 --iterations 100

# Specific device
openruntime execute compute --device device_0
```

#### `list`

List recent tasks.

```bash
openruntime list [OPTIONS]

Options:
  --limit INTEGER    Number of tasks to show [default: 10]
  --status TEXT      Filter by status (pending|running|completed|failed)
  --format TEXT      Output format (table|json|csv)
```

### AI Operations

#### `ai`

Execute AI-powered tasks.

```bash
openruntime ai WORKFLOW PROMPT [OPTIONS]

Arguments:
  WORKFLOW  AI workflow type (code_generation|system_analysis|optimization|model_inference)
  PROMPT    Input prompt for the AI task

Options:
  --model TEXT       AI model to use
  --temperature FLOAT Temperature setting (0.0-1.0)
  --max-tokens INTEGER Maximum tokens to generate
  --save FILE        Save output to file
```

Examples:
```bash
# Code generation
openruntime ai code_generation "Create a Python web scraper"

# System analysis
openruntime ai system_analysis "Analyze GPU memory usage patterns"

# Save output
openruntime ai optimization "Optimize this matrix multiplication" --save output.py
```

#### `ai agents`

List available AI agents.

```bash
openruntime ai agents
```

#### `ai insights`

Generate AI insights for system metrics.

```bash
openruntime ai insights [OPTIONS]

Options:
  --metric TEXT      Metric type (performance|memory|efficiency|errors)
  --timeframe TEXT   Analysis timeframe (5m|1h|24h|7d)
```

### Benchmarking

#### `benchmark`

Run performance benchmarks.

```bash
openruntime benchmark [OPTIONS]

Options:
  --suite TEXT       Benchmark suite (standard|ai|stress|all)
  --output FILE      Save results to file
  --compare TEXT...  Compare backends
  --continuous       Run continuous benchmark
  --duration INTEGER Duration for continuous mode (seconds)
```

Examples:
```bash
# Standard benchmark
openruntime benchmark

# AI workload benchmark
openruntime benchmark --suite ai

# Compare backends
openruntime benchmark --compare metal cuda cpu

# Continuous monitoring
openruntime benchmark --continuous --duration 300
```

### Monitoring

#### `monitor`

Real-time system monitoring.

```bash
openruntime monitor [OPTIONS]

Options:
  --interval INTEGER  Update interval in seconds [default: 1]
  --metrics TEXT...   Specific metrics to monitor
```

#### `logs`

View service logs.

```bash
openruntime logs [OPTIONS]

Options:
  --follow         Follow log output
  --lines INTEGER  Number of lines to show
  --level TEXT     Filter by log level
  --grep TEXT      Search pattern
```

## Configuration

### Configuration File

Create `~/.openruntime/config.yml`:

```yaml
# Default service URL
service_url: http://localhost:8000

# CLI preferences
output:
  format: table
  color: true
  timestamps: true

# Default parameters
defaults:
  task_timeout: 300
  priority: 5
  device: auto

# AI settings
ai:
  default_model: gpt-4
  temperature: 0.7
  max_tokens: 2000
```

### Environment Variables

```bash
# Service URL
export OPENRUNTIME_URL="http://localhost:8000"

# API key for AI features
export OPENAI_API_KEY="your-key"

# Default output format
export OPENRUNTIME_OUTPUT="json"

# Enable debug mode
export OPENRUNTIME_DEBUG="1"
```

## Examples

### Complete Workflows

#### 1. Performance Testing Workflow

```bash
# Start service
python -m openruntime.main --host 0.0.0.0 --port 8000 &

# Check status
openruntime status

# Run benchmark
openruntime benchmark --suite standard --output results.json

# Analyze results with AI
openruntime ai system_analysis "Analyze the benchmark results" < results.json
```

#### 2. Development Workflow

```bash
# Generate code
python openruntime_cli.py ai code_generation "Create a Python web scraper"

# Optimize the code
openruntime ai optimization "Optimize for GPU performance" < train.py --save train_optimized.py

# Test performance
python openruntime_cli.py run --operation compute --size 10000
```

#### 3. Monitoring Workflow

```bash
# Start monitoring
openruntime monitor --metrics gpu_utilization memory_usage

# In another terminal, run workload
openruntime execute matrix_multiply --size 5000 --iterations 100

# Generate insights
openruntime ai insights --metric performance --timeframe 1h
```

### Shell Scripting

#### Batch Processing

```bash
#!/bin/bash
# batch_process.sh

# Process multiple tasks
for size in 100 500 1000 2000 5000; do
  echo "Processing size: $size"
  python openruntime_cli.py run --operation matrix_multiply --size $size --json | \
    jq -r '.execution_time'
done
```

#### Health Check Script

```bash
#!/bin/bash
# health_check.sh

while true; do
  if ! python openruntime_cli.py status --json | jq -e '.status == "healthy"' > /dev/null; then
    echo "Service unhealthy!"
    # Send alert
  fi
  sleep 60
done
```

## Advanced Usage

### Piping and Redirection

```bash
# Pipe AI output to another command
python openruntime_cli.py ai code_generation "SQL query optimizer" | python

# Save metrics to file
python openruntime_cli.py status --json > metrics.json

# Process log output
python openruntime_cli.py logs --follow | grep ERROR | tee errors.log
```

### JSON Processing with jq

```bash
# Extract execution time
python openruntime_cli.py run --operation matrix_multiply --size 1000 --json | jq '.execution_time'

# Get GPU utilization
python openruntime_cli.py status --json | jq '.metrics.gpu_utilization'

# List device IDs
python openruntime_cli.py devices --json | jq -r '.devices[].id'
```

### Automation with cron

```cron
# Run benchmark every hour
0 * * * * /usr/local/bin/python /usr/local/bin/openruntime_cli.py benchmark --output /var/log/openruntime/bench_$(date +\%Y\%m\%d_\%H).json

# Generate daily report
0 0 * * * /usr/local/bin/python /usr/local/bin/openruntime_cli.py ai insights --timeframe 24h > /var/reports/daily_$(date +\%Y\%m\%d).txt
```

### Custom Aliases

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Quick status
alias ors='python openruntime_cli.py status'

# Quick benchmark
alias orb='python openruntime_cli.py benchmark'

# AI code helper
orai() {
  python openruntime_cli.py ai code_generation "$*" | tee generated_$(date +%s).py
}

# Execute with timing
orexec() {
  time python openruntime_cli.py run "$@"
}
```

## Troubleshooting

### Common Issues

1. **Connection refused**
   ```bash
   # Check if service is running
   python openruntime_cli.py status
   
   # Check URL
   python openruntime_cli.py --url http://localhost:8001 status
   ```

2. **Command not found**
   ```bash
   # Verify installation
   pip show openruntime
   
   # Check PATH
   which openruntime_cli.py
   ```

3. **Slow performance**
   ```bash
   # Check system load
   python openruntime_cli.py status --detailed
   
   # Run diagnostics
   python openruntime_cli.py diagnose
   ```

### Debug Mode

Enable detailed debug output:

```bash
# Via flag
python openruntime_cli.py --debug run --operation matrix_multiply --size 1000

# Via environment
export OPENRUNTIME_DEBUG=1
python openruntime_cli.py status
```

## Tips and Best Practices

1. **Use JSON output for scripting**
   ```bash
   openruntime status --json | python -m json.tool
   ```

2. **Monitor long-running tasks**
   ```bash
   openruntime execute large_task --async
   openruntime monitor
   ```

3. **Save frequently used commands**
   ```bash
   # Create command file
   echo "execute matrix_multiply --size 5000" > my_task.cmd
   openruntime < my_task.cmd
   ```

4. **Use configuration files for complex setups**
   ```bash
   openruntime --config production.yml server start
   ```