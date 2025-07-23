#!/usr/bin/env python3
"""
OpenRuntime CLI: Command Line Interface for OpenRuntime GPU Computing Platform

Author: Nik Jois <nikjois@llamasearch.ai>
Version: 2.0.0

A comprehensive CLI for managing GPU computing tasks, monitoring devices,
running benchmarks, and interacting with the OpenRuntime API.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import httpx
import rich.console
import rich.table
import rich.progress
import rich.panel
import rich.text
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OpenRuntime CLI")

# Version information
__version__ = "2.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

# Rich console for beautiful output
console = Console()


class OpenRuntimeCLI:
    """Main CLI class for OpenRuntime operations"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.session_id = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def health_check(self) -> bool:
        """Check if OpenRuntime server is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            console.print(f"[red]Health check failed: {e}[/red]")
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            response = await self.client.get(f"{self.base_url}/")
            return response.json()
        except Exception as e:
            console.print(f"[red]Failed to get system info: {e}[/red]")
            return {}
    
    async def list_devices(self) -> List[Dict[str, Any]]:
        """List all available devices"""
        try:
            response = await self.client.get(f"{self.base_url}/devices")
            data = response.json()
            return data.get("devices", [])
        except Exception as e:
            console.print(f"[red]Failed to list devices: {e}[/red]")
            return []
    
    async def get_device_metrics(self, device_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get metrics for a specific device"""
        try:
            response = await self.client.get(f"{self.base_url}/devices/{device_id}/metrics?limit={limit}")
            data = response.json()
            return data.get("metrics", [])
        except Exception as e:
            console.print(f"[red]Failed to get device metrics: {e}[/red]")
            return []
    
    async def create_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and execute a new task"""
        try:
            response = await self.client.post(f"{self.base_url}/tasks", json=task_data)
            return response.json()
        except Exception as e:
            console.print(f"[red]Failed to create task: {e}[/red]")
            return {}
    
    async def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active tasks"""
        try:
            response = await self.client.get(f"{self.base_url}/tasks")
            data = response.json()
            return data.get("active_tasks", [])
        except Exception as e:
            console.print(f"[red]Failed to get active tasks: {e}[/red]")
            return []
    
    async def run_benchmark(self, benchmark_type: str = "comprehensive", device_id: Optional[str] = None) -> Dict[str, Any]:
        """Run benchmarks"""
        try:
            params = {"benchmark_type": benchmark_type}
            if device_id:
                params["device_id"] = device_id
            
            response = await self.client.post(f"{self.base_url}/benchmark", params=params)
            return response.json()
        except Exception as e:
            console.print(f"[red]Failed to run benchmark: {e}[/red]")
            return {}
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            response = await self.client.get(f"{self.base_url}/metrics/summary")
            return response.json()
        except Exception as e:
            console.print(f"[red]Failed to get metrics summary: {e}[/red]")
            return {}


def display_system_info(info: Dict[str, Any]):
    """Display system information in a beautiful format"""
    console.print(Panel.fit(
        f"[bold blue]OpenRuntime System Information[/bold blue]\n"
        f"Version: [green]{info.get('version', 'Unknown')}[/green]\n"
        f"Author: [cyan]{info.get('author', 'Unknown')}[/cyan]\n"
        f"Status: [green]{info.get('status', 'Unknown')}[/green]\n"
        f"Devices: [yellow]{info.get('devices', 0)}[/yellow]\n"
        f"Active Tasks: [yellow]{info.get('active_tasks', 0)}[/yellow]\n"
        f"MLX Available: [green]{'Yes' if info.get('mlx_available') else 'No'}[/green]\n"
        f"PyTorch Available: [green]{'Yes' if info.get('torch_available') else 'No'}[/green]",
        title="System Status"
    ))


def display_devices(devices: List[Dict[str, Any]]):
    """Display devices in a table format"""
    table = Table(title="Available Devices")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Memory", style="blue")
    table.add_column("Status", style="red")
    table.add_column("Capabilities", style="magenta")
    
    for device in devices:
        memory_gb = device.get("memory_total", 0) / (1024**3)
        status = "🟢 Available" if device.get("is_available") else "🔴 Busy"
        capabilities = ", ".join(device.get("capabilities", [])[:3])
        
        table.add_row(
            device.get("id", "Unknown"),
            device.get("name", "Unknown"),
            device.get("type", "Unknown"),
            f"{memory_gb:.1f} GB",
            status,
            capabilities
        )
    
    console.print(table)


def display_tasks(tasks: List[Dict[str, Any]]):
    """Display active tasks in a table format"""
    if not tasks:
        console.print("[yellow]No active tasks[/yellow]")
        return
    
    table = Table(title="Active Tasks")
    table.add_column("Task ID", style="cyan", no_wrap=True)
    table.add_column("Operation", style="green")
    table.add_column("Device", style="yellow")
    table.add_column("Status", style="blue")
    table.add_column("Running Time", style="magenta")
    
    for task in tasks:
        running_time = task.get("running_time", 0)
        table.add_row(
            task.get("task_id", "Unknown")[:8] + "...",
            task.get("operation", "Unknown"),
            task.get("device_id", "Unknown"),
            task.get("status", "Unknown"),
            f"{running_time:.1f}s"
        )
    
    console.print(table)


def display_benchmark_results(results: Dict[str, Any]):
    """Display benchmark results"""
    if not results:
        console.print("[red]No benchmark results available[/red]")
        return
    
    console.print(Panel.fit(
        f"[bold blue]Benchmark Results[/bold blue]\n"
        f"Task ID: [cyan]{results.get('task_id', 'Unknown')}[/cyan]\n"
        f"Status: [green]{results.get('status', 'Unknown')}[/green]\n"
        f"Execution Time: [yellow]{results.get('execution_time', 0):.2f}s[/yellow]",
        title="Benchmark Summary"
    ))
    
    if "result" in results and "results" in results["result"]:
        benchmark_results = results["result"]["results"]
        
        for category, category_results in benchmark_results.items():
            if isinstance(category_results, list):
                table = Table(title=f"{category.title()} Benchmarks")
                table.add_column("Test", style="cyan")
                table.add_column("Size", style="green")
                table.add_column("Performance", style="yellow")
                table.add_column("Device", style="blue")
                
                for result in category_results:
                    if isinstance(result, dict):
                        size = result.get("size", "N/A")
                        gflops = result.get("gflops", 0)
                        device = result.get("device", "Unknown")
                        
                        table.add_row(
                            category.title(),
                            str(size),
                            f"{gflops:.2f} GFLOPS" if gflops else "N/A",
                            device
                        )
                
                console.print(table)


@click.group()
@click.version_option(version=__version__, prog_name="OpenRuntime CLI")
@click.option("--url", default="http://localhost:8000", help="OpenRuntime server URL")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, url: str, verbose: bool):
    """OpenRuntime CLI - Advanced GPU Computing Platform
    
    A comprehensive command-line interface for managing GPU computing tasks,
    monitoring devices, running benchmarks, and interacting with the OpenRuntime API.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj["url"] = url


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and information"""
    async def run_status():
        async with OpenRuntimeCLI(ctx.obj["url"]) as cli_instance:
            # Check health
            with console.status("[bold green]Checking system health..."):
                is_healthy = await cli_instance.health_check()
            
            if not is_healthy:
                console.print("[red]OpenRuntime server is not responding[/red]")
                return
            
            # Get system info
            with console.status("[bold green]Fetching system information..."):
                info = await cli_instance.get_system_info()
            
            display_system_info(info)
            
            # Get devices
            with console.status("[bold green]Fetching device information..."):
                devices = await cli_instance.list_devices()
            
            display_devices(devices)
    
    asyncio.run(run_status())


@cli.command()
@click.pass_context
def devices(ctx):
    """List all available devices with detailed information"""
    async def run_devices():
        async with OpenRuntimeCLI(ctx.obj["url"]) as cli_instance:
            with console.status("[bold green]Fetching device information..."):
                devices = await cli_instance.list_devices()
            
            display_devices(devices)
            
            # Show detailed metrics for each device
            for device in devices:
                device_id = device.get("id")
                if device_id:
                    console.print(f"\n[bold blue]Metrics for {device.get('name', device_id)}:[/bold blue]")
                    
                    with console.status(f"[bold green]Fetching metrics for {device_id}..."):
                        metrics = await cli_instance.get_device_metrics(device_id, limit=5)
                    
                    if metrics:
                        table = Table(title=f"Recent Metrics - {device_id}")
                        table.add_column("Timestamp", style="cyan")
                        table.add_column("Memory Usage", style="green")
                        table.add_column("GPU Utilization", style="yellow")
                        table.add_column("Temperature", style="red")
                        table.add_column("Power Usage", style="magenta")
                        
                        for metric in metrics[-5:]:  # Last 5 metrics
                            table.add_row(
                                metric.get("timestamp", "Unknown")[:19],
                                f"{metric.get('memory_usage', 0):.1f}%",
                                f"{metric.get('gpu_utilization', 0):.1f}%",
                                f"{metric.get('temperature', 0):.1f}°C",
                                f"{metric.get('power_usage', 0):.1f}W"
                            )
                        
                        console.print(table)
    
    asyncio.run(run_devices())


@cli.command()
@click.pass_context
def tasks(ctx):
    """Show active tasks and their status"""
    async def run_tasks():
        async with OpenRuntimeCLI(ctx.obj["url"]) as cli_instance:
            with console.status("[bold green]Fetching active tasks..."):
                tasks = await cli_instance.get_active_tasks()
            
            display_tasks(tasks)
    
    asyncio.run(run_tasks())


@cli.command()
@click.option("--operation", "-o", default="compute", 
              type=click.Choice(["compute", "inference", "benchmark", "mlx_compute"]),
              help="Type of operation to perform")
@click.option("--size", "-s", default=1024, help="Matrix size for compute operations")
@click.option("--device", "-d", help="Preferred device ID")
@click.option("--model", "-m", default="resnet50", help="Model for inference")
@click.option("--batch-size", "-b", default=1, help="Batch size for inference")
@click.pass_context
def run(ctx, operation: str, size: int, device: str, model: str, batch_size: int):
    """Run a computational task"""
    async def run_task():
        async with OpenRuntimeCLI(ctx.obj["url"]) as cli_instance:
            # Prepare task data
            task_data = {
                "operation": operation,
                "data": {}
            }
            
            if device:
                task_data["device_preference"] = device
            
            if operation == "compute":
                task_data["data"] = {
                    "type": "matrix_multiply",
                    "size": size
                }
            elif operation == "inference":
                task_data["data"] = {
                    "model": model,
                    "batch_size": batch_size
                }
            elif operation == "mlx_compute":
                task_data["data"] = {
                    "type": "matrix_multiply",
                    "size": size,
                    "dtype": "float32"
                }
            
            console.print(f"[bold blue]Running {operation} task...[/bold blue]")
            console.print(f"Task data: {json.dumps(task_data, indent=2)}")
            
            with console.status("[bold green]Executing task..."):
                result = await cli_instance.create_task(task_data)
            
            if result.get("status") == "completed":
                console.print("[green]Task completed successfully![/green]")
                
                # Display results
                if "result" in result:
                    console.print(Panel.fit(
                        f"[bold blue]Task Results[/bold blue]\n"
                        f"Operation: [cyan]{result['result'].get('operation', 'Unknown')}[/cyan]\n"
                        f"Device Used: [green]{result.get('device_used', 'Unknown')}[/green]\n"
                        f"Execution Time: [yellow]{result.get('execution_time', 0):.2f}s[/yellow]",
                        title="Task Summary"
                    ))
                    
                    # Show detailed results
                    if "gflops" in result["result"]:
                        console.print(f"Performance: [bold green]{result['result']['gflops']:.2f} GFLOPS[/bold green]")
                    if "throughput_fps" in result["result"]:
                        console.print(f"Throughput: [bold green]{result['result']['throughput_fps']:.2f} FPS[/bold green]")
            else:
                console.print(f"[red]Task failed: {result.get('error', 'Unknown error')}[/red]")
    
    asyncio.run(run_task())


@cli.command()
@click.option("--type", "-t", default="comprehensive", 
              type=click.Choice(["comprehensive", "compute", "memory", "ml"]),
              help="Type of benchmark to run")
@click.option("--device", "-d", help="Device ID to benchmark")
@click.pass_context
def benchmark(ctx, type: str, device: str):
    """Run performance benchmarks"""
    async def run_benchmark():
        async with OpenRuntimeCLI(ctx.obj["url"]) as cli_instance:
            console.print(f"[bold blue]Running {type} benchmark...[/bold blue]")
            
            with console.status("[bold green]Executing benchmark..."):
                result = await cli_instance.run_benchmark(type, device)
            
            display_benchmark_results(result)
    
    asyncio.run(run_benchmark())


@cli.command()
@click.pass_context
def metrics(ctx):
    """Show system metrics summary"""
    async def run_metrics():
        async with OpenRuntimeCLI(ctx.obj["url"]) as cli_instance:
            with console.status("[bold green]Fetching metrics summary..."):
                summary = await cli_instance.get_metrics_summary()
            
            if "summary" in summary:
                console.print(Panel.fit(
                    "[bold blue]System Metrics Summary[/bold blue]",
                    title="Metrics"
                ))
                
                for device_id, metrics in summary["summary"].items():
                    table = Table(title=f"Device: {device_id}")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")
                    
                    table.add_row("Average Utilization", f"{metrics.get('avg_utilization', 0):.1f}%")
                    table.add_row("Average Memory Usage", f"{metrics.get('avg_memory_usage', 0):.1f}%")
                    table.add_row("Average Temperature", f"{metrics.get('avg_temperature', 0):.1f}°C")
                    table.add_row("Average Power Usage", f"{metrics.get('avg_power', 0):.1f}W")
                    table.add_row("Total Throughput", f"{metrics.get('total_throughput', 0):.1f} GFLOPS")
                    
                    console.print(table)
            else:
                console.print("[yellow]No metrics available[/yellow]")
    
    asyncio.run(run_metrics())


@cli.command()
@click.pass_context
def monitor(ctx):
    """Start real-time monitoring of system metrics"""
    async def run_monitor():
        async with OpenRuntimeCLI(ctx.obj["url"]) as cli_instance:
            console.print("[bold blue]Starting real-time monitoring...[/bold blue]")
            console.print("Press Ctrl+C to stop")
            
            try:
                while True:
                    # Get current status
                    info = await cli_instance.get_system_info()
                    devices = await cli_instance.list_devices()
                    tasks = await cli_instance.get_active_tasks()
                    
                    # Clear screen and redraw
                    console.clear()
                    
                    # Display current status
                    display_system_info(info)
                    display_devices(devices)
                    display_tasks(tasks)
                    
                    # Wait before next update
                    await asyncio.sleep(5)
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped[/yellow]")
    
    asyncio.run(run_monitor())


@cli.command()
@click.pass_context
def server(ctx):
    """Start the OpenRuntime server"""
    import subprocess
    import sys
    
    console.print("[bold blue]Starting OpenRuntime server...[/bold blue]")
    
    try:
        # Start the server
        subprocess.run([sys.executable, "openruntime.py", "--host", "0.0.0.0", "--port", "8000"], check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start server: {e}[/red]")


@cli.command()
@click.pass_context
def docs(ctx):
    """Open API documentation in browser"""
    import webbrowser
    
    url = f"{ctx.obj['url']}/docs"
    console.print(f"[bold blue]Opening API documentation: {url}[/bold blue]")
    
    try:
        webbrowser.open(url)
    except Exception as e:
        console.print(f"[red]Failed to open browser: {e}[/red]")
        console.print(f"Please visit: {url}")


def main():
    """Main entry point for the CLI"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
