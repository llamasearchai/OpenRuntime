#!/usr/bin/env python3
"""
OpenRuntime CLI: Command Line Interface for OpenRuntime GPU Computing Platform

Author: Nik Jois <nikjois@llamasearch.ai>
Version: 2.0.0

A comprehensive CLI for managing GPU computing tasks, monitoring devices,
running benchmarks, and interacting with the OpenRuntime API.
"""

import asyncio
import json
import logging
import sys
import webbrowser
from typing import Any, Dict, List, Optional

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import SpinnerColumn, TextColumn, Progress
from rich.syntax import Syntax
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OpenRuntime CLI")

# Version information
__version__ = "2.0.0"

# Rich console for beautiful output
console = Console()


class OpenRuntimeCLI:
    """Main CLI class for OpenRuntime operations"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Helper for making API requests."""
        try:
            response = await self.client.request(method, f"{self.base_url}{endpoint}", **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        except httpx.RequestError as e:
            console.print(f"[red]Connection error: {e}[/red]")
        except json.JSONDecodeError:
            console.print("[red]Error: Failed to decode JSON response from server.[/red]")
        return {}

    async def get_system_info(self) -> Dict[str, Any]:
        return await self._request("GET", "/")

    async def list_devices(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/devices")
        return data.get("devices", [])
    
    async def create_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/tasks", json=task_data)

    async def create_enhanced_task(self, gpu_task: Dict[str, Any], ai_task: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/tasks/enhanced", json={"gpu_task": gpu_task, "ai_task": ai_task})

    async def create_ai_task(self, ai_task: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/ai/tasks", json=ai_task)


def display_system_info(info: Dict[str, Any]):
    """Display system information in a beautiful format"""
    if not info: return
    table = Table(title="System Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in info.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


def display_devices(devices: List[Dict[str, Any]]):
    """Display devices in a table format"""
    if not devices: return
    table = Table(title="Available Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Memory (GB)", style="blue")
    for device in devices:
        memory_gb = device.get("memory_total", 0) / (1024**3)
        table.add_row(device.get("id"), device.get("name"), device.get("type"), f"{memory_gb:.2f}")
    console.print(table)


@click.group()
@click.version_option(version=__version__, prog_name="OpenRuntime CLI")
@click.option("--url", default="http://localhost:8000", help="OpenRuntime server URL.")
@click.pass_context
def cli(ctx, url: str):
    """A CLI for interacting with the OpenRuntime server."""
    ctx.obj = {"URL": url}


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and information."""
    async def run():
        async with OpenRuntimeCLI(ctx.obj["URL"]) as client:
            with console.status("Fetching system info..."):
                info = await client.get_system_info()
            display_system_info(info)
            with console.status("Fetching devices..."):
                devices = await client.list_devices()
            display_devices(devices)
    asyncio.run(run())


@cli.command()
@click.option("--operation", "-o", default="compute", type=click.Choice(["compute", "inference"]), help="Operation type.")
@click.option("--size", "-s", default=1024, help="Matrix size for compute operations.")
@click.option("--ai-workflow", help="AI workflow to enhance the task.")
@click.option("--ai-prompt", help="AI prompt for enhancement.")
@click.pass_context
def run(ctx, operation, size, ai_workflow, ai_prompt):
    """Run a computational task, with optional AI enhancement."""
    gpu_task = {"operation": operation, "data": {"size": size}}
    
    async def _run():
        async with OpenRuntimeCLI(ctx.obj["URL"]) as client:
            if ai_workflow and ai_prompt:
                console.print(f"Running enhanced task with AI workflow: [bold]{ai_workflow}[/bold]")
                ai_task = {"workflow_type": ai_workflow, "prompt": ai_prompt}
                result = await client.create_enhanced_task(gpu_task, ai_task)
            else:
                console.print("Running standard GPU task...")
                result = await client.create_task(gpu_task)
            
            console.print(Panel(json.dumps(result, indent=2), title="Task Result", border_style="green"))

    asyncio.run(_run())


@cli.command()
@click.option("--workflow", "-w", required=True, help="Type of AI workflow.")
@click.option("--prompt", "-p", required=True, help="AI prompt or task description.")
@click.pass_context
def ai(ctx, workflow, prompt):
    """Execute a standalone AI-powered task."""
    ai_task = {"workflow_type": workflow, "prompt": prompt}
    
    async def _run():
        async with OpenRuntimeCLI(ctx.obj["URL"]) as client:
            with console.status("Executing AI task..."):
                result = await client.create_ai_task(ai_task)
            console.print(Panel(json.dumps(result, indent=2), title="AI Task Result", border_style="green"))

    asyncio.run(_run())


@cli.command()
def server():
    """Start the OpenRuntime server."""
    import subprocess
    console.print("[bold blue]Starting OpenRuntime server...[/bold blue]")
    try:
        subprocess.run([sys.executable, "-m", "openruntime.main"], check=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")


@cli.command()
@click.pass_context
def docs(ctx):
    """Open API documentation in browser."""
    url = f"{ctx.obj['URL']}/docs"
    console.print(f"Opening API documentation at [link={url}]{url}[/link]")
    webbrowser.open(url)


if __name__ == "__main__":
    cli()