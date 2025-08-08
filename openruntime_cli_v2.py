#!/usr/bin/env python3
"""
OpenRuntime CLI v2 - Command-line interface for OpenRuntime
"""

import asyncio
import json
import sys
import argparse
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from typing import Optional, Dict, Any

console = Console()


class OpenRuntimeCLI:
    """CLI for OpenRuntime API v2"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def health_check(self) -> None:
        """Check runtime health"""
        try:
            response = await self.client.get(f"{self.base_url}/v2/health")
            if response.status_code == 200:
                data = response.json()
                console.print("[green]Runtime is healthy[/green]")
                console.print(json.dumps(data, indent=2))
            else:
                console.print(f"[red]Health check failed: {response.status_code}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            
    async def list_backends(self) -> None:
        """List available backends"""
        try:
            response = await self.client.get(f"{self.base_url}/v2/backends")
            if response.status_code == 200:
                data = response.json()
                
                table = Table(title="Available Backends")
                table.add_column("Backend", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Details")
                
                for backend in data["backends"]:
                    status = "Ready" if backend["initialized"] else "Not Ready"
                    details = json.dumps(backend["metrics"], indent=2)[:100] + "..."
                    table.add_row(backend["type"], status, details)
                    
                console.print(table)
            else:
                console.print(f"[red]Failed to list backends: {response.status_code}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            
    async def list_models(self) -> None:
        """List available models"""
        try:
            response = await self.client.get(f"{self.base_url}/v2/models")
            if response.status_code == 200:
                data = response.json()
                
                table = Table(title="Available Models")
                table.add_column("Backend", style="cyan")
                table.add_column("Models")
                
                for backend, models in data["models"].items():
                    model_list = ", ".join(models) if models else "None"
                    table.add_row(backend, model_list)
                    
                console.print(table)
            else:
                console.print(f"[red]Failed to list models: {response.status_code}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            
    async def complete(
        self,
        prompt: str,
        model: str = "gpt-4-turbo-preview",
        backend: Optional[str] = None
    ) -> None:
        """Generate completion"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Generating completion...", total=None)
            
            try:
                request_data = {
                    "prompt": prompt,
                    "model": model,
                    "stream": False
                }
                
                if backend:
                    request_data["backend"] = backend
                    
                response = await self.client.post(
                    f"{self.base_url}/v2/completions",
                    json=request_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    task_id = data["task_id"]
                    
                    # Wait for task completion (simplified)
                    await asyncio.sleep(2)
                    
                    console.print(f"[green]Task {task_id} completed[/green]")
                else:
                    console.print(f"[red]Completion failed: {response.text}[/red]")
                    
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            finally:
                progress.stop()
                
    async def embed(
        self,
        text: str,
        model: str = "text-embedding-3-small",
        backend: Optional[str] = None
    ) -> None:
        """Generate embeddings"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Generating embeddings...", total=None)
            
            try:
                request_data = {
                    "texts": text,
                    "model": model
                }
                
                if backend:
                    request_data["backend"] = backend
                    
                response = await self.client.post(
                    f"{self.base_url}/v2/embeddings",
                    json=request_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    task_id = data["task_id"]
                    
                    console.print(f"[green]Embedding task {task_id} submitted[/green]")
                else:
                    console.print(f"[red]Embedding failed: {response.text}[/red]")
                    
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            finally:
                progress.stop()
                
    async def run_agent(
        self,
        task: str,
        agent_type: str = "developer",
        backend: str = "openai"
    ) -> None:
        """Run agent"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress_task = progress.add_task(f"Running {agent_type} agent...", total=None)
            
            try:
                request_data = {
                    "task": task,
                    "agent_type": agent_type,
                    "backend": backend
                }
                
                response = await self.client.post(
                    f"{self.base_url}/v2/agents",
                    json=request_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    task_id = data["task_id"]
                    
                    console.print(f"[green]Agent task {task_id} submitted[/green]")
                else:
                    console.print(f"[red]Agent failed: {response.text}[/red]")
                    
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            finally:
                progress.stop()
                
    async def run_workflow(
        self,
        workflow_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Run workflow"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Running {workflow_type} workflow...", total=None)
            
            try:
                request_data = {
                    "workflow_type": workflow_type,
                    "data": data
                }
                
                response = await self.client.post(
                    f"{self.base_url}/v2/workflows",
                    json=request_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    task_id = data["task_id"]
                    
                    console.print(f"[green]Workflow {task_id} started[/green]")
                else:
                    console.print(f"[red]Workflow failed: {response.text}[/red]")
                    
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            finally:
                progress.stop()
                
    async def close(self) -> None:
        """Close client"""
        await self.client.aclose()


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="OpenRuntime CLI v2")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Health command
    subparsers.add_parser("health", help="Check runtime health")
    
    # Backends command
    subparsers.add_parser("backends", help="List available backends")
    
    # Models command
    subparsers.add_parser("models", help="List available models")
    
    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Generate completion")
    complete_parser.add_argument("prompt", help="Prompt text")
    complete_parser.add_argument("--model", default="gpt-4-turbo-preview", help="Model to use")
    complete_parser.add_argument("--backend", help="Backend to use")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    embed_parser.add_argument("text", help="Text to embed")
    embed_parser.add_argument("--model", default="text-embedding-3-small", help="Model to use")
    embed_parser.add_argument("--backend", help="Backend to use")
    
    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run agent")
    agent_parser.add_argument("task", help="Task for agent")
    agent_parser.add_argument("--type", default="developer", help="Agent type")
    agent_parser.add_argument("--backend", default="openai", help="Backend to use")
    
    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run workflow")
    workflow_parser.add_argument("type", help="Workflow type")
    workflow_parser.add_argument("--data", default="{}", help="Workflow data (JSON)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    cli = OpenRuntimeCLI(args.url)
    
    try:
        if args.command == "health":
            await cli.health_check()
        elif args.command == "backends":
            await cli.list_backends()
        elif args.command == "models":
            await cli.list_models()
        elif args.command == "complete":
            await cli.complete(args.prompt, args.model, args.backend)
        elif args.command == "embed":
            await cli.embed(args.text, args.model, args.backend)
        elif args.command == "agent":
            await cli.run_agent(args.task, args.type, args.backend)
        elif args.command == "workflow":
            data = json.loads(args.data)
            await cli.run_workflow(args.type, data)
        else:
            console.print(f"[red]Unknown command: {args.command}[/red]")
            
    finally:
        await cli.close()


if __name__ == "__main__":
    asyncio.run(main())


def entrypoint():
    """Synchronous entrypoint for console_scripts."""
    asyncio.run(main())