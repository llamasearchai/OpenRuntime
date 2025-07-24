#!/usr/bin/env python3
"""
OpenRuntime Benchmark Script - Comprehensive performance testing suite.
"""

import asyncio
import time
import statistics
import json
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.chart import BarChart
import argparse

console = Console()


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    operation: str
    parameters: Dict[str, Any]
    execution_time: float
    throughput: Optional[float] = None
    memory_used: Optional[float] = None
    device: Optional[str] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None


class OpenRuntimeBenchmark:
    """Comprehensive benchmark suite for OpenRuntime."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results: List[BenchmarkResult] = []
        
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Make HTTP request with proper error handling."""
        url = f"{self.base_url}{endpoint}"
        
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            try:
                async with session.request(method, url, json=data) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        return {"error": f"HTTP {response.status}: {error_text}"}
                    return await response.json()
            except asyncio.TimeoutError:
                return {"error": f"Request timeout after {timeout}s"}
            except aiohttp.ClientError as e:
                return {"error": f"Connection error: {str(e)}"}
    
    async def benchmark_operation(
        self,
        operation: str,
        parameters: Dict[str, Any],
        iterations: int = 5
    ) -> BenchmarkResult:
        """Benchmark a single operation with multiple iterations."""
        times = []
        errors = []
        
        for i in range(iterations):
            start_time = time.time()
            
            result = await self._make_request(
                "POST",
                "/tasks",
                data={
                    "operation": operation,
                    **parameters
                }
            )
            
            execution_time = time.time() - start_time
            
            if "error" in result:
                errors.append(result["error"])
            else:
                times.append(execution_time)
        
        if not times and errors:
            return BenchmarkResult(
                operation=operation,
                parameters=parameters,
                execution_time=0,
                error=errors[0],
                timestamp=datetime.now().isoformat()
            )
        
        avg_time = statistics.mean(times) if times else 0
        
        # Calculate throughput based on operation type
        throughput = None
        if operation == "matrix_multiply" and "size" in parameters:
            # GFLOPS for matrix multiplication
            size = parameters["size"]
            flops = 2 * size ** 3  # 2n³ operations for n×n matrix multiply
            throughput = (flops / avg_time) / 1e9 if avg_time > 0 else 0
        elif operation == "vector_operations" and "size" in parameters:
            # Elements per second
            size = parameters["size"]
            throughput = size / avg_time if avg_time > 0 else 0
        elif operation == "memory_bandwidth" and "size" in parameters:
            # GB/s for memory operations
            size = parameters["size"]
            bytes_processed = size * 4 * 2  # float32, read + write
            throughput = (bytes_processed / avg_time) / 1e9 if avg_time > 0 else 0
        
        return BenchmarkResult(
            operation=operation,
            parameters=parameters,
            execution_time=avg_time,
            throughput=throughput,
            timestamp=datetime.now().isoformat()
        )
    
    async def run_benchmark_suite(self, suite: str = "standard"):
        """Run a predefined benchmark suite."""
        console.print(Panel.fit(f"[bold cyan]Running {suite.upper()} Benchmark Suite[/bold cyan]"))
        
        # Define benchmark suites
        suites = {
            "standard": [
                ("matrix_multiply", [
                    {"size": 100},
                    {"size": 500},
                    {"size": 1000},
                    {"size": 2000}
                ]),
                ("vector_operations", [
                    {"size": 1000},
                    {"size": 10000},
                    {"size": 100000},
                    {"size": 1000000}
                ]),
                ("memory_bandwidth", [
                    {"size": 1_000_000},
                    {"size": 10_000_000},
                    {"size": 100_000_000}
                ])
            ],
            "ai": [
                ("ai_inference", [
                    {"model": "small", "batch_size": 1},
                    {"model": "small", "batch_size": 16},
                    {"model": "medium", "batch_size": 1},
                    {"model": "medium", "batch_size": 8}
                ]),
                ("ai_training", [
                    {"model": "small", "epochs": 1},
                    {"model": "small", "epochs": 5},
                    {"model": "medium", "epochs": 1}
                ])
            ],
            "stress": [
                ("matrix_multiply", [
                    {"size": 5000},
                    {"size": 10000}
                ]),
                ("memory_stress", [
                    {"size": 1_000_000_000},
                    {"size": 2_000_000_000}
                ]),
                ("concurrent_tasks", [
                    {"tasks": 10, "size": 1000},
                    {"tasks": 50, "size": 1000},
                    {"tasks": 100, "size": 1000}
                ])
            ]
        }
        
        if suite not in suites:
            console.print(f"[red]Unknown suite: {suite}[/red]")
            console.print(f"Available suites: {', '.join(suites.keys())}")
            return
        
        benchmarks = suites[suite]
        total_benchmarks = sum(len(params) for _, params in benchmarks)
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running benchmarks...", total=total_benchmarks)
            
            for operation, param_sets in benchmarks:
                for params in param_sets:
                    desc = f"{operation} ({', '.join(f'{k}={v}' for k, v in params.items())})"
                    progress.update(task, description=desc)
                    
                    result = await self.benchmark_operation(operation, params)
                    self.results.append(result)
                    
                    progress.advance(task)
        
        self._display_results()
    
    def _display_results(self):
        """Display benchmark results in a formatted table."""
        if not self.results:
            console.print("[yellow]No benchmark results to display.[/yellow]")
            return
        
        # Group results by operation
        grouped_results: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            if result.operation not in grouped_results:
                grouped_results[result.operation] = []
            grouped_results[result.operation].append(result)
        
        for operation, results in grouped_results.items():
            table = Table(
                title=f"{operation.replace('_', ' ').title()} Benchmark Results",
                show_header=True,
                header_style="bold magenta"
            )
            
            table.add_column("Parameters", style="cyan")
            table.add_column("Avg Time (s)", justify="right", style="yellow")
            table.add_column("Throughput", justify="right", style="green")
            table.add_column("Status", justify="center")
            
            for result in results:
                param_str = ", ".join(f"{k}={v}" for k, v in result.parameters.items())
                
                if result.error:
                    table.add_row(
                        param_str,
                        "N/A",
                        "N/A",
                        "[red]ERROR[/red]"
                    )
                else:
                    throughput_str = "N/A"
                    if result.throughput is not None:
                        if "matrix" in operation:
                            throughput_str = f"{result.throughput:.2f} GFLOPS"
                        elif "vector" in operation:
                            throughput_str = f"{result.throughput:.2e} elem/s"
                        elif "memory" in operation:
                            throughput_str = f"{result.throughput:.2f} GB/s"
                    
                    table.add_row(
                        param_str,
                        f"{result.execution_time:.4f}",
                        throughput_str,
                        "[green]OK[/green]"
                    )
            
            console.print(table)
            console.print()
    
    async def run_comparison(self, backends: List[str]):
        """Compare performance across different backends."""
        console.print(Panel.fit("[bold cyan]Backend Comparison Benchmark[/bold cyan]"))
        
        operations = [
            ("matrix_multiply", {"size": 1000}),
            ("vector_operations", {"size": 100000}),
            ("memory_bandwidth", {"size": 10_000_000})
        ]
        
        comparison_results: Dict[str, Dict[str, BenchmarkResult]] = {}
        
        for backend in backends:
            console.print(f"\n[bold]Testing backend: {backend}[/bold]")
            comparison_results[backend] = {}
            
            for operation, params in operations:
                # Add backend to parameters
                params_with_backend = {**params, "backend": backend}
                result = await self.benchmark_operation(operation, params_with_backend)
                comparison_results[backend][operation] = result
        
        # Display comparison table
        table = Table(
            title="Backend Performance Comparison",
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Operation", style="cyan")
        for backend in backends:
            table.add_column(backend, justify="right")
        
        for operation, _ in operations:
            row = [operation.replace('_', ' ').title()]
            
            for backend in backends:
                result = comparison_results[backend].get(operation)
                if result and not result.error:
                    if result.throughput:
                        row.append(f"{result.throughput:.2f}")
                    else:
                        row.append(f"{result.execution_time:.4f}s")
                else:
                    row.append("[red]ERROR[/red]")
            
            table.add_row(*row)
        
        console.print(table)
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        console.print(f"[green]Results saved to {filename}[/green]")
    
    async def continuous_benchmark(self, duration: int = 300, interval: int = 10):
        """Run continuous benchmarks for monitoring."""
        console.print(Panel.fit(f"[bold cyan]Continuous Benchmark - {duration}s duration[/bold cyan]"))
        
        start_time = time.time()
        iteration = 0
        
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="results"),
            Layout(name="chart", size=10)
        )
        
        with Live(layout, refresh_per_second=1) as live:
            while time.time() - start_time < duration:
                iteration += 1
                elapsed = time.time() - start_time
                
                # Update header
                header_text = f"[bold]Iteration {iteration} | Elapsed: {elapsed:.0f}s / {duration}s[/bold]"
                layout["header"].update(Panel(header_text))
                
                # Run benchmark
                result = await self.benchmark_operation(
                    "matrix_multiply",
                    {"size": 1000},
                    iterations=1
                )
                self.results.append(result)
                
                # Update results display
                recent_results = self.results[-10:]  # Last 10 results
                results_table = Table(show_header=True, header_style="bold magenta")
                results_table.add_column("Time", style="cyan")
                results_table.add_column("Execution (s)", justify="right")
                results_table.add_column("Throughput (GFLOPS)", justify="right")
                
                for r in recent_results:
                    if r.timestamp:
                        time_str = r.timestamp.split('T')[1].split('.')[0]
                        results_table.add_row(
                            time_str,
                            f"{r.execution_time:.4f}",
                            f"{r.throughput:.2f}" if r.throughput else "N/A"
                        )
                
                layout["results"].update(results_table)
                
                # Update chart (simple ASCII chart)
                if len(self.results) > 1:
                    throughputs = [r.throughput for r in self.results[-20:] if r.throughput]
                    if throughputs:
                        chart_text = self._create_simple_chart(throughputs)
                        layout["chart"].update(Panel(chart_text, title="Throughput Trend"))
                
                await asyncio.sleep(interval)
        
        console.print("\n[bold green]Continuous benchmark completed![/bold green]")
        self._display_summary_stats()
    
    def _create_simple_chart(self, values: List[float]) -> str:
        """Create a simple ASCII chart."""
        if not values:
            return "No data"
        
        max_val = max(values)
        min_val = min(values)
        height = 8
        width = len(values)
        
        chart_lines = []
        
        for h in range(height, 0, -1):
            line = ""
            threshold = min_val + (max_val - min_val) * (h / height)
            
            for v in values:
                if v >= threshold:
                    line += "█"
                else:
                    line += " "
            
            chart_lines.append(f"{threshold:6.1f} |{line}")
        
        chart_lines.append(f"{'':6s} └{'─' * width}")
        
        return "\n".join(chart_lines)
    
    def _display_summary_stats(self):
        """Display summary statistics of all benchmarks."""
        if not self.results:
            return
        
        console.print("\n[bold]Summary Statistics[/bold]")
        
        # Group by operation
        op_stats: Dict[str, List[float]] = {}
        for result in self.results:
            if not result.error and result.execution_time > 0:
                if result.operation not in op_stats:
                    op_stats[result.operation] = []
                op_stats[result.operation].append(result.execution_time)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Operation", style="cyan")
        table.add_column("Runs", justify="right")
        table.add_column("Avg Time (s)", justify="right")
        table.add_column("Min Time (s)", justify="right")
        table.add_column("Max Time (s)", justify="right")
        table.add_column("Std Dev", justify="right")
        
        for operation, times in op_stats.items():
            if len(times) >= 2:
                table.add_row(
                    operation,
                    str(len(times)),
                    f"{statistics.mean(times):.4f}",
                    f"{min(times):.4f}",
                    f"{max(times):.4f}",
                    f"{statistics.stdev(times):.4f}"
                )
            elif times:
                table.add_row(
                    operation,
                    str(len(times)),
                    f"{times[0]:.4f}",
                    f"{times[0]:.4f}",
                    f"{times[0]:.4f}",
                    "N/A"
                )
        
        console.print(table)


async def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="OpenRuntime Performance Benchmark Suite"
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="OpenRuntime service URL"
    )
    
    parser.add_argument(
        "--suite",
        choices=["standard", "ai", "stress", "all"],
        default="standard",
        help="Benchmark suite to run"
    )
    
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous benchmarks"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration for continuous benchmarks (seconds)"
    )
    
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare performance across backends"
    )
    
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    benchmark = OpenRuntimeBenchmark(args.url)
    
    try:
        if args.continuous:
            await benchmark.continuous_benchmark(args.duration)
        elif args.compare:
            await benchmark.run_comparison(args.compare)
        elif args.suite == "all":
            for suite in ["standard", "ai", "stress"]:
                await benchmark.run_benchmark_suite(suite)
                console.print()
        else:
            await benchmark.run_benchmark_suite(args.suite)
        
        if args.output:
            benchmark.save_results(args.output)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        if benchmark.results and args.output:
            benchmark.save_results(args.output)


if __name__ == "__main__":
    asyncio.run(main())