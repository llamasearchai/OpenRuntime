#!/usr/bin/env python3
"""
Stress testing script for OpenRuntime Enhanced
Advanced load testing with concurrent requests and performance analysis
"""

import asyncio
import aiohttp
import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import statistics
import sys

@dataclass
class TestResult:
    endpoint: str
    method: str
    response_time: float
    status_code: int
    response_size: int
    success: bool
    timestamp: str

@dataclass
class EndpointStats:
    endpoint: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    error_rate: float

class StressTester:
    def __init__(self, base_url: str, concurrent: int = 10, total: int = 1000):
        self.base_url = base_url.rstrip('/')
        self.concurrent = concurrent
        self.total = total
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
    
    async def make_request(self, session: aiohttp.ClientSession, method: str, endpoint: str, data: Dict = None) -> TestResult:
        """Make a single request and record metrics"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        try:
            if method == "POST":
                async with session.post(url, json=data) as response:
                    response_size = len(await response.text())
                    return TestResult(
                        endpoint=endpoint,
                        method=method,
                        response_time=time.time() - start_time,
                        status_code=response.status,
                        response_size=response_size,
                        success=response.status == 200,
                        timestamp=datetime.now().isoformat()
                    )
            else:  # GET
                async with session.get(url) as response:
                    response_size = len(await response.text())
                    return TestResult(
                        endpoint=endpoint,
                        method=method,
                        response_time=time.time() - start_time,
                        status_code=response.status,
                        response_size=response_size,
                        success=response.status == 200,
                        timestamp=datetime.now().isoformat()
                    )
        except Exception as e:
            return TestResult(
                endpoint=endpoint,
                method=method,
                response_time=time.time() - start_time,
                status_code=0,
                response_size=0,
                success=False,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_endpoint(self, endpoint: str, payload: Dict[str, Any], method: str = "POST") -> TestResult:
        """Test a single endpoint"""
        start_time = time.time()
        
        try:
            if method == "POST":
                async with self.session.post(
                    f"{self.base_url}{endpoint}",
                    json=payload
                ) as response:
                    await response.text()  # Consume response
                    return TestResult(
                        endpoint=endpoint,
                        success=response.status == 200,
                        response_time=time.time() - start_time,
                        status_code=response.status
                    )
            else:  # GET
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    await response.text()
                    return TestResult(
                        endpoint=endpoint,
                        success=response.status == 200,
                        response_time=time.time() - start_time,
                        status_code=response.status
                    )
        except Exception as e:
            return TestResult(
                endpoint=endpoint,
                success=False,
                response_time=time.time() - start_time,
                status_code=0,
                error=str(e)
            )

    async def run_concurrent_tests(self, endpoint: str, payload: Dict[str, Any], 
                                 concurrent_requests: int, total_requests: int):
        """Run concurrent tests on an endpoint"""
        print(f"Testing {endpoint} with {concurrent_requests} concurrent requests, {total_requests} total")
        
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def limited_test():
            async with semaphore:
                return await self.test_endpoint(endpoint, payload)
        
        # Create tasks
        tasks = [limited_test() for _ in range(total_requests)]
        
        # Run tests
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Collect results
        self.results.extend(results)
        
        # Print statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if successful:
            response_times = [r.response_time for r in successful]
            print(f"SUCCESS: {len(successful)}/{len(results)} requests successful")
            print(f"Response time stats:")
            print(f"   Average: {statistics.mean(response_times):.3f}s")
            print(f"   Median:  {statistics.median(response_times):.3f}s")
            print(f"   Min:     {min(response_times):.3f}s")
            print(f"   Max:     {max(response_times):.3f}s")
            print(f"Throughput: {len(successful)/total_time:.2f} requests/second")
        
        if failed:
            print(f"FAILED: {len(failed)} requests failed")
            error_counts = {}
            for result in failed:
                key = f"HTTP {result.status_code}" if result.status_code else "Network Error"
                error_counts[key] = error_counts.get(key, 0) + 1
            
            for error_type, count in error_counts.items():
                print(f"   {error_type}: {count}")
        
        print()

async def main():
    parser = argparse.ArgumentParser(description="OpenRuntime Stress Tester")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--total", type=int, default=100, help="Total requests per test")
    
    args = parser.parse_args()
    
    test_scenarios = [
        {
            "endpoint": "/",
            "payload": {},
            "method": "GET",
            "name": "Health Check"
        },
        {
            "endpoint": "/devices",
            "payload": {},
            "method": "GET",
            "name": "Device List"
        },
        {
            "endpoint": "/tasks",
            "payload": {
                "operation": "compute",
                "data": {
                    "type": "matrix_multiply",
                    "size": 512
                }
            },
            "method": "POST",
            "name": "Compute Task"
        },
        {
            "endpoint": "/ai/tasks",
            "payload": {
                "workflow_type": "system_analysis",
                "prompt": "Analyze system performance",
                "context": {
                    "metrics": {
                        "cpu_usage": 45,
                        "memory_usage": 60,
                        "gpu_utilization": 75
                    }
                }
            },
            "method": "POST",
            "name": "AI Analysis Task"
        },
        {
            "endpoint": "/benchmark",
            "payload": {},
            "method": "POST",
            "name": "Benchmark"
        }
    ]
    
    async with StressTester(args.url) as tester:
        print("OpenRuntime Enhanced Stress Test")
        print("=" * 50)
        
        # Test server availability
        try:
            result = await tester.test_endpoint("/", {}, "GET")
            if not result.success:
                print(f"ERROR: Server not available at {args.url}")
                return
            print(f"SUCCESS: Server is available at {args.url}")
            print()
        except Exception as e:
            print(f"ERROR: Failed to connect to server: {e}")
            return
        
        # Run stress tests
        for scenario in test_scenarios:
            await tester.run_concurrent_tests(
                scenario["endpoint"],
                scenario["payload"],
                args.concurrent,
                args.total
            )
            
            # Brief pause between tests
            await asyncio.sleep(1)
        
        # Overall statistics
        all_results = tester.results
        successful_results = [r for r in all_results if r.success]
        
        print("Overall Test Results")
        print("=" * 30)
        print(f"Total requests: {len(all_results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(all_results) - len(successful_results)}")
        print(f"Success rate: {len(successful_results)/len(all_results)*100:.1f}%")
        
        if successful_results:
            response_times = [r.response_time for r in successful_results]
            print(f"Average response time: {statistics.mean(response_times):.3f}s")
            print(f"95th percentile: {sorted(response_times)[int(len(response_times)*0.95)]:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())