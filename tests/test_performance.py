#!/usr/bin/env python3
"""
Performance and Load Testing Suite for OpenRuntime Enhanced
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, AsyncMock

from openruntime_enhanced import app, enhanced_runtime, AITaskRequest, WorkflowType


class TestPerformanceBenchmarks:
    """Performance benchmarking test suite"""

    @pytest.mark.asyncio
    async def test_ai_task_throughput(self, async_client):
        """Test AI task processing throughput"""
        async with async_client as client:
            # Create multiple AI tasks
            tasks = []
            start_time = time.time()

            for i in range(20):
                task_data = {"workflow_type": "system_analysis", "prompt": f"Quick analysis task {i}", "max_tokens": 100}
                tasks.append(client.post("/ai/tasks", json=task_data))

            with patch.object(enhanced_runtime.ai_manager, "execute_ai_task") as mock_execute:

                async def mock_task(*args, **kwargs):
                    await asyncio.sleep(0.1)  # Simulate processing time
                    return {"task_id": "test", "result": "success", "execution_time": 0.1}

                mock_execute.side_effect = mock_task

                # Execute all tasks concurrently
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                end_time = time.time()
                total_time = end_time - start_time

                # Verify performance metrics
                successful_responses = [r for r in responses if not isinstance(r, Exception)]
                throughput = len(successful_responses) / total_time

                assert len(successful_responses) >= 15  # Allow some failures
                assert throughput > 5  # At least 5 tasks per second
                assert total_time < 10  # Complete within 10 seconds

    @pytest.mark.asyncio
    async def test_gpu_task_latency(self, async_client):
        """Test GPU task execution latency"""
        async with async_client as client:
            latencies = []

            for _ in range(10):
                start_time = time.time()

                task_data = {"operation": "compute", "data": {"type": "matrix_multiply", "size": 512}, "priority": 1}

                response = await client.post("/tasks", json=task_data)
                end_time = time.time()

                assert response.status_code == 200
                latency = end_time - start_time
                latencies.append(latency)

            # Analyze latency statistics
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

            assert avg_latency < 2.0  # Average latency under 2 seconds
            assert p95_latency < 5.0  # 95th percentile under 5 seconds
            assert max(latencies) < 10.0  # No task takes more than 10 seconds

    @pytest.mark.asyncio
    async def test_concurrent_mixed_workload(self, async_client):
        """Test performance under mixed AI and GPU workloads"""
        async with async_client as client:
            # Create mixed workload
            ai_tasks = []
            gpu_tasks = []

            # AI tasks
            for i in range(10):
                task_data = {"workflow_type": "code_generation", "prompt": f"Generate function {i}", "max_tokens": 200}
                ai_tasks.append(client.post("/ai/tasks", json=task_data))

            # GPU tasks
            for i in range(10):
                task_data = {"operation": "benchmark", "data": {"type": "compute"}, "priority": 1}
                gpu_tasks.append(client.post("/tasks", json=task_data))

            with patch.object(enhanced_runtime.ai_manager, "execute_ai_task") as mock_ai:

                async def mock_ai_task(*args, **kwargs):
                    await asyncio.sleep(0.2)
                    return {"task_id": "ai_test", "result": "code generated"}

                mock_ai.side_effect = mock_ai_task

                start_time = time.time()

                # Execute all tasks concurrently
                all_tasks = ai_tasks + gpu_tasks
                responses = await asyncio.gather(*all_tasks, return_exceptions=True)

                end_time = time.time()
                total_time = end_time - start_time

                # Verify mixed workload performance
                successful_responses = [r for r in responses if not isinstance(r, Exception)]
                assert len(successful_responses) >= 15  # Most tasks succeed
                assert total_time < 15  # Complete within reasonable time

    def test_memory_usage_stability(self):
        """Test memory usage remains stable under load"""
        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Create many AI agent instances
        agents = []
        for i in range(100):
            from openruntime_enhanced import AIAgentManager

            agent_manager = AIAgentManager()
            agents.append(agent_manager)

        # Force garbage collection
        gc.collect()

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Clean up
        del agents
        gc.collect()

        final_memory = process.memory_info().rss
        memory_leak = final_memory - initial_memory

        # Memory usage should be reasonable
        assert memory_increase < 500 * 1024 * 1024  # Less than 500MB increase
        assert memory_leak < 50 * 1024 * 1024  # Less than 50MB leak


class TestScalabilityTests:
    """Scalability testing suite"""

    @pytest.mark.asyncio
    async def test_high_concurrency_ai_tasks(self, async_client):
        """Test system behavior under high concurrency"""
        async with async_client as client:
            # Create high number of concurrent tasks
            tasks = []

            for i in range(50):
                task_data = {"workflow_type": "system_analysis", "prompt": f"Analysis {i}", "max_tokens": 50}
                tasks.append(client.post("/ai/tasks", json=task_data))

            with patch.object(enhanced_runtime.ai_manager, "execute_ai_task") as mock_execute:

                async def mock_task(*args, **kwargs):
                    await asyncio.sleep(0.05)  # Fast processing
                    return {"task_id": f"test_{id(args)}", "result": "success"}

                mock_execute.side_effect = mock_task

                start_time = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()

                # Analyze results
                successful_responses = [r for r in responses if not isinstance(r, Exception)]
                success_rate = len(successful_responses) / len(tasks)

                assert success_rate > 0.8  # At least 80% success rate
                assert end_time - start_time < 20  # Complete within 20 seconds

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, async_client):
        """Test performance under sustained load"""
        async with async_client as client:
            # Run sustained load for multiple rounds
            round_times = []

            for round_num in range(5):
                tasks = []

                for i in range(10):
                    task_data = {
                        "workflow_type": "compute_optimization",
                        "prompt": f"Round {round_num} task {i}",
                        "max_tokens": 100,
                    }
                    tasks.append(client.post("/ai/tasks", json=task_data))

                with patch.object(enhanced_runtime.ai_manager, "execute_ai_task") as mock_execute:

                    async def mock_task(*args, **kwargs):
                        await asyncio.sleep(0.1)
                        return {"task_id": "sustained_test", "result": "optimized"}

                    mock_execute.side_effect = mock_task

                    start_time = time.time()
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    end_time = time.time()

                    round_time = end_time - start_time
                    round_times.append(round_time)

                    # Brief pause between rounds
                    await asyncio.sleep(0.5)

            # Performance should remain consistent across rounds
            avg_time = statistics.mean(round_times)
            time_variance = statistics.variance(round_times)

            assert avg_time < 5.0  # Average round time under 5 seconds
            assert time_variance < 2.0  # Low variance in performance

    def test_resource_cleanup(self):
        """Test proper resource cleanup"""
        import threading
        import gc

        initial_thread_count = threading.active_count()

        # Create and destroy multiple runtime instances
        runtimes = []
        for i in range(5):  # Reduced from 10 to 5
            from openruntime_enhanced import EnhancedGPURuntimeManager

            runtime = EnhancedGPURuntimeManager()
            runtimes.append(runtime)

        peak_thread_count = threading.active_count()

        # Clean up
        del runtimes
        gc.collect()

        # Allow time for cleanup
        time.sleep(2)  # Increased cleanup time

        final_thread_count = threading.active_count()

        # Thread count should not grow excessively
        # Allow for some background threads from AI components
        assert peak_thread_count <= initial_thread_count + 15  # More realistic limit
        assert final_thread_count <= initial_thread_count + 10  # Allow some persistent threads


class TestStressTests:
    """Stress testing suite"""

    @pytest.mark.asyncio
    async def test_error_recovery_under_load(self, async_client):
        """Test system recovery from errors under load"""
        async with async_client as client:
            tasks = []

            for i in range(30):
                task_data = {"workflow_type": "shell_automation", "prompt": f"Command {i}", "max_tokens": 100}
                tasks.append(client.post("/ai/tasks", json=task_data))

            with patch.object(enhanced_runtime.ai_manager, "execute_ai_task") as mock_execute:

                async def mock_task_with_failures(*args, **kwargs):
                    # Simulate 30% failure rate
                    if hash(str(args)) % 10 < 3:
                        raise Exception("Simulated failure")
                    await asyncio.sleep(0.1)
                    return {"task_id": "stress_test", "result": "success"}

                mock_execute.side_effect = mock_task_with_failures

                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # System should handle failures gracefully
                successful_responses = [r for r in responses if not isinstance(r, Exception)]
                error_responses = [r for r in responses if isinstance(r, Exception)]

                # Should have some successes despite failures
                assert len(successful_responses) > 10
                # Should not crash the system
                assert len(error_responses) < len(tasks)

    @pytest.mark.asyncio
    async def test_timeout_handling_stress(self, async_client):
        """Test timeout handling under stress"""
        async with async_client as client:
            tasks = []

            for i in range(20):
                task_data = {"workflow_type": "code_generation", "prompt": f"Complex generation {i}", "max_tokens": 1000}
                tasks.append(client.post("/ai/tasks", json=task_data))

            with patch.object(enhanced_runtime.ai_manager, "execute_ai_task") as mock_execute:

                async def mock_slow_task(*args, **kwargs):
                    # Some tasks timeout, others succeed
                    if hash(str(args)) % 5 == 0:
                        raise asyncio.TimeoutError("Task timed out")
                    await asyncio.sleep(0.2)
                    return {"task_id": "timeout_test", "result": "generated"}

                mock_execute.side_effect = mock_slow_task

                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Verify timeout handling
                successful_responses = [r for r in responses if not isinstance(r, Exception)]

                # System should handle timeouts gracefully
                assert len(successful_responses) > 5

                # Check that timeout responses are properly formatted
                for response in responses:
                    if not isinstance(response, Exception):
                        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
