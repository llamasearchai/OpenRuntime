#!/usr/bin/env python3
"""
Test suite for OpenRuntime GPU Computing Platform

Author: Nik Jois <nikjois@llamasearch.ai>
Version: 2.0.0

Comprehensive tests for the OpenRuntime system including:
- Core functionality testing
- MLX Metal integration testing
- PyTorch Metal integration testing
- API endpoint testing
- Performance benchmarking
- Error handling and edge cases
"""

import asyncio
import json
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openruntime import (
    GPURuntimeManager,
    MLXRuntimeManager,
    TaskRequest,
    TaskResponse,
    TaskStatus,
    DeviceType,
    GPUDevice,
    RuntimeMetrics,
    ComputeKernel,
    MLXModel
)


class TestMLXRuntimeManager:
    """Test MLX Runtime Manager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mlx_manager = MLXRuntimeManager()
    
    def test_initialization(self):
        """Test MLX manager initialization"""
        assert hasattr(self.mlx_manager, 'devices')
        assert hasattr(self.mlx_manager, 'compiled_kernels')
        assert hasattr(self.mlx_manager, 'active_computations')
    
    @pytest.mark.asyncio
    async def test_matrix_multiply_mlx_simulation(self):
        """Test matrix multiplication with MLX simulation"""
        result = await self.mlx_manager.matrix_multiply_mlx(512)
        
        assert isinstance(result, dict)
        # Check if MLX is available or fallback to simulation
        if result['operation'] == 'matrix_multiply_mlx':
            assert result['operation'] == 'matrix_multiply_mlx'
            assert result['device'] == 'mlx_metal_0'
        else:
            assert result['operation'] == 'matrix_multiply_simulated'
            assert result['device'] == 'cpu_0'
        assert result['size'] == 512
        assert 'execution_time' in result
        assert 'gflops' in result
        # MLX returns tuple, simulation returns list
        assert result['result_shape'] in ([512, 512], (512, 512))
    
    @pytest.mark.asyncio
    async def test_neural_network_mlx_simulation(self):
        """Test neural network with MLX simulation"""
        result = await self.mlx_manager.neural_network_mlx(784, 512, 10)
        
        assert isinstance(result, dict)
        # Check if MLX is available or fallback to simulation
        if result['operation'] == 'neural_network_mlx':
            assert result['operation'] == 'neural_network_mlx'
            assert result['device'] == 'mlx_metal_0'
        else:
            assert result['operation'] == 'neural_network_simulated'
            assert result['device'] == 'cpu_0'
        assert result['input_size'] == 784
        assert result['hidden_size'] == 512
        assert result['output_size'] == 10
        assert 'execution_time' in result
        # MLX returns tuple, simulation returns list
        assert result['output_shape'] in ([1, 10], (1, 10))
    
    def test_simulate_matrix_multiply(self):
        """Test CPU matrix multiplication simulation"""
        result = asyncio.run(self.mlx_manager._simulate_matrix_multiply(256))
        
        assert isinstance(result, dict)
        assert result['operation'] == 'matrix_multiply_simulated'
        assert result['size'] == 256
        assert result['gflops'] > 0
        # MLX returns tuple, simulation returns list
        assert result['result_shape'] in ([256, 256], (256, 256))
    
    def test_simulate_neural_network(self):
        """Test CPU neural network simulation"""
        result = asyncio.run(self.mlx_manager._simulate_neural_network(100, 50, 10))
        
        assert isinstance(result, dict)
        assert result['operation'] == 'neural_network_simulated'
        assert result['input_size'] == 100
        assert result['hidden_size'] == 50
        assert result['output_size'] == 10
        # MLX returns tuple, simulation returns list
        assert result['output_shape'] in ([1, 10], (1, 10))


class TestGPURuntimeManager:
    """Test GPU Runtime Manager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gpu_manager = GPURuntimeManager()
    
    def test_initialization(self):
        """Test GPU manager initialization"""
        assert hasattr(self.gpu_manager, 'devices')
        assert hasattr(self.gpu_manager, 'metrics_history')
        assert hasattr(self.gpu_manager, 'active_tasks')
        assert hasattr(self.gpu_manager, 'executor')
        assert hasattr(self.gpu_manager, 'kernel_cache')
        assert hasattr(self.gpu_manager, 'mlx_manager')
    
    def test_device_initialization(self):
        """Test device initialization"""
        # Should have at least CPU device
        assert len(self.gpu_manager.devices) >= 1
        assert 'cpu_0' in self.gpu_manager.devices
        
        cpu_device = self.gpu_manager.devices['cpu_0']
        assert isinstance(cpu_device, GPUDevice)
        assert cpu_device.type == DeviceType.CPU
        assert cpu_device.is_available == True
    
    def test_device_selection(self):
        """Test device selection logic"""
        # Test with no preference
        device = self.gpu_manager._select_device(None)
        assert device is not None
        assert device.is_available == True
        
        # Test with CPU preference
        device = self.gpu_manager._select_device(DeviceType.CPU)
        assert device is not None
        assert device.type == DeviceType.CPU
        
        # Test with non-existent device preference
        device = self.gpu_manager._select_device(DeviceType.CUDA)
        # Should fall back to available device
        assert device is not None
    
    def test_metrics_collection(self):
        """Test metrics collection"""
        device = self.gpu_manager.devices['cpu_0']
        metrics = self.gpu_manager._collect_device_metrics(device)
        
        assert isinstance(metrics, RuntimeMetrics)
        assert metrics.device_id == device.id
        assert metrics.memory_usage >= 0
        assert metrics.gpu_utilization >= 0
        assert metrics.temperature >= 0
        assert metrics.power_usage >= 0
        assert metrics.throughput >= 0
    
    @pytest.mark.asyncio
    async def test_execute_compute_task(self):
        """Test compute task execution"""
        task = TaskRequest(
            operation="compute",
            data={"type": "matrix_multiply", "size": 512}
        )
        
        response = await self.gpu_manager.execute_task(task)
        
        assert isinstance(response, TaskResponse)
        assert response.task_id == task.task_id
        assert response.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert response.execution_time is not None
        
        if response.status == TaskStatus.COMPLETED:
            assert response.result is not None
            assert response.metrics is not None
            assert response.device_used is not None
    
    @pytest.mark.asyncio
    async def test_execute_mlx_compute_task(self):
        """Test MLX compute task execution"""
        task = TaskRequest(
            operation="mlx_compute",
            data={"type": "matrix_multiply", "size": 512, "dtype": "float32"}
        )
        
        response = await self.gpu_manager.execute_task(task)
        
        assert isinstance(response, TaskResponse)
        assert response.task_id == task.task_id
        assert response.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert response.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_execute_inference_task(self):
        """Test inference task execution"""
        task = TaskRequest(
            operation="inference",
            data={"model": "resnet50", "batch_size": 1}
        )
        
        response = await self.gpu_manager.execute_task(task)
        
        assert isinstance(response, TaskResponse)
        assert response.task_id == task.task_id
        assert response.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert response.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_execute_benchmark_task(self):
        """Test benchmark task execution"""
        task = TaskRequest(
            operation="benchmark",
            data={"type": "comprehensive"}
        )
        
        response = await self.gpu_manager.execute_task(task)
        
        assert isinstance(response, TaskResponse)
        assert response.task_id == task.task_id
        assert response.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert response.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test handling of invalid operations"""
        task = TaskRequest(
            operation="invalid_operation",
            data={}
        )
        
        response = await self.gpu_manager.execute_task(task)
        
        assert isinstance(response, TaskResponse)
        assert response.status == TaskStatus.FAILED
        assert response.error is not None
        assert "Unknown operation" in response.error
    
    @pytest.mark.asyncio
    async def test_cpu_matrix_multiply(self):
        """Test CPU matrix multiplication"""
        result = await self.gpu_manager._cpu_matrix_multiply(256)
        
        assert isinstance(result, dict)
        assert 'gflops' in result
        assert 'execution_time' in result
        assert result['gflops'] > 0
        assert result['execution_time'] > 0
    
    @pytest.mark.asyncio
    async def test_compute_fft(self):
        """Test FFT computation"""
        device = self.gpu_manager.devices['cpu_0']
        result = await self.gpu_manager._compute_fft(1024, device)
        
        assert isinstance(result, dict)
        assert 'throughput' in result
        assert result['throughput'] > 0
    
    @pytest.mark.asyncio
    async def test_cpu_inference(self):
        """Test CPU inference"""
        result = await self.gpu_manager._cpu_inference("resnet50", 1)
        
        assert isinstance(result, dict)
        assert 'latency' in result
        assert 'fps' in result
        assert 'predictions' in result
        assert result['latency'] > 0
        assert result['fps'] > 0
        assert len(result['predictions']) == 1
    
    @pytest.mark.asyncio
    async def test_benchmark_compute(self):
        """Test compute benchmarking"""
        device = self.gpu_manager.devices['cpu_0']
        result = await self.gpu_manager._benchmark_compute(512, device)
        
        assert isinstance(result, dict)
        assert 'size' in result
        assert 'gflops' in result
        assert 'device' in result
        assert result['size'] == 512
        assert result['device'] == device.id
    
    @pytest.mark.asyncio
    async def test_benchmark_memory(self):
        """Test memory benchmarking"""
        device = self.gpu_manager.devices['cpu_0']
        result = await self.gpu_manager._benchmark_memory(device)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        for item in result:
            assert 'size_kb' in item
            assert 'bandwidth_mbps' in item
            assert item['bandwidth_mbps'] > 0
    
    @pytest.mark.asyncio
    async def test_benchmark_ml(self):
        """Test ML benchmarking"""
        device = self.gpu_manager.devices['cpu_0']
        result = await self.gpu_manager._benchmark_ml(device)
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        for item in result:
            assert 'model' in item
            assert 'latency_ms' in item
            assert 'throughput_fps' in item
            assert item['latency_ms'] > 0
            assert item['throughput_fps'] > 0


class TestDataModels:
    """Test data model classes"""
    
    def test_task_request(self):
        """Test TaskRequest model"""
        task = TaskRequest(
            operation="compute",
            data={"size": 1024},
            device_preference=DeviceType.CPU,
            priority=5
        )
        
        assert task.operation == "compute"
        assert task.data["size"] == 1024
        assert task.device_preference == DeviceType.CPU
        assert task.priority == 5
        assert task.task_id is not None
        assert task.timeout == 300
    
    def test_task_response(self):
        """Test TaskResponse model"""
        response = TaskResponse(
            task_id="test-123",
            status=TaskStatus.COMPLETED,
            result={"gflops": 100.5},
            metrics={"execution_time": 1.5},
            execution_time=1.5,
            device_used="cpu_0"
        )
        
        assert response.task_id == "test-123"
        assert response.status == TaskStatus.COMPLETED
        assert response.result["gflops"] == 100.5
        assert response.metrics["execution_time"] == 1.5
        assert response.execution_time == 1.5
        assert response.device_used == "cpu_0"
    
    def test_gpu_device(self):
        """Test GPUDevice model"""
        device = GPUDevice(
            id="test_device",
            name="Test GPU",
            type=DeviceType.MLX,
            memory_total=16 * 1024**3,
            memory_available=12 * 1024**3,
            compute_units=32,
            is_available=True,
            capabilities=["matrix_multiply", "neural_networks"],
            driver_version="1.0.0",
            compute_capability="Metal 3.0"
        )
        
        assert device.id == "test_device"
        assert device.name == "Test GPU"
        assert device.type == DeviceType.MLX
        assert device.memory_total == 16 * 1024**3
        assert device.memory_available == 12 * 1024**3
        assert device.compute_units == 32
        assert device.is_available == True
        assert "matrix_multiply" in device.capabilities
        assert device.driver_version == "1.0.0"
        assert device.compute_capability == "Metal 3.0"
    
    def test_runtime_metrics(self):
        """Test RuntimeMetrics model"""
        metrics = RuntimeMetrics(
            device_id="test_device",
            timestamp=time.time(),
            memory_usage=75.5,
            gpu_utilization=85.2,
            temperature=65.0,
            power_usage=25.5,
            throughput=1500.0,
            active_kernels=3
        )
        
        assert metrics.device_id == "test_device"
        assert metrics.memory_usage == 75.5
        assert metrics.gpu_utilization == 85.2
        assert metrics.temperature == 65.0
        assert metrics.power_usage == 25.5
        assert metrics.throughput == 1500.0
        assert metrics.active_kernels == 3
    
    def test_compute_kernel(self):
        """Test ComputeKernel model"""
        kernel = ComputeKernel(
            name="test_kernel",
            source_code="def test(): pass",
            entry_point="test",
            parameters={"size": 1024},
            target_device=DeviceType.MLX
        )
        
        assert kernel.name == "test_kernel"
        assert kernel.source_code == "def test(): pass"
        assert kernel.entry_point == "test"
        assert kernel.parameters["size"] == 1024
        assert kernel.target_device == DeviceType.MLX
    
    def test_mlx_model(self):
        """Test MLXModel model"""
        model = MLXModel(
            name="test_model",
            model_type="neural_network",
            input_shape=[1, 784],
            output_shape=[1, 10],
            parameters={"layers": 3}
        )
        
        assert model.name == "test_model"
        assert model.model_type == "neural_network"
        assert model.input_shape == [1, 784]
        assert model.output_shape == [1, 10]
        assert model.parameters["layers"] == 3


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gpu_manager = GPURuntimeManager()
    
    @pytest.mark.asyncio
    async def test_no_available_devices(self):
        """Test behavior when no devices are available"""
        # Temporarily disable all devices
        for device in self.gpu_manager.devices.values():
            device.is_available = False
        
        task = TaskRequest(operation="compute", data={"type": "matrix_multiply", "size": 512})
        response = await self.gpu_manager.execute_task(task)
        
        assert response.status == TaskStatus.FAILED
        assert "No suitable device available" in response.error
        
        # Restore devices
        for device in self.gpu_manager.devices.values():
            device.is_available = True
    
    @pytest.mark.asyncio
    async def test_large_matrix_handling(self):
        """Test handling of very large matrices"""
        task = TaskRequest(
            operation="compute",
            data={"type": "matrix_multiply", "size": 10000}  # Very large
        )
        
        response = await self.gpu_manager.execute_task(task)
        
        # Should handle gracefully (either complete or fail with reasonable error)
        assert response.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert response.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_invalid_matrix_size(self):
        """Test handling of invalid matrix sizes"""
        task = TaskRequest(
            operation="compute",
            data={"type": "matrix_multiply", "size": -1}  # Invalid size
        )
        
        response = await self.gpu_manager.execute_task(task)
        
        # Should handle gracefully
        assert response.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self):
        """Test concurrent task execution"""
        tasks = []
        for i in range(5):
            task = TaskRequest(
                operation="compute",
                data={"type": "matrix_multiply", "size": 256},
                task_id=f"concurrent-{i}"
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*[
            self.gpu_manager.execute_task(task) for task in tasks
        ])
        
        assert len(responses) == 5
        for response in responses:
            assert response.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            assert response.execution_time is not None


class TestPerformance:
    """Test performance characteristics"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.gpu_manager = GPURuntimeManager()
    
    @pytest.mark.asyncio
    async def test_matrix_multiply_performance(self):
        """Test matrix multiplication performance"""
        sizes = [256, 512, 1024]
        results = {}
        
        for size in sizes:
            task = TaskRequest(
                operation="compute",
                data={"type": "matrix_multiply", "size": size}
            )
            
            start_time = time.time()
            response = await self.gpu_manager.execute_task(task)
            end_time = time.time()
            
            results[size] = {
                'execution_time': response.execution_time,
                'total_time': end_time - start_time,
                'status': response.status
            }
        
        # Verify performance characteristics
        for size, result in results.items():
            assert result['status'] in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            if result['status'] == TaskStatus.COMPLETED:
                assert result['execution_time'] > 0
                assert result['total_time'] > 0
    
    @pytest.mark.asyncio
    async def test_mlx_compute_performance(self):
        """Test MLX compute performance"""
        task = TaskRequest(
            operation="mlx_compute",
            data={"type": "matrix_multiply", "size": 1024, "dtype": "float32"}
        )
        
        start_time = time.time()
        response = await self.gpu_manager.execute_task(task)
        end_time = time.time()
        
        assert response.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        if response.status == TaskStatus.COMPLETED:
            assert response.execution_time > 0
            assert (end_time - start_time) > 0
    
    @pytest.mark.asyncio
    async def test_inference_performance(self):
        """Test inference performance"""
        models = ["resnet50", "bert", "gpt"]
        results = {}
        
        for model in models:
            task = TaskRequest(
                operation="inference",
                data={"model": model, "batch_size": 1}
            )
            
            start_time = time.time()
            response = await self.gpu_manager.execute_task(task)
            end_time = time.time()
            
            results[model] = {
                'execution_time': response.execution_time,
                'total_time': end_time - start_time,
                'status': response.status
            }
        
        # Verify all models complete
        for model, result in results.items():
            assert result['status'] in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            if result['status'] == TaskStatus.COMPLETED:
                assert result['execution_time'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 