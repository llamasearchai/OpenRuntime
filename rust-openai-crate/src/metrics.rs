//! Metrics collection and monitoring

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    /// Request counters
    request_counts: Arc<RwLock<std::collections::HashMap<String, AtomicU64>>>,
    /// Success counters
    success_counts: Arc<RwLock<std::collections::HashMap<String, AtomicU64>>>,
    /// Error counters
    error_counts: Arc<RwLock<std::collections::HashMap<String, AtomicU64>>>,
    /// Response time tracking
    response_times: Arc<RwLock<std::collections::HashMap<String, Vec<Duration>>>>,
    /// Start time
    start_time: Instant,
}

/// Metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Request metrics by operation
    pub operations: std::collections::HashMap<String, OperationMetrics>,
    /// Overall statistics
    pub overall: OverallMetrics,
}

/// Operation-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Minimum response time in milliseconds
    pub min_response_time_ms: f64,
    /// Maximum response time in milliseconds
    pub max_response_time_ms: f64,
    /// 95th percentile response time in milliseconds
    pub p95_response_time_ms: f64,
}

/// Overall metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallMetrics {
    /// Total requests across all operations
    pub total_requests: u64,
    /// Total successful requests
    pub total_successful: u64,
    /// Total failed requests
    pub total_failed: u64,
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Requests per second
    pub requests_per_second: f64,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            request_counts: Arc::new(RwLock::new(std::collections::HashMap::new())),
            success_counts: Arc::new(RwLock::new(std::collections::HashMap::new())),
            error_counts: Arc::new(RwLock::new(std::collections::HashMap::new())),
            response_times: Arc::new(RwLock::new(std::collections::HashMap::new())),
            start_time: Instant::now(),
        }
    }

    /// Record a request
    pub fn record_request(&self, operation: &str, duration: Duration, success: bool) {
        // Update request count
        {
            let mut counts = self.request_counts.write();
            counts.entry(operation.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }

        // Update success/error counts
        if success {
            let mut counts = self.success_counts.write();
            counts.entry(operation.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        } else {
            let mut counts = self.error_counts.write();
            counts.entry(operation.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(1, Ordering::Relaxed);
        }

        // Update response times (keep last 1000 entries per operation)
        {
            let mut times = self.response_times.write();
            let operation_times = times.entry(operation.to_string()).or_insert_with(Vec::new);
            operation_times.push(duration);
            if operation_times.len() > 1000 {
                operation_times.drain(..operation_times.len() - 1000);
            }
        }
    }

    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        let uptime = self.start_time.elapsed();
        let mut operations = std::collections::HashMap::new();
        
        let request_counts = self.request_counts.read();
        let success_counts = self.success_counts.read();
        let error_counts = self.error_counts.read();
        let response_times = self.response_times.read();

        let mut total_requests = 0u64;
        let mut total_successful = 0u64;
        let mut total_failed = 0u64;

        for (operation, request_count) in request_counts.iter() {
            let requests = request_count.load(Ordering::Relaxed);
            let successes = success_counts.get(operation)
                .map(|c| c.load(Ordering::Relaxed))
                .unwrap_or(0);
            let failures = error_counts.get(operation)
                .map(|c| c.load(Ordering::Relaxed))
                .unwrap_or(0);

            total_requests += requests;
            total_successful += successes;
            total_failed += failures;

            let success_rate = if requests > 0 {
                successes as f64 / requests as f64
            } else {
                0.0
            };

            let times = response_times.get(operation).cloned().unwrap_or_default();
            let (avg_ms, min_ms, max_ms, p95_ms) = if !times.is_empty() {
                let mut sorted_times: Vec<f64> = times.iter()
                    .map(|d| d.as_secs_f64() * 1000.0)
                    .collect();
                sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let avg = sorted_times.iter().sum::<f64>() / sorted_times.len() as f64;
                let min = sorted_times[0];
                let max = sorted_times[sorted_times.len() - 1];
                let p95_idx = (sorted_times.len() as f64 * 0.95) as usize;
                let p95 = sorted_times.get(p95_idx).copied().unwrap_or(max);

                (avg, min, max, p95)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

            operations.insert(operation.clone(), OperationMetrics {
                total_requests: requests,
                successful_requests: successes,
                failed_requests: failures,
                success_rate,
                avg_response_time_ms: avg_ms,
                min_response_time_ms: min_ms,
                max_response_time_ms: max_ms,
                p95_response_time_ms: p95_ms,
            });
        }

        let overall_success_rate = if total_requests > 0 {
            total_successful as f64 / total_requests as f64
        } else {
            0.0
        };

        let requests_per_second = if uptime.as_secs() > 0 {
            total_requests as f64 / uptime.as_secs() as f64
        } else {
            0.0
        };

        let overall = OverallMetrics {
            total_requests,
            total_successful,
            total_failed,
            overall_success_rate,
            requests_per_second,
        };

        MetricsSnapshot {
            uptime_seconds: uptime.as_secs(),
            operations,
            overall,
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.request_counts.write().clear();
        self.success_counts.write().clear();
        self.error_counts.write().clear();
        self.response_times.write().clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}