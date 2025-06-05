//! # OpenAI Runtime RS
//!
//! A high-performance Rust client for OpenAI API with GPU runtime integration.
//!
//! ## Features
//!
//! - Async/await support with Tokio
//! - GPU-accelerated inference
//! - Streaming responses
//! - Comprehensive error handling
//! - Performance monitoring
//! - Thread-safe design
//!
//! ## Quick Start
//!
//! ,no_run
//! use openai_runtime_rs::{OpenAIClient, ChatRequest, Message, Role};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = OpenAIClient::new("your-api-key")?;
//!     
//!     let request = ChatRequest::new("gpt-4o-mini")
//!         .add_message(Message::user("Hello, world!"));
//!     
//!     let response = client.chat_completion(request).await?;
//!     println!("Response: {}", response.content());
//!     
//!     Ok(())
//! }
//! 

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Semaphore;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod gpu;
pub mod metrics;
pub mod streaming;

pub use chat::*;
pub use completions::*;
pub use embeddings::*;
pub use error::*;
pub use gpu::*;
pub use metrics::*;
pub use streaming::*;

/// OpenAI API base URL
const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// Default timeout for API requests
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Maximum concurrent requests
const MAX_CONCURRENT_REQUESTS: usize = 10;

/// OpenAI API client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// API key for authentication
    pub api_key: String,
    /// Base URL for API requests
    pub base_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Enable GPU acceleration
    pub gpu_acceleration: bool,
    /// Organization ID
    pub organization: Option<String>,
}

impl ClientConfig {
    /// Create a new client configuration
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: OPENAI_API_BASE.to_string(),
            timeout: DEFAULT_TIMEOUT,
            max_concurrent_requests: MAX_CONCURRENT_REQUESTS,
            gpu_acceleration: false,
            organization: None,
        }
    }

    /// Set the base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable GPU acceleration
    pub fn with_gpu_acceleration(mut self, enabled: bool) -> Self {
        self.gpu_acceleration = enabled;
        self
    }

    /// Set organization ID
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }
}

/// Main OpenAI client
#[derive(Debug)]
pub struct OpenAIClient {
    /// HTTP client
    client: Client,
    /// Configuration
    config: ClientConfig,
    /// Request semaphore for rate limiting
    semaphore: Arc<Semaphore>,
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
    /// GPU runtime manager
    gpu_manager: Option<Arc<GpuRuntimeManager>>,
    /// Request cache
    cache: Arc<DashMap<String, CachedResponse>>,
}

/// Cached response data
#[derive(Debug, Clone)]
struct CachedResponse {
    data: String,
    timestamp: DateTime<Utc>,
    ttl: Duration,
}

impl OpenAIClient {
    /// Create a new OpenAI client
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        let config = ClientConfig::new(api_key);
        Self::with_config(config)
    }

    /// Create a new client with custom configuration
    pub fn with_config(config: ClientConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", config.api_key).parse()?,
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse()?,
        );

        if let Some(org) = &config.organization {
            headers.insert("OpenAI-Organization", org.parse()?);
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()
            .context("Failed to create HTTP client")?;

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));
        let metrics = Arc::new(MetricsCollector::new());
        
        let gpu_manager = if config.gpu_acceleration {
            Some(Arc::new(GpuRuntimeManager::new()?))
        } else {
            None
        };

        Ok(Self {
            client,
            config,
            semaphore,
            metrics,
            gpu_manager,
            cache: Arc::new(DashMap::new()),
        })
    }

    /// Execute a chat completion request
    #[instrument(skip(self, request))]
    pub async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse> {
        let _permit = self.semaphore.acquire().await?;
        let start_time = Instant::now();

        // Check cache first
        let cache_key = format!("chat:{}", request.cache_key());
        if let Some(cached) = self.get_cached_response(&cache_key) {
            debug!("Cache hit for chat completion");
            return Ok(serde_json::from_str(&cached.data)?);
        }

        let url = format!("{}/chat/completions", self.config.base_url);
        
        debug!("Sending chat completion request to: {}", url);
        
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let response = self.handle_response(response).await?;
        let chat_response: ChatResponse = response.json().await?;

        // Cache the response
        self.cache_response(cache_key, &chat_response, Duration::from_secs(300));

        // Record metrics
        let duration = start_time.elapsed();
        self.metrics.record_request("chat_completion", duration, true);

        info!("Chat completion completed in {:?}", duration);
        
        Ok(chat_response)
    }

    /// Execute a completion request
    #[instrument(skip(self, request))]
    pub async fn completion(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let _permit = self.semaphore.acquire().await?;
        let start_time = Instant::now();

        let url = format!("{}/completions", self.config.base_url);
        
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let response = self.handle_response(response).await?;
        let completion_response: CompletionResponse = response.json().await?;

        let duration = start_time.elapsed();
        self.metrics.record_request("completion", duration, true);

        Ok(completion_response)
    }

    /// Execute an embeddings request
    #[instrument(skip(self, request))]
    pub async fn embeddings(&self, request: EmbeddingsRequest) -> Result<EmbeddingsResponse> {
        let _permit = self.semaphore.acquire().await?;
        let start_time = Instant::now();

        // Use GPU acceleration if available
        if let Some(gpu_manager) = &self.gpu_manager {
            if gpu_manager.can_accelerate_embeddings() {
                debug!("Using GPU acceleration for embeddings");
                return gpu_manager.compute_embeddings(request).await;
            }
        }

        let url = format!("{}/embeddings", self.config.base_url);
        
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let response = self.handle_response(response).await?;
        let embeddings_response: EmbeddingsResponse = response.json().await?;

        let duration = start_time.elapsed();
        self.metrics.record_request("embeddings", duration, true);

        Ok(embeddings_response)
    }

    /// Stream a chat completion
    #[instrument(skip(self, request))]
    pub async fn stream_chat_completion(
        &self,
        request: ChatRequest,
    ) -> Result<impl futures::Stream<Item = Result<ChatStreamResponse>>> {
        let _permit = self.semaphore.acquire().await?;
        
        let mut request = request;
        request.stream = Some(true);

        let url = format!("{}/chat/completions", self.config.base_url);
        
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        let response = self.handle_response(response).await?;
        
        Ok(StreamingResponse::new(response, self.metrics.clone()))
    }

    /// Get client metrics
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Clear the response cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Handle HTTP response and check for errors
    async fn handle_response(&self, response: Response) -> Result<Response> {
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            
            error!("API request failed with status {}: {}", status, error_text);
            
            return Err(OpenAIError::ApiError {
                status: status.as_u16(),
                message: error_text,
            }.into());
        }

        Ok(response)
    }

    /// Get cached response if valid
    fn get_cached_response(&self, key: &str) -> Option<CachedResponse> {
        self.cache.get(key).and_then(|entry| {
            let cached = entry.value();
            if Utc::now().signed_duration_since(cached.timestamp) < cached.ttl.into() {
                Some(cached.clone())
            } else {
                drop(entry);
                self.cache.remove(key);
                None
            }
        })
    }

    /// Cache a response
    fn cache_response<T: Serialize>(&self, key: String, data: &T, ttl: Duration) {
        if let Ok(json) = serde_json::to_string(data) {
            let cached = CachedResponse {
                data: json,
                timestamp: Utc::now(),
                ttl,
            };
            self.cache.insert(key, cached);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_client_creation() {
        let client = OpenAIClient::new("test-key");
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_config_builder() {
        let config = ClientConfig::new("test-key")
            .with_timeout(Duration::from_secs(30))
            .with_gpu_acceleration(true)
            .with_organization("test-org");

        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.gpu_acceleration);
        assert_eq!(config.organization, Some("test-org".to_string()));
    }
}