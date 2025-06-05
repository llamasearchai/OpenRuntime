//! Error types for the OpenAI client

use thiserror::Error;

/// Main error type for OpenAI operations
#[derive(Error, Debug)]
pub enum OpenAIError {
    /// HTTP request error
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    /// API error response
    #[error("API error (status {status}): {message}")]
    ApiError {
        status: u16,
        message: String,
    },

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Rate limit exceeded
    #[error("Rate limit exceeded. Retry after: {retry_after:?}")]
    RateLimitExceeded {
        retry_after: Option<std::time::Duration>,
    },

    /// Authentication error
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// GPU runtime error
    #[error("GPU runtime error: {0}")]
    GpuRuntimeError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Timeout error
    #[error("Request timed out")]
    TimeoutError,

    /// Invalid input error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),
}

/// Result type alias for OpenAI operations
pub type Result<T> = std::result::Result<T, OpenAIError>;

impl OpenAIError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            OpenAIError::RequestError(_) => true,
            OpenAIError::ApiError { status, .. } => {
                // Retry on server errors and some client errors
                *status >= 500 || *status == 429 || *status == 408
            }
            OpenAIError::TimeoutError => true,
            OpenAIError::NetworkError(_) => true,
            _ => false,
        }
    }

    /// Get retry delay suggestion
    pub fn retry_delay(&self) -> Option<std::time::Duration> {
        match self {
            OpenAIError::RateLimitExceeded { retry_after } => *retry_after,
            OpenAIError::ApiError { status: 429, .. } => {
                Some(std::time::Duration::from_secs(60))
            }
            _ if self.is_retryable() => Some(std::time::Duration::from_secs(1)),
            _ => None,
        }
    }
}