//! Embeddings functionality

use serde::{Deserialize, Serialize};

/// Embeddings request
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingsRequest {
    /// Model to use for embeddings
    pub model: String,
    /// Input text or array of texts
    pub input: EmbeddingsInput,
    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Embeddings input
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    /// Single text input
    Single(String),
    /// Multiple text inputs
    Multiple(Vec<String>),
}

/// Embeddings response
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsResponse {
    /// Object type
    pub object: String,
    /// Embedding data
    pub data: Vec<EmbeddingData>,
    /// Model used
    pub model: String,
    /// Token usage
    pub usage: EmbeddingsUsage,
}

/// Individual embedding data
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingData {
    /// Object type
    pub object: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Index in the input array
    pub index: u32,
}

/// Token usage for embeddings
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingsUsage {
    /// Prompt tokens
    pub prompt_tokens: u32,
    /// Total tokens
    pub total_tokens: u32,
}

impl EmbeddingsRequest {
    /// Create a new embeddings request
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: EmbeddingsInput::Single(input.into()),
            user: None,
        }
    }

    /// Create embeddings request for multiple inputs
    pub fn new_multiple(model: impl Into<String>, inputs: Vec<String>) -> Self {
        Self {
            model: model.into(),
            input: EmbeddingsInput::Multiple(inputs),
            user: None,
        }
    }

    /// Set user identifier
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

impl EmbeddingsResponse {
    /// Get the first embedding vector
    pub fn embedding(&self) -> Option<&[f32]> {
        self.data.first().map(|d| d.embedding.as_slice())
    }

    /// Get all embedding vectors
    pub fn embeddings(&self) -> Vec<&[f32]> {
        self.data.iter().map(|d| d.embedding.as_slice()).collect()
    }
}