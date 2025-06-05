//! Text completion functionality

use serde::{Deserialize, Serialize};

/// Completion request
#[derive(Debug, Clone, Serialize)]
pub struct CompletionRequest {
    /// Model to use
    pub model: String,
    /// Prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Temperature for randomness
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// Completion response
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionResponse {
    /// Unique identifier
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used
    pub model: String,
    /// Completion choices
    pub choices: Vec<CompletionChoice>,
    /// Token usage
    pub usage: crate::chat::Usage,
}

/// Completion choice
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionChoice {
    /// Generated text
    pub text: String,
    /// Choice index
    pub index: u32,
    /// Log probabilities
    pub logprobs: Option<serde_json::Value>,
    /// Finish reason
    pub finish_reason: Option<String>,
}

impl CompletionRequest {
    /// Create a new completion request
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
        }
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

impl CompletionResponse {
    /// Get the text from the first choice
    pub fn text(&self) -> Option<&str> {
        self.choices.first().map(|choice| choice.text.as_str())
    }
}