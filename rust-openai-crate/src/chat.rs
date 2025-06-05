//! Chat completion functionality

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chat completion request
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    /// Model to use for completion
    pub model: String,
    /// List of messages
    pub messages: Vec<Message>,
    /// Temperature for randomness
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Function definitions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<Function>>,
    /// Function call behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender
    pub role: Role,
    /// Content of the message
    pub content: String,
    /// Optional name for the message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Function call information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCallData>,
}

/// Message role
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Function,
}

/// Function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    /// Name of the function
    pub name: String,
    /// Description of the function
    pub description: String,
    /// Parameters schema
    pub parameters: serde_json::Value,
}

/// Function call specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FunctionCall {
    /// No function calls
    None,
    /// Auto function calling
    Auto,
    /// Specific function name
    Name(String),
}

/// Function call data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallData {
    /// Function name
    pub name: String,
    /// Function arguments
    pub arguments: String,
}

/// Chat completion response
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    /// Unique identifier
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used
    pub model: String,
    /// Response choices
    pub choices: Vec<ChatChoice>,
    /// Token usage information
    pub usage: Usage,
}

/// Chat choice
#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    /// Choice index
    pub index: u32,
    /// Message content
    pub message: Message,
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// Token usage information
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    /// Prompt tokens
    pub prompt_tokens: u32,
    /// Completion tokens
    pub completion_tokens: u32,
    /// Total tokens
    pub total_tokens: u32,
}

/// Streaming chat response
#[derive(Debug, Clone, Deserialize)]
pub struct ChatStreamResponse {
    /// Unique identifier
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: u64,
    /// Model used
    pub model: String,
    /// Response choices
    pub choices: Vec<ChatStreamChoice>,
}

/// Streaming chat choice
#[derive(Debug, Clone, Deserialize)]
pub struct ChatStreamChoice {
    /// Choice index
    pub index: u32,
    /// Delta message
    pub delta: MessageDelta,
    /// Finish reason
    pub finish_reason: Option<String>,
}

/// Message delta for streaming
#[derive(Debug, Clone, Deserialize)]
pub struct MessageDelta {
    /// Role (only present in first chunk)
    pub role: Option<Role>,
    /// Content delta
    pub content: Option<String>,
    /// Function call delta
    pub function_call: Option<FunctionCallDelta>,
}

/// Function call delta
#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name
    pub name: Option<String>,
    /// Arguments delta
    pub arguments: Option<String>,
}

impl ChatRequest {
    /// Create a new chat request
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            stop: None,
            functions: None,
            function_call: None,
        }
    }

    /// Add a message to the request
    pub fn add_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set streaming
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Add function
    pub fn with_function(mut self, function: Function) -> Self {
        if self.functions.is_none() {
            self.functions = Some(Vec::new());
        }
        self.functions.as_mut().unwrap().push(function);
        self
    }

    /// Generate cache key for this request
    pub fn cache_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.model.hash(&mut hasher);
        
        for message in &self.messages {
            message.role.to_string().hash(&mut hasher);
            message.content.hash(&mut hasher);
        }
        
        if