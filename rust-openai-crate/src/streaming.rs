//! Streaming response handling

use crate::{ChatStreamResponse, MetricsCollector, OpenAIError, Result};
use futures::{Stream, StreamExt};
use reqwest::Response;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{debug, error};

/// Streaming response wrapper
pub struct StreamingResponse {
    inner: Pin<Box<dyn Stream<Item = Result<ChatStreamResponse>> + Send>>,
}

impl StreamingResponse {
    /// Create a new streaming response
    pub fn new(response: Response, metrics: Arc<MetricsCollector>) -> Self {
        let stream = async_stream::stream! {
            let mut lines = BufReader::new(response.bytes_stream().map(|result| {
                result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
            }).into_async_read()).lines();

            while let Ok(Some(line)) = lines.next_line().await {
                if line.is_empty() {
                    continue;
                }

                if line == "data: [DONE]" {
                    debug!("Stream completed");
                    break;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    match serde_json::from_str::<ChatStreamResponse>(data) {
                        Ok(response) => yield Ok(response),
                        Err(e) => {
                            error!("Failed to parse streaming response: {}", e);
                            yield Err(OpenAIError::JsonError(e));
                        }
                    }
                }
            }
        };

        Self {
            inner: Box::pin(stream),
        }
    }
}

impl Stream for StreamingResponse {
    type Item = Result<ChatStreamResponse>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl futures::stream::FusedStream for StreamingResponse {
    fn is_terminated(&self) -> bool {
        false
    }
}