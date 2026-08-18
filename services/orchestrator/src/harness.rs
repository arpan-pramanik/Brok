use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Instant;
use reqwest::Client;
use std::sync::Arc;
use crate::guardrails::{GuardrailEngine, GuardrailDecision};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCallRecord {
    pub tool_name: String,
    pub status: String, // "success" | "retried" | "failed"
    pub execution_time_ms: f64,
    pub retry_count: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HarnessResponse {
    pub trace_id: String,
    pub query: String,
    pub answer: String,
    pub abstained: bool,
    pub abstention_reason: Option<String>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub sources: Vec<String>,
    pub total_time_ms: f64,
}

pub struct OrchestrationHarness {
    pub http_client: Client,
    pub retrieval_url: String,
    pub generation_url: String,
    pub asr_url: String,
    pub guardrails: Arc<GuardrailEngine>,
    pub abstain_threshold: f64,
}

impl OrchestrationHarness {
    pub fn new(
        http_client: Client,
        retrieval_url: String,
        generation_url: String,
        asr_url: String,
        guardrails: Arc<GuardrailEngine>,
        abstain_threshold: f64,
    ) -> Self {
        Self {
            http_client,
            retrieval_url,
            generation_url,
            asr_url,
            guardrails,
            abstain_threshold,
        }
    }

    /// Resilient Tool Call Helper with Automatic Retries
    pub async fn execute_tool_with_retry<F, Fut, T>(&self, tool_name: &str, mut action: F) -> (Option<T>, ToolCallRecord)
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
    {
        let start = Instant::now();
        let max_retries = 2;
        let mut retries = 0;

        while retries <= max_retries {
            match action().await {
                Ok(val) => {
                    let elapsed = start.elapsed().as_millis() as f64;
                    let status = if retries > 0 { "retried".to_string() } else { "success".to_string() };
                    return (
                        Some(val),
                        ToolCallRecord {
                            tool_name: tool_name.to_string(),
                            status,
                            execution_time_ms: elapsed,
                            retry_count: retries,
                        },
                    );
                }
                Err(_err) => {
                    retries += 1;
                    if retries <= max_retries {
                        tokio::time::sleep(tokio::time::Duration::from_millis(10 * retries as u64)).await;
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_millis() as f64;
        (
            None,
            ToolCallRecord {
                tool_name: tool_name.to_string(),
                status: "failed".to_string(),
                execution_time_ms: elapsed,
                retry_count: retries - 1,
            },
        )
    }
}
