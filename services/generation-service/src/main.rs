use axum::{
    extract::State,
    response::{sse::{Event, Sse}, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use futures::stream::StreamExt;
use serde::Deserialize;
use std::{convert::Infallible, env};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use reqwest::Client;
use aws_sdk_bedrockruntime::types::{Message, ContentBlock, ConversationRole};

#[derive(Debug, Deserialize)]
struct GenerationRequest {
    query: String,
    context_chunks: Vec<String>,
    #[serde(default = "default_max_tokens")]
    max_tokens: i32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    source_docs: Vec<String>,
}

fn default_max_tokens() -> i32 { 512 }
fn default_temperature() -> f32 { 0.3 }

fn format_prompt(query: &str, context_chunks: &[String], _source_docs: &[String]) -> String {
    if context_chunks.is_empty() {
        return "sorry i dont have any information regarding that.".to_string();
    }
    let ctx = context_chunks.join(" ");
    // /no_think disables qwen3's internal reasoning chain for fastest TTFT
    format!("Context: {}\nQuestion: {}\nAnswer concisely: /no_think", ctx, query)
}

#[derive(Clone)]
enum LlmBackend {
    Groq {
        client: Client,
        api_key: String,
        model: String,
    },
    AwsNova {
        client: aws_sdk_bedrockruntime::Client,
        model_id: String,
    },
}

#[derive(Clone)]
struct AppState {
    backend: LlmBackend,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let env = env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string());
    
    let backend = if env == "production" {
        println!("Using AWS Nova Micro (Production - us-east-1)");
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_config::Region::new("us-east-1"))
            .load()
            .await;
        let client = aws_sdk_bedrockruntime::Client::new(&config);
        LlmBackend::AwsNova {
            client,
            model_id: "amazon.nova-micro-v1:0".to_string(),
        }
    } else {
        let groq_api_key = env::var("GROQ_API_KEY").unwrap_or_default();
        let groq_model = env::var("GROQ_MODEL").unwrap_or_else(|_| "llama-3.2-3b-preview".to_string());
        println!("Using Groq LPU API ({})", groq_model);
        LlmBackend::Groq {
            client: Client::builder().tcp_nodelay(true).build().unwrap_or_default(),
            api_key: groq_api_key,
            model: groq_model,
        }
    };

    let state = AppState { backend };

    let app = Router::new()
        .route("/health", get(health))
        .route("/generate_stream", post(generate_stream))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8004").await?;
    println!("Listening on 0.0.0.0:8004");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn generate_stream(
    State(state): State<AppState>,
    Json(req): Json<GenerationRequest>,
) -> impl IntoResponse {
    let prompt = format_prompt(&req.query, &req.context_chunks, &req.source_docs);
    
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(100);

    match state.backend {
        LlmBackend::Groq { client, api_key, model } => {
            let max_tokens = req.max_tokens;
            tokio::spawn(async move {
                let payload = serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": true,
                    "max_tokens": max_tokens,
                    "temperature": 0.0
                });

                if let Ok(resp) = client.post("https://api.groq.com/openai/v1/chat/completions")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .header("Connection", "keep-alive")
                    .json(&payload)
                    .send()
                    .await 
                {
                    let mut stream = resp.bytes_stream();
                    while let Some(chunk_res) = stream.next().await {
                        if let Ok(bytes) = chunk_res {
                            if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                                for line in text.lines() {
                                    if let Some(data_str) = line.strip_prefix("data: ") {
                                        if data_str.trim() == "[DONE]" { break; }
                                        if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data_str) {
                                            if let Some(token) = chunk["choices"][0]["delta"]["content"].as_str() {
                                                if !token.is_empty() {
                                                    let _ = tx.send(Ok(Event::default().data(token))).await;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
        LlmBackend::AwsNova { client, model_id } => {
            let max_tokens = req.max_tokens;
            let temperature = req.temperature;
            tokio::spawn(async move {
                let message = Message::builder()
                    .role(ConversationRole::User)
                    .content(ContentBlock::Text(prompt.clone()))
                    .build()
                    .unwrap();

                let inference_config = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                    .max_tokens(max_tokens)
                    .temperature(temperature)
                    .build();

                match client.converse_stream()
                    .model_id(model_id)
                    .messages(message)
                    .inference_config(inference_config)
                    .send()
                    .await 
                {
                    Ok(mut resp) => {
                        while let Ok(Some(event)) = resp.stream.recv().await {
                            if let aws_sdk_bedrockruntime::types::ConverseStreamOutput::ContentBlockDelta(delta) = event {
                                if let Some(aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(text)) = delta.delta {
                                    let _ = tx.send(Ok(Event::default().data(text))).await;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Bedrock converse_stream error: {:?}", e);
                        let _ = tx.send(Ok(Event::default().data(format!("Service temporarily unavailable: {}", e)))).await;
                    }
                }
            });
        }
    }

    Sse::new(ReceiverStream::new(rx))
}
