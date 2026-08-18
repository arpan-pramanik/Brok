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

    let context_str = context_chunks
        .iter()
        .map(|c| format!("- {}", c))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are a strict, helpful AI assistant.\n\
        Answer the User Query STRICTLY AND ONLY using the provided Context.\n\
        Keep your answer extremely concise (1 short sentence, under 20 words).\n\
        If the Context does not contain the answer to the User Query (e.g. if the user asks an unrelated question like which game has spiderman in it), you MUST reply EXACTLY with:\n\
        sorry i dont have any information regarding that.\n\
        Do not add any other text, apologies, or explanations. Do not use outside knowledge.\n\n\
        Context:\n\
        {}\n\n\
        User Query: {}",
        context_str, query
    )
}

#[derive(Clone)]
enum LlmBackend {
    Ollama {
        client: Client,
        base_url: String,
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
        let ollama_host = env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        println!("Using Ollama Local ({})", ollama_host);
        LlmBackend::Ollama {
            client: Client::new(),
            base_url: ollama_host,
            model: "llama3.2:latest".to_string(),
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
        LlmBackend::Ollama { client, base_url, model } => {
            let max_tokens = req.max_tokens;
            tokio::spawn(async move {
                let payload = serde_json::json!({
                    "model": model,
                    "prompt": prompt,
                    "stream": true,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_tokens.min(15),
                        "num_ctx": 512,
                        "num_thread": 16,
                        "keep_alive": "60m"
                    }
                });

                if let Ok(resp) = client.post(format!("{}/api/generate", base_url))
                    .json(&payload)
                    .send()
                    .await 
                {
                    let mut stream = resp.bytes_stream();
                    while let Some(chunk_res) = stream.next().await {
                        if let Ok(bytes) = chunk_res {
                            if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                                for line in text.lines() {
                                    if line.trim().is_empty() { continue; }
                                    if let Ok(data) = serde_json::from_str::<serde_json::Value>(line) {
                                        if let Some(resp) = data.get("response").and_then(|r| r.as_str()) {
                                            if !resp.is_empty() {
                                                let _ = tx.send(Ok(Event::default().data(resp))).await;
                                            }
                                        }
                                        if data.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                                            break;
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
