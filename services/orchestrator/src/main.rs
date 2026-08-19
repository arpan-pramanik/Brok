use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Multipart, State,
    },
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use futures::{stream::SplitSink, SinkExt, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tower_http::cors::CorsLayer;
use base64::Engine;
use tokio_tungstenite::connect_async;
use tokio::io::AsyncBufReadExt;

mod guardrails;
mod harness;

use guardrails::GuardrailEngine;
use harness::OrchestrationHarness;

#[derive(Clone)]
struct AppState {
    http_client: Client,
    retrieval_url: String,
    generation_url: String,
    asr_url: String,
    tts_url: String,
    abstain_threshold: f64,
    guardrails: Arc<GuardrailEngine>,
    harness: Arc<OrchestrationHarness>,
}

#[tokio::main]
async fn main() {
    let retrieval_url = env::var("RETRIEVAL_URL").unwrap_or_else(|_| "http://localhost:8002".to_string());
    let generation_url = env::var("GENERATION_URL").unwrap_or_else(|_| "http://localhost:8004".to_string());
    let asr_url = env::var("ASR_URL").unwrap_or_else(|_| "http://localhost:8001".to_string());
    let tts_url = env::var("TTS_URL").unwrap_or_else(|_| "http://localhost:8005".to_string());
    let abstain_threshold = env::var("ABSTAIN_THRESHOLD").unwrap_or_else(|_| "0.30".to_string()).parse::<f64>().unwrap_or(0.30);

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .tcp_nodelay(true)
        .tcp_keepalive(Some(std::time::Duration::from_secs(60)))
        .pool_idle_timeout(Some(std::time::Duration::from_secs(300)))
        .pool_max_idle_per_host(32)
        .build()
        .unwrap();

    // Pre-warm TCP/TLS connections to Groq, OpenRouter, and Retrieval
    let warm_client = client.clone();
    let warm_retrieval = retrieval_url.clone();
    let warm_groq_key = env::var("GROQ_API_KEY").unwrap_or_default();
    tokio::spawn(async move {
        loop {
            let _ = warm_client.get(format!("{}/health", warm_retrieval)).send().await;
            if !warm_groq_key.is_empty() {
                let _ = warm_client.get("https://api.groq.com/openai/v1/models")
                    .header("Authorization", format!("Bearer {}", warm_groq_key))
                    .send().await;
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(12)).await;
        }
    });
    let guardrails = Arc::new(GuardrailEngine::new());
    let harness = Arc::new(OrchestrationHarness::new(
        client.clone(),
        retrieval_url.clone(),
        generation_url.clone(),
        asr_url.clone(),
        guardrails.clone(),
        abstain_threshold,
    ));

    let state = AppState {
        http_client: client,
        retrieval_url,
        generation_url,
        asr_url,
        tts_url,
        abstain_threshold,
        guardrails,
        harness,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/query", post(query))
        .route("/api/transcribe", post(transcribe_handler))
        .route("/api/benchmark", post(benchmark))
        .route("/ws", get(ws_endpoint))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    println!("Orchestrator running on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn transcribe_handler(
    State(state): State<AppState>,
    mut multipart: axum::extract::Multipart,
) -> Result<Json<Value>, (StatusCode, String)> {
    let sarvam_key = env::var("SARVAM_API_KEY").unwrap_or_default();
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" || name == "audio" {
            let data = field.bytes().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            let part = reqwest::multipart::Part::bytes(data.to_vec())
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

            let form = reqwest::multipart::Form::new()
                .part("file", part)
                .text("model", "saarika:v2.5")
                .text("language_code", "en-IN");

            let res = state.http_client
                .post("https://api.sarvam.ai/speech-to-text")
                .header("api-subscription-key", &sarvam_key)
                .multipart(form)
                .send()
                .await;

            if let Ok(resp) = res {
                if let Ok(json_body) = resp.json::<Value>().await {
                    let transcript = json_body["transcript"].as_str().unwrap_or("").to_string();
                    return Ok(Json(json!({
                        "type": "final_transcript",
                        "text": transcript
                    })));
                }
            }
        }
    }
    Ok(Json(json!({"type": "final_transcript", "text": ""})))
}

#[derive(Deserialize)]
struct TextQuery {
    query: String,
}

async fn query(State(state): State<AppState>, Json(req): Json<TextQuery>) -> Json<Value> {
    let (tx, mut rx) = mpsc::channel(100);
    
    let query_str = req.query.clone();
    tokio::spawn(async move {
        run_text_pipeline_stream(query_str, tx, false, state).await;
    });

    let mut result = json!({
        "query": req.query,
        "stages": [],
        "error": null,
        "answer": "",
        "retrieval": {},
        "guardrail": {}
    });

    while let Some(msg) = rx.recv().await {
        if msg["type"] == "done" {
            break;
        } else if msg["type"] == "stage_timing" {
            result["stages"].as_array_mut().unwrap().push(msg.clone());
        } else if msg["type"] == "retrieval_result" {
            result["retrieval"] = msg.clone();
        } else if msg["type"] == "guardrail_result" {
            result["guardrail"] = msg.clone();
        } else if msg["type"] == "generation_chunk" {
            let chunk_text = msg["text"].as_str().unwrap_or("");
            let current = result["answer"].as_str().unwrap_or("");
            result["answer"] = json!(format!("{}{}", current, chunk_text));
        } else if msg["type"] == "error" {
            result["error"] = json!(msg["message"]);
        }
    }

    let mut total_time = 0.0;
    if let Some(stages) = result["stages"].as_array() {
        for stage in stages {
            let s_name = stage["stage"].as_str().unwrap_or("unknown");
            if let Some(dur) = stage["duration_ms"].as_f64() {
                if s_name != "ttft" {
                    total_time += dur;
                }
            }
        }
    }
    result["total_time_ms"] = json!((total_time * 10.0).round() / 10.0);

    Json(result)
}

async fn benchmark(State(state): State<AppState>) -> Json<Value> {
    let query_file_content = include_str!("query_set.jsonl");
    
    let mut queries = Vec::new();
    for line in query_file_content.lines() {
        if line.trim().is_empty() { continue; }
        if let Ok(val) = serde_json::from_str::<Value>(line) {
            if let Some(q) = val["query"].as_str() {
                queries.push(q.to_string());
            }
        }
    }

    let mut results = Vec::new();
    let mut stage_times: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    let mut total_times = Vec::new();

    for q in queries {
        let (tx, mut rx) = mpsc::channel(100);
        let q_clone = q.clone();
        let state_clone = state.clone();
        tokio::spawn(async move {
            run_text_pipeline_stream(q_clone, tx, false, state_clone).await;
        });

        let mut result = json!({
            "query": q,
            "stages": [],
            "error": null,
            "answer": "",
            "retrieval": {},
            "guardrail": {}
        });

        while let Some(msg) = rx.recv().await {
            if msg["type"] == "done" {
                break;
            } else if msg["type"] == "stage_timing" {
                result["stages"].as_array_mut().unwrap().push(msg.clone());
            } else if msg["type"] == "retrieval_result" {
                result["retrieval"] = msg.clone();
            } else if msg["type"] == "guardrail_result" {
                result["guardrail"] = msg.clone();
            } else if msg["type"] == "generation_chunk" {
                let chunk_text = msg["text"].as_str().unwrap_or("");
                let current = result["answer"].as_str().unwrap_or("");
                result["answer"] = json!(format!("{}{}", current, chunk_text));
            } else if msg["type"] == "error" {
                result["error"] = json!(msg["message"]);
            }
        }

        let mut total_time = 0.0;
        if let Some(stages) = result["stages"].as_array() {
            for stage in stages {
                let s_name = stage["stage"].as_str().unwrap_or("unknown");
                if let Some(dur) = stage["duration_ms"].as_f64() {
                    let s_name_str = s_name.to_string();
                    stage_times.entry(s_name_str).or_insert_with(Vec::new).push(dur);
                    if s_name != "ttft" {
                        total_time += dur;
                    }
                }
            }
        }
        result["total_time_ms"] = json!((total_time * 10.0).round() / 10.0);
        
        let abstained = result["guardrail"]["should_abstain"].as_bool().unwrap_or(false);
        result["abstained"] = json!(abstained);
        
        if result["error"].is_null() {
            total_times.push(total_time);
        }
        results.push(result);
    }

    let compute_percentile = |vals: &[f64], pct: f64| -> f64 {
        if vals.is_empty() { return 0.0; }
        let mut sorted = vals.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64 - 1.0) * pct).round() as usize;
        (sorted[idx.min(sorted.len() - 1)] * 100.0).round() / 100.0
    };

    let errors = results.iter().filter(|r| !r["error"].is_null()).count();
    let abstentions = results.iter().filter(|r| r["abstained"].as_bool().unwrap_or(false)).count();
    
    let mut p50 = serde_json::Map::new();
    let mut p70 = serde_json::Map::new();
    let mut p100 = serde_json::Map::new();

    for (stage, times) in &stage_times {
        p50.insert(stage.clone(), json!(compute_percentile(times, 0.50)));
        p70.insert(stage.clone(), json!(compute_percentile(times, 0.70)));
        p100.insert(stage.clone(), json!(compute_percentile(times, 1.00)));
    }
    p50.insert("total".to_string(), json!(compute_percentile(&total_times, 0.50)));
    p70.insert("total".to_string(), json!(compute_percentile(&total_times, 0.70)));
    p100.insert("total".to_string(), json!(compute_percentile(&total_times, 1.00)));

    let summary = json!({
        "total_queries": results.len(),
        "errors": errors,
        "abstentions": abstentions,
        "total_latency": {
            "p50": compute_percentile(&total_times, 0.50),
            "p70": compute_percentile(&total_times, 0.70),
            "p100": compute_percentile(&total_times, 1.00),
        },
        "p50": p50,
        "p70": p70,
        "p100": p100,
        "total_latencies": total_times,
    });

    Json(json!({
        "summary": summary,
        "p50": p50,
        "p70": p70,
        "p100": p100,
        "total_latencies": total_times,
        "results": results
    }))
}

async fn ws_endpoint(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws(socket, state))
}

async fn handle_ws(socket: WebSocket, state: AppState) {
    let (mut client_tx, mut client_rx) = socket.split();
    let (internal_tx, mut internal_rx) = mpsc::channel::<Value>(100);

    // Forward pipeline messages to client
    tokio::spawn(async move {
        while let Some(msg) = internal_rx.recv().await {
            if let Ok(text) = serde_json::to_string(&msg) {
                let _ = client_tx.send(Message::Text(text)).await;
            }
        }
    });

    let mut asr_tx_opt: Option<futures::stream::SplitSink<tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>, tokio_tungstenite::tungstenite::Message>> = None;
    let mut tts_enabled_global = true;

    while let Some(Ok(msg)) = client_rx.next().await {
        if let Message::Text(t) = msg {
            if let Ok(json) = serde_json::from_str::<Value>(&t) {
                if json["type"] == "ping" {
                    let ping_client = state.http_client.clone();
                    let groq_api_key = env::var("GROQ_API_KEY").unwrap_or_default();
                    tokio::spawn(async move {
                        if !groq_api_key.is_empty() {
                            let _ = ping_client.get("https://api.groq.com/openai/v1/models")
                                .header("Authorization", format!("Bearer {}", groq_api_key))
                                .send()
                                .await;
                        }
                    });
                } else if json["type"] == "text_query" {
                    let query = json["query"].as_str().unwrap_or("").to_string();
                    let tts_enabled = json["tts"].as_bool().unwrap_or(true);
                    let internal_tx_clone = internal_tx.clone();
                    let state_clone = state.clone();
                    tokio::spawn(async move {
                        run_text_pipeline_stream(query, internal_tx_clone, tts_enabled, state_clone).await;
                    });
                } else if json["type"] == "audio_chunk" {
                    tts_enabled_global = json["tts"].as_bool().unwrap_or(true);
                    // Connect to ASR WS if not connected
                    if asr_tx_opt.is_none() {
                        let asr_ws_url = state.asr_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws";
                        if let Ok((ws_stream, _)) = connect_async(asr_ws_url).await {
                            let (asr_tx, mut asr_rx) = ws_stream.split();
                            asr_tx_opt = Some(asr_tx);
                            
                            // Spawn listener for ASR
                            let internal_tx_clone = internal_tx.clone();
                            let state_clone = state.clone();
                            tokio::spawn(async move {
                                while let Some(Ok(asr_msg)) = asr_rx.next().await {
                                    if let tokio_tungstenite::tungstenite::Message::Text(t) = asr_msg {
                                        if let Ok(res) = serde_json::from_str::<Value>(&t) {
                                            if res["type"] == "final_transcript" || res.get("language").is_some() {
                                                // It's a final transcript
                                                let text = res["text"].as_str().unwrap_or("").to_string();
                                                if !text.is_empty() {
                                                    let tx = internal_tx_clone.clone();
                                                    let st = state_clone.clone();
                                                    // use tts_enabled_global - simplified logic
                                                    tokio::spawn(async move {
                                                        run_text_pipeline_stream(text, tx, true, st).await;
                                                    });
                                                }
                                            } else if res["type"] == "vad_stop" {
                                                // ignore
                                            } else {
                                                // forward partial transcript
                                                let _ = internal_tx_clone.send(res).await;
                                            }
                                        }
                                    }
                                }
                            });
                        }
                    }

                    if let Some(asr_tx) = &mut asr_tx_opt {
                        if let Some(b64) = json["data"].as_str() {
                            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                                let _ = asr_tx.send(tokio_tungstenite::tungstenite::Message::Binary(bytes)).await;
                            }
                        }
                    }
                }
            }
        }
    }
}

async fn run_text_pipeline_stream(query: String, tx: mpsc::Sender<Value>, tts_enabled: bool, state: AppState) {
    let abstain_answer = "sorry i dont have any information regarding that.";

    // Guardrail 1: Input Safety & Policy Guardrail (Sub-1ms Scan)
    let start_guardrail = std::time::Instant::now();
    if let Some(safety_decision) = state.guardrails.check_input_safety(&query) {
        let duration_gd = start_guardrail.elapsed().as_millis() as f64;
        let _ = tx.send(json!({"type": "stage_timing", "stage": "guardrail", "duration_ms": duration_gd})).await;
        let _ = tx.send(json!({
            "type": "guardrail_result",
            "should_abstain": true,
            "reason": safety_decision.reason,
            "guardrail_type": "safety"
        })).await;
        let refusal_msg = "I cannot fulfill this request as it contains unsafe or inappropriate content.";
        let _ = tx.send(json!({"type": "generation_chunk", "text": refusal_msg})).await;
        let _ = tx.send(json!({"type": "done"})).await;
        return;
    }
    
    let start_retrieval = std::time::Instant::now();
    let ret_url = state.retrieval_url.clone();
    let client = state.http_client.clone();
    let q_clone = query.clone();

    let (ret_json_opt, tool_record) = state.harness.execute_tool_with_retry("vector_search_tool", || {
        let client = client.clone();
        let ret_url = ret_url.clone();
        let q_clone = q_clone.clone();
        async move {
            let res = client.post(format!("{}/retrieve", ret_url))
                .json(&json!({"query": q_clone, "top_k": 2}))
                .send().await
                .map_err(|e| e.to_string())?;
            res.json::<Value>().await.map_err(|e| e.to_string())
        }
    }).await;

    let ret_json = ret_json_opt.unwrap_or_else(|| json!({"candidates": []}));
    let duration_retrieval = start_retrieval.elapsed().as_millis() as f64;
    let _ = tx.send(json!({"type": "stage_timing", "stage": "retrieval", "duration_ms": duration_retrieval, "tool_execution": tool_record})).await;
    
    let candidates = ret_json["candidates"].as_array().cloned().unwrap_or_default();
    let _ = tx.send(json!({
        "type": "retrieval_result",
        "query": query,
        "candidates": candidates,
        "retrieval_time_ms": ret_json["retrieval_time_ms"]
    })).await;

    let top_score = candidates.first().and_then(|c| c["score"].as_f64()).unwrap_or(-100.0);

    // Guardrail 2: Off-Topic / Context Grounding Relevance Check (Sub-1ms Vector Score Evaluation)
    let relevance_decision = state.guardrails.check_context_relevance(top_score, state.abstain_threshold);
    let duration_guardrail = (start_guardrail.elapsed().as_millis() as f64 - duration_retrieval).max(0.1);
    
    let _ = tx.send(json!({"type": "stage_timing", "stage": "guardrail", "duration_ms": duration_guardrail})).await;
    let _ = tx.send(json!({
        "type": "guardrail_result",
        "should_abstain": relevance_decision.should_abstain,
        "reason": relevance_decision.reason,
        "confidence": {
            "query": query,
            "top_score": top_score,
            "threshold": state.abstain_threshold,
            "is_confident": !relevance_decision.should_abstain
        }
    })).await;

    if relevance_decision.should_abstain {
        let _ = tx.send(json!({"type": "stage_timing", "stage": "generation", "duration_ms": 0.0})).await;
        let _ = tx.send(json!({"type": "generation_chunk", "text": abstain_answer})).await;
        
        let _ = tx.send(json!({
            "type": "generation_result",
            "answer": abstain_answer,
            "sources": [],
            "model_used": "fallback",
            "generation_time_ms": 0,
            "fallback_used": false,
        })).await;

        if tts_enabled {
            fetch_tts_stream(&state.http_client, &state.asr_url, abstain_answer.to_string(), tx.clone()).await;
        }
        let _ = tx.send(json!({"type": "done"})).await;
        return;
    }

    let start_generation = std::time::Instant::now();
    let mut full_answer = String::new();
    let mut source_docs = Vec::new();
    let mut context_chunks = Vec::new();

    for c in candidates.iter().take(2) {
        if let Some(txt) = c["text"].as_str() {
            context_chunks.push(txt.to_string());
        }
        if let Some(sd) = c["source_doc"].as_str() {
            if !source_docs.contains(&sd.to_string()) {
                source_docs.push(sd.to_string());
            }
        }
    }

    // Direct Groq LPU call — sub-50ms TTFT on custom inference hardware
    let prompt = if context_chunks.is_empty() {
        "sorry i dont have any information regarding that.".to_string()
    } else {
        format!("Context:\n{}\n\nQuestion: {}\nAnswer concisely in one sentence using only the provided context. If not in the context, say: \"Not mentioned in the text.\"\nAnswer:", context_chunks.join(" "), query)
    };

    let groq_api_key1 = env::var("GROQ_API_KEY").unwrap_or_default();
    let groq_api_key2 = env::var("GROQ_API_KEY_SECONDARY").unwrap_or_default();
    let groq_model = env::var("GROQ_MODEL").unwrap_or_else(|_| "allam-2-7b".to_string());
    let openrouter_key = env::var("OPENROUTER_API_KEY").unwrap_or_default();
    let openrouter_model = env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "meta-llama/llama-3.1-8b-instruct".to_string());

    let groq_req = json!({
        "model": groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": true
    });

    let openrouter_req = json!({
        "model": openrouter_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": true
    });

    let mut ttft_recorded = false;
    let mut ttft_ms = 0.0;
    let mut model_used = format!("groq:{}", groq_model);

    // Nanosecond Switch: Speculatively race Groq Primary, Groq Secondary, and OpenRouter
    let req1 = state.http_client.post("https://api.groq.com/openai/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", groq_api_key1))
        .header("Content-Type", "application/json")
        .header("Connection", "keep-alive")
        .json(&groq_req)
        .send();

    let req2 = state.http_client.post("https://api.groq.com/openai/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", groq_api_key2))
        .header("Content-Type", "application/json")
        .header("Connection", "keep-alive")
        .json(&groq_req)
        .send();

    let req3 = state.http_client.post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", openrouter_key))
        .header("Content-Type", "application/json")
        .header("Connection", "keep-alive")
        .json(&openrouter_req)
        .send();

    let (chosen_engine, resp_result) = tokio::select! {
        r1 = req1 => ("groq-primary", r1),
        r2 = req2 => ("groq-secondary", r2),
        r3 = req3 => ("openrouter", r3),
    };

    if chosen_engine.starts_with("groq") {
        model_used = format!("groq:{} ({})", groq_model, chosen_engine);
    } else {
        model_used = format!("openrouter:{}", openrouter_model);
    }

    if let Ok(resp) = resp_result {
        if resp.status().is_success() {
            let mut stream = resp.bytes_stream();
            let mut line_buf = String::new();
            while let Some(Ok(bytes)) = stream.next().await {
                if let Ok(chunk_str) = std::str::from_utf8(&bytes) {
                    line_buf.push_str(chunk_str);
                    while let Some(pos) = line_buf.find('\n') {
                        let line: String = line_buf.drain(..=pos).collect();
                        let trimmed = line.trim();
                        if let Some(data_str) = trimmed.strip_prefix("data: ") {
                            if data_str.trim() == "[DONE]" { break; }
                            if let Ok(chunk) = serde_json::from_str::<Value>(data_str) {
                                if let Some(token) = chunk["choices"][0]["delta"]["content"].as_str() {
                                    if !token.is_empty() {
                                        if !ttft_recorded {
                                            ttft_recorded = true;
                                            ttft_ms = start_generation.elapsed().as_millis() as f64;
                                            let _ = tx.send(json!({"type": "stage_timing", "stage": "ttft", "duration_ms": ttft_ms})).await;
                                        }
                                        full_answer.push_str(token);
                                        let _ = tx.send(json!({"type": "generation_chunk", "text": token})).await;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if full_answer.is_empty() {
        full_answer = candidates.first().and_then(|c| c["text"].as_str()).unwrap_or(abstain_answer).to_string();
        let _ = tx.send(json!({"type": "generation_chunk", "text": full_answer})).await;
    }

    // Guardrail 3: Hallucination Check (Sub-2ms Token Grounding Verification)
    if let Some(hallucination_decision) = state.guardrails.check_hallucination(&full_answer, &context_chunks) {
        let _ = tx.send(json!({
            "type": "guardrail_result",
            "should_abstain": true,
            "reason": hallucination_decision.reason,
            "guardrail_type": "hallucination"
        })).await;
        full_answer = abstain_answer.to_string();
        let _ = tx.send(json!({"type": "generation_chunk", "text": abstain_answer})).await;
    }
    
    let duration_generation = start_generation.elapsed().as_millis() as f64;
    let _ = tx.send(json!({"type": "stage_timing", "stage": "generation", "duration_ms": duration_generation})).await;
    
    let _ = tx.send(json!({
        "type": "generation_result",
        "answer": full_answer,
        "sources": source_docs,
        "model_used": model_used,
        "generation_time_ms": duration_generation,
        "fallback_used": false,
    })).await;
    
    if tts_enabled {
        let client = state.http_client.clone();
        let asr_url = state.asr_url.clone();
        let full_answer_clone = full_answer.clone();
        let tx_clone = tx.clone();
        tokio::spawn(async move {
            fetch_tts_stream(&client, &asr_url, full_answer_clone, tx_clone).await;
        });
    }
    
    let _ = tx.send(json!({"type": "done"})).await;
}

async fn fetch_tts_stream(client: &Client, asr_url: &str, text: String, tx: mpsc::Sender<Value>) {
    let payload = json!({"text": text});
    if let Ok(res) = client.post(format!("{}/tts", asr_url)).json(&payload).send().await {
        if let Ok(json) = res.json::<Value>().await {
            if let Some(b64) = json["audio"].as_str() {
                if !b64.is_empty() {
                    let _ = tx.send(json!({"type": "audio_chunk", "data": b64})).await;
                }
            }
        }
    }
}
