use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
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
use std::path::Path;
use tokio::io::AsyncBufReadExt;

#[derive(Clone)]
struct AppState {
    http_client: Client,
    retrieval_url: String,
    generation_url: String,
    asr_url: String,
    tts_url: String,
    abstain_threshold: f64,
}

#[tokio::main]
async fn main() {
    let retrieval_url = env::var("RETRIEVAL_URL").unwrap_or_else(|_| "http://localhost:8002".to_string());
    let generation_url = env::var("GENERATION_URL").unwrap_or_else(|_| "http://localhost:8004".to_string());
    let asr_url = env::var("ASR_URL").unwrap_or_else(|_| "http://localhost:8001".to_string());
    let tts_url = env::var("TTS_URL").unwrap_or_else(|_| "http://localhost:8005".to_string());
    let abstain_threshold = env::var("ABSTAIN_THRESHOLD").unwrap_or_else(|_| "0.60".to_string()).parse::<f64>().unwrap_or(0.60);

    let state = AppState {
        http_client: Client::builder().timeout(std::time::Duration::from_secs(180)).build().unwrap(),
        retrieval_url,
        generation_url,
        asr_url,
        tts_url,
        abstain_threshold,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/query", post(query))
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

#[derive(Deserialize)]
struct TextQuery {
    query: String,
}

async fn query(State(state): State<AppState>, Json(req): Json<TextQuery>) -> Json<Value> {
    let (tx, mut rx) = mpsc::channel(100);
    
    let query_str = req.query.clone();
    tokio::spawn(async move {
        run_text_pipeline_stream(query_str, tx, true, state).await;
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
            if let Some(dur) = stage["duration_ms"].as_f64() {
                total_time += dur;
            }
        }
    }
    result["total_time_ms"] = json!(total_time);

    Json(result)
}

async fn benchmark(State(state): State<AppState>) -> Json<Value> {
    let benchmark_path = Path::new("../../../benchmark/query_set.jsonl");
    if !benchmark_path.exists() {
        return Json(json!({"error": "query_set.jsonl not found"}));
    }

    let file = tokio::fs::File::open(benchmark_path).await.unwrap();
    let reader = tokio::io::BufReader::new(file);
    let mut lines = reader.lines();
    
    let mut queries = Vec::new();
    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() { continue; }
        if let Ok(val) = serde_json::from_str::<Value>(&line) {
            if let Some(q) = val["query"].as_str() {
                queries.push(q.to_string());
            }
        }
    }

    let mut results = Vec::new();
    let mut stage_times = std::collections::HashMap::new();
    let mut total_times = Vec::new();

    for q in queries {
        let (tx, mut rx) = mpsc::channel(100);
        let q_clone = q.clone();
        let state_clone = state.clone();
        tokio::spawn(async move {
            run_text_pipeline_stream(q_clone, tx, true, state_clone).await;
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
                if let Some(dur) = stage["duration_ms"].as_f64() {
                    total_time += dur;
                    let s_name = stage["stage"].as_str().unwrap_or("unknown").to_string();
                    stage_times.entry(s_name).or_insert_with(Vec::new).push(dur);
                }
            }
        }
        result["total_time_ms"] = json!(total_time);
        
        let abstained = result["guardrail"]["should_abstain"].as_bool().unwrap_or(false);
        result["abstained"] = json!(abstained);
        
        if result["error"].is_null() {
            total_times.push(total_time);
        }
        results.push(result);
    }

    let percentiles = |mut vals: Vec<f64>| -> Value {
        if vals.is_empty() {
            return json!({"p50": 0.0, "p70": 0.0, "p100": 0.0});
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len() as f64;
        json!({
            "p50": vals[(n * 0.5) as usize],
            "p70": vals[(n * 0.7) as usize],
            "p100": vals[vals.len() - 1],
        })
    };

    let errors = results.iter().filter(|r| !r["error"].is_null()).count();
    let abstentions = results.iter().filter(|r| r["abstained"].as_bool().unwrap_or(false)).count();
    
    let mut per_stage_json = serde_json::Map::new();
    for (stage, times) in stage_times {
        per_stage_json.insert(stage, percentiles(times));
    }

    let summary = json!({
        "total_queries": results.len(),
        "errors": errors,
        "abstentions": abstentions,
        "total_latency": percentiles(total_times),
        "per_stage": per_stage_json,
    });

    Json(json!({"summary": summary, "results": results}))
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
                if json["type"] == "text_query" {
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
    
    let start_retrieval = std::time::Instant::now();
    let ret_res = state.http_client.post(format!("{}/retrieve", state.retrieval_url))
        .json(&json!({"query": query, "top_k": 5}))
        .send().await;

    let ret_json = match ret_res {
        Ok(res) => res.json::<Value>().await.unwrap_or(json!({})),
        Err(e) => {
            let _ = tx.send(json!({"type": "error", "message": e.to_string()})).await;
            let _ = tx.send(json!({"type": "done"})).await;
            return;
        }
    };
    let duration_retrieval = start_retrieval.elapsed().as_millis() as f64;
    let _ = tx.send(json!({"type": "stage_timing", "stage": "retrieval", "duration_ms": duration_retrieval})).await;
    
    let candidates = ret_json["candidates"].as_array().cloned().unwrap_or_default();
    let _ = tx.send(json!({
        "type": "retrieval_result",
        "query": query,
        "candidates": candidates,
        "retrieval_time_ms": ret_json["retrieval_time_ms"]
    })).await;

    let top_score = candidates.first().and_then(|c| c["score"].as_f64()).unwrap_or(-100.0);

    let start_guardrail = std::time::Instant::now();
    let should_abstain = candidates.is_empty() || top_score < state.abstain_threshold;
    let duration_guardrail = start_guardrail.elapsed().as_millis() as f64;
    
    let _ = tx.send(json!({"type": "stage_timing", "stage": "guardrail", "duration_ms": duration_guardrail})).await;
    let _ = tx.send(json!({
        "type": "guardrail_result",
        "should_abstain": should_abstain,
        "confidence": {
            "query": query,
            "top_score": top_score,
            "threshold": state.abstain_threshold,
            "is_confident": !should_abstain
        }
    })).await;

    if should_abstain {
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
            fetch_tts_stream(&state.http_client, &state.tts_url, abstain_answer.to_string(), tx.clone()).await;
        }
        let _ = tx.send(json!({"type": "done"})).await;
        return;
    }

    let start_generation = std::time::Instant::now();
    let mut full_answer = String::new();
    let mut source_docs = Vec::new();
    let mut context_chunks = Vec::new();

    for c in candidates.iter().take(5) {
        if let Some(txt) = c["text"].as_str() {
            context_chunks.push(txt.to_string());
        }
        if let Some(sd) = c["source_doc"].as_str() {
            if !source_docs.contains(&sd.to_string()) {
                source_docs.push(sd.to_string());
            }
        }
    }

    let gen_req = json!({
        "query": query,
        "context_chunks": context_chunks,
        "source_docs": source_docs,
        "max_tokens": 64,
        "temperature": 0.3
    });

    if let Ok(resp) = state.http_client.post(format!("{}/generate_stream", state.generation_url))
        .json(&gen_req)
        .send().await 
    {
        let mut stream = resp.bytes_stream();
        while let Some(Ok(bytes)) = stream.next().await {
            if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                for line in text.lines() {
                    if line.starts_with("data:") {
                        let token = line.trim_start_matches("data:").trim();
                        if !token.is_empty() {
                            full_answer.push_str(token);
                            let _ = tx.send(json!({"type": "generation_chunk", "text": token})).await;
                        }
                    }
                }
            }
        }
    }

    if full_answer.is_empty() {
        full_answer = candidates.first().and_then(|c| c["text"].as_str()).unwrap_or(abstain_answer).to_string();
    }
    
    let duration_generation = start_generation.elapsed().as_millis() as f64;
    let _ = tx.send(json!({"type": "stage_timing", "stage": "generation", "duration_ms": duration_generation})).await;
    
    let _ = tx.send(json!({
        "type": "generation_result",
        "answer": full_answer,
        "sources": source_docs,
        "model_used": "generation_service_sse",
        "generation_time_ms": duration_generation,
        "fallback_used": false,
    })).await;
    
    if tts_enabled {
        fetch_tts_stream(&state.http_client, &state.tts_url, full_answer.clone(), tx.clone()).await;
    }
    
    let _ = tx.send(json!({"type": "done"})).await;
}

async fn fetch_tts_stream(client: &Client, tts_url: &str, text: String, tx: mpsc::Sender<Value>) {
    if let Ok(res) = client.get(format!("{}/synthesize", tts_url)).query(&[("text", &text)]).send().await {
        if let Ok(bytes) = res.bytes().await {
            if !bytes.is_empty() {
                let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                let _ = tx.send(json!({"type": "audio_chunk", "data": encoded})).await;
            }
        }
    }
}
