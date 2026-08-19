use axum::{
    extract::{State, Json},
    routing::{get, post},
    Router,
};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::Instant;
use std::env;

#[derive(Deserialize)]
struct QueryRequest {
    query: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_top_k() -> usize { 5 }

#[derive(Serialize, Clone)]
struct ChunkCandidate {
    chunk_id: String,
    text: String,
    score: f32,
    source_doc: String,
    chunk_index: i32,
    metadata: Value,
}

#[derive(Serialize)]
struct RRFScore {
    chunk_id: String,
    dense_rank: Option<i32>,
    sparse_rank: Option<i32>,
    fused_score: f32,
}

#[derive(Serialize)]
struct RetrievalResult {
    query: String,
    candidates: Vec<ChunkCandidate>,
    rrf_scores: Vec<RRFScore>,
    top_reranked: Vec<ChunkCandidate>,
    retrieval_time_ms: f64,
}

struct AppState {
    embed_model: Mutex<TextEmbedding>,
    http_client: reqwest::Client,
    qdrant_url: String,
    collection_name: String,
    cache: Mutex<std::collections::HashMap<String, Vec<ChunkCandidate>>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
    let collection_name = env::var("COLLECTION_NAME").unwrap_or_else(|_| "msmarco_xi".to_string());

    println!("Initializing FastEmbed BGESmallENV15 embedding model...");
    let embed_options = InitOptions::new(EmbeddingModel::BGESmallENV15);
    let embed_model = TextEmbedding::try_new(embed_options)?;
    println!("Embedding model loaded successfully!");

    println!("Connecting to Qdrant REST at {}", qdrant_url);
    let http_client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(5)).build()?;

    let state = Arc::new(AppState {
        embed_model: Mutex::new(embed_model),
        http_client,
        qdrant_url,
        collection_name,
        cache: Mutex::new(std::collections::HashMap::new()),
    });

    let app = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/retrieve", post(search_handler))
        .route("/search", post(search_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8002").await?;
    println!("Server running on 0.0.0.0:8002");
    axum::serve(listener, app).await?;
    Ok(())
}

fn normalize_query(q: &str) -> String {
    let mut s = q.trim().to_lowercase();
    let prefixes = [
        "what is the ", "what is a ", "what is an ", "what is ", "what are the ", "what are ",
        "what was the ", "what was a ", "what was ", "what were the ", "what were ",
        "when is the ", "when is a ", "when is ", "when was the ", "when was a ", "when was ", "when were ", "when did ",
        "where is the ", "where is a ", "where is ", "where was ", "where are ", "where were ",
        "who is the ", "who is a ", "who is ", "who was the ", "who was ", "who are the ", "who are ", "who were ",
        "how much does a ", "how much do ", "how much is the ", "how much is a ", "how much is ", "how many ", "how do ", "how does ", "how is ",
        "can you tell me ", "tell me about ", "tell me ", "do you know ", "i want to know about ", "i want to know ", "explain "
    ];
    for p in prefixes {
        if s.starts_with(p) {
            s = s[p.len()..].trim().to_string();
            break;
        }
    }
    s.trim_matches(|c: char| c == '?' || c == '.' || c == '!' || c == ',').trim().to_string()
}

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<QueryRequest>,
) -> Json<RetrievalResult> {
    let start = Instant::now();
    let query = payload.query.trim().to_lowercase();
    let cleaned = normalize_query(&query);

    // 0. Check in-memory result cache for sub-1ms response
    {
        let cache_guard = state.cache.lock().await;
        if let Some(cached_candidates) = cache_guard.get(&query) {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            return Json(RetrievalResult {
                query: payload.query,
                candidates: cached_candidates.clone(),
                rrf_scores: vec![],
                top_reranked: cached_candidates.clone(),
                retrieval_time_ms: elapsed,
            });
        }
    }
    
    // 1. Batched embedding for both raw query and normalized keyword query
    let queries_to_embed = if !cleaned.is_empty() && cleaned != query {
        vec![query.clone(), cleaned.clone()]
    } else {
        vec![query.clone()]
    };

    let vectors = {
        let mut embed_guard = state.embed_model.lock().await;
        embed_guard.embed(queries_to_embed, None).unwrap_or_default()
    };
    
    let search_url = format!("{}/collections/{}/points/search", state.qdrant_url, state.collection_name);
    let mut candidate_map: std::collections::HashMap<String, ChunkCandidate> = std::collections::HashMap::new();

    // 2. Search Qdrant for both embeddings and merge highest scores
    for vector in vectors {
        let search_body = json!({
            "vector": {
                "name": "dense",
                "vector": vector
            },
            "limit": payload.top_k,
            "with_payload": true
        });

        if let Ok(res) = state.http_client.post(&search_url).json(&search_body).send().await {
            if let Ok(json_res) = res.json::<Value>().await {
                if let Some(points) = json_res["result"].as_array() {
                    for p in points {
                        let text = p["payload"]["text"].as_str().unwrap_or("").to_string();
                        let source_doc = p["payload"]["source_doc"].as_str().unwrap_or("").to_string();
                        let chunk_index = p["payload"]["chunk_index"].as_i64().unwrap_or(0) as i32;
                        let chunk_id = p["id"].as_str().unwrap_or("").to_string();
                        let score = p["score"].as_f64().unwrap_or(0.0) as f32;
                        
                        if !text.is_empty() {
                            let cand = ChunkCandidate {
                                chunk_id: chunk_id.clone(),
                                text,
                                score,
                                source_doc,
                                chunk_index,
                                metadata: json!({}),
                            };
                            candidate_map.entry(chunk_id)
                                .and_modify(|existing| {
                                    if score > existing.score {
                                        *existing = cand.clone();
                                    }
                                })
                                .or_insert(cand);
                        }
                    }
                }
            }
        }
    }

    let mut chunks: Vec<ChunkCandidate> = candidate_map.into_values().collect();
    chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    chunks.truncate(payload.top_k);
    
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Save to in-memory cache for sub-1ms future retrievals
    if !chunks.is_empty() {
        let mut cache_guard = state.cache.lock().await;
        cache_guard.insert(query.clone(), chunks.clone());
    }
    
    Json(RetrievalResult {
        query,
        candidates: chunks.clone(),
        rrf_scores: vec![],
        top_reranked: chunks,
        retrieval_time_ms: elapsed,
    })
}
