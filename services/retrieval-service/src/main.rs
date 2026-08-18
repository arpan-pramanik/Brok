use axum::{
    extract::{State, Json},
    routing::{get, post},
    Router,
};
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel, TextRerank, RerankInitOptions, RerankerModel};
use reqwest::Client;
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
    rerank_model: Mutex<TextRerank>,
    http_client: reqwest::Client,
    qdrant_url: String,
    collection_name: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
    let collection_name = env::var("COLLECTION_NAME").unwrap_or_else(|_| "msmarco_xi".to_string());

    println!("Initializing embedding model BGESmallENV15...");
    let embed_options = InitOptions::new(EmbeddingModel::BGESmallENV15);
    let embed_model = TextEmbedding::try_new(embed_options)?;
    println!("Embedding model loaded successfully!");
    
    println!("Initializing reranker model BGERerankerBase...");
    let rerank_options = RerankInitOptions::new(RerankerModel::BGERerankerBase);
    let rerank_model = TextRerank::try_new(rerank_options)?;
    println!("Reranker model loaded successfully!");

    println!("Connecting to Qdrant REST at {}", qdrant_url);
    let http_client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(5)).build()?;

    let state = Arc::new(AppState {
        embed_model: Mutex::new(embed_model),
        rerank_model: Mutex::new(rerank_model),
        http_client,
        qdrant_url,
        collection_name,
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

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<QueryRequest>,
) -> Json<RetrievalResult> {
    let start = Instant::now();
    let query = payload.query.clone();
    
    // 1. Embed query
    let vector = {
        let mut embed_guard = state.embed_model.lock().await;
        let embeddings = embed_guard.embed(vec![query.clone()], None).unwrap_or_default();
        embeddings.into_iter().next().unwrap_or_default()
    };
    
    // 2. Search Qdrant via HTTP REST API
    let search_url = format!("{}/collections/{}/points/search", state.qdrant_url, state.collection_name);
    let search_body = json!({
        "vector": vector,
        "limit": 5,
        "with_payload": true
    });
    
    let mut chunks = Vec::new();
    let mut texts = Vec::new();
    
    if let Ok(res) = state.http_client.post(&search_url).json(&search_body).send().await {
        if let Ok(json_res) = res.json::<Value>().await {
            if let Some(points) = json_res["result"].as_array() {
                for p in points {
                    let text = p["payload"]["text"].as_str().unwrap_or("").to_string();
                    let source_doc = p["payload"]["source_doc"].as_str().unwrap_or("").to_string();
                    let chunk_index = p["payload"]["chunk_index"].as_i64().unwrap_or(0) as i32;
                    let chunk_id = p["id"].as_str().unwrap_or("").to_string();
                    let score = p["score"].as_f64().unwrap_or(0.0) as f32;
                    
                    chunks.push(ChunkCandidate {
                        chunk_id,
                        text: text.clone(),
                        score,
                        source_doc,
                        chunk_index,
                        metadata: json!({}),
                    });
                    texts.push(text);
                }
            }
        }
    }
    
    // 3. Rerank
    let mut reranked_chunks = chunks.clone();
    if !texts.is_empty() {
        let mut rerank_guard = state.rerank_model.lock().await;
        // Fix for inference: S is &str.
        // Convert texts to Vec<&str>
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        if let Ok(rerank_results) = rerank_guard.rerank(query.as_str(), text_refs, true, None) {
            let mut new_chunks = Vec::new();
            for res in rerank_results {
                if let Some(chunk) = chunks.get(res.index) {
                    let mut new_chunk = chunk.clone();
                    new_chunk.score = res.score;
                    new_chunks.push(new_chunk);
                }
            }
            new_chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            reranked_chunks = new_chunks.into_iter().take(payload.top_k).collect();
        }
    }
    
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    
    Json(RetrievalResult {
        query,
        candidates: reranked_chunks.clone(),
        rrf_scores: vec![],
        top_reranked: reranked_chunks,
        retrieval_time_ms: elapsed,
    })
}
