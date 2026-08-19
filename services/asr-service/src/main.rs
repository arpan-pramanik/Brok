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
use serde::{Deserialize, Serialize};
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

mod sarvam;
mod vad;

use sarvam::SarvamClient;
use vad::Vad;

#[derive(Serialize, Deserialize)]
struct FinalTranscript {
    #[serde(rename = "type")]
    msg_type: String,
    text: String,
    confidence: f32,
    duration_seconds: f32,
    language: String,
}

#[derive(Serialize, Deserialize)]
struct PartialTranscript {
    #[serde(rename = "type")]
    msg_type: String,
    text: String,
}

#[derive(Clone)]
struct AppState {
    vad: Arc<Vad>,
    sarvam: SarvamClient,
}

#[tokio::main]
async fn main() {
    let _ = dotenvy::from_filename("../../../.env");
    let sarvam_key = env::var("SARVAM_API_KEY").unwrap_or_default();
    
    let state = AppState {
        vad: Arc::new(Vad::new(0.005)),
        sarvam: SarvamClient::new(sarvam_key),
    };

    let app = Router::new()
        .route("/health", get(|| async { Json(serde_json::json!({"status": "ok"})) }))
        .route("/transcribe", post(transcribe_http))
        .route("/tts", post(tts_http))
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8001));
    println!("ASR Service running on {}", addr);
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.set_reuseaddr(true).unwrap();
    #[cfg(unix)]
    let _ = socket.set_reuseport(true);
    socket.bind(addr).unwrap();
    let listener = socket.listen(1024).unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn transcribe_http(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<FinalTranscript>, (StatusCode, String)> {
    if let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            let data = field.bytes().await.unwrap();
            
            let cursor = std::io::Cursor::new(data);
            let mut reader = hound::WavReader::new(cursor).map_err(|_| (StatusCode::BAD_REQUEST, "Invalid WAV".into()))?;
            let spec = reader.spec();
            if spec.sample_rate != 16000 || spec.channels != 1 || spec.bits_per_sample != 16 {
                return Err((StatusCode::BAD_REQUEST, "Must be 16kHz mono 16-bit WAV".into()));
            }
            
            let mut audio_f32 = Vec::new();
            for sample in reader.samples::<i16>() {
                if let Ok(s) = sample {
                    audio_f32.push(s as f32 / 32768.0);
                }
            }
            
            let text = state.sarvam.transcribe(&audio_f32).await;
            let duration = audio_f32.len() as f32 / 16000.0;
            return Ok(Json(FinalTranscript {
                msg_type: "final_transcript".into(),
                text,
                confidence: 0.99,
                duration_seconds: duration,
                language: "en-IN".into(),
            }));
        }
    }
    Err((StatusCode::BAD_REQUEST, "Missing file".into()))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let mut audio_buffer = Vec::<f32>::new();
    let mut speech_detected = false;
    let mut silence_chunks = 0;
    let mut chunk_count = 0;

    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(_) => break,
        };

        match msg {
            Message::Text(t) => {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&t) {
                    if json["type"] == "stop_recording" {
                        if !audio_buffer.is_empty() {
                            let text = state.sarvam.transcribe(&audio_buffer).await;
                            let duration = audio_buffer.len() as f32 / 16000.0;
                            let res = FinalTranscript {
                                msg_type: "final_transcript".into(),
                                text,
                                confidence: 0.99,
                                duration_seconds: duration,
                                language: "en-IN".into(),
                            };
                            let _ = socket.send(Message::Text(serde_json::to_string(&res).unwrap())).await;
                        }
                        audio_buffer.clear();
                    } else if json["type"] == "start_recording" {
                        audio_buffer.clear();
                    }
                }
            }
            Message::Binary(b) => {
                let chunk: Vec<f32> = b
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap_or([0; 4])))
                    .collect();
                
                audio_buffer.extend_from_slice(&chunk);
                chunk_count += 1;
            }
            _ => {}
        }
    }
}

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
}

async fn tts_http(
    State(state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if let Some(audio_b64) = state.sarvam.synthesize_tts(&req.text).await {
        return Ok(Json(serde_json::json!({
            "audio": audio_b64,
            "format": "base64_wav"
        })));
    }
    Err((StatusCode::BAD_REQUEST, "Sarvam TTS synthesis unavailable or rejected key".into()))
}
