use hound::{SampleFormat, WavSpec, WavWriter};
use reqwest::multipart;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use regex::Regex;

#[derive(Serialize, Deserialize, Debug)]
pub struct SarvamResponse {
    pub transcript: Option<String>,
}

#[derive(Clone)]
pub struct SarvamClient {
    client: reqwest::Client,
    api_key: String,
}

impl SarvamClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
        }
    }

    fn preprocess_audio(audio: &[f32]) -> Vec<f32> {
        let mut data = audio.to_vec();

        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;
        for x in &mut data {
            *x -= mean;
        }

        let max_peak = data.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        if max_peak > 1e-4 {
            for x in &mut data {
                *x = (*x / max_peak) * 0.9;
            }
        }
        data
    }

    fn postprocess_text(text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        let re_fillers = Regex::new(r"(?i)\b(um|uh|er|ah|hmm)\b").unwrap();
        let mut cleaned = re_fillers.replace_all(text, "").to_string();

        let re_space = Regex::new(r"\s+").unwrap();
        cleaned = re_space.replace_all(&cleaned, " ").to_string();

        cleaned = cleaned.trim().to_string();

        if !cleaned.is_empty() {
            let mut c = cleaned.chars();
            if let Some(first) = c.next() {
                cleaned = first.to_uppercase().collect::<String>() + c.as_str();
            }
        }

        cleaned
    }

    pub async fn transcribe(&self, audio_data: &[f32]) -> String {
        if audio_data.is_empty() {
            return String::new();
        }

        let processed = Self::preprocess_audio(audio_data);

        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut buffer = Cursor::new(Vec::new());
        {
            let mut writer = WavWriter::new(&mut buffer, spec).unwrap();
            for &sample in &processed {
                let clamped = sample.clamp(-1.0, 1.0);
                let pcm = (clamped * 32767.0) as i16;
                writer.write_sample(pcm).unwrap();
            }
            writer.finalize().unwrap();
        }

        let wav_bytes = buffer.into_inner();

        let part = multipart::Part::bytes(wav_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .unwrap();

        let form = multipart::Form::new()
            .part("file", part)
            .text("model", "saarika:v2.5")
            .text("language_code", "en-IN");

        let res = self
            .client
            .post("https://api.sarvam.ai/speech-to-text")
            .header("api-subscription-key", &self.api_key)
            .multipart(form)
            .send()
            .await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<SarvamResponse>().await {
                    if let Some(text) = json.transcript {
                        return Self::postprocess_text(&text);
                    }
                }
            }
            Ok(resp) => {
                println!("Sarvam Error HTTP {}", resp.status());
            }
            Err(e) => {
                println!("Request error: {}", e);
            }
        }
        String::new()
    }

    pub async fn synthesize_tts(&self, text: &str) -> Option<String> {
        if text.trim().is_empty() {
            return None;
        }

        let payload = serde_json::json!({
            "inputs": [text],
            "target_language_code": "en-IN",
            "speaker": "karun",
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "speech_sample_rate": 16000,
            "enable_preprocessing": true,
            "model": "bulbul:v2"
        });

        let res = self
            .client
            .post("https://api.sarvam.ai/text-to-speech")
            .header("api-subscription-key", &self.api_key)
            .json(&payload)
            .send()
            .await;

        if let Ok(resp) = res {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(audios) = json.get("audios").and_then(|a| a.as_array()) {
                        if let Some(first) = audios.first().and_then(|a| a.as_str()) {
                            return Some(first.to_string());
                        }
                    }
                }
            } else {
                println!("Sarvam TTS HTTP Error: {}", resp.status());
            }
        }
        None
    }
}
