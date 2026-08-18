pub struct Vad {
    energy_threshold: f32,
}

impl Vad {
    pub fn new(energy_threshold: f32) -> Self {
        Self { energy_threshold }
    }

    pub fn detect(&self, audio_chunk: &[f32]) -> bool {
        if audio_chunk.is_empty() {
            return false;
        }
        
        let sum_sq: f32 = audio_chunk.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / audio_chunk.len() as f32).sqrt();
        
        rms > self.energy_threshold
    }
}
