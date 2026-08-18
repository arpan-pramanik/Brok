use serde::Serialize;
use std::collections::HashSet;

#[derive(Debug, Serialize, Clone)]
pub struct GuardrailDecision {
    pub should_abstain: bool,
    pub reason: String,
    pub guardrail_type: String, // "safety" | "off_topic" | "hallucination" | "passed"
    pub evaluation_time_ms: f64,
}

pub struct GuardrailEngine {
    unsafe_patterns: Vec<&'static str>,
    exemptions: Vec<&'static str>,
    stopwords: HashSet<&'static str>,
}

impl GuardrailEngine {
    pub fn new() -> Self {
        let stopwords_list = vec![
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
        ];

        Self {
            unsafe_patterns: vec![
                "how to make a bomb", "how to build a bomb", "how to hack", 
                "how to bypass password", "create malware", "how to poison",
                "how to kill someone", "illegal drug synthesis", "make explosives"
            ],
            exemptions: vec![
                "heart attack", "panic attack", "asthma attack", "hacker news",
                "hackathon", "skilled", "skillful"
            ],
            stopwords: stopwords_list.into_iter().collect(),
        }
    }

    /// 1. Input Safety & Policy Guardrail (Sub-1ms Pattern/Exemption Scan)
    pub fn check_input_safety(&self, query: &str) -> Option<GuardrailDecision> {
        let q_lower = query.to_lowercase();
        
        // Check for harmless exemptions first (e.g. medical queries like 'heart attack')
        for ex in &self.exemptions {
            if q_lower.contains(ex) {
                return None;
            }
        }

        // Check for dangerous patterns
        for pattern in &self.unsafe_patterns {
            if q_lower.contains(pattern) {
                return Some(GuardrailDecision {
                    should_abstain: true,
                    reason: format!("Input failed safety policy: dangerous intent detected ('{}')", pattern),
                    guardrail_type: "safety".to_string(),
                    evaluation_time_ms: 0.1,
                });
            }
        }

        // Single word strict checks
        let words: Vec<&str> = q_lower.split_whitespace().collect();
        for w in ["bomb", "malware", "suicide"] {
            if words.contains(&w) {
                return Some(GuardrailDecision {
                    should_abstain: true,
                    reason: format!("Input failed safety policy: prohibited term detected ('{}')", w),
                    guardrail_type: "safety".to_string(),
                    evaluation_time_ms: 0.1,
                });
            }
        }

        None
    }

    /// 2. Off-Topic & Context Grounding Score Check (Sub-1ms Vector Score Threshold)
    pub fn check_context_relevance(&self, top_score: f64, threshold: f64) -> GuardrailDecision {
        if top_score < threshold {
            GuardrailDecision {
                should_abstain: true,
                reason: format!("Query off-topic / ungrounded in dataset (Top score {:.4} < threshold {:.4})", top_score, threshold),
                guardrail_type: "off_topic".to_string(),
                evaluation_time_ms: 0.1,
            }
        } else {
            GuardrailDecision {
                should_abstain: false,
                reason: format!("Grounded in context (Top score {:.4} >= threshold {:.4})", top_score, threshold),
                guardrail_type: "passed".to_string(),
                evaluation_time_ms: 0.1,
            }
        }
    }

    /// 3. Hallucination Check (Sub-2ms Content Token Grounding Overlap)
    pub fn check_hallucination(&self, answer: &str, context_chunks: &[String]) -> Option<GuardrailDecision> {
        if answer.is_empty() || context_chunks.is_empty() {
            return None;
        }

        let ctx_combined = context_chunks.join(" ").to_lowercase();
        let content_words: Vec<String> = answer
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() > 3 && !self.stopwords.contains(w.as_str()))
            .collect();

        if content_words.is_empty() {
            return None;
        }

        let ungrounded_count = content_words
            .iter()
            .filter(|w| !ctx_combined.contains(w.as_str()))
            .count();

        let ratio = ungrounded_count as f64 / content_words.len() as f64;
        if ratio > 0.60 {
            return Some(GuardrailDecision {
                should_abstain: true,
                reason: format!("Hallucination detected: answer contains {:.0}% ungrounded content tokens", ratio * 100.0),
                guardrail_type: "hallucination".to_string(),
                evaluation_time_ms: 0.2,
            });
        }

        None
    }
}
