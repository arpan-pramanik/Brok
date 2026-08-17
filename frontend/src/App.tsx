import { useState, useRef, useEffect } from "react";
import { Send, User, Loader2, Mic, MicOff, Square } from "lucide-react";
import { Waveform } from "./components/Waveform";
import { PartialTranscript } from "./components/PartialTranscript";
import { LatencyWaterfall } from "./components/LatencyWaterfall";
import { ConfidenceGauge } from "./components/ConfidenceGauge";
import BenchmarkRunner from "./components/BenchmarkRunner";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [mode, setMode] = useState<"user" | "developer">("user");
  const [view, setView] = useState<"chat" | "benchmark">("chat");
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState({ partial: "", final: "" });
  const [metrics, setMetrics] = useState<any>(null);
  
  const [isRecording, setIsRecording] = useState(false);
  
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const currentAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);

  const initAudio = () => {
    if (!audioContextRef.current) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      audioContextRef.current = new AudioContextClass();
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
  };

  // Initialize WebSocket
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8001/ws");
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("WebSocket message:", data);
      
      if (data.type === "partial_transcript") {
        setTranscript(prev => ({ ...prev, partial: data.text }));
      } else if (data.type === "final_transcript") {
        setTranscript({ partial: "", final: data.text });
        if (data.text.trim()) {
          handleTextQuery(data.text);
        }
      } else if (data.type === "vad_stop") {
        setIsRecording(false);
      }
    };
    
    wsRef.current = ws;
    
    return () => {
      ws.close();
    };
  }, []);

  const handleAudioData = (data: ArrayBuffer) => {
    if (!isRecording) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  };

  const startRecording = () => {
    initAudio();
    if (!isRecording) {
      setTranscript({ partial: "", final: "" });
      setIsRecording(true);
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "start_recording" }));
      }
    }
  };

  const stopRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "stop_recording" }));
      }
    }
  };

  const fallbackTTS = (text: string) => {
    console.warn("Falling back to Web Speech API");
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  };

  const speakResponse = async (text: string) => {
    if (!text) return;
    try {
      initAudio();
      
      if (currentAudioSourceRef.current) {
        try { currentAudioSourceRef.current.stop(); } catch (e) {}
      }
      
      const ttsUrl = `http://localhost:8005/synthesize?text=${encodeURIComponent(text)}`;
      const res = await fetch(ttsUrl);
      if (res.ok) {
        const arrayBuffer = await res.arrayBuffer();
        const ctx = audioContextRef.current!;
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        
        if (currentAudioSourceRef.current) {
          try { currentAudioSourceRef.current.stop(); } catch (e) {}
        }
        
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        source.start(0);
        currentAudioSourceRef.current = source;
        return;
      } else {
        throw new Error("TTS fetch failed");
      }
    } catch (e) {
      console.error("Audio setup error:", e);
      fallbackTTS(text);
    }
  };

  const handleTextQuery = async (text: string) => {
    if (!text.trim()) return;
    
    setIsProcessing(true);
    setAnswer("");
    try {
      const res = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, conversation_id: "test" })
      });
      const data = await res.json();
      setAnswer(data.answer);
      if (data.answer) {
        speakResponse(data.answer);
      }
      setMetrics({
        latency: data.latency,
        guardrail: data.guardrail
      });
    } catch (e) {
      console.error(e);
      setAnswer("Error connecting to backend.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    initAudio();
    if (query) {
      handleTextQuery(query);
      setQuery("");
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div>
          <div className="brand-logo">BRAGI</div>
          <div className="brand-sub">VOICE-ENABLED RAG</div>
        </div>
        
        <div className="nav-tabs">
          <button 
            onClick={() => setMode("user")}
            className={`nav-btn ${mode === "user" ? "active" : ""}`}
          >
            User Mode
          </button>
          <button 
            onClick={() => setMode("developer")}
            className={`nav-btn ${mode === "developer" ? "active" : ""}`}
          >
            Developer Mode
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main>
        
        {mode === "developer" && (
          <div className="nav-tabs" style={{ marginBottom: '1.5rem', justifyContent: 'center', alignItems: 'center', gap: '0.75rem' }}>
            <button 
              onClick={() => setView("chat")}
              className={`nav-btn ${view === "chat" ? "active" : ""}`}
            >
              Interactive Mode
            </button>
            <button 
              onClick={() => setView("benchmark")}
              className={`nav-btn ${view === "benchmark" ? "active" : ""}`}
            >
              Benchmark Suite
            </button>
            <button 
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onMouseLeave={stopRecording}
              onTouchStart={startRecording}
              onTouchEnd={stopRecording}
              className={`nav-btn ${isRecording ? "active" : ""}`}
              style={{
                borderColor: isRecording ? '#c85a5a' : 'var(--border-sand)',
                background: isRecording ? '#c85a5a' : 'transparent',
                color: isRecording ? '#fff' : 'var(--text-sand)',
                userSelect: 'none'
              }}
            >
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
                <Mic size={13} />
                {isRecording ? "LISTENING..." : "HOLD TO SPEAK"}
              </span>
            </button>
          </div>
        )}

        {view === "chat" ? (
          <div className="workspace-grid">
            {/* Left Panel: Input */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              
              {/* Voice Card */}
              <div className="card-sand" style={{ alignItems: 'center', textAlign: 'center', minHeight: '350px', justifyContent: 'center', gap: '1rem' }}>
                <div className="card-label" style={{ width: '100%' }}>VOICE ENGINE</div>
                
                <div className="waveform-frame" style={{ width: '100%', maxWidth: '300px', margin: '0 auto' }}>
                  {isRecording ? (
                    <Waveform isRecording={isRecording} onAudioData={handleAudioData} />
                  ) : (
                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.3)', fontSize: '0.8rem', fontFamily: 'var(--font-mono)' }}>
                      WAITING FOR VOICE...
                    </div>
                  )}
                </div>

                <div style={{ minHeight: '3rem', width: '100%' }}>
                  <PartialTranscript partialText={transcript.partial} finalText={transcript.final} />
                </div>
                
                <button
                  onMouseDown={startRecording}
                  onMouseUp={stopRecording}
                  onMouseLeave={stopRecording}
                  onTouchStart={startRecording}
                  onTouchEnd={stopRecording}
                  className="btn-pill-sand"
                  style={{ 
                    marginTop: '1rem',
                    background: isRecording ? '#c85a5a' : 'var(--bg-dark)',
                    color: isRecording ? '#fff' : 'var(--text-sand)',
                    userSelect: 'none'
                  }}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Mic size={14} />
                    {isRecording ? "LISTENING..." : "HOLD TO SPEAK"}
                  </span>
                </button>
              </div>

              {/* Text Input (Developer Mode only) */}
              {mode === "developer" && (
                <form onSubmit={handleSubmit} style={{ position: 'relative' }}>
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Or type your query here..."
                    className="input-field"
                    style={{ paddingRight: '3rem' }}
                  />
                  <button 
                    type="submit"
                    disabled={!query.trim() || isProcessing}
                    style={{
                      position: 'absolute',
                      right: '0.75rem',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none',
                      border: 'none',
                      color: 'var(--text-sand)',
                      cursor: (!query.trim() || isProcessing) ? 'default' : 'pointer',
                      opacity: (!query.trim() || isProcessing) ? 0.3 : 1
                    }}
                  >
                    <Send size={18} />
                  </button>
                </form>
              )}
            </div>

            {/* Right Panel: Output & Metrics */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <div className="card-dark" style={{ flex: 1, minHeight: '350px' }}>
                <div className="card-label">
                  <span>RESPONSE LOG</span>
                  <User size={14} />
                </div>
                
                <div className="response-view">
                  {isProcessing ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)' }}>
                      <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                      THINKING...
                    </div>
                  ) : answer ? (
                    <div>{answer}</div>
                  ) : (
                    <div style={{ color: 'var(--text-muted)' }}>
                      Awaiting query...
                    </div>
                  )}
                </div>
              </div>

              {/* Metrics (Developer Mode only) */}
              {mode === "developer" && metrics && (
                <div className="metrics-row">
                  <div className="card-dark">
                    <div className="card-label">LATENCY PROFILE</div>
                    <LatencyWaterfall timings={metrics.latency || []} />
                  </div>
                  <div className="card-dark" style={{ alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                    <div className="card-label" style={{ width: '100%' }}>CONFIDENCE</div>
                    <ConfidenceGauge score={metrics.guardrail?.confidence_score ?? 0} shouldAbstain={metrics.guardrail?.abstained ?? false} threshold={0.7} />
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <BenchmarkRunner />
        )}
      </main>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default App;
