import { useState, useRef, useEffect } from "react";
import { Send, User, Loader2, Mic } from "lucide-react";
import { Waveform } from "./components/Waveform";
import { PartialTranscript } from "./components/PartialTranscript";
import { LatencyWaterfall } from "./components/LatencyWaterfall";
import { ConfidenceGauge } from "./components/ConfidenceGauge";
import BenchmarkRunner from "./components/BenchmarkRunner";
import { DeveloperStats } from "./components/DeveloperStats";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [mode, setMode] = useState<"user" | "developer">("user");
  const [view, setView] = useState<"chat" | "benchmark">("chat");
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState({ partial: "", final: "" });
  const [metrics, setMetrics] = useState<any>(null);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  
  const [isRecording, setIsRecording] = useState(false);
  
  const asrWsRef = useRef<WebSocket | null>(null);
  const orchWsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextAudioTimeRef = useRef<number>(0);

  const initAudio = () => {
    if (!audioContextRef.current) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      audioContextRef.current = new AudioContextClass();
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
  };

  const playAudioChunk = async (base64Audio: string) => {
    try {
      initAudio();
      const ctx = audioContextRef.current!;
      
      const binaryString = window.atob(base64Audio);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      const audioBuffer = await ctx.decodeAudioData(bytes.buffer);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      
      const currentTime = ctx.currentTime;
      if (nextAudioTimeRef.current < currentTime) {
        nextAudioTimeRef.current = currentTime + 0.1; // Add slight buffer
      }
      
      source.start(nextAudioTimeRef.current);
      nextAudioTimeRef.current += audioBuffer.duration;
    } catch (e) {
      console.error("Failed to play audio chunk", e);
    }
  };

  // Initialize WebSockets
  useEffect(() => {
    // Helper to upgrade ws:// to wss:// if site is on HTTPS
    const getWsUrl = (envUrl: string, fallback: string) => {
      let url = envUrl || fallback;
      if (window.location.protocol === 'https:' && url.startsWith('ws://')) {
        url = url.replace('ws://', 'wss://');
      }
      return url;
    };

    // 1. ASR WebSocket (Port 8001)
    const asrUrl = getWsUrl(import.meta.env.VITE_ASR_WS_URL, "ws://localhost:8001/ws");
    console.log("Connecting ASR WS to:", asrUrl);
    const asrWs = new WebSocket(asrUrl);
    asrWs.onerror = (e) => console.error("ASR WS Error:", e);
    asrWs.onopen = () => console.log("ASR WS Connected!");

    asrWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
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
    asrWsRef.current = asrWs;

    // 2. Orchestrator WebSocket (Port 8000)
    const orchUrl = getWsUrl(import.meta.env.VITE_ORCHESTRATOR_WS_URL, "ws://localhost:8000/ws");
    console.log("Connecting Orchestrator WS to:", orchUrl);
    const orchWs = new WebSocket(orchUrl);
    orchWs.onerror = (e) => console.error("Orchestrator WS Error:", e);
    orchWs.onopen = () => console.log("Orchestrator WS Connected!");
    orchWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Orchestrator WS message:", data.type);
      
      if (data.type === "generation_chunk") {
        setIsProcessing(false);
        setAnswer(prev => prev + data.text);
      } else if (data.type === "audio_chunk") {
        playAudioChunk(data.data);
      } else if (data.type === "generation_result") {
        setIsProcessing(false);
        if (!answer && data.answer) {
          setAnswer(data.answer);
        }
        if (data.latency || data.guardrail) {
           setMetrics((prev: any) => ({
              ...prev,
              guardrail: data.guardrail
           }));
        }
      } else if (data.type === "retrieval_result") {
          setMetrics((prev: any) => ({ ...prev, retrieval: data }));
      } else if (data.type === "guardrail_result") {
          setMetrics((prev: any) => ({ ...prev, guardrail: data }));
      } else if (data.type === "stage_timing") {
          setMetrics((prev: any) => {
              const latency = prev?.latency || [];
              return { ...prev, latency: [...latency, data] };
          });
      } else if (data.type === "error") {
          setIsProcessing(false);
          setAnswer("Error: " + data.message);
      }
    };
    orchWsRef.current = orchWs;
    
    return () => {
      asrWs.close();
      orchWs.close();
    };
  }, []);

  const handleAudioData = (data: ArrayBuffer) => {
    if (!isRecording) return;
    if (asrWsRef.current?.readyState === WebSocket.OPEN) {
      asrWsRef.current.send(data);
    }
  };

  const startRecording = () => {
    initAudio();
    if (!isRecording) {
      setTranscript({ partial: "", final: "" });
      setIsRecording(true);
      if (asrWsRef.current?.readyState === WebSocket.OPEN) {
        asrWsRef.current.send(JSON.stringify({ type: "start_recording" }));
      }
    }
  };

  const stopRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      if (asrWsRef.current?.readyState === WebSocket.OPEN) {
        asrWsRef.current.send(JSON.stringify({ type: "stop_recording" }));
      }
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };



  const handleTextQuery = async (text: string) => {
    if (!text.trim()) return;
    
    setIsProcessing(true);
    setAnswer("");
    setMetrics({ latency: [], guardrail: null });
    nextAudioTimeRef.current = 0;
    
    if (orchWsRef.current?.readyState === WebSocket.OPEN) {
       orchWsRef.current.send(JSON.stringify({ type: "text_query", query: text, tts: ttsEnabled }));
    } else {
       setAnswer("Error: Orchestrator WebSocket (port 8000) not connected.");
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
              onClick={toggleRecording}
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
                {isRecording ? "STOP RECORDING" : "START RECORDING"}
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
                
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
                  <button
                    onClick={toggleRecording}
                    className="btn-pill-sand"
                    style={{ 
                      flex: 1,
                      background: isRecording ? '#c85a5a' : 'var(--bg-dark)',
                      color: isRecording ? '#fff' : 'var(--text-sand)',
                      userSelect: 'none'
                    }}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                      <Mic size={14} />
                      {isRecording ? "STOP RECORDING" : "START SPEAKING"}
                    </span>
                  </button>
                  <button
                    onClick={() => setTtsEnabled(!ttsEnabled)}
                    className="btn-pill-sand"
                    style={{ 
                      flex: 1,
                      background: ttsEnabled ? 'var(--bg-dark)' : 'transparent',
                      color: ttsEnabled ? 'var(--text-sand)' : 'var(--text-muted)',
                      borderColor: ttsEnabled ? 'var(--border-sand)' : 'var(--border-sand)'
                    }}
                  >
                    VOICE: {ttsEnabled ? "ON" : "OFF"}
                  </button>
                </div>
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

              {/* Metrics & Comprehensive Stats (Developer Mode only) */}
              {mode === "developer" && (
                <>
                  {metrics && (
                    <div className="metrics-row">
                      <div className="card-dark">
                        <div className="card-label">LATENCY PROFILE</div>
                        <LatencyWaterfall timings={metrics.latency || []} />
                      </div>
                      <div className="card-dark" style={{ alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                        <div className="card-label" style={{ width: '100%' }}>CONFIDENCE</div>
                        <ConfidenceGauge score={metrics.guardrail?.top_rerank_score ?? metrics.guardrail?.confidence_score ?? 0} shouldAbstain={metrics.guardrail?.should_abstain ?? false} threshold={0.3} />
                      </div>
                    </div>
                  )}
                  <DeveloperStats metrics={metrics} ttsEnabled={ttsEnabled} />
                </>
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
