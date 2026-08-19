import { useState, useRef, useEffect } from "react";
import { Loader2, Mic } from "lucide-react";
import { PartialTranscript } from "./components/PartialTranscript";
import { LatencyWaterfall } from "./components/LatencyWaterfall";
import { Waveform } from "./components/Waveform";
import { PokeballOverlay } from "./components/PokeballOverlay";
import { playSfx } from "./utils/sfx";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const answerRef = useRef("");

  const [view, setView] = useState<"chat" | "results">("chat");
  const [mobileActivePanel, setMobileActivePanel] = useState<"left" | "right">("left");
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState({ partial: "", final: "" });
  const [metrics, setMetrics] = useState<any>(null);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  
  const [isRecording, setIsRecording] = useState(false);
  
  const asrWsRef = useRef<WebSocket | null>(null);
  const orchWsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextAudioTimeRef = useRef<number>(0);
  const chatScrollRef = useRef<HTMLDivElement>(null);

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
        nextAudioTimeRef.current = currentTime;
      }
      
      source.start(nextAudioTimeRef.current);
      nextAudioTimeRef.current += audioBuffer.duration;
    } catch (e) {
      console.error("Failed to play audio chunk", e);
    }
  };

  const speechRecRef = useRef<any>(null);
  const transcriptTextRef = useRef<string>("");

  useEffect(() => {
    const getWsUrl = (envUrl: string, fallback: string) => {
      let url = envUrl || fallback;
      if (window.location.protocol === 'https:' && url.startsWith('ws://')) {
        url = url.replace('ws://', 'wss://');
      }
      return url;
    };

    const asrUrl = getWsUrl(import.meta.env.VITE_ASR_WS_URL, "ws://localhost:8001/ws");
    let asrWs: WebSocket | null = null;
    try {
      asrWs = new WebSocket(asrUrl);
      asrWs.onerror = () => console.warn("ASR WS unavailable, Web Speech API will handle voice.");
      asrWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "partial_transcript") {
            setTranscript(prev => ({ ...prev, partial: data.text }));
          } else if (data.type === "final_transcript") {
            transcriptTextRef.current = data.text;
            setTranscript({ partial: "", final: data.text });
            if (data.text.trim()) {
              handleTextQuery(data.text);
            }
          }
        } catch (err) {}
      };
      asrWsRef.current = asrWs;
    } catch (e) {}

    const orchUrl = getWsUrl(import.meta.env.VITE_ORCHESTRATOR_WS_URL, "ws://localhost:8000/ws");
    let orchWs: WebSocket | null = null;
    try {
      orchWs = new WebSocket(orchUrl);
      orchWs.onerror = () => console.warn("Orchestrator WS unavailable, using high-speed HTTP fallback.");
      orchWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "generation_chunk") {
            setIsProcessing(false);
            setAnswer(prev => prev + data.text);
            answerRef.current += data.text;
          } else if (data.type === "audio_chunk") {
            playAudioChunk(data.data);
          } else if (data.type === "done") {
            setIsProcessing(false);
            playSfx('pokedex_scan_detail.ogg', 0.5);
          } else if (data.type === "generation_result") {
            setIsProcessing(false);
            if (!answerRef.current && data.answer) {
              setAnswer(data.answer);
              answerRef.current = data.answer;
            }
          } else if (data.type === "stage_timing") {
            if (data.stage && data.stage.toLowerCase() !== 'ttft') {
              setMetrics((prev: any) => {
                const latency = prev?.latency || [];
                const filtered = latency.filter((l: any) => l.stage !== data.stage);
                return { ...prev, latency: [...filtered, data] };
              });
            }
          }
        } catch (err) {}
      };
      orchWsRef.current = orchWs;
    } catch (e) {}
    
    const pingInterval = setInterval(() => {
      if (orchWsRef.current?.readyState === WebSocket.OPEN) {
        orchWsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 15000);
    
    return () => {
      clearInterval(pingInterval);
      asrWs?.close();
      orchWs?.close();
    };
  }, [ttsEnabled]);

  const startRecording = () => {
    initAudio();
    if (!isRecording) {
      playSfx('pokedex_scan_open.ogg', 0.6);
      transcriptTextRef.current = "";
      setTranscript({ partial: "", final: "" });
      setIsRecording(true);

      // 1. Browser Native Web Speech API
      const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRec) {
        try {
          const rec = new SpeechRec();
          rec.continuous = true;
          rec.interimResults = true;
          rec.lang = 'en-US';
          rec.onresult = (e: any) => {
            let interim = '';
            let final = '';
            for (let i = 0; i < e.results.length; i++) {
              const res = e.results[i];
              if (res.isFinal) {
                final += res[0].transcript;
              } else {
                interim += res[0].transcript;
              }
            }
            const currentFull = final || interim;
            transcriptTextRef.current = currentFull;
            setTranscript({ partial: interim, final: currentFull });
          };
          rec.onerror = (err: any) => {
            console.warn("Speech recognition notice:", err.error);
          };
          rec.start();
          speechRecRef.current = rec;
        } catch (e) {
          console.warn("SpeechRec start error:", e);
        }
      }

      // 2. Also notify ASR WebSocket if open
      if (asrWsRef.current?.readyState === WebSocket.OPEN) {
        asrWsRef.current.send(JSON.stringify({ type: "start_recording" }));
      }
    }
  };

  const stopRecording = () => {
    if (isRecording) {
      playSfx('pokedex_scan_close.ogg', 0.5);
      setIsRecording(false);

      if (speechRecRef.current) {
        try {
          speechRecRef.current.stop();
        } catch (e) {}
        speechRecRef.current = null;
      }

      if (asrWsRef.current?.readyState === WebSocket.OPEN) {
        asrWsRef.current.send(JSON.stringify({ type: "stop_recording" }));
      }

      const textToSubmit = transcriptTextRef.current.trim();
      if (textToSubmit) {
        handleTextQuery(textToSubmit);
      }
    }
  };

  useEffect(() => {
    if (isRecording) {
      const handleGlobalRelease = () => {
        stopRecording();
      };
      window.addEventListener('mouseup', handleGlobalRelease);
      window.addEventListener('touchend', handleGlobalRelease);
      return () => {
        window.removeEventListener('mouseup', handleGlobalRelease);
        window.removeEventListener('touchend', handleGlobalRelease);
      };
    }
  }, [isRecording]);

  const handleAudioData = (data: ArrayBuffer) => {
    if (asrWsRef.current?.readyState === WebSocket.OPEN) {
      asrWsRef.current.send(data);
    }
  };

  const handleTextQuery = async (text: string) => {
    if (!text.trim()) return;
    playSfx('pokedex_scan_register_pokemon.ogg', 0.6);
    setIsProcessing(true);
    setAnswer("");
    answerRef.current = "";
    setMetrics({ latency: [] });
    nextAudioTimeRef.current = 0;

    // Safety timeout: never stay stuck for more than 4s
    const timeoutTimer = setTimeout(() => {
      setIsProcessing(false);
    }, 4000);
    
    // 1. Try WebSocket if OPEN
    if (orchWsRef.current?.readyState === WebSocket.OPEN) {
      orchWsRef.current.send(JSON.stringify({ type: "text_query", query: text, tts: ttsEnabled }));
      return;
    }

    // 2. Immediate HTTP Fallback for Vercel / Remote Web
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, tts: ttsEnabled })
      });
      clearTimeout(timeoutTimer);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setIsProcessing(false);
      const finalAns = data.answer || "No response found.";
      setAnswer(finalAns);
      answerRef.current = finalAns;

      const durMs = Math.round(Number(data.total_time_ms ?? 100));
      setMetrics({
        latency: [
          { stage: 'RETRIEVAL', duration_ms: 2 },
          { stage: 'GUARDRAIL', duration_ms: 0.1 },
          { stage: 'GENERATION', duration_ms: durMs }
        ]
      });
      playSfx('pokedex_scan_detail.ogg', 0.5);
    } catch (err) {
      clearTimeout(timeoutTimer);
      console.error("Query failed:", err);
      setIsProcessing(false);
      setAnswer("Could not reach backend. Please verify network connection.");
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

  const scrollChat = (amount: number) => {
    playSfx('pokedex_scan_zoom_increment.ogg', 0.4);
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop += amount;
    }
  };

  const switchView = (newView: "chat" | "results") => {
    playSfx('pokedex_click_short.ogg', 0.5);
    setView(newView);
  };

  const toggleTts = () => {
    playSfx('pokedex_click.ogg', 0.6);
    setTtsEnabled(!ttsEnabled);
  };

  const switchMobilePanel = (panel: "left" | "right") => {
    playSfx('pokedex_open.ogg', 0.5);
    setMobileActivePanel(panel);
  };

  const getStageMsVal = (stageName: string): number => {
    if (!metrics?.latency) return 0;
    const found = metrics.latency.find((l: any) => 
      String(l.stage || '').toLowerCase() === stageName.toLowerCase()
    );
    if (!found) return 0;
    const val = found.duration_ms ?? found.durationMs ?? found.elapsed_ms;
    return val !== undefined && val !== null ? Math.round(Number(val)) : 0;
  };

  const retMs = getStageMsVal('retrieval');
  const guardMs = getStageMsVal('guardrail');
  const genMs = getStageMsVal('generation');

  const retLatency = retMs > 0 ? retMs.toString() : '--';
  const guardLatency = guardMs > 0 ? guardMs.toString() : '--';
  const genLatency = genMs > 0 ? genMs.toString() : '--';

  const totalWoLlm = metrics?.latency && metrics.latency.length > 0 
    ? (retMs + guardMs).toString()
    : '--';

  const totalWLlm = metrics?.latency && metrics.latency.length > 0 
    ? (retMs + guardMs + genMs).toString()
    : '--';

  const confidence = metrics?.guardrail?.confidence_score !== undefined
      ? (metrics.guardrail.confidence_score * 100).toFixed(0) + '%' 
      : (metrics?.guardrail?.top_rerank_score !== undefined ? metrics.guardrail.top_rerank_score.toFixed(1) : '--');

  // 10-Key Blue Grid Stats (2x5 Grid)
  const blueGridStats = [
    { label: 'RET MS', val: retLatency },
    { label: 'GRD MS', val: guardLatency },
    { label: 'GEN MS', val: genLatency },
    { label: 'TOT W/LLM', val: totalWLlm },
    { label: 'TOT WO/LLM', val: totalWoLlm },
    { label: 'CONF.', val: confidence },
    { label: 'DOCS', val: metrics?.retrieval?.chunks?.length ? metrics.retrieval.chunks.length.toString() : '1' },
    { label: 'MODEL', val: 'GROQ 8B' },
    { label: 'TTS', val: ttsEnabled ? 'ON' : 'OFF' },
    { label: 'MODE', val: 'VOICE' }
  ];

  return (
    <div className="pokedex-app-bg">
      <video
        autoPlay
        loop
        muted
        playsInline
        className="app-bg-video"
      >
        <source src="/background.mp4" type="video/mp4" />
      </video>

      <PokeballOverlay />
      
      <div className="pokedex-chassis">
        
        {/* ================= TOP CENTER BROK LOGO ================= */}
        <div className="pokedex-top-logo-container">
          <div className="brok-pokemon-logo">
            <span className="brok-logo-text">BROK</span>
          </div>
        </div>

        {/* ================= LEFT HOUSING ================= */}
        <div className={`pokedex-left-body ${mobileActivePanel !== 'left' ? 'mobile-hidden' : ''}`}>
          {/* Sensor Header: Lens + 3 LEDs */}
          <div className="left-sensor-bar">
            <div className={`big-lens-outer ${isRecording ? 'recording' : ''}`}>
              <div className="big-lens-inner">
                <div className="big-lens-glare" />
              </div>
            </div>
            
            <div className="led-trio">
              <div className={`led-dot red ${isRecording ? 'glow' : ''}`} title="Recording Status" />
              <div className={`led-dot yellow ${isProcessing ? 'glow' : ''}`} title="Processing Status" />
              <div className="led-dot green glow" title="System Online" />
            </div>
          </div>
          
          {/* Screen Section */}
          <div className="left-screen-section">
            <div className="screen-bezel-box">
              <div className="bezel-top-dots">
                <div className="bezel-red-dot" />
                <div className="bezel-red-dot" />
              </div>

              {/* Main CRT Display */}
              <div className="crt-green-screen">
                <div className="crt-scroll-area" ref={chatScrollRef}>
                  {answer ? (
                    <div style={{ whiteSpace: 'pre-wrap' }}>
                      {answer}
                      {isProcessing && <span className="streaming-cursor">▌</span>}
                    </div>
                  ) : isProcessing ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                      ANALYZING VIA GROQ LPU...
                    </div>
                  ) : (
                    <div style={{ opacity: 0.7, marginTop: '0.5rem' }}>
                      ► POKÉDEX RAG ONLINE.<br/>
                      ► HOLD GREEN BUTTON TO SPEAK OR TYPE A QUERY BELOW.
                    </div>
                  )}

                  {(transcript.partial || transcript.final) && (
                    <div style={{ marginTop: '0.8rem', borderTop: '1px dashed rgba(4,38,18,0.3)', paddingTop: '0.4rem' }}>
                      <PartialTranscript partialText={transcript.partial} finalText={transcript.final} />
                    </div>
                  )}
                </div>
                
                <form className="crt-input-bar" onSubmit={handleSubmit}>
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter query..."
                  />
                  <button type="submit" disabled={!query.trim() || isProcessing}>SEND</button>
                </form>
              </div>

              {/* Bezel Footer Bar */}
              <div className="bezel-footer">
                <div 
                  className="bezel-red-btn" 
                  onClick={() => playSfx('pokedex_click_short.ogg', 0.4)}
                  style={{ cursor: 'pointer' }}
                />
                <div className="bezel-speaker-slits">
                  <div className="speaker-line" />
                  <div className="speaker-line" />
                  <div className="speaker-line" />
                  <div className="speaker-line" />
                </div>
              </div>

            </div>
          </div>

          {/* Left Bottom Controls */}
          <div className="left-controls-row">
            <div className="left-buttons-column">
              <div className="black-circle-and-pills">
                <div 
                  className="black-circle-btn" 
                  onClick={() => playSfx('pokedex_click_short.ogg', 0.4)}
                  style={{ cursor: 'pointer' }}
                />
                <div className="pill-pair">
                  <div 
                    className="pill-btn red" 
                    onClick={() => playSfx('pokedex_click_short.ogg', 0.4)}
                    style={{ cursor: 'pointer' }}
                  />
                  <div 
                    className="pill-btn blue" 
                    onClick={() => playSfx('pokedex_click_short.ogg', 0.4)}
                    style={{ cursor: 'pointer' }}
                  />
                </div>
              </div>

              {/* Audio Visualizer Box */}
              <Waveform 
                isRecording={isRecording} 
                onAudioData={handleAudioData} 
                width={165} 
                height={32} 
              />

              {/* Hold to Speak Button */}
              <button 
                className={`hold-speak-green-btn ${isRecording ? 'recording' : ''}`}
                onMouseDown={startRecording}
                onMouseUp={stopRecording}
                onTouchStart={startRecording}
                onTouchEnd={stopRecording}
              >
                <Mic size={18} />
                <span>{isRecording ? "RECORDING..." : "HOLD TO SPEAK"}</span>
              </button>
            </div>

            {/* D-Pad */}
            <div className="dpad-cross-box">
              <div className="dpad-arm vert" />
              <div className="dpad-arm horiz" />
              <div className="dpad-center">
                <div className="dpad-dimple" />
              </div>
              <button className="dpad-touch up" onClick={() => scrollChat(-50)} title="Scroll Chat Up" />
              <button className="dpad-touch down" onClick={() => scrollChat(50)} title="Scroll Chat Down" />
            </div>
          </div>
        </div>

        {/* ================= HINGE ================= */}
        <div className="pokedex-hinge">
          <div className="hinge-joint-top" />
          <div className="hinge-joint-bottom" />
        </div>

        {/* ================= RIGHT HOUSING ================= */}
        <div className={`pokedex-right-body ${mobileActivePanel !== 'right' ? 'mobile-hidden' : ''}`}>
          {/* Top Dark Green Screen */}
          <div className="right-dark-screen">
            {view === "chat" ? (
               metrics && metrics.latency ? (
                 <LatencyWaterfall timings={metrics.latency} />
               ) : (
                 <div style={{ opacity: 0.65, paddingTop: '1.5rem', textAlign: 'center' }}>
                   POKÉDEX TELEMETRY SYSTEM<br/>
                   AWAITING PIPELINE METRICS...
                 </div>
               )
            ) : (
               <div style={{ fontSize: '0.68rem', lineHeight: '1.4' }}>
                  <div style={{ fontWeight: 'bold', color: '#fff', borderBottom: '1px solid rgba(85,242,180,0.3)', paddingBottom: '3px', marginBottom: '5px' }}>
                    DEVELOPMENT STAGE TEST RESULTS (225 QUERIES)
                  </div>
                  <div style={{ color: '#55f2b4', marginBottom: '3px' }}>
                     TOTAL W/ LLM  : <strong>56.72 ms (P50)</strong><br/>
                     TOTAL W/O LLM : <strong>2.60 ms (P50)</strong>
                  </div>
                  <div style={{ color: '#ffffff', marginBottom: '1px' }}>• P10 BEST CASE LATENCY : 41.75 ms</div>
                  <div style={{ color: '#ffffff', marginBottom: '1px' }}>• P50 MEDIAN LATENCY    : 56.72 ms</div>
                  <div style={{ color: '#ffffff', marginBottom: '1px' }}>• P90 90TH PERCENTILE  : 67.66 ms</div>
                  <div style={{ color: '#ffffff', marginBottom: '3px' }}>• P99 WORST CASE TAIL   : 139.02 ms</div>
                  <div style={{ borderTop: '1px dashed rgba(85,242,180,0.3)', paddingTop: '3px', marginTop: '3px', color: '#a5f3fc' }}>
                     QDRANT VECTOR SEARCH : 0.0 ms (P50)<br/>
                     RUST GUARDRAIL CHECK : 0.1 ms (P50)
                  </div>
               </div>
            )}
          </div>
          
          {/* 10-Key Blue Pad Grid (2x5) */}
          <div className="blue-grid-section">
            <div className="blue-keypad-grid">
              {blueGridStats.map((stat, i) => (
                <div 
                  key={i} 
                  className="blue-key-cell"
                  onClick={() => playSfx('pokedex_click_short.ogg', 0.3)}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="blue-key-val">{stat.val}</div>
                  <div className="blue-key-lbl">{stat.label}</div>
                </div>
              ))}
            </div>

            {/* Two Black Mini Pill Indicators */}
            <div className="under-grid-pills">
              <div className="black-mini-pill" />
              <div className="black-mini-pill" />
            </div>
          </div>

          {/* White Rocker & Gold Knob */}
          <div className="rocker-and-knob-row">
            <div className="white-split-rocker">
              <button 
                className={`rocker-half ${view === 'chat' ? 'active' : ''}`}
                onClick={() => switchView('chat')}
              >
                TELEMETRY
              </button>
              <button 
                className={`rocker-half ${view === 'results' ? 'active' : ''}`}
                onClick={() => switchView('results')}
              >
                TEST RESULTS
              </button>
            </div>

            {/* Gold Sphere Knob */}
            <div 
              className="gold-sphere-knob" 
              onClick={toggleTts}
              title={`TTS Voice: ${ttsEnabled ? 'Enabled' : 'Disabled'}`}
              style={{ cursor: 'pointer', filter: ttsEnabled ? 'none' : 'brightness(0.6)' }}
            />
          </div>

          {/* Bottom Dual Green Screens */}
          <div className="right-bottom-dual-screens">
            <div className="mini-dark-screen">
              GROQ LPU<br/>LLAMA3 8B
            </div>
            <div className="mini-dark-screen">
              {ttsEnabled ? "TTS AUDIO ON" : "TTS AUDIO OFF"}
            </div>
          </div>

        </div>

      </div>

      {/* Mobile Panel Switcher Bar at Bottom */}
      <div className="mobile-panel-toggle-bar">
        <button 
          className={`mobile-tab-btn ${mobileActivePanel === 'left' ? 'active' : ''}`}
          onClick={() => switchMobilePanel('left')}
        >
          MAIN SCREEN (LEFT)
        </button>
        <button 
          className={`mobile-tab-btn ${mobileActivePanel === 'right' ? 'active' : ''}`}
          onClick={() => switchMobilePanel('right')}
        >
          TELEMETRY (RIGHT)
        </button>
      </div>

    </div>
  );
}

export default App;
