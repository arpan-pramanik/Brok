import React, { useState, useEffect, useRef } from 'react';
import './index.css';
import { Waveform } from './components/Waveform';
import { PartialTranscript } from './components/PartialTranscript';
import { LatencyWaterfall, type StageTiming } from './components/LatencyWaterfall';
import { ConfidenceGauge } from './components/ConfidenceGauge';
import { BenchmarkRunner } from './components/BenchmarkRunner';

type ViewMode = 'chat' | 'benchmark';

function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('chat');
  const [isRecording, setIsRecording] = useState(false);
  const [textInput, setTextInput] = useState('');
  
  const [partialText, setPartialText] = useState('');
  const [finalText, setFinalText] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const [timings, setTimings] = useState<StageTiming[]>([]);
  const [confidence, setConfidence] = useState({ score: 0, threshold: 0.3, shouldAbstain: false });
  const [errorMsg, setErrorMsg] = useState('');

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onopen = () => {
        console.log('WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (e) {
          console.error('Error parsing WS message:', e);
        }
      };

      ws.onclose = () => {
        setTimeout(connectWebSocket, 3000);
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('WebSocket connection failed:', e);
    }
  };

  const handleWebSocketMessage = (data: any) => {
    if (data.type === 'partial_transcript') {
      if (data.is_final) {
        setFinalText(prev => prev + (prev ? ' ' : '') + data.text);
        setPartialText('');
      } else {
        setPartialText(data.text);
      }
    } else if (data.type === 'guardrail_result') {
      setConfidence({
        score: data.confidence.top_score,
        threshold: data.confidence.threshold,
        shouldAbstain: data.should_abstain
      });
    } else if (data.type === 'generation_result') {
      setAnswer(data.answer);
      setSources(data.sources || []);
      setIsLoading(false);
    } else if (data.type === 'stage_timing') {
      setTimings(prev => {
        const existing = prev.filter(t => t.stage !== data.stage);
        return [...existing, { stage: data.stage as any, durationMs: data.duration_ms }];
      });
    } else if (data.type === 'error') {
      setErrorMsg(data.message);
      setIsLoading(false);
    }
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim() || isLoading) return;

    const query = textInput;
    setTextInput('');
    setFinalText(query);
    setPartialText('');
    setAnswer('');
    setSources([]);
    setTimings([]);
    setErrorMsg('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      
      const data = await response.json();
      if (data.answer) {
        setAnswer(data.answer);
        setSources(data.sources || []);
        if (data.stages) {
          setTimings(data.stages.map((s: any) => ({
            stage: s.stage,
            durationMs: s.duration_ms
          })));
        }
        if (data.guardrail) {
          setConfidence({
            score: data.guardrail.confidence.top_score,
            threshold: data.guardrail.confidence.threshold,
            shouldAbstain: data.guardrail.should_abstain
          });
        }
      } else if (data.error) {
        setErrorMsg(data.error);
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Error executing query');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      setFinalText('');
      setPartialText('');
      setAnswer('');
      setSources([]);
      setTimings([]);
      setErrorMsg('');
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'start_recording' }));
      }
    } else {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'stop_recording' }));
      }
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div>
          <div className="brand-logo">BRAGI</div>
          <div className="brand-sub">DEVELOPER EDITION</div>
        </div>

        <nav className="nav-tabs">
          <button 
            className={`nav-btn ${viewMode === 'chat' ? 'active' : ''}`}
            onClick={() => setViewMode('chat')}
          >
            INTERACTION
          </button>
          <button 
            className={`nav-btn ${viewMode === 'benchmark' ? 'active' : ''}`}
            onClick={() => setViewMode('benchmark')}
          >
            BENCHMARK
          </button>
        </nav>
      </header>

      <section className="hero-banner">
        <div className="hero-meta">+++ VOICE RAG ARCHITECTURE +++</div>
        <h1 className="hero-heading">
          A LEGEND BROUGHT TO LIFE IN THE HEART OF INTELLIGENCE, UNITING VOICE AND RETRIEVAL
        </h1>
      </section>

      {viewMode === 'chat' ? (
        <>
          <main className="workspace-grid">
            <div className="card-dark">
              <div className="card-label">
                <span>01 // INPUT STREAM & SPEECH</span>
                <span>[ 16KHZ VAD ]</span>
              </div>

              <div className="waveform-frame">
                <Waveform isRecording={isRecording} />
              </div>

              <div style={{ margin: '1rem 0' }}>
                <button 
                  className={isRecording ? 'btn-pill-sand' : 'btn-pill-outline'} 
                  onClick={toggleRecording}
                  style={{ width: '100%' }}
                >
                  {isRecording ? 'PAUSE VOICE STREAM' : 'START VOICE STREAM'}
                </button>
              </div>

              <div style={{ flex: 1, minHeight: '120px', background: 'rgba(10,9,8,0.5)', padding: '1rem', border: '1px solid var(--border-sand)', marginBottom: '1rem' }}>
                <PartialTranscript partialText={partialText} finalText={finalText} />
              </div>

              <form style={{ display: 'flex', gap: '0.5rem' }} onSubmit={handleTextSubmit}>
                <input 
                  type="text" 
                  className="input-field" 
                  placeholder="Enter query..." 
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  disabled={isRecording || isLoading}
                />
                <button type="submit" className="btn-pill-sand" disabled={isRecording || isLoading || !textInput.trim()}>
                  {isLoading ? '...' : 'SEND'}
                </button>
              </form>
            </div>

            <div className="card-sand">
              <div className="card-label">
                <span>02 // RAG SYNTHESIS ENGINE</span>
                <span>[ QWEN2.5 + BEDROCK ]</span>
              </div>

              {errorMsg && (
                <div style={{ color: '#8b0000', marginBottom: '1rem', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                  {errorMsg}
                </div>
              )}

              <div className="response-view">
                {isLoading ? (
                  <span>SYNTHESIZING ANSWER FROM RETRIEVED CHUNKS...</span>
                ) : answer ? (
                  answer
                ) : (
                  <span>AWAITING STREAM OR TEXT INPUT...</span>
                )}
              </div>

              {sources.length > 0 && (
                <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(20,18,16,0.2)' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', letterSpacing: '0.2em', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
                    REFERENCED SOURCES:
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                    {sources.map((s, i) => (
                      <span key={i} style={{ background: 'rgba(20,18,16,0.1)', padding: '0.2rem 0.6rem', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', border: '1px solid rgba(20,18,16,0.2)' }}>
                        {typeof s === 'string' ? s : s.source_doc || 'Doc'}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </main>

          <footer className="metrics-row">
            <div className="card-dark">
              <div className="card-label">
                <span>STAGE LATENCY WATERFALL</span>
                <span>[ MILLISECONDS ]</span>
              </div>
              <LatencyWaterfall timings={timings} />
            </div>

            <div className="card-dark">
              <div className="card-label">
                <span>GUARDRAIL CONFIDENCE</span>
                <span>[ CALIBRATED ]</span>
              </div>
              <ConfidenceGauge 
                score={confidence.score} 
                threshold={confidence.threshold} 
                shouldAbstain={confidence.shouldAbstain} 
              />
            </div>
          </footer>
        </>
      ) : (
        <BenchmarkRunner />
      )}
    </div>
  );
}

export default App;
