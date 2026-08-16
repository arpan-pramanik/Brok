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
  
  // Chat state
  const [partialText, setPartialText] = useState('');
  const [finalText, setFinalText] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  // Metrics state
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

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
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
        <div className="brand">
          <div className="logo-badge">B</div>
          <div>
            <div className="brand-name">BRAGI</div>
            <div className="brand-tagline">Voice-Enabled RAG System</div>
          </div>
        </div>

        <nav className="tabs-nav">
          <button 
            className={`nav-tab ${viewMode === 'chat' ? 'active' : ''}`}
            onClick={() => setViewMode('chat')}
          >
            Chat
          </button>
          <button 
            className={`nav-tab ${viewMode === 'benchmark' ? 'active' : ''}`}
            onClick={() => setViewMode('benchmark')}
          >
            Benchmark
          </button>
        </nav>
      </header>

      {viewMode === 'chat' ? (
        <>
          <main className="main-grid">
            <div className="glass-panel">
              <div className="panel-title">
                <span className="panel-title-dot"></span> Input Stream
              </div>

              <div className="waveform-box">
                <Waveform isRecording={isRecording} />
              </div>

              <button 
                className={`btn-voice ${isRecording ? 'recording' : ''}`} 
                onClick={toggleRecording}
              >
                {isRecording ? 'Stop Recording' : 'Start Voice Input'}
              </button>

              <div className="transcript-area">
                <PartialTranscript partialText={partialText} finalText={finalText} />
              </div>

              <form className="input-bar" onSubmit={handleTextSubmit}>
                <input 
                  type="text" 
                  className="text-input" 
                  placeholder="Ask any question..." 
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  disabled={isRecording || isLoading}
                />
                <button type="submit" className="btn-primary" disabled={isRecording || isLoading || !textInput.trim()}>
                  {isLoading ? '...' : 'Send'}
                </button>
              </form>
            </div>

            <div className="glass-panel">
              <div className="panel-title">
                <span className="panel-title-dot" style={{ background: 'var(--accent-cyan)' }}></span> Synthesis Engine
              </div>
              
              {errorMsg && (
                <div style={{ color: 'var(--status-error)', marginBottom: '1rem', fontSize: '0.9rem' }}>
                  {errorMsg}
                </div>
              )}

              <div className="response-container">
                {isLoading ? (
                  <span style={{ color: 'var(--text-muted)' }}>Retrieving and generating answer...</span>
                ) : answer ? (
                  answer
                ) : (
                  <span style={{ color: 'var(--text-muted)' }}>Ready for query input</span>
                )}

                {sources.length > 0 && (
                  <div className="source-list">
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>CITED SOURCES</div>
                    {sources.map((s, i) => (
                      <span key={i} className="source-tag">{typeof s === 'string' ? s : s.source_doc || 'Doc'}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </main>

          <footer className="bottom-metrics">
            <div className="glass-panel">
              <div className="panel-title" style={{ fontSize: '0.875rem' }}>Latency Waterfall (ms)</div>
              <LatencyWaterfall timings={timings} />
            </div>
            <div className="glass-panel">
              <div className="panel-title" style={{ fontSize: '0.875rem' }}>Guardrail Threshold</div>
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
