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
  
  // Metrics state
  const [timings, setTimings] = useState<StageTiming[]>([]);
  const [confidence, setConfidence] = useState({ score: 0, threshold: 0.5, shouldAbstain: false });
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
        console.log('WebSocket closed, reconnecting in 3s...');
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
    } else if (data.type === 'retrieval_result') {
      // Could show retrieval results
    } else if (data.type === 'guardrail_result') {
      setConfidence({
        score: data.confidence.top_score,
        threshold: data.confidence.threshold,
        shouldAbstain: data.should_abstain
      });
    } else if (data.type === 'generation_result') {
      setAnswer(data.answer);
      setSources(data.sources || []);
    } else if (data.type === 'stage_timing') {
      setTimings(prev => {
        const existing = prev.filter(t => t.stage !== data.stage);
        return [...existing, { stage: data.stage as any, durationMs: data.duration_ms }];
      });
    } else if (data.type === 'error') {
      setErrorMsg(data.message);
    }
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!textInput.trim()) return;

    const query = textInput;
    setTextInput('');
    setFinalText(query);
    setPartialText('');
    setAnswer('');
    setSources([]);
    setTimings([]);
    setErrorMsg('');

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
        if (data.timings) {
          setTimings(Object.entries(data.timings).map(([stage, durationMs]) => ({
            stage: stage as any,
            durationMs: durationMs as number
          })));
        }
        if (data.guardrail) {
          setConfidence({
            score: data.guardrail.confidence,
            threshold: 0.5,
            shouldAbstain: data.guardrail.should_abstain
          });
        }
      } else if (data.error) {
        setErrorMsg(data.error);
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Error executing query');
    }
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    if (!isRecording) {
      // Starting fresh recording
      setFinalText('');
      setPartialText('');
      setAnswer('');
      setSources([]);
      setTimings([]);
      setErrorMsg('');
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Send a start signal if protocol requires it
        wsRef.current.send(JSON.stringify({ type: 'start_recording' }));
      }
    } else {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Send a stop signal
        wsRef.current.send(JSON.stringify({ type: 'stop_recording' }));
      }
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Bragi</h1>
        <p>voice-enabled RAG</p>
      </header>

      <div className="tabs">
        <button 
          className={`tab-btn ${viewMode === 'chat' ? 'active' : ''}`}
          onClick={() => setViewMode('chat')}
        >
          Chat Interaction
        </button>
        <button 
          className={`tab-btn ${viewMode === 'benchmark' ? 'active' : ''}`}
          onClick={() => setViewMode('benchmark')}
        >
          System Benchmark
        </button>
      </div>

      {viewMode === 'chat' ? (
        <>
          <main className="main-content">
            <div className="left-panel">
              <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                <Waveform isRecording={isRecording} />
                
                <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
                  <button 
                    className="btn" 
                    onClick={toggleRecording}
                    style={{ backgroundColor: isRecording ? 'var(--error)' : 'var(--accent-primary)' }}
                  >
                    {isRecording ? 'Stop Recording' : 'Start Voice Input'}
                  </button>
                </div>

                <div style={{ marginTop: '1rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
                  <PartialTranscript partialText={partialText} finalText={finalText} />
                </div>

                <form className="input-group" onSubmit={handleTextSubmit}>
                  <input 
                    type="text" 
                    className="text-input" 
                    placeholder="Or type your query here..." 
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    disabled={isRecording}
                  />
                  <button type="submit" className="btn" disabled={isRecording || !textInput.trim()}>
                    Send
                  </button>
                </form>
              </div>
            </div>

            <div className="right-panel">
              <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
                <h2 style={{ marginTop: 0, marginBottom: '1rem', fontSize: '1.2rem' }}>Response</h2>
                
                {errorMsg && (
                  <div className="error-message">
                    {errorMsg}
                  </div>
                )}
                
                <div className="answer-box">
                  {answer || <span style={{ color: 'var(--text-secondary)' }}>Awaiting response...</span>}
                  
                  {sources.length > 0 && (
                    <div className="answer-sources">
                      <strong>Sources:</strong>
                      <ul style={{ paddingLeft: '1.2rem', margin: '0.5rem 0 0 0' }}>
                        {sources.map((s, i) => (
                          <li key={i}>{s.source_doc || s}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </main>

          <div className="bottom-panel">
            <div className="glass-panel">
              <LatencyWaterfall timings={timings} />
            </div>
            <div className="glass-panel">
              <ConfidenceGauge 
                score={confidence.score} 
                threshold={confidence.threshold} 
                shouldAbstain={confidence.shouldAbstain} 
              />
            </div>
          </div>
        </>
      ) : (
        <BenchmarkRunner />
      )}
    </div>
  );
}

export default App;
