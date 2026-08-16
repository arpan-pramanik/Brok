import React, { useState } from 'react';

interface BenchmarkResult {
  p50: Record<string, number>;
  p70: Record<string, number>;
  p100: Record<string, number>;
  total_latencies: number[];
}

export const BenchmarkRunner: React.FC = () => {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runBenchmark = async () => {
    setRunning(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/api/benchmark', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Benchmark failed');
      }
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Error running benchmark');
    } finally {
      setRunning(false);
    }
  };

  const stages = result ? Object.keys(result.p50) : [];

  return (
    <div className="benchmark-container">
      <div>
        <button 
          className="btn" 
          onClick={runBenchmark} 
          disabled={running}
        >
          {running ? 'Running Benchmark...' : 'Run Benchmark'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {result && (
        <>
          <div className="benchmark-stats">
            <div className="stat-card">
              <div className="stat-label">Total Queries</div>
              <div className="stat-value">{result.total_latencies.length}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Median Total Latency</div>
              <div className="stat-value">
                {result.total_latencies.length > 0 
                  ? result.total_latencies.sort((a,b)=>a-b)[Math.floor(result.total_latencies.length/2)] 
                  : 0}ms
              </div>
            </div>
          </div>

          <div className="glass-panel" style={{ overflowX: 'auto' }}>
            <h3>Percentiles by Stage (ms)</h3>
            <table>
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>P50</th>
                  <th>P70</th>
                  <th>P100</th>
                </tr>
              </thead>
              <tbody>
                {stages.map(stage => (
                  <tr key={stage}>
                    <td>{stage}</td>
                    <td>{result.p50[stage]}</td>
                    <td>{result.p70[stage]}</td>
                    <td>{result.p100[stage]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};
