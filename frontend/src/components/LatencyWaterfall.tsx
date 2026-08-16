import React from 'react';

export interface StageTiming {
  stage: 'ASR' | 'Retrieval' | 'Guardrail' | 'Generation';
  durationMs: number;
}

interface LatencyWaterfallProps {
  timings: StageTiming[];
}

const STAGE_COLORS: Record<string, string> = {
  ASR: 'var(--accent-primary)',
  Retrieval: 'var(--accent-secondary)',
  Guardrail: 'var(--warning)',
  Generation: 'var(--success)'
};

export const LatencyWaterfall: React.FC<LatencyWaterfallProps> = ({ timings }) => {
  const totalLatency = timings.reduce((acc, curr) => acc + curr.durationMs, 0);
  const maxScale = Math.max(totalLatency, 1000); // at least 1s scale for better viz

  let currentOffset = 0;

  return (
    <div className="waterfall-container">
      <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1rem', color: 'var(--text-primary)' }}>
        Latency Waterfall
      </h3>
      {timings.map((timing) => {
        const leftPercent = (currentOffset / maxScale) * 100;
        const widthPercent = (timing.durationMs / maxScale) * 100;
        const color = STAGE_COLORS[timing.stage] || 'var(--text-secondary)';
        
        currentOffset += timing.durationMs;

        return (
          <div key={timing.stage} className="waterfall-row">
            <div className="waterfall-label">{timing.stage}</div>
            <div className="waterfall-track">
              <div 
                className="waterfall-bar" 
                style={{ 
                  left: `${leftPercent}%`, 
                  width: `${widthPercent}%`,
                  backgroundColor: color
                }} 
              />
            </div>
            <div className="waterfall-time">{timing.durationMs}ms</div>
          </div>
        );
      })}
      {timings.length === 0 && (
        <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', textAlign: 'center' }}>
          No timing data available yet.
        </div>
      )}
    </div>
  );
};
