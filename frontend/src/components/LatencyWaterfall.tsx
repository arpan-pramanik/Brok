import React from 'react';

export interface StageTiming {
  stage: string;
  duration_ms?: number;
  durationMs?: number;
  elapsed_ms?: number;
}

interface LatencyWaterfallProps {
  timings: StageTiming[];
}

export const LatencyWaterfall: React.FC<LatencyWaterfallProps> = ({ timings }) => {
  const normalized = (timings || [])
    .map((t: any) => ({
      stage: String(t.stage || '').toUpperCase(),
      ms: Math.round(Number(t.duration_ms ?? t.durationMs ?? t.elapsed_ms ?? 0))
    }))
    .filter(t => (t.ms > 0 || t.stage !== '') && !['ASR', 'STT', 'SPEECH', 'TRANSCRIPTION', 'TTFT'].includes(t.stage));

  const withoutLlm = normalized
    .filter(t => t.stage !== 'GENERATION' && t.stage !== 'TTFT')
    .reduce((acc, curr) => acc + curr.ms, 0);

  const withLlm = normalized
    .filter(t => t.stage !== 'TTFT')
    .reduce((acc, curr) => acc + curr.ms, 0);

  const maxScale = Math.max(withLlm, 120);

  let currentOffset = 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', width: '100%' }}>
      <div style={{ borderBottom: '1px solid rgba(85,242,180,0.3)', paddingBottom: '3px', marginBottom: '2px' }}>
        <div style={{ fontWeight: 'bold', color: '#fff', fontSize: '0.72rem' }}>REAL-TIME TELEMETRY</div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#55f2b4', marginBottom: '2px' }}>
        <span>TOT W/ LLM: <strong style={{ color: '#ffffff' }}>{withLlm}ms</strong></span>
        <span>TOT W/O LLM: <strong style={{ color: '#ffffff' }}>{withoutLlm}ms</strong></span>
      </div>

      {normalized.map((timing, idx) => {
        const leftPercent = Math.min(90, (currentOffset / maxScale) * 100);
        const widthPercent = Math.max(8, (timing.ms / maxScale) * 100);
        currentOffset += timing.ms;

        return (
          <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.7rem' }}>
            <div style={{ width: '70px', fontWeight: 'bold', color: '#55f2b4' }}>{timing.stage}</div>
            <div style={{ flex: 1, height: '10px', background: 'rgba(0,0,0,0.5)', borderRadius: '3px', position: 'relative', overflow: 'hidden' }}>
              <div 
                style={{ 
                  position: 'absolute',
                  top: 0, bottom: 0,
                  left: `${leftPercent}%`, 
                  width: `${widthPercent}%`,
                  backgroundColor: '#1bc5ed',
                  borderRadius: '2px'
                }} 
              />
            </div>
            <div style={{ width: '45px', textAlign: 'right', fontWeight: 'bold', color: '#fff' }}>{timing.ms}ms</div>
          </div>
        );
      })}

      {normalized.length === 0 && (
        <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.72rem', textAlign: 'center', paddingTop: '0.5rem' }}>
          No real-time timing data recorded yet.
        </div>
      )}
    </div>
  );
};
