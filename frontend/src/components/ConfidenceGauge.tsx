import React from 'react';

interface ConfidenceGaugeProps {
  score: number; // 0 to 1
  threshold: number;
  shouldAbstain: boolean;
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({ score, shouldAbstain }) => {
  const percentage = Math.round(score * 100);
  const color = shouldAbstain ? 'var(--error)' : 'var(--success)';
  
  // Create a conic gradient for the gauge
  const backgroundStyle = {
    background: `conic-gradient(${color} ${percentage}%, var(--panel-border) ${percentage}%)`
  };

  return (
    <div className="gauge-container">
      <div className="gauge-circle" style={backgroundStyle}>
        <div className="gauge-value" style={{ color }}>
          {percentage}%
        </div>
      </div>
      <div className="gauge-label">
        {shouldAbstain ? 'Abstained (Below Threshold)' : 'Confident (Above Threshold)'}
      </div>
    </div>
  );
};
