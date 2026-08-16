import React from 'react';

interface PartialTranscriptProps {
  partialText: string;
  finalText: string;
}

export const PartialTranscript: React.FC<PartialTranscriptProps> = ({ partialText, finalText }) => {
  return (
    <div className="transcript-box">
      <span className="final-text">{finalText}</span>
      {finalText && partialText && ' '}
      <span className="partial-text">{partialText}</span>
      {!finalText && !partialText && (
        <span style={{ color: 'var(--text-secondary)' }}>Waiting for speech...</span>
      )}
    </div>
  );
};
