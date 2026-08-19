import React, { useState, useEffect } from 'react';

interface PokeballOverlayProps {
  onPowerOn?: () => void;
}

export const PokeballOverlay: React.FC<PokeballOverlayProps> = ({ onPowerOn }) => {
  const [isOpened, setIsOpened] = useState<boolean>(() => {
    return sessionStorage.getItem('pokeball_opened') === 'true';
  });

  const [isOpening, setIsOpening] = useState<boolean>(false);

  useEffect(() => {
    if (isOpened) {
      // API Warmup if already opened in session
      wakeUpApi();
    }
  }, [isOpened]);

  const wakeUpApi = () => {
    const orchestratorUrl = import.meta.env.VITE_ORCHESTRATOR_URL || "http://localhost:8000";
    const asrUrl = import.meta.env.VITE_ASR_URL || "http://localhost:8001";

    fetch(`${orchestratorUrl}/health`).catch(() => {});
    fetch(`${asrUrl}/health`).catch(() => {});
  };

  const handlePowerOn = () => {
    if (isOpening || isOpened) return;

    // Play Pokeball opening sound effect
    try {
      const audio = new Audio('/pokeball-opening.mp3');
      audio.currentTime = 0;
      audio.play().catch((err) => console.log('Audio autoplay prevented:', err));
    } catch (e) {
      console.error('Failed to play Pokeball opening audio:', e);
    }

    setIsOpening(true);
    sessionStorage.setItem('pokeball_opened', 'true');

    // Fire background API wakeup to handle cold start
    wakeUpApi();

    if (onPowerOn) {
      onPowerOn();
    }

    // Unmount overlay after CSS slide animation finishes (850ms)
    setTimeout(() => {
      setIsOpened(true);
    }, 850);
  };

  if (isOpened) {
    return null;
  }

  return (
    <div 
      className={`pokeball-overlay-container ${isOpening ? 'opening' : ''}`}
      onClick={handlePowerOn}
      title="Click anywhere to power on Pokédex"
    >
      {/* Top Red Half (with Center Circles Attached) */}
      <div className="pokeball-half-top">
        <div className="pokeball-center-button-wrapper">
          <div className="pokeball-center-outer-circle">
            <div className="pokeball-center-inner-circle" />
          </div>
        </div>
      </div>

      {/* Bottom White Half */}
      <div className="pokeball-half-bottom" />
    </div>
  );
};
