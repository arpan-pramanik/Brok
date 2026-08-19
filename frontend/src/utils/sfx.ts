export const playSfx = (name: string, volume = 0.5) => {
  try {
    const audio = new Audio(`/sfx/${name}`);
    audio.volume = volume;
    audio.currentTime = 0;
    audio.play().catch(() => {});
  } catch (e) {
    // Ignore audio play errors
  }
};
