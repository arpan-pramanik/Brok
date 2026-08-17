import React, { useEffect, useRef } from 'react';

interface WaveformProps {
  isRecording: boolean;
  onAudioData?: (data: ArrayBuffer) => void;
}

export const Waveform: React.FC<WaveformProps> = ({ isRecording, onAudioData }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const isRecordingRef = useRef<boolean>(isRecording);

  useEffect(() => {
    isRecordingRef.current = isRecording;
    if (isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
    return () => {
      stopRecording();
    };
  }, [isRecording]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const audioCtx = new window.AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;
      
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      
      // Capture audio for ASR
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e) => {
        if (!isRecordingRef.current) return;
        const inputData = e.inputBuffer.getChannelData(0); // Float32Array natively
        if (onAudioData) {
            // Need to copy the buffer, otherwise ScriptProcessor reuses the same buffer for next event
            onAudioData(new Float32Array(inputData).buffer);
        }
      };
      
      source.connect(processor);
      processor.connect(audioCtx.destination);
      processorRef.current = processor;
      
      const bufferLength = analyser.frequencyBinCount;
      dataArrayRef.current = new Uint8Array(bufferLength);
      
      draw();
    } catch (err) {
      console.error("Error accessing microphone:", err);
    }
  };

  const stopRecording = () => {
    isRecordingRef.current = false;
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.enabled = false;
        track.stop();
      });
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    // Clear canvas
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  };

  const draw = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    const dataArray = dataArrayRef.current;
    
    if (!canvas || !analyser || !dataArray) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    animationRef.current = requestAnimationFrame(draw);
    
    // @ts-ignore
    analyser.getByteFrequencyData(dataArray);
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    const barWidth = (canvas.width / dataArray.length) * 2.5;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height;
      
      ctx.fillStyle = `rgb(34, 197, 94)`; // success color
      ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
      
      x += barWidth + 1;
    }
  };

  return (
    <div className="waveform-container">
      <canvas ref={canvasRef} className="waveform-canvas" width={800} height={120} />
    </div>
  );
};
