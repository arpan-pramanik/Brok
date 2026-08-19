import React, { useEffect, useRef } from 'react';

interface WaveformProps {
  isRecording: boolean;
  onAudioData?: (data: ArrayBuffer) => void;
  width?: number;
  height?: number;
}

export const Waveform: React.FC<WaveformProps> = ({ 
  isRecording, 
  onAudioData,
  width = 300,
  height = 36
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const isRecordingRef = useRef<boolean>(isRecording);

  // Update recording ref synchronously
  useEffect(() => {
    isRecordingRef.current = isRecording;
    if (isRecording && !animationRef.current) {
      draw();
    }
  }, [isRecording]);

  // Pre-initialize microphone stream once on mount for INSTANT 0ms activation
  useEffect(() => {
    let isMounted = true;

    const initMicrophone = async () => {
      try {
        if (streamRef.current) return;
        
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (!isMounted) {
          stream.getTracks().forEach(t => t.stop());
          return;
        }
        streamRef.current = stream;

        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        const audioCtx = new AudioContextClass({ sampleRate: 16000 });
        audioContextRef.current = audioCtx;

        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 128;
        analyserRef.current = analyser;

        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);

        const processor = audioCtx.createScriptProcessor(2048, 1, 1);
        processor.onaudioprocess = (e) => {
          if (!isRecordingRef.current) return;
          const inputData = e.inputBuffer.getChannelData(0);
          if (onAudioData) {
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
        console.error("Microphone pre-warm error:", err);
      }
    };

    initMicrophone();

    return () => {
      isMounted = false;
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (processorRef.current) {
        processorRef.current.onaudioprocess = null;
        processorRef.current.disconnect();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, []);

  const draw = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    const dataArray = dataArrayRef.current;
    
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (isRecordingRef.current) {
      animationRef.current = requestAnimationFrame(draw);
    } else {
      animationRef.current = null;
    }
    
    if (analyser && dataArray) {
      // @ts-ignore
      analyser.getByteFrequencyData(dataArray);
      
      ctx.fillStyle = '#042612';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      const barWidth = (canvas.width / dataArray.length) * 1.8;
      let x = 0;
      
      for (let i = 0; i < dataArray.length; i++) {
        const val = isRecordingRef.current ? dataArray[i] : 0;
        const barHeight = (val / 255) * canvas.height;
        
        const greenVal = Math.min(255, 180 + val);
        ctx.fillStyle = `rgb(40, ${greenVal}, 100)`;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
        
        x += barWidth + 1;
      }
    } else {
      ctx.fillStyle = '#042612';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'rgba(72, 184, 104, 0.2)';
      ctx.fillRect(0, canvas.height / 2 - 1, canvas.width, 2);
    }
  };

  return (
    <div style={{
      width: '100%',
      height: `${height}px`,
      background: '#042612',
      border: '2.5px solid #000000',
      borderRadius: '6px',
      overflow: 'hidden',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <canvas 
        ref={canvasRef} 
        width={width} 
        height={height} 
        style={{ width: '100%', height: '100%', display: 'block' }} 
      />
    </div>
  );
};
