import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react()
  ],
  optimizeDeps: {
    exclude: ['@ricky0123/vad-web', 'onnxruntime-web']
  }
})
