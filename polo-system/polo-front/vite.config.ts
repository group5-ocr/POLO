import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // ★ /static을 FastAPI 쪽으로 프록시 (백엔드 포트 8000)
      '/static': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})


