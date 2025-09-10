/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string
  // 다른 환경변수도 여기서 선언 가능 (반드시 VITE_ 접두어)
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}