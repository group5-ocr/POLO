import React, { useEffect, useMemo, useState } from 'react'

// Vite 환경 변수 (루트 .env: VITE_API_BASE=http://localhost:8000)
const BASE_URL = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

// ---------------- Types ----------------
interface UploadResult {
  filename: string
  file_size: number
  extracted_text_length: number
  extracted_text_preview: string
  easy_json: any
  status: string
  raw_file_path?: string
  json_file_path?: string
}

interface ProcessingInfo {
  gpu_used?: boolean
  inference_time?: number
  total_time?: number
  input_length?: number
  output_length?: number
}

interface RecentItem {
  filename: string
  original_filename: string
  processed_at: string
  title?: string
  plain_summary?: string
  processing_info?: ProcessingInfo
}

// ---------------- Helpers ----------------
const box: React.CSSProperties = { padding: 15, borderRadius: 10, border: '1px solid #e5e7eb', background: '#fff' }
const btn: React.CSSProperties = { padding: '10px 16px', borderRadius: 8, border: '1px solid #d1d5db', cursor: 'pointer', background: '#fff' }
const btnPrimary: React.CSSProperties = { ...btn, background: '#2563eb', borderColor: '#2563eb', color: '#fff' }
const btnMuted: React.CSSProperties = { ...btn, color: '#6b7280', cursor: 'not-allowed', background: '#e5e7eb', borderColor: '#e5e7eb' }

const fmtDate = (s?: string) => {
  if (!s) return '-'
  try { return new Date(s).toLocaleString() } catch { return s }
}

const isNum = (v: any): v is number => typeof v === 'number' && Number.isFinite(v)

async function safeParseJson(resp: Response) {
  const text = await resp.text()
  try { return JSON.parse(text) } catch { return { detail: text || 'Unknown error' } }
}

// ---------------- Recent Inline ----------------
function RecentInline() {
  const [limit, setLimit] = useState(10)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [items, setItems] = useState<RecentItem[]>([])
  const [detail, setDetail] = useState<any | null>(null)
  const [detailName, setDetailName] = useState<string>('')

  const fetchRecent = async () => {
    setLoading(true); setError(null)
    try {
      const r = await fetch(`${BASE_URL}/easy/results/recent?limit=${limit}`)
      const data = await r.json()
      setItems(data.results || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : '최근 결과 조회 실패')
    } finally { setLoading(false) }
  }

  useEffect(() => { fetchRecent() }, [])
  const list = useMemo(() => items, [items])

  const seeDetail = async (filename: string) => {
    setDetail(null); setDetailName(filename)
    try {
      const r = await fetch(`${BASE_URL}/easy/results/${encodeURIComponent(filename)}`)
      const data = await r.json()
      setDetail(data.data)
    } catch (e) {
      setDetail({ error: e instanceof Error ? e.message : '조회 실패' })
    }
  }

  return (
    <div style={{ ...box }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <h3 style={{ margin: 0, fontSize: 18 }}>최근 변환 결과</h3>
        <div style={{ display: 'flex', gap: 8 }}>
          <select value={limit} onChange={(e) => setLimit(Number(e.target.value))} style={{ padding: '6px 10px', borderRadius: 8, border: '1px solid #d1d5db' }}>
            {[5, 10, 20, 30, 50].map(n => <option key={n} value={n}>{n}개</option>)}
          </select>
          <button onClick={fetchRecent} style={btn}>{loading ? '불러오는 중…' : '새로고침'}</button>
        </div>
      </div>

      {error && (
        <div style={{ ...box, background: '#FEF2F2', borderColor: '#FCA5A5', color: '#991B1B' }}>오류: {error}</div>
      )}

      {list.length === 0 && !loading && <div style={{ color: '#6b7280' }}>데이터가 없습니다.</div>}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {list.map((it) => {
          const gpu = it.processing_info?.gpu_used ? '✅ GPU' : 'CPU'
          const tInf = it.processing_info?.inference_time
          const tTot = it.processing_info?.total_time
          const docId = (it as any)?.doc_id || (it as any)?.processing_info?.doc_id
          return (
            <div key={it.filename} style={{ ...box, background: '#f9fafb' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 600, wordBreak: 'break-all' }}>{it.filename}</div>
                  <div style={{ color: '#6b7280', fontSize: 14 }}>원본: {it.original_filename || 'Unknown'}</div>
                  <div style={{ color: '#6b7280', fontSize: 14 }}>처리시각: {fmtDate(it.processed_at)}</div>
                  {it.title && <div style={{ marginTop: 4 }}>제목: {it.title}</div>}
                  {it.plain_summary && <div style={{ marginTop: 4, color: '#374151' }}>요약: {it.plain_summary}</div>}
                  <div style={{ marginTop: 4, fontSize: 12, color: '#6b7280' }}>
                    {gpu}
                    {isNum(tInf) && ` · 추론 ${tInf.toFixed(2)}s`}
                    {isNum(tTot) && ` · 총 ${tTot.toFixed(2)}s`}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
                  <button style={btn} onClick={() => seeDetail(it.filename)}>상세보기</button>
                  <button style={btn} onClick={() => window.open(`${BASE_URL}/easy/download/${encodeURIComponent(it.filename)}`, '_blank')}>JSON 다운로드</button>
                  {docId && (
                    <button style={btn} onClick={() => window.open(`${BASE_URL}/easy/download/raw/${encodeURIComponent(docId)}`, '_blank')}>원본 PDF</button>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {detail && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>상세: {detailName}</div>
          <pre style={{ padding: 12, background: '#111827', color: '#e5e7eb', borderRadius: 10, maxHeight: 420, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(detail, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// ---------------- Main Upload Page ----------------
export default function Upload() {
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<UploadResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const onChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    if (f.type !== 'application/pdf') {
      setError('PDF 파일만 업로드할 수 있습니다.')
      e.currentTarget.value = ''
      return
    }
    if (f.size > 50 * 1024 * 1024) {
      setError('파일은 50MB 이하만 가능합니다.')
      e.currentTarget.value = ''
      return
    }
    setSelectedFile(f); setError(null)
  }

  const checkModelStatus = async () => {
    try {
      const response = await fetch(`${BASE_URL}/easy/model-status`)
      const data = await response.json()
      if (data.model_available) {
        alert('✅ AI 모델이 정상적으로 연결되어 있습니다!')
      } else {
        alert('❌ AI 모델 서비스가 사용 불가능합니다. 서버/도커를 확인해주세요.')
      }
    } catch {
      alert('❌ 서버 연결에 실패했습니다.')
    }
  }

  const uploadFile = async (file: File) => {
    setUploading(true); setError(null); setResult(null); setProgress('파일 업로드 중...')
    try {
      const formData = new FormData(); formData.append('file', file)
      setProgress('PDF 텍스트 추출 중...')
      const response = await fetch(`${BASE_URL}/easy/convert`, { method: 'POST', body: formData })
      if (!response.ok) {
        const err = await safeParseJson(response)
        throw new Error(err.detail || '업로드 실패')
      }
      setProgress('AI 모델로 변환 중... (시간이 걸릴 수 있습니다)')
      const data = await safeParseJson(response) as UploadResult
      setResult(data); setProgress('완료!')
    } catch (err) {
      setError(err instanceof Error ? err.message : '업로드 중 오류가 발생했습니다.')
      setProgress('')
    } finally { setUploading(false) }
  }

  const processing = result?.easy_json?.processing_info as ProcessingInfo | undefined

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto', padding: 16 }}>
      <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 12 }}>A!POLO</h2>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <button onClick={checkModelStatus} style={btn}>모델 상태 확인하기</button>
        <label style={{ ...btn, display: 'inline-flex', alignItems: 'center', gap: 8 }}>
          <input type="file" accept="application/pdf" onChange={onChange} disabled={uploading} style={{ display: 'none' }} />
          <span role="img" aria-label="file">📄</span> 파일 선택
        </label>
        <button
          onClick={() => selectedFile && uploadFile(selectedFile)}
          disabled={!selectedFile || uploading}
          style={!selectedFile || uploading ? btnMuted : btnPrimary}
        >
          변환하기
        </button>
      </div>

      {selectedFile && (
        <div style={{ marginBottom: 8, color: '#374151' }}>선택된 파일: <b>{selectedFile.name}</b> ({(selectedFile.size/1024/1024).toFixed(2)} MB)</div>
      )}

      {/* Progress */}
      {uploading && (
        <div style={{ ...box, background: '#E0F2FE', borderColor: '#93C5FD' }}>
          <div style={{ marginBottom: 8 }}>
            <div style={{ width: '100%', height: 8, background: '#e5e7eb', borderRadius: 999, overflow: 'hidden' }}>
              <div style={{ width: '100%', height: '100%', background: '#3B82F6', animation: 'pulse 1.5s ease-in-out infinite' }} />
            </div>
          </div>
          <div style={{ fontWeight: 600, color: '#1D4ED8' }}>{progress}</div>
          <div style={{ marginTop: 4, fontSize: 13, color: '#6b7280' }}>GPU 가속으로 처리 중입니다. 잠시만 기다려주세요…</div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ ...box, background: '#FEF2F2', borderColor: '#FCA5A5', color: '#991B1B', marginTop: 12 }}>
          <b>오류:</b> {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div style={{ ...box, marginTop: 16 }}>
          <h3 style={{ marginTop: 0 }}>처리 결과</h3>
          <div style={{ ...box, background: '#DCFCE7', borderColor: '#86EFAC' }}>
            <p><b>파일명:</b> {result.filename}</p>
            <p><b>파일 크기:</b> {(result.file_size / 1024).toFixed(2)} KB</p>
            <p><b>추출된 텍스트 길이:</b> {result.extracted_text_length} 문자</p>
            <p><b>상태:</b> {result.status}</p>
          </div>

          <div style={{ margin: '12px 0' }}>
            <h4>추출된 텍스트 미리보기</h4>
            <div style={{ padding: 10, background: '#F8FAFC', border: '1px solid #E5E7EB', borderRadius: 8, maxHeight: 200, overflowY: 'auto', whiteSpace: 'pre-wrap' }}>
              {result.extracted_text_preview}
            </div>
          </div>

          <div>
            <h4>AI 변환 결과 (JSON)</h4>
            <pre style={{ padding: 12, background: '#F1F5F9', border: '1px solid #DBEAFE', borderRadius: 8, maxHeight: 420, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(result.easy_json, null, 2)}
            </pre>
          </div>

          {processing && (
            <div style={{ marginTop: 12, ...box, background: '#F8FAFC' }}>
              <h4 style={{ marginTop: 0 }}>처리 정보</h4>
              <p><b>GPU 사용:</b> {processing.gpu_used ? '✅ 예' : '❌ 아니오'}</p>
              <p><b>추론 시간:</b> {isNum(processing.inference_time) ? processing.inference_time!.toFixed(2) + '초' : '-'} </p>
              <p><b>전체 처리 시간:</b> {isNum(processing.total_time) ? processing.total_time!.toFixed(2) + '초' : '-'}</p>
              <p><b>입력 길이:</b> {isNum(processing.input_length) ? processing.input_length : '-'}</p>
              <p><b>출력 길이:</b> {isNum(processing.output_length) ? processing.output_length : '-'}</p>
            </div>
          )}
        </div>
      )}

      {/* Recent List */}
      <div style={{ marginTop: 24 }}>
        <RecentInline />
      </div>
    </div>
  )
}
