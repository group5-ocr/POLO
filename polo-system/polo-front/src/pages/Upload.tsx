import React, { useState } from 'react'

interface UploadResult {
  filename: string
  file_size: number
  extracted_text_length: number
  extracted_text_preview: string
  easy_json: any
  status: string
  raw_file_path?: string
  json_file_path?: string
  processing_info?: any
}

export default function Upload() {
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<UploadResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const uploadFile = async (file: File) => {
    setUploading(true)
    setError(null)
    setResult(null)
    setProgress('파일 업로드 중...')

    try {
      const formData = new FormData()
      formData.append('file', file)

      setProgress('PDF 텍스트 추출 중...')
      
      const response = await fetch('http://localhost:8000/api/convert', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || '업로드 실패')
      }

      setProgress('AI 모델로 변환 중... (시간이 걸릴 수 있습니다)')
      
      const data = await response.json()
      setResult(data)
      setProgress('완료!')
    } catch (err) {
      setError(err instanceof Error ? err.message : '업로드 중 오류가 발생했습니다.')
      setProgress('')
    } finally {
      setUploading(false)
    }
  }

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
    setSelectedFile(f)
  }

  const checkModelStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/model-status')
      const data = await response.json()
      
      if (data.model_available) {
        alert('✅ AI 모델이 정상적으로 연결되어 있습니다!')
      } else {
        alert('❌ AI 모델 서비스가 사용 불가능합니다. 도커 서비스를 확인해주세요.')
      }
    } catch (err) {
      alert('❌ 서버 연결에 실패했습니다.')
    }
  }

  return (
    <div className="page upload">
      <h2>논문 업로드 (PDF 전용)</h2>
      
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
        <button 
          onClick={checkModelStatus}
          style={{
            padding: '10px 20px',
            backgroundColor: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          모델 연결 테스트 (health)
        </button>
        <button
          onClick={() => selectedFile && uploadFile(selectedFile)}
          disabled={!selectedFile || uploading}
          style={{
            padding: '10px 20px',
            backgroundColor: selectedFile && !uploading ? '#007bff' : '#9bbcf1',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: selectedFile && !uploading ? 'pointer' : 'not-allowed'
          }}
        >
          변환하기 (generate)
        </button>
      </div>

      <label className="upload-box">
        <input 
          type="file" 
          accept="application/pdf" 
          onChange={onChange}
          disabled={uploading}
        />
        <div>
          {uploading ? '처리 중...' : '여기를 클릭하거나 파일을 드래그하여 PDF를 업로드하세요.'}
        </div>
      </label>

      {uploading && (
        <div style={{
          marginTop: '20px',
          padding: '15px',
          backgroundColor: '#e3f2fd',
          border: '1px solid #90caf9',
          borderRadius: '5px',
          textAlign: 'center'
        }}>
          <div style={{ marginBottom: '10px' }}>
            <div style={{
              width: '100%',
              height: '20px',
              backgroundColor: '#e0e0e0',
              borderRadius: '10px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: '100%',
                height: '100%',
                backgroundColor: '#2196f3',
                animation: 'pulse 1.5s ease-in-out infinite'
              }}></div>
            </div>
          </div>
          <p style={{ margin: 0, fontWeight: 'bold', color: '#1976d2' }}>
            {progress}
          </p>
          <p style={{ margin: '5px 0 0 0', fontSize: '14px', color: '#666' }}>
            GPU 가속으로 처리 중입니다. 잠시만 기다려주세요...
          </p>
        </div>
      )}

      {error && (
        <div style={{
          marginTop: '20px',
          padding: '15px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          border: '1px solid #f5c6cb',
          borderRadius: '5px'
        }}>
          <strong>오류:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: '20px' }}>
          <h3>처리 결과</h3>
          <div style={{
            padding: '15px',
            backgroundColor: '#d4edda',
            border: '1px solid #c3e6cb',
            borderRadius: '5px',
            marginBottom: '15px'
          }}>
            <p><strong>파일명:</strong> {result.filename}</p>
            <p><strong>파일 크기:</strong> {(result.file_size / 1024).toFixed(2)} KB</p>
            <p><strong>추출된 텍스트 길이:</strong> {result.extracted_text_length} 문자</p>
            <p><strong>상태:</strong> {result.status}</p>
          </div>

          <div style={{ marginBottom: '15px' }}>
            <h4>추출된 텍스트 미리보기:</h4>
            <div style={{
              padding: '10px',
              backgroundColor: '#f8f9fa',
              border: '1px solid #dee2e6',
              borderRadius: '5px',
              maxHeight: '200px',
              overflowY: 'auto',
              whiteSpace: 'pre-wrap',
              fontSize: '14px'
            }}>
              {result.extracted_text_preview}
            </div>
          </div>

          <div>
            <h4>AI 변환 결과 (JSON):</h4>
            <div style={{
              padding: '15px',
              backgroundColor: '#e7f3ff',
              border: '1px solid #b3d9ff',
              borderRadius: '5px',
              maxHeight: '400px',
              overflowY: 'auto',
              fontSize: '14px'
            }}>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(result.easy_json, null, 2)}
              </pre>
            </div>
          </div>

          {result.processing_info && (
            <div style={{ marginTop: '15px' }}>
              <h4>처리 정보:</h4>
              <div style={{
                padding: '10px',
                backgroundColor: '#f8f9fa',
                border: '1px solid #dee2e6',
                borderRadius: '5px',
                fontSize: '14px'
              }}>
                <p><strong>GPU 사용:</strong> {result.easy_json.processing_info?.gpu_used ? '✅ 예' : '❌ 아니오'}</p>
                <p><strong>추론 시간:</strong> {result.easy_json.processing_info?.inference_time?.toFixed(2)}초</p>
                <p><strong>전체 처리 시간:</strong> {result.easy_json.processing_info?.total_time?.toFixed(2)}초</p>
                <p><strong>입력 길이:</strong> {result.easy_json.processing_info?.input_length} 문자</p>
                <p><strong>출력 길이:</strong> {result.easy_json.processing_info?.output_length} 문자</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
