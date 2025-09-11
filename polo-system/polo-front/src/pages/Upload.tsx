import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

interface UploadResult {
  filename: string;
  file_size: number;
  extracted_text_length: number;
  extracted_text_preview: string;
  easy_text: string;
  status: string;
  doc_id?: string;
  json_file_path?: string;
}

export default function Upload() {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const navigate = useNavigate();

  const uploadFile = async (file: File) => {
    setUploading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/api/convert", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "업로드 실패");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "업로드 중 오류가 발생했습니다."
      );
    } finally {
      setUploading(false);
    }
  };

  const handleFile = (file: File) => {
    if (file.type !== "application/pdf") {
      setError("PDF 파일만 업로드할 수 있습니다.");
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError("파일은 50MB 이하만 가능합니다.");
      return;
    }

    uploadFile(file);
  };

  const onChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    handleFile(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const checkModelStatus = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/model-status");
      const data = await response.json();

      if (data.model_available) {
        alert("✅ AI 모델이 정상적으로 연결되어 있습니다!");
      } else {
        alert(
          "❌ AI 모델 서비스가 사용 불가능합니다. 도커 서비스를 확인해주세요."
        );
      }
    } catch (err) {
      alert("❌ 서버 연결에 실패했습니다.");
    }
  };

  const showTestResult = () => {
    setResult({
      filename: "test_paper.pdf",
      file_size: 1024000,
      extracted_text_length: 5000,
      extracted_text_preview:
        "이것은 테스트용 논문의 미리보기 텍스트입니다. 실제 변환된 논문에서는 더 긴 텍스트가 표시됩니다...",
      easy_text:
        "이것은 AI가 변환한 쉬운 버전의 논문입니다. 복잡한 학술 용어들이 일반인도 이해할 수 있는 쉬운 말로 바뀌었습니다.",
      status: "success",
      doc_id: "20250101_123456_test_paper.pdf",
    });
  };

  const showTestError = () => {
    setResult({
      filename: "error_paper.pdf",
      file_size: 512000,
      extracted_text_length: 0,
      extracted_text_preview: "텍스트를 추출할 수 없습니다.",
      easy_text: "변환에 실패했습니다.",
      status: "error",
      doc_id: undefined,
    });
  };

  const downloadFile = async (
    filename: string,
    fileType: "json" | "pdf" | "math" | "easy"
  ) => {
    try {
      let endpoint;
      if (fileType === "json") {
        endpoint = `http://localhost:8000/api/download/${filename}`;
      } else if (fileType === "easy") {
        // 변환된 논문 PDF
        endpoint = `http://localhost:8000/api/download/easy/${filename}`;
      } else if (fileType === "math") {
        // 수식 설명 PDF
        endpoint = `http://localhost:8000/api/download/math/${filename}`;
      } else {
        // 원본 PDF
        endpoint = `http://localhost:8000/api/download/raw/${filename}`;
      }

      const response = await fetch(endpoint);

      if (!response.ok) {
        throw new Error("다운로드 실패");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      alert("다운로드 중 오류가 발생했습니다.");
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-header">
          <h1>논문 변환하기</h1>
          <p>PDF 파일을 업로드하면 AI가 쉽게 이해할 수 있도록 변환해드려요!</p>
        </div>

        <div className="upload-actions">
          <button onClick={checkModelStatus} className="btn-secondary">
            AI 모델 상태 확인
          </button>
          <button onClick={showTestResult} className="btn-secondary">
            성공 테스트
          </button>
          <button onClick={showTestError} className="btn-secondary">
            실패 테스트
          </button>
        </div>

        <div
          className={`upload-area ${dragActive ? "drag-active" : ""} ${
            uploading ? "uploading" : ""
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="application/pdf"
            onChange={onChange}
            disabled={uploading}
            className="file-input"
          />
          <div className="upload-content">
            {uploading ? (
              <>
                <div className="upload-spinner"></div>
                <h3>AI가 논문을 분석하고 있습니다...</h3>
                <p>잠시만 기다려주세요!</p>
              </>
            ) : (
              <>
                <div className="upload-icon">📁</div>
                <h3>PDF 파일을 업로드하세요</h3>
                <p>여기를 클릭하거나 파일을 드래그하여 업로드하세요</p>
                <div className="upload-info">
                  <span>• PDF 파일만 지원</span>
                  <span>• 최대 50MB</span>
                </div>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="error-message">
            <div className="error-icon">⚠️</div>
            <div className="error-content">
              <strong>오류가 발생했습니다</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {result && (
          <div className="result-container">
            <div className="result-header">
              <h3>
                {result.status === "success" ? "변환 완료!" : "변환 실패"}
              </h3>
              <p>
                {result.status === "success"
                  ? "논문이 성공적으로 변환되었습니다"
                  : "논문 변환 중 오류가 발생했습니다"}
              </p>
              <div
                className={`status-badge ${
                  result.status === "success"
                    ? "status-success"
                    : "status-error"
                }`}
              >
                <span className="status-icon">
                  {result.status === "success" ? "✅" : "❌"}
                </span>
                <span className="status-text">
                  {result.status === "success" ? "변환 성공" : "변환 실패"}
                </span>
              </div>
            </div>

            <div className="result-info">
              <div className="info-item">
                <span className="info-label">파일명</span>
                <span className="info-value">{result.filename}</span>
              </div>
              <div className="info-item">
                <span className="info-label">파일 크기</span>
                <span className="info-value">
                  {(result.file_size / 1024).toFixed(2)} KB
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">추출된 텍스트</span>
                <span className="info-value">
                  {result.extracted_text_length} 문자
                </span>
              </div>
            </div>

            <div className="result-sections">
              <div className="result-section">
                <h4>원본 텍스트 미리보기</h4>
                <div className="text-preview">
                  {result.extracted_text_preview}
                </div>
              </div>

              <div className="result-section">
                <h4>AI 변환 결과</h4>
                <div className="converted-text">{result.easy_text}</div>
              </div>
            </div>

            <div className="result-actions">
              <div className="download-buttons">
                <button
                  className="btn-download"
                  onClick={() => {
                    if (result.doc_id) {
                      // 변환된 논문 PDF 다운로드
                      downloadFile(result.doc_id, "easy");
                    }
                  }}
                  disabled={!result.doc_id}
                >
                  📄 쉬운 버전 논문 PDF 다운로드
                </button>
                <button
                  className="btn-download"
                  onClick={() => {
                    if (result.doc_id) {
                      // 수식 설명 PDF는 doc_id를 그대로 사용 (서버에서 _math.pdf로 변환)
                      downloadFile(result.doc_id, "math");
                    }
                  }}
                  disabled={!result.doc_id}
                >
                  📐 수식 설명 PDF 다운로드
                </button>
              </div>
              <div className="action-buttons">
                <button className="btn-secondary" onClick={() => navigate("/")}>
                  홈으로 돌아가기
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
