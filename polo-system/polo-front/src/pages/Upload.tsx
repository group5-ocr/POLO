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
  arxiv_id?: string;
  is_arxiv_paper?: boolean;
  // JSONL 데이터 추가
  jsonl_data?: Array<{
    index: number;
    text: string;
    easy_text?: string;
    image_path?: string;
  }>;
  // Math 결과 추가
  math_result?: {
    overview: string;
    items: Array<{
      index: number;
      line_start: number;
      line_end: number;
      kind: string;
      env: string;
      equation: string;
      explanation: string;
    }>;
  };
  // arXiv 결과 추가
  arxiv_result?: {
    arxiv_id: string;
    title: string;
    tex_id: string;
    paths: any;
  };
}

export default function Upload() {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [arxivId, setArxivId] = useState("");
  const [arxivTitle, setArxivTitle] = useState("");
  const [showArxivForm, setShowArxivForm] = useState(false);
  const [activeTab, setActiveTab] = useState<"preview" | "jsonl" | "math">(
    "preview"
  );
  const [downloadInfo, setDownloadInfo] = useState<any>(null);
  const navigate = useNavigate();

  const uploadFile = async (file: File) => {
    setUploading(true);
    setError(null);
    setResult(null);

    try {
      const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
      console.log("[Upload] API Base URL:", apiBase);

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase}/api/upload/convert`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "업로드 실패");
      }

      const data = await response.json();

      // 서버에서 반환된 실제 논문 ID 사용
      setResult(data);

      // 다운로드 정보 조회 (실제 논문 ID가 있을 때만)
      if (data.doc_id) {
        try {
          const infoResponse = await fetch(
            `${
              import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
            }/api/upload/download/info/${data.doc_id}`
          );
          if (infoResponse.ok) {
            const infoData = await infoResponse.json();
            setDownloadInfo(infoData);
          }
        } catch (err) {
          console.warn("다운로드 정보 조회 실패:", err);
        }
      }
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
      const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
      console.log("API Base URL:", apiBase);

      const response = await fetch(`${apiBase}/api/upload/model-status`);
      const data = await response.json();

      if (data.model_available) {
        alert(
          `✅ AI 모델이 정상적으로 연결되어 있습니다!\n\nAPI Base: ${apiBase}`
        );
      } else {
        alert(
          `❌ AI 모델 서비스가 사용 불가능합니다.\nAPI Base: ${apiBase}\n도커 서비스를 확인해주세요.`
        );
      }
    } catch (err) {
      const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
      alert(
        `❌ 서버 연결에 실패했습니다.\nAPI Base: ${apiBase}\nError: ${err}`
      );
    }
  };

  const uploadFromArxiv = async (arxivId: string, title: string) => {
    setUploading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(
        `${
          import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
        }/api/upload/from-arxiv`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            user_id: 1, // 임시 사용자 ID
            arxiv_id: arxivId,
            title: title,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "arXiv 업로드 실패");
      }

      const data = await response.json();

      // 서버에서 반환된 실제 논문 ID 사용
      const docId = data.tex_id;

      // arXiv 업로드는 비동기 처리이므로 성공 메시지만 표시
      setResult({
        filename: `${arxivId}.pdf`,
        file_size: 0,
        extracted_text_length: 0,
        extracted_text_preview: `arXiv 논문 처리 시작: ${title}\n논문 ID: ${docId}\n\n처리 중입니다...`,
        easy_text:
          "논문이 다운로드되고 처리 중입니다. 완료되면 결과가 표시됩니다.",
        status: "processing",
        doc_id: docId,
        json_file_path: `/api/download/${docId}.json`,
        // arXiv 처리 결과 추가
        arxiv_result: {
          arxiv_id: arxivId,
          title: title,
          tex_id: data.tex_id,
          paths: data.paths,
        },
      });

      // 다운로드 정보 조회 (실제 논문 ID가 있을 때만)
      if (docId) {
        try {
          const infoResponse = await fetch(
            `${
              import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
            }/download/info/${docId}`
          );
          if (infoResponse.ok) {
            const infoData = await infoResponse.json();
            setDownloadInfo(infoData);
          }
        } catch (err) {
          console.warn("다운로드 정보 조회 실패:", err);
        }
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "arXiv 업로드 중 오류가 발생했습니다."
      );
    } finally {
      setUploading(false);
    }
  };

  const downloadFile = async (
    filename: string,
    fileType: "json" | "pdf" | "math" | "easy" | "raw"
  ) => {
    try {
      const baseUrl = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
      let endpoint;
      if (fileType === "json") {
        endpoint = `${baseUrl}/download/${filename}`;
      } else if (fileType === "easy") {
        // 변환된 논문 PDF
        endpoint = `${baseUrl}/download/easy/${filename}`;
      } else if (fileType === "math") {
        // 수식 설명 PDF
        endpoint = `${baseUrl}/download/math/${filename}`;
      } else {
        // 원본 PDF
        endpoint = `${baseUrl}/download/raw/${filename}`;
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
          <button
            onClick={() => setShowArxivForm(!showArxivForm)}
            className="btn-secondary"
          >
            {showArxivForm ? "PDF 업로드" : "arXiv 논문"}
          </button>
        </div>

        {showArxivForm && (
          <div className="arxiv-form">
            <h3>arXiv 논문 업로드</h3>
            <div className="form-group">
              <label htmlFor="arxivId">arXiv ID (예: 2408.12345)</label>
              <input
                type="text"
                id="arxivId"
                value={arxivId}
                onChange={(e) => setArxivId(e.target.value)}
                placeholder="2408.12345"
                disabled={uploading}
              />
            </div>
            <div className="form-group">
              <label htmlFor="arxivTitle">논문 제목</label>
              <input
                type="text"
                id="arxivTitle"
                value={arxivTitle}
                onChange={(e) => setArxivTitle(e.target.value)}
                placeholder="논문 제목을 입력하세요"
                disabled={uploading}
              />
            </div>
            <button
              onClick={() => uploadFromArxiv(arxivId, arxivTitle)}
              disabled={!arxivId || !arxivTitle || uploading}
              className="btn-primary"
            >
              arXiv 논문 처리하기
            </button>
          </div>
        )}

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
              {result.is_arxiv_paper && result.arxiv_id && (
                <div className="arxiv-info">
                  <span className="arxiv-badge">📄 arXiv 논문</span>
                  <span className="arxiv-id">ID: {result.arxiv_id}</span>
                </div>
              )}
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

            <div className="result-tabs">
              <button
                className={`tab-button ${
                  activeTab === "preview" ? "active" : ""
                }`}
                onClick={() => setActiveTab("preview")}
              >
                미리보기
              </button>
              <button
                className={`tab-button ${
                  activeTab === "jsonl" ? "active" : ""
                }`}
                onClick={() => setActiveTab("jsonl")}
              >
                JSONL 데이터
              </button>
              <button
                className={`tab-button ${activeTab === "math" ? "active" : ""}`}
                onClick={() => setActiveTab("math")}
              >
                수식 해설
              </button>
            </div>

            <div className="result-sections">
              {activeTab === "preview" && (
                <>
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
                </>
              )}

              {activeTab === "jsonl" && (
                <div className="result-section">
                  <h4>JSONL 데이터 (Easy 모델 출력)</h4>
                  {result.jsonl_data && result.jsonl_data.length > 0 ? (
                    <div className="jsonl-container">
                      {result.jsonl_data.map((item, index) => (
                        <div key={index} className="jsonl-item">
                          <div className="jsonl-header">
                            <span className="jsonl-index">#{item.index}</span>
                            {item.image_path && (
                              <span className="jsonl-image">
                                🖼️ 이미지 생성됨
                              </span>
                            )}
                          </div>
                          <div className="jsonl-content">
                            <div className="jsonl-original">
                              <strong>원본:</strong>
                              <p>{item.text}</p>
                            </div>
                            {item.easy_text && (
                              <div className="jsonl-easy">
                                <strong>쉬운 설명:</strong>
                                <p>{item.easy_text}</p>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="no-data">
                      <p>JSONL 데이터가 아직 처리되지 않았습니다.</p>
                    </div>
                  )}
                </div>
              )}

              {activeTab === "math" && (
                <div className="result-section">
                  <h4>수식 해설 (Math 모델 출력)</h4>
                  {result.math_result ? (
                    <div className="math-container">
                      <div className="math-overview">
                        <h5>문서 개요</h5>
                        <p>{result.math_result.overview}</p>
                      </div>
                      <div className="math-equations">
                        <h5>수식 해설</h5>
                        {result.math_result.items.map((item, index) => (
                          <div key={index} className="math-item">
                            <div className="math-header">
                              <span className="math-index">#{item.index}</span>
                              <span className="math-kind">{item.kind}</span>
                              <span className="math-lines">
                                라인 {item.line_start}-{item.line_end}
                              </span>
                            </div>
                            <div className="math-equation">
                              <code>{item.equation}</code>
                            </div>
                            <div className="math-explanation">
                              <p>{item.explanation}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="no-data">
                      <p>수식 해설이 아직 처리되지 않았습니다.</p>
                    </div>
                  )}
                </div>
              )}
            </div>

            {downloadInfo && (
              <div className="download-info">
                <h4>다운로드 가능한 파일</h4>
                <div className="file-list">
                  {downloadInfo.files.easy.length > 0 && (
                    <div className="file-category">
                      <h5>
                        🖼️ 쉬운 버전 이미지 ({downloadInfo.files.easy.length}개)
                      </h5>
                      <button
                        className="btn-download"
                        onClick={() =>
                          result.doc_id && downloadFile(result.doc_id, "easy")
                        }
                      >
                        이미지 다운로드
                      </button>
                    </div>
                  )}

                  {downloadInfo.files.math.length > 0 && (
                    <div className="file-category">
                      <h5>📐 수식 해설 ({downloadInfo.files.math.length}개)</h5>
                      <div className="file-items">
                        {downloadInfo.files.math.map(
                          (file: any, index: number) => (
                            <button
                              key={index}
                              className="btn-download-small"
                              onClick={() =>
                                result.doc_id &&
                                downloadFile(result.doc_id, "math")
                              }
                            >
                              {file.name} ({(file.size / 1024).toFixed(1)}KB)
                            </button>
                          )
                        )}
                      </div>
                    </div>
                  )}

                  {downloadInfo.files.preprocess.length > 0 && (
                    <div className="file-category">
                      <h5>
                        📄 전처리 파일 ({downloadInfo.files.preprocess.length}
                        개)
                      </h5>
                      <div className="file-items">
                        {downloadInfo.files.preprocess.map(
                          (file: any, index: number) => (
                            <button
                              key={index}
                              className="btn-download-small"
                              onClick={() => downloadFile(file.name, "json")}
                            >
                              {file.name} ({(file.size / 1024).toFixed(1)}KB)
                            </button>
                          )
                        )}
                      </div>
                    </div>
                  )}

                  {downloadInfo.files.raw.length > 0 && (
                    <div className="file-category">
                      <h5>📁 원본 파일 ({downloadInfo.files.raw.length}개)</h5>
                      <div className="file-items">
                        {downloadInfo.files.raw.map(
                          (file: any, index: number) => (
                            <button
                              key={index}
                              className="btn-download-small"
                              onClick={() => downloadFile(file.name, "raw")}
                            >
                              {file.name} ({(file.size / 1024).toFixed(1)}KB)
                            </button>
                          )
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="result-actions">
              <div className="download-buttons">
                <button
                  className="btn-download"
                  onClick={() => {
                    if (result.doc_id) {
                      // Easy 모델 출력 이미지 다운로드
                      downloadFile(result.doc_id, "easy");
                    }
                  }}
                  disabled={!result.doc_id}
                >
                  🖼️ 쉬운 버전 이미지 다운로드
                </button>
                <button
                  className="btn-download"
                  onClick={() => {
                    if (result.doc_id) {
                      // Math 모델 출력 파일 다운로드
                      downloadFile(result.doc_id, "math");
                    }
                  }}
                  disabled={!result.doc_id}
                >
                  📐 수식 해설 파일 다운로드
                </button>
                <button
                  className="btn-download"
                  onClick={() => {
                    if (result.doc_id) {
                      // 원본 파일 다운로드
                      downloadFile(result.filename, "raw");
                    }
                  }}
                  disabled={!result.doc_id}
                >
                  📄 원본 파일 다운로드
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
