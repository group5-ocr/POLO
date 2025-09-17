import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

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
  const { user, isLoading } = useAuth();
  const navigate = useNavigate();
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [easyResults, setEasyResults] = useState<any>(null);
  const [isLoadingEasy, setIsLoadingEasy] = useState(false);
  const [easyReady, setEasyReady] = useState(false);
  const [progress, setProgress] = useState(0);

  // 진행률 업데이트 함수
  const updateProgress = (value: number) => {
    setProgress(Math.min(100, Math.max(0, value)));
  };

  // 동적 로딩 게이지 애니메이션
  const [progressAnimation, setProgressAnimation] = useState<NodeJS.Timeout | null>(null);

  const startProgressAnimation = () => {
    if (progressAnimation) {
      clearInterval(progressAnimation);
    }
    
    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += Math.random() * 3; // 0-3% 랜덤 증가
      if (currentProgress >= 95) {
        currentProgress = 95; // 95%에서 멈춤
      }
      setProgress(currentProgress);
    }, 200); // 200ms마다 업데이트
    
    setProgressAnimation(interval);
  };

  const stopProgressAnimation = () => {
    if (progressAnimation) {
      clearInterval(progressAnimation);
      setProgressAnimation(null);
    }
  };

  // Easy 결과 로드 함수
  const loadEasyResults = async (paperId: string) => {
    setIsLoadingEasy(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE ?? "http://localhost:8000"}/api/upload/download/easy-json/${paperId}`);
      if (response.ok) {
        const data = await response.json();
        setEasyResults(data);
        console.log(`[Easy 결과] 로드 완료: ${data.total_chunks}개 청크`);
      } else {
        console.log(`[Easy 결과] 로드 실패: ${response.status}`);
      }
    } catch (error) {
      console.log(`[Easy 결과] 로드 에러: ${error}`);
    } finally {
      setIsLoadingEasy(false);
    }
  };

  // 쉬운 논문 생성 함수 (통합된 기능)
  const handleGenerateEasyPaper = async (paperIdOverride?: string) => {
    const finalPaperId = paperIdOverride ?? result?.doc_id;
    if (!finalPaperId) {
      alert("먼저 논문을 업로드해주세요.");
      return;
    }

    setIsLoadingEasy(true);
    setProgress(0);
    setEasyReady(false);
    startProgressAnimation(); // 동적 애니메이션 시작
    
    try {
      const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
      
      // 1단계: Easy 모델로 전송
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 25 * 60 * 1000); // 25분 타임아웃
      
      const response = await fetch(`${apiBase}/api/upload/send-to-easy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          paper_id: finalPaperId
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        console.log("Easy 모델 전송 성공:", data);
        
        // 2단계: 결과 파일 생성 폴링 (로컬 파일 존재 여부만 확인)
        const maxWaitMs = 25 * 60 * 1000; // 25분
        const intervalMs = 3000; // 3초 폴링
        const start = Date.now();
        let ready = false;
        while (Date.now() - start < maxWaitMs) {
          try {
            const r = await fetch(`${apiBase}/api/results/${finalPaperId}/ready`);
            if (r.ok) {
              const j = await r.json();
              if (j.ok) { 
                ready = true; 
                break; 
              }
            }
          } catch {}
          await new Promise(res => setTimeout(res, intervalMs));
        }

        if (ready) {
          setEasyReady(true);
          stopProgressAnimation(); // 애니메이션 중지
          updateProgress(100);
        } else {
          console.warn('결과 파일 폴링 타임아웃');
          stopProgressAnimation();
        }
      } else {
        const errorData = await response.json();
        console.error("Easy 모델 전송 실패:", errorData);
        alert(`쉬운 논문 생성 실패: ${errorData.detail || response.statusText}`);
        stopProgressAnimation();
        setProgress(0);
      }
    } catch (error) {
      console.error("쉬운 논문 생성 에러:", error);
      alert('쉬운 논문 생성 중 오류가 발생했습니다.');
      stopProgressAnimation();
      setProgress(0);
    } finally {
      setIsLoadingEasy(false);
    }
  };

  // Easy 결과를 HTML로 다운로드하는 함수
  const downloadEasyResultsAsHTML = () => {
    if (!result?.doc_id) return;
    
    // 서버에서 생성된 HTML 파일 다운로드
    const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
    const downloadUrl = `${apiBase}/api/results/${result.doc_id}/html`;
    
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `polo_easy_explanation_${result.doc_id}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Easy 결과를 브라우저에서 보는 함수
  const viewEasyResultsInBrowser = () => {
    if (!result?.doc_id) return;
    
    // 새 탭에서 HTML 결과 열기
    const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
    const viewUrl = `${apiBase}/api/results/${result.doc_id}/html`;
    window.open(viewUrl, '_blank');
  };

  // Easy 결과 HTML 생성 함수
  const generateEasyResultsHTML = (easyResults: any) => {
    const sections = easyResults.sections || easyResults.chunks || [];
    const totalSections = easyResults.total_sections || easyResults.total_chunks || 0;
    
    return `
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Easy 결과 - 논문 ${easyResults.paper_id}</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }
        .header h1 { margin: 0; font-size: 2.5em; }
        .stats { display: flex; justify-content: center; gap: 30px; margin-top: 20px; }
        .stat { text-align: center; }
        .stat-number { font-size: 2em; font-weight: bold; }
        .content { padding: 30px; }
        .section { margin-bottom: 30px; padding: 25px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }
        .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #ddd; }
        .section-title { font-size: 18px; font-weight: bold; color: #2c3e50; }
        .section-status { padding: 6px 12px; border-radius: 4px; font-size: 12px; }
        .status-success { background: #4caf50; color: white; }
        .status-failed { background: #f44336; color: white; }
        .original-content { background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9em; line-height: 1.6; }
        .korean-translation { background: #f3e5f5; padding: 15px; border-radius: 5px; margin-bottom: 15px; line-height: 1.8; }
        .image-container { text-align: center; margin-top: 15px; }
        .image-container img { max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .no-image { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Easy 모델 결과</h1>
            <p>논문 ID: ${easyResults.paper_id}</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-number">${totalSections}</div>
                    <div>총 섹션</div>
                </div>
                <div class="stat">
                    <div class="stat-number">${easyResults.success_count}</div>
                    <div>성공</div>
                </div>
                <div class="stat">
                    <div class="stat-number">${easyResults.failed_count}</div>
                    <div>실패</div>
                </div>
            </div>
        </div>
        <div class="content">
            ${sections.map((section: any, index: number) => `
                <div class="section">
                    <div class="section-header">
                        <span class="section-title">${section.title || `섹션 ${section.index + 1}`}</span>
                        <span class="section-status ${section.status === 'success' ? 'status-success' : 'status-failed'}">
                            ${section.status === 'success' ? '✅ 성공' : '❌ 실패'}
                        </span>
                    </div>
                    <div class="original-content">
                        <strong>원본 내용:</strong><br>
                        ${(section.original_content || section.original_text || '').substring(0, 500)}${(section.original_content || section.original_text || '').length > 500 ? '...' : ''}
                    </div>
                    ${section.korean_translation ? `
                        <div class="korean-translation">
                            <strong>쉬운 설명:</strong><br>
                            ${section.korean_translation}
                        </div>
                    ` : ''}
                    <div class="image-container">
                        ${section.image_path ? 
                            `<img src="${section.image_path}" alt="시각화 이미지">` : 
                            '<div class="no-image">이미지 없음</div>'
                        }
                    </div>
                </div>
            `).join('')}
        </div>
    </div>
</body>
</html>`;
  };
  const [dragActive, setDragActive] = useState(false);
  const [arxivId, setArxivId] = useState("");
  const [arxivTitle, setArxivTitle] = useState("");
  const [showArxivForm, setShowArxivForm] = useState(false);
  const [activeTab, setActiveTab] = useState<"preview" | "jsonl" | "math">(
    "preview"
  );
  const [downloadInfo, setDownloadInfo] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // 로그인 체크
  useEffect(() => {
    if (!isLoading && !user) {
      alert("로그아웃 되었습니다.");
      navigate("/");
    }
  }, [user, isLoading, navigate]);

  const uploadFile = async (file: File): Promise<UploadResult | null> => {
    setUploading(true);
    setError(null);
    setResult(null);

    try {
      const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
      console.log("[Upload] API Base URL:", apiBase);
      console.log(`[convert] 호출 시작 → ${apiBase}/api/upload/convert`);

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase}/api/upload/convert`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = "업로드 실패";
        try { const j = await response.json(); detail = j.detail || detail; } catch {}
        console.log(`[convert] 실패: ${response.status} ${detail}`);
        throw new Error(`[convert] ${detail}`);
      }

      const data = await response.json();

      // 서버에서 반환된 실제 논문 ID 사용
      setResult({ ...data, status: data.status ?? "processing" });
      console.log(`[convert] 성공: doc_id=${data?.doc_id ?? "-"}`);

      // 다운로드 정보 조회 (실제 논문 ID가 있을 때만)
      if (data.doc_id) {
        try {
          console.log(`[download/info] 호출 → ${apiBase}/api/upload/download/info/${data.doc_id}`);
          const infoResponse = await fetch(
            `${
              import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
            }/api/upload/download/info/${data.doc_id}`
          );
          if (infoResponse.ok) {
            const infoData = await infoResponse.json();
            setDownloadInfo(infoData);
            console.log(`[download/info] 성공`);
          } else {
            console.log(`[download/info] 실패: ${infoResponse.status}`);
          }
        } catch (err) {
          console.warn("다운로드 정보 조회 실패:", err);
          console.log(`[download/info] 예외: ${String(err)}`);
        }
      }
      return data as UploadResult;
    } catch (err) {
      setError(err instanceof Error ? err.message : "업로드 중 오류가 발생했습니다.");
      return null;
    } finally {
      setUploading(false);
    }
  };

  // 전처리 → Easy 모델까지 한 번에 실행
  const handleConvertAndGenerate = async () => {
    try {
      if (!selectedFile && !result?.doc_id) {
        alert("먼저 PDF를 선택해주세요.");
        return;
      }
      let docId = result?.doc_id;
      if (!docId && selectedFile) {
        const r = await uploadFile(selectedFile);
        docId = r?.doc_id || undefined;
      }
      if (!docId) {
        alert("전처리 실패: 논문 ID를 가져오지 못했습니다.");
        return;
      }
      await handleGenerateEasyPaper(docId);
    } catch (e) {
      console.error("통합 실행 실패", e);
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

    setSelectedFile(file);
    setError(null);
    setResult(null);
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

  const startConversion = () => {
    if (selectedFile) {
      uploadFile(selectedFile);
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
        let detail = "arXiv 업로드 실패";
        try { const j = await response.json(); detail = j.detail || detail; } catch {}
        console.log(`[from-arxiv] 실패: ${response.status} ${detail}`);
        throw new Error(`[from-arxiv] ${detail}`);
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
            }/api/upload/download/info/${docId}`
          );
          if (infoResponse.ok) {
            const infoData = await infoResponse.json();
            setDownloadInfo(infoData);
            console.log(`[download/info] 성공`);
          } else {
            console.log(`[download/info] 실패: ${infoResponse.status}`);
          }
        } catch (err) {
          console.warn("다운로드 정보 조회 실패:", err);
          console.log(`[download/info] 예외: ${String(err)}`);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "arXiv 업로드 중 오류가 발생했습니다.");
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
        endpoint = `${baseUrl}/api/upload/download/info/${filename}`;
      } else if (fileType === "easy") {
        endpoint = `${baseUrl}/api/upload/download/easy/${filename}`;
      } else if (fileType === "math") {
        endpoint = `${baseUrl}/api/upload/download/math/${filename}`;
      } else {
        endpoint = `${baseUrl}/api/upload/download/raw/${filename}`;
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

  // 로딩 중이거나 로그인하지 않은 경우 로딩 화면 표시
  if (isLoading) {
    return (
      <div className="upload-page">
        <div className="upload-container">
          <div style={{ textAlign: "center", padding: "40px" }}>
            <div className="upload-spinner"></div>
            <p>로딩 중...</p>
          </div>
        </div>
      </div>
    );
  }

  // 로그인하지 않은 경우 빈 화면 (useEffect에서 리다이렉트 처리)
  if (!user) {
    return null;
  }

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-header">
          <h1>쉬운 논문 생성</h1>
          <p>PDF 파일을 업로드하면 AI가 쉽게 이해할 수 있도록 변환해드려요!</p>
        </div>

        <div className="upload-actions">
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
            ) : selectedFile ? (
              <>
                <div className="upload-icon">📄</div>
                <h3>선택된 파일</h3>
                <p className="selected-file-name">{selectedFile.name}</p>
                <p className="selected-file-size">
                  {(selectedFile.size / 1024).toFixed(2)} KB
                </p>
                <div className="upload-info">
                  <span>• PDF 파일만 지원</span>
                  <span>• 최대 50MB</span>
                </div>
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

        {selectedFile && !uploading && (
          <div className="conversion-actions">
            <button
              onClick={handleConvertAndGenerate}
              className="btn-primary"
              style={{ 
                marginRight: 15,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                border: 'none',
                borderRadius: '8px',
                padding: '12px 24px',
                fontSize: '16px',
                fontWeight: '600',
                color: 'white',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 15px rgba(102, 126, 234, 0.3)'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 20px rgba(102, 126, 234, 0.4)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 15px rgba(102, 126, 234, 0.3)';
              }}
            >
              쉬운 논문 생성
            </button>
            <button
              onClick={() => {
                if (!result?.doc_id) {
                  alert("먼저 논문을 업로드해주세요.");
                  return;
                }
                // 수학 모델 전송 로직 (기존 send-to-math 엔드포인트 호출)
                const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
                fetch(`${apiBase}/api/upload/send-to-math`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ paper_id: result.doc_id })
                })
                .then(response => response.json())
                .then(data => {
                  console.log("수학 모델 전송 성공:", data);
                  alert("수학 모델로 전송되었습니다.");
                })
                .catch(error => {
                  console.error("수학 모델 전송 실패:", error);
                  alert("수학 모델 전송에 실패했습니다.");
                });
              }}
              style={{ 
                marginRight: 15,
                background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)',
                border: 'none',
                borderRadius: '8px',
                padding: '12px 24px',
                fontSize: '16px',
                fontWeight: '600',
                color: 'white',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 15px rgba(25, 118, 210, 0.3)'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 20px rgba(25, 118, 210, 0.4)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 15px rgba(25, 118, 210, 0.3)';
              }}
            >
              수학 모델
            </button>
            <button
              onClick={() => {
                setSelectedFile(null);
                setError(null);
                setResult(null);
              }}
              style={{
                background: 'transparent',
                border: '2px solid #e0e0e0',
                borderRadius: '8px',
                padding: '10px 22px',
                fontSize: '16px',
                fontWeight: '500',
                color: '#666',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.borderColor = '#999';
                e.currentTarget.style.color = '#333';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.borderColor = '#e0e0e0';
                e.currentTarget.style.color = '#666';
              }}
            >
              파일 다시 선택
            </button>
          </div>
        )}

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
              <h3>전처리 완료</h3>
              <p>논문이 분석되었습니다. 아래 버튼으로 쉬운 논문 생성을 시작하세요.</p>
              {result.is_arxiv_paper && result.arxiv_id && (
                <div className="arxiv-info">
                  <span className="arxiv-badge">📄 arXiv 논문</span>
                  <span className="arxiv-id">ID: {result.arxiv_id}</span>
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

            {/* 진행률 표시 */}
            {isLoadingEasy && (
              <div className="progress-section">
                <h4>🔄 쉬운 논문 생성 중...</h4>
                <div className="progress-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                  <span className="progress-text">{progress}%</span>
                </div>
                <p style={{ textAlign: 'center', marginTop: '10px', color: '#666' }}>
                  AI가 논문을 쉬운 언어로 변환하고 있습니다...
                </p>
              </div>
            )}

            {/* Easy 모델 완료 시 결과 보기 버튼들 */}
            {easyReady && (
              <div className="model-buttons">
                <h4 style={{ textAlign: 'center', marginBottom: '20px', color: '#2c3e50', fontSize: '18px' }}>
                  쉬운 논문 생성 완료!
                </h4>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '15px' }}>
                  <button
                    onClick={viewEasyResultsInBrowser}
                    style={{
                      background: 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)',
                      border: 'none',
                      borderRadius: '8px',
                      padding: '12px 24px',
                      fontSize: '16px',
                      fontWeight: '600',
                      color: 'white',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      boxShadow: '0 4px 15px rgba(76, 175, 80, 0.3)'
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 6px 20px rgba(76, 175, 80, 0.4)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 4px 15px rgba(76, 175, 80, 0.3)';
                    }}
                  >
                    결과 보러가기
                  </button>
                  <button
                    onClick={downloadEasyResultsAsHTML}
                    style={{
                      background: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
                      border: 'none',
                      borderRadius: '8px',
                      padding: '12px 24px',
                      fontSize: '16px',
                      fontWeight: '600',
                      color: 'white',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      boxShadow: '0 4px 15px rgba(255, 152, 0, 0.3)'
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 6px 20px rgba(255, 152, 0, 0.4)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 4px 15px rgba(255, 152, 0, 0.3)';
                    }}
                  >
                    HTML 다운로드
                  </button>
                </div>
              </div>
            )}

            {/* 상세 결과/통계 UI는 간소화 요청에 따라 제거 */}

            {/* 로딩 상태 */}
            {isLoadingEasy && (
              <div className="loading-easy">
                <p>🔄 Easy 모델이 논문을 쉬운 언어로 변환 중입니다...</p>
              </div>
            )}

              <div className="action-buttons">
                <button className="btn-secondary" onClick={() => navigate("/")}>
                  홈으로 돌아가기
                </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
