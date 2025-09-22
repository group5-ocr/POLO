import React, { useState, useEffect, useRef } from "react";
import "./Result.css";

interface PaperInfo {
  paper_id: string;
  paper_title: string;
  paper_authors: string;
  paper_venue: string;
  paper_date?: string;
  paper_doi?: string;
  total_sections: number;
  total_equations: number;
}

interface EasySection {
  easy_section_id: string;
  easy_section_title: string;
  easy_section_type: "section" | "subsection";
  easy_section_order: number;
  easy_section_level?: number;
  easy_section_parent?: string;
  easy_content: string;
  easy_paragraphs: EasyParagraph[];
  easy_subsections?: EasySection[];
  easy_visualizations?: EasyVisualization[];
  viz_api_result?: VizApiResult; // 섹션별 Viz API 결과
}

interface EasyParagraph {
  easy_paragraph_id: string;
  easy_paragraph_text: string;
  easy_paragraph_order: number;
  easy_visualization_trigger?: boolean; // 클릭 시 시각화 표시 여부
}

interface EasyVisualization {
  easy_viz_id: string;
  easy_viz_title: string;
  easy_viz_description?: string;
  easy_viz_image_path?: string;
  easy_viz_type: 'chart' | 'diagram' | 'graph' | 'table';
}

interface VizApiResult {
  viz_api_id: string;
  viz_api_title: string;
  viz_api_description?: string;
  viz_api_image_url?: string;
  viz_api_type: 'section_visualization';
  viz_api_status: 'success' | 'error' | 'loading';
  viz_api_error?: string;
}

interface MathEquation {
  math_equation_id: string;
  math_equation_index: string;
  math_equation_latex: string;
  math_equation_explanation: string;
  math_equation_context?: string;
  math_equation_section_ref?: string; // 어떤 섹션에 속하는지
}

interface IntegratedData {
  paper_info: PaperInfo;
  easy_sections: EasySection[];
  math_equations: MathEquation[];
  model_errors?: {
    easy_model_error?: string;
    math_model_error?: string;
    viz_api_error?: string;
  };
  processing_logs?: string[];
}

interface ResultProps {
  data?: IntegratedData;
  onDownload?: () => void;
  onPreview?: () => void;
}

const Result: React.FC<ResultProps> = ({ data, onDownload, onPreview }) => {
  const [integratedData, setIntegratedData] = useState<IntegratedData | null>(
    data || null
  );
  const [loading, setLoading] = useState(!data);
  const [error, setError] = useState<string | null>(null);
  const [activeViz, setActiveViz] = useState<{ [key: string]: boolean }>({});
  const [activeVizApi, setActiveVizApi] = useState<{ [key: string]: boolean }>({});
  const [activeEquation, setActiveEquation] = useState<string | null>(null);
  const [loadingVizApi, setLoadingVizApi] = useState<{ [key: string]: boolean }>({});
  const [isDownloading, setIsDownloading] = useState(false);
  const mathJaxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!data) {
      loadIntegratedData();
    }
  }, [data]);

  useEffect(() => {
    if (integratedData) {
      renderMathJax();
    }
  }, [integratedData, activeEquation]);

  const loadIntegratedData = async () => {
    try {
      setLoading(true);
      // URL에서 paper_id 추출 (경로 파라미터에서)
      const pathParts = window.location.pathname.split('/');
      const paper_id = pathParts[pathParts.length - 1];
      
      console.log(`[Result] paper_id: ${paper_id}`);
      
      // 통합 결과 API 호출
      const response = await fetch(`/api/integrated-result/${paper_id}`);
      if (!response.ok) {
        console.warn(`[Result] 통합 결과 API 실패: ${response.status}, Easy 결과만 로드 시도`);
        
        // 통합 결과 실패 시 Easy 결과만 로드
        const easyResponse = await fetch(`/api/results/${paper_id}/easy_results.json`);
        if (easyResponse.ok) {
          const easyData = await easyResponse.json();
          const partialData = {
            paper_info: easyData.paper_info || {
              paper_id: paper_id,
              paper_title: `논문 ${paper_id}`,
              paper_authors: "Unknown",
              paper_venue: "Unknown",
              total_sections: easyData.easy_sections?.length || 0,
              total_equations: 0
            },
            easy_sections: easyData.easy_sections || [],
            math_equations: [],
            model_errors: {
              easy_model_error: undefined,
              math_model_error: "Math 모델이 아직 처리되지 않았습니다",
              viz_api_error: "Viz API가 아직 처리되지 않았습니다"
            },
            processing_logs: [
              "✅ Easy 모델 완료 - 중학생도 이해할 수 있는 쉬운 설명 생성됨",
              "⏳ Math 모델 처리 중 - 수식 분석 및 상세 해설 생성 중",
              "⏳ Viz API 처리 중 - 섹션별 시각화 이미지 생성 중"
            ]
          };
          setIntegratedData(partialData);
          console.log("✅ [Result] Easy 결과만 로드 완료");
          return;
        }
        
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setIntegratedData(result);
      console.log("✅ [Result] 통합 결과 로드 완료");
    } catch (err) {
      console.error("❌ [Result] 데이터 로드 실패:", err);
      setError(err instanceof Error ? err.message : '데이터 로드 실패');
      
      // 에러 시에도 기본 데이터 표시
      const pathParts = window.location.pathname.split('/');
      const paper_id = pathParts[pathParts.length - 1];
      const fallbackData = {
        paper_info: {
          paper_id: paper_id,
          paper_title: `논문 ${paper_id}`,
          paper_authors: "Unknown",
          paper_venue: "Unknown",
          total_sections: 0,
          total_equations: 0
        },
        easy_sections: [],
        math_equations: [],
        model_errors: {
          easy_model_error: "Easy 모델 처리 실패",
          math_model_error: "Math 모델 처리 실패",
          viz_api_error: "Viz API 처리 실패"
        },
        processing_logs: ["모든 모델 처리 실패", "데이터를 불러올 수 없습니다"]
      };
      setIntegratedData(fallbackData);
    } finally {
      setLoading(false);
    }
  };

  const renderMathJax = () => {
    if (
      (window as any).MathJax &&
      typeof (window as any).MathJax.typeset === "function"
    ) {
      (window as any).MathJax.typeset();
    }
  };

  const toggleVisualization = (sectionId: string, paragraphId: string) => {
    const key = `${sectionId}-${paragraphId}`;
    setActiveViz(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const toggleVizApi = (sectionId: string) => {
    setActiveVizApi(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const toggleEquation = (equationId: string) => {
    setActiveEquation(prev => prev === equationId ? null : equationId);
  };

  // Viz API 호출 함수 (임시)
  const callVizApi = async (sectionId: string, sectionTitle: string, sectionContent: string) => {
    const key = sectionId;
    setLoadingVizApi(prev => ({ ...prev, [key]: true }));
    
    try {
      // 임시 Viz API 호출 (실제 API 엔드포인트로 교체 필요)
      const response = await fetch('/api/viz-api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          section_id: sectionId,
          section_title: sectionTitle,
          section_content: sectionContent,
        }),
      });

      if (!response.ok) {
        throw new Error(`Viz API 호출 실패: ${response.status}`);
      }

      const vizResult: VizApiResult = await response.json();
      
      // 결과를 integratedData에 업데이트
      setIntegratedData(prev => {
        if (!prev) return prev;
        
        const updatedSections = prev.easy_sections.map(section => {
          if (section.easy_section_id === sectionId) {
            return {
              ...section,
              viz_api_result: vizResult
            };
          }
          return section;
        });

        return {
          ...prev,
          easy_sections: updatedSections
        };
      });

    } catch (error) {
      console.error('Viz API 호출 오류:', error);
      
      // 에러 상태로 Viz API 결과 설정
      const errorResult: VizApiResult = {
        viz_api_id: `${sectionId}_error`,
        viz_api_title: `${sectionTitle} 시각화`,
        viz_api_description: '시각화 생성 중 오류가 발생했습니다.',
        viz_api_type: 'section_visualization',
        viz_api_status: 'error',
        viz_api_error: error instanceof Error ? error.message : '알 수 없는 오류'
      };

      setIntegratedData(prev => {
        if (!prev) return prev;
        
        const updatedSections = prev.easy_sections.map(section => {
          if (section.easy_section_id === sectionId) {
            return {
              ...section,
              viz_api_result: errorResult
            };
          }
          return section;
        });

        return {
          ...prev,
          easy_sections: updatedSections
        };
      });
    } finally {
      setLoadingVizApi(prev => ({ ...prev, [key]: false }));
    }
  };

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  const createTableOfContents = () => {
    if (!integratedData?.easy_sections) return null;

    const renderTocItem = (section: EasySection, level: number = 0) => {
      const indentClass =
        level > 0 ? `toc-subsection toc-level-${level}` : "toc-section";

      return (
        <li key={section.easy_section_id} className={indentClass}>
          <a
            href={`#${section.easy_section_id}`}
            onClick={(e) => {
              e.preventDefault();
              scrollToSection(section.easy_section_id);
            }}
          >
            {level > 0 && <span className="toc-indent">└ </span>}
            {section.easy_section_title}
          </a>
          {section.easy_subsections && section.easy_subsections.length > 0 && (
            <ul className="toc-sublist">
              {section.easy_subsections.map((subsection) =>
                renderTocItem(subsection, level + 1)
              )}
            </ul>
          )}
        </li>
      );
    };

    return (
      <div className="sidebar">
        <nav className="table-of-contents" id="table-of-contents">
          <h3>목차</h3>
          <ul id="toc-list">
            {integratedData.easy_sections.map((section) =>
              renderTocItem(section)
            )}
          </ul>
        </nav>

        {/* HTML 다운로드 버튼 */}
        <div className="download-section">
          <button
            id="download-html-btn"
            className="download-btn"
            onClick={downloadAsHTML}
            disabled={isDownloading}
          >
            <span className="download-icon">🌐</span>
            {isDownloading ? "HTML 생성 중..." : "HTML로 다운로드"}
          </button>
        </div>
      </div>
    );
  };

  const createSectionElement = (section: EasySection, index: number) => {
    const isSubsection = section.easy_section_type === "subsection";
    const sectionClass = isSubsection ? "paper-subsection" : "paper-section";
    const headerClass = isSubsection ? "subsection-header" : "section-header";
    const titleClass = isSubsection ? "subsection-title" : "section-title";

    return (
      <div
        key={section.easy_section_id}
        className={sectionClass}
        id={section.easy_section_id}
      >
        <div className={headerClass}>
          <div className={titleClass}>
            <span className="section-order">
              {isSubsection
                ? `${section.easy_section_order}.`
                : section.easy_section_order}
            </span>
            <span>{section.easy_section_title}</span>
          </div>
          
          {/* Viz API 버튼 (섹션에만 표시) */}
          {!isSubsection && (
            <div className="section-actions">
              <button
                className="viz-api-button"
                onClick={() => {
                  if (!section.viz_api_result) {
                    // 첫 호출 시 Viz API 호출
                    callVizApi(section.easy_section_id, section.easy_section_title, section.easy_content);
                  }
                  toggleVizApi(section.easy_section_id);
                }}
                disabled={loadingVizApi[section.easy_section_id]}
              >
                {loadingVizApi[section.easy_section_id] ? (
                  <>
                    <span className="spinner-small"></span>
                    생성 중...
                  </>
                ) : (
                  <>
                    🎨 {activeVizApi[section.easy_section_id] ? '시각화 숨기기' : '시각화 보기'}
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        <div className="easy-content">
          {section.easy_paragraphs.map((paragraph) => (
            <div
              key={paragraph.easy_paragraph_id}
              className="paragraph-container"
            >
              <p
                className={`paragraph-text ${
                  paragraph.easy_visualization_trigger
                    ? "clickable-paragraph"
                    : ""
                }`}
                onClick={() => {
                  if (paragraph.easy_visualization_trigger) {
                    toggleVisualization(
                      section.easy_section_id,
                      paragraph.easy_paragraph_id
                    );
                  }
                }}
                dangerouslySetInnerHTML={{
                  __html: formatText(paragraph.easy_paragraph_text),
                }}
              />

              {/* 시각화 표시 영역 */}
              {paragraph.easy_visualization_trigger &&
                activeViz[
                  `${section.easy_section_id}-${paragraph.easy_paragraph_id}`
                ] && (
                  <div className="visualization-container">
                    {section.easy_visualizations?.map((viz) => (
                      <div key={viz.easy_viz_id} className="visualization-item">
                        <h4>{viz.easy_viz_title}</h4>
                        {viz.easy_viz_description && (
                          <p className="viz-description">
                            {viz.easy_viz_description}
                          </p>
                        )}
                        {viz.easy_viz_image_path && (
                          <img
                            src={viz.easy_viz_image_path}
                            alt={viz.easy_viz_title}
                            className="viz-image"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.style.display = "none";
                              const fallback = document.createElement("div");
                              fallback.className = "image-fallback";
                              fallback.textContent = viz.easy_viz_title;
                              fallback.style.cssText =
                                "padding: 40px; text-align: center; background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; color: #6c757d;";
                              target.parentNode?.appendChild(fallback);
                            }}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                )}
            </div>
          ))}
        </div>

        {/* Viz API 결과 표시 영역 */}
        {!isSubsection && activeVizApi[section.easy_section_id] && section.viz_api_result && (
          <div className="viz-api-container">
            <div className="viz-api-header">
              <h4>🎨 {section.viz_api_result.viz_api_title}</h4>
              {section.viz_api_result.viz_api_description && (
                <p className="viz-api-description">{section.viz_api_result.viz_api_description}</p>
              )}
            </div>
            
            {section.viz_api_result.viz_api_status === 'success' && section.viz_api_result.viz_api_image_url && (
              <div className="viz-api-image-container">
                <img
                  src={section.viz_api_result.viz_api_image_url}
                  alt={section.viz_api_result.viz_api_title}
                  className="viz-api-image"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.style.display = 'none';
                    const fallback = document.createElement('div');
                    fallback.className = 'viz-api-fallback';
                    fallback.textContent = '이미지를 불러올 수 없습니다';
                    fallback.style.cssText = 'padding: 40px; text-align: center; background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; color: #6c757d;';
                    target.parentNode?.appendChild(fallback);
                  }}
                />
              </div>
            )}
            
            {section.viz_api_result.viz_api_status === 'error' && (
              <div className="viz-api-error">
                <div className="error-icon">⚠️</div>
                <div className="error-message">
                  <strong>시각화 생성 실패</strong>
                  <p>{section.viz_api_result.viz_api_error || '알 수 없는 오류가 발생했습니다.'}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Subsection들 렌더링 */}
        {section.easy_subsections && section.easy_subsections.length > 0 && (
          <div className="subsections-container">
            {section.easy_subsections.map((subsection) =>
              createSectionElement(subsection, 0)
            )}
          </div>
        )}

        {/* 수식 섹션 */}
        {integratedData?.math_equations && (
          <div className="math-equations">
            {integratedData.math_equations
              .filter(eq => {
                // easy_section_id와 math_equation_section_ref 매핑
                // Math 모델에서 생성된 수식들을 해당 섹션에 맞게 필터링
                return eq.math_equation_section_ref === section.easy_section_id ||
                       eq.math_equation_section_ref === `section_${section.easy_section_order}` ||
                       eq.math_equation_section_ref === section.easy_section_title;
              })
              .map((equation) => (
                <div key={equation.math_equation_id} className="equation-item">
                  <div className="equation-header">
                    <div className="equation-index">
                      {equation.math_equation_index}
                    </div>
                    <div className="equation-title">
                      수식 {equation.math_equation_index}
                    </div>
                    <button
                      className="equation-toggle"
                      onClick={() => toggleEquation(equation.math_equation_id)}
                    >
                      {activeEquation === equation.math_equation_id
                        ? "숨기기"
                        : "설명 보기"}
                    </button>
                  </div>
                  
                  <div 
                    className={`equation ${activeEquation === equation.math_equation_id ? 'equation-active' : ''}`}
                    ref={mathJaxRef}
                    onClick={() => toggleEquation(equation.math_equation_id)}
                    style={{ cursor: 'pointer' }}
                    title="수식을 클릭하면 설명을 볼 수 있습니다"
                  >
                    {`$$${equation.math_equation_latex}$$`}
                  </div>

                  {activeEquation === equation.math_equation_id && (
                    <div className="equation-explanation">
                      <div className="explanation-header">
                        <span className="explanation-icon">💡</span>
                        <span className="explanation-title">수식 설명</span>
                      </div>
                      <div className="explanation-content" dangerouslySetInnerHTML={{ __html: formatText(equation.math_equation_explanation) }} />
                    </div>
                  )}
                </div>
              ))}
          </div>
        )}
      </div>
    );
  };

  const formatText = (text: string) => {
    if (!text) return "";
    return text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  };

  const downloadAsHTML = async () => {
    if (!integratedData) return;

    try {
      setIsDownloading(true);

      // HTML 내용 생성
      const htmlContent = generateHTMLContent(integratedData);

      // 이미지를 Base64로 변환
      const processedHtml = await convertImagesToBase64(htmlContent);

      // 파일 다운로드
      const blob = new Blob([processedHtml], {
        type: "text/html;charset=utf-8",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `YOLOv1_논문분석_${
        new Date().toISOString().split("T")[0]
      }.html`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("HTML 다운로드 오류:", error);
      alert(
        "HTML 다운로드 중 오류가 발생했습니다: " + (error as Error).message
      );
    } finally {
      setIsDownloading(false);
    }
  };

  const generateHTMLContent = (data: IntegratedData) => {
    const sectionsHtml = data.easy_sections
      .map((section, index) => {
        const mathEquations = data.math_equations.filter(
          (eq) => eq.math_equation_section_ref === section.easy_section_id
        );

        return `
        <div class="paper-section" id="${section.easy_section_id}">
          <div class="section-header">
            <div class="section-title">
              <span class="section-order">${section.easy_section_order}</span>
              <span>${section.easy_section_title}</span>
            </div>
          </div>
          
          <div class="easy-content">
            ${section.easy_paragraphs
              .map(
                (paragraph) =>
                  `<p class="paragraph-text">${formatText(
                    paragraph.easy_paragraph_text
                  )}</p>`
              )
              .join("")}
          </div>

          ${
            mathEquations.length > 0
              ? `
            <div class="math-equations">
              ${mathEquations
                .map(
                  (equation) => `
                <div class="equation-item">
                  <div class="equation-header">
                    <div class="equation-index">${
                      equation.math_equation_index
                    }</div>
                    <div class="equation-title">수식 ${
                      equation.math_equation_index
                    }</div>
                  </div>
                  <div class="equation">$$${
                    equation.math_equation_latex
                  }$$</div>
                  <div class="equation-explanation">${formatText(
                    equation.math_equation_explanation
                  )}</div>
                </div>
              `
                )
                .join("")}
            </div>
          `
              : ""
          }
        </div>
      `;
      })
      .join("");

    return `
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv1 논문 분석 결과</title>
    <style>
        ${getInlineStyles()}
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [["$", "$"], ["\\(", "\\)"]],
                displayMath: [["$$", "$$"], ["\\[", "\\]"]]
            }
        };
    </script>
</head>
<body>
    <div class="container">
        <div class="main-content">
            <header class="paper-header">
                <h1>${data.paper_info.paper_title}</h1>
                <div class="paper-info">
                    <p><strong>논문 제목:</strong> ${
                      data.paper_info.paper_title
                    }</p>
                    <p><strong>저자:</strong> ${
                      data.paper_info.paper_authors
                    }</p>
                    <p><strong>발표:</strong> ${data.paper_info.paper_venue}</p>
                    <p><strong>논문 ID:</strong> ${data.paper_info.paper_id}</p>
                </div>
            </header>
            <div class="integrated-paper">
                <div class="paper-sections">
                    ${sectionsHtml}
                </div>
            </div>
            <footer class="paper-footer">
                <p>AI 통합 분석 시스템 | YOLOv1 논문 분석 결과</p>
                <div class="stats">
                    <div class="stat-item">
                        <span class="stat-number">${
                          data.paper_info.total_sections
                        }</span>
                        <span class="stat-label">총 섹션</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${
                          data.paper_info.total_equations
                        }</span>
                        <span class="stat-label">총 수식</span>
                    </div>
                </div>
            </footer>
        </div>
    </div>
</body>
</html>`;
  };

  const convertImagesToBase64 = async (htmlContent: string) => {
    // 임시 DOM 요소 생성
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = htmlContent;

    // 모든 이미지 요소 찾기
    const images = tempDiv.querySelectorAll("img");

    for (let img of images) {
      try {
        // 이미지가 로드되었는지 확인
        if (img.complete && img.naturalHeight !== 0) {
          // Canvas를 사용하여 이미지를 Base64로 변환
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");

          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;

          ctx?.drawImage(img, 0, 0);
          const base64 = canvas.toDataURL("image/png");

          // src를 Base64로 교체
          img.src = base64;
        } else {
          // 이미지가 로드되지 않은 경우 원본 경로 유지
          console.warn("이미지 로드 실패:", img.src);
        }
      } catch (error) {
        console.error("이미지 변환 오류:", error);
        // 오류 발생 시 원본 경로 유지
      }
    }

    return tempDiv.innerHTML;
  };

  const getInlineStyles = () => {
    return `
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body { font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; }
      .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
      .main-content { flex: 1; min-width: 0; }
      .paper-header { background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
      .paper-header h1 { font-size: 2.5em; margin-bottom: 20px; text-align: center; font-weight: 700; color: white; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); }
      .paper-info { background: rgba(0, 0, 0, 0.3); padding: 20px; border-radius: 8px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
      .paper-info p { margin-bottom: 8px; font-size: 1.1em; color: rgba(255, 255, 255, 0.95); font-weight: 500; }
      .paper-info strong { color: #ffd700; font-weight: 700; }
      .integrated-paper { background: white; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); overflow: hidden; }
      .paper-sections { padding: 0; }
      .paper-section { border-bottom: 1px solid #e9ecef; padding: 40px; transition: background-color 0.3s ease; }
      .paper-section:hover { background-color: #f8f9fa; }
      .paper-section:last-child { border-bottom: none; }
      .section-header { margin-bottom: 30px; padding-bottom: 15px; border-bottom: 2px solid #f59e0b; }
      .section-title { font-size: 2em; color: #2c3e50; margin-bottom: 10px; font-weight: 600; }
      .section-order { display: inline-block; background: #f59e0b; color: white; width: 30px; height: 30px; border-radius: 50%; text-align: center; line-height: 30px; font-weight: bold; margin-right: 15px; vertical-align: middle; }
      .easy-content { margin-bottom: 30px; padding: 25px; background: #fef3c7; border-radius: 8px; border-left: 4px solid #f59e0b; }
      .easy-content p { font-size: 1.1em; line-height: 1.8; color: #424242; }
      .easy-content strong { color: #d97706; font-weight: 600; }
      .math-equations { margin-top: 30px; }
      .equation-item { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); transition: box-shadow 0.3s ease; }
      .equation-item:hover { box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
      .equation-header { display: flex; align-items: center; margin-bottom: 15px; }
      .equation-index { background: #d32f2f; color: white; width: 25px; height: 25px; border-radius: 50%; text-align: center; line-height: 25px; font-weight: bold; margin-right: 10px; font-size: 0.9em; }
      .equation-title { color: #d32f2f; font-weight: 600; font-size: 1.1em; }
      .equation { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 15px; text-align: center; border: 1px solid #e9ecef; overflow-x: auto; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
      .equation-explanation { color: #424242; line-height: 1.7; font-size: 1.05em; }
      .equation-explanation strong { color: #d32f2f; font-weight: 600; }
      .paper-footer { background: #2c3e50; color: white; padding: 30px; border-radius: 12px; margin-top: 30px; text-align: center; }
      .paper-footer p { margin-bottom: 15px; font-size: 1.1em; }
      .stats { display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; }
      .stat-item { text-align: center; padding: 15px; background: rgba(255, 255, 255, 0.1); border-radius: 8px; min-width: 100px; }
      .stat-number { font-size: 2em; font-weight: bold; color: #ffd700; display: block; margin-bottom: 5px; }
      .stat-label { font-size: 0.9em; opacity: 0.8; }
    `;
  };

  const displayStats = () => {
    if (!integratedData?.paper_info) return null;

    const { paper_info } = integratedData;
    const sectionsWithEquations = integratedData.easy_sections.filter(
      (section) =>
        integratedData.math_equations.some(
          (eq) => eq.math_equation_section_ref === section.easy_section_id
        )
    ).length;

    return (
      <div className="stats" id="stats">
        <div className="stat-item">
          <span className="stat-number">{paper_info.total_sections}</span>
          <span className="stat-label">총 섹션</span>
        </div>
        <div className="stat-item">
          <span className="stat-number">{paper_info.total_equations}</span>
          <span className="stat-label">총 수식</span>
        </div>
        <div className="stat-item">
          <span className="stat-number">{sectionsWithEquations}</span>
          <span className="stat-label">수식 포함 섹션</span>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="container">
        <div id="loading" className="loading">
          <div className="spinner"></div>
          <p>데이터를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container">
        <div id="error" className="error">
          <h3>오류 발생</h3>
          <p>{error}</p>
          {integratedData?.model_errors && (
            <div className="model-errors">
              <h4>모델별 오류 정보:</h4>
              {integratedData.model_errors.easy_model_error && (
                <div className="model-error-item">
                  <strong>Easy 모델:</strong> {integratedData.model_errors.easy_model_error}
                </div>
              )}
              {integratedData.model_errors.math_model_error && (
                <div className="model-error-item">
                  <strong>Math 모델:</strong> {integratedData.model_errors.math_model_error}
                </div>
              )}
              {integratedData.model_errors.viz_api_error && (
                <div className="model-error-item">
                  <strong>Viz API:</strong> {integratedData.model_errors.viz_api_error}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (!integratedData) {
    return (
      <div className="container">
        <div id="error" className="error">
          <h3>데이터 없음</h3>
          <p>표시할 데이터가 없습니다.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      {createTableOfContents()}

      <div className="main-content">
        <header className="paper-header">
          <h1 id="paper-title">{integratedData.paper_info.paper_title}</h1>
          <div className="paper-info">
            <p>
              <strong>논문 제목:</strong>
              <span id="paper-title-text">
                {integratedData.paper_info.paper_title}
              </span>
            </p>
            <p>
              <strong>저자:</strong>{" "}
              <span id="paper-authors">
                {integratedData.paper_info.paper_authors}
              </span>
            </p>
            <p>
              <strong>발표:</strong>{" "}
              <span id="paper-venue">
                {integratedData.paper_info.paper_venue}
              </span>
            </p>
            <p>
              <strong>논문 ID:</strong>{" "}
              <span id="paper-id">{integratedData.paper_info.paper_id}</span>
            </p>
          </div>
        </header>

        <div id="integrated-paper" className="integrated-paper">
          <div className="paper-sections" id="paper-sections">
            {integratedData.easy_sections.map((section, index) =>
              createSectionElement(section, index)
            )}
          </div>
        </div>

        {/* 모델 에러 및 로그 표시 */}
        {integratedData?.model_errors && (
          <div className="model-status">
            <h3>모델 처리 상태</h3>
            <div className="model-status-grid">
              {integratedData.model_errors.easy_model_error ? (
                <div className="status-item error">
                  <span className="status-icon">❌</span>
                  <div className="status-content">
                    <strong>Easy 모델</strong>
                    <p>{integratedData.model_errors.easy_model_error}</p>
                  </div>
                </div>
              ) : (
                <div className="status-item success">
                  <span className="status-icon">✅</span>
                  <div className="status-content">
                    <strong>Easy 모델</strong>
                    <p>정상 처리 완료</p>
                  </div>
                </div>
              )}
              
              {integratedData.model_errors.math_model_error ? (
                <div className="status-item error">
                  <span className="status-icon">❌</span>
                  <div className="status-content">
                    <strong>Math 모델</strong>
                    <p>{integratedData.model_errors.math_model_error}</p>
                  </div>
                </div>
              ) : (
                <div className="status-item success">
                  <span className="status-icon">✅</span>
                  <div className="status-content">
                    <strong>Math 모델</strong>
                    <p>정상 처리 완료</p>
                  </div>
                </div>
              )}
              
              {integratedData.model_errors.viz_api_error ? (
                <div className="status-item error">
                  <span className="status-icon">❌</span>
                  <div className="status-content">
                    <strong>Viz API</strong>
                    <p>{integratedData.model_errors.viz_api_error}</p>
                  </div>
                </div>
              ) : (
                <div className="status-item success">
                  <span className="status-icon">✅</span>
                  <div className="status-content">
                    <strong>Viz API</strong>
                    <p>정상 처리 완료</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 처리 로그 표시 */}
        {integratedData?.processing_logs && integratedData.processing_logs.length > 0 && (
          <div className="processing-logs">
            <h3>처리 로그</h3>
            <div className="logs-container">
              {integratedData.processing_logs.map((log, index) => (
                <div key={index} className="log-item">
                  <span className="log-time">{new Date().toLocaleTimeString()}</span>
                  <span className="log-message">{log}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <footer className="paper-footer">
          <p>AI 통합 분석 시스템 | YOLOv1 논문 분석 결과</p>
          {displayStats()}
        </footer>
      </div>
    </div>
  );
};

export default Result;
