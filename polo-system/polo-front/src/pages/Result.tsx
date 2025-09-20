import React, { useState, useEffect, useRef } from 'react';
import './Result.css';

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
  easy_section_type: 'section' | 'subsection';
  easy_section_order: number;
  easy_section_level?: number;
  easy_section_parent?: string;
  easy_content: string;
  easy_paragraphs: EasyParagraph[];
  easy_subsections?: EasySection[];
  easy_visualizations?: EasyVisualization[];
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
}

interface ResultProps {
  data?: IntegratedData;
  onDownload?: () => void;
  onPreview?: () => void;
}

const Result: React.FC<ResultProps> = ({ data, onDownload, onPreview }) => {
  const [integratedData, setIntegratedData] = useState<IntegratedData | null>(data || null);
  const [loading, setLoading] = useState(!data);
  const [error, setError] = useState<string | null>(null);
  const [activeViz, setActiveViz] = useState<{ [key: string]: boolean }>({});
  const [activeEquation, setActiveEquation] = useState<string | null>(null);
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
      // 실제 API 호출로 변경
      const response = await fetch('/api/integrated-result');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setIntegratedData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : '데이터 로드 실패');
    } finally {
      setLoading(false);
    }
  };

  const renderMathJax = () => {
    if ((window as any).MathJax && typeof (window as any).MathJax.typeset === 'function') {
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

  const toggleEquation = (equationId: string) => {
    setActiveEquation(prev => prev === equationId ? null : equationId);
  };

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const createTableOfContents = () => {
    if (!integratedData?.easy_sections) return null;

    const renderTocItem = (section: EasySection, level: number = 0) => {
      const indentClass = level > 0 ? `toc-subsection toc-level-${level}` : 'toc-section';
      
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
      <nav className="table-of-contents" id="table-of-contents">
        <h3>목차</h3>
        <ul id="toc-list">
          {integratedData.easy_sections.map((section) => renderTocItem(section))}
        </ul>
      </nav>
    );
  };

  const createSectionElement = (section: EasySection, index: number) => {
    const isSubsection = section.easy_section_type === 'subsection';
    const sectionClass = isSubsection ? 'paper-subsection' : 'paper-section';
    const headerClass = isSubsection ? 'subsection-header' : 'section-header';
    const titleClass = isSubsection ? 'subsection-title' : 'section-title';

    return (
      <div key={section.easy_section_id} className={sectionClass} id={section.easy_section_id}>
        <div className={headerClass}>
          <div className={titleClass}>
            <span className="section-order">
              {isSubsection ? `${section.easy_section_order}.` : section.easy_section_order}
            </span>
            <span>{section.easy_section_title}</span>
          </div>
        </div>

        <div className="easy-content">
          {section.easy_paragraphs.map((paragraph) => (
            <div key={paragraph.easy_paragraph_id} className="paragraph-container">
              <p
                className={`paragraph-text ${paragraph.easy_visualization_trigger ? 'clickable-paragraph' : ''}`}
                onClick={() => {
                  if (paragraph.easy_visualization_trigger) {
                    toggleVisualization(section.easy_section_id, paragraph.easy_paragraph_id);
                  }
                }}
                dangerouslySetInnerHTML={{ __html: formatText(paragraph.easy_paragraph_text) }}
              />
              
              {/* 시각화 표시 영역 */}
              {paragraph.easy_visualization_trigger && 
               activeViz[`${section.easy_section_id}-${paragraph.easy_paragraph_id}`] && (
                <div className="visualization-container">
                  {section.easy_visualizations?.map((viz) => (
                    <div key={viz.easy_viz_id} className="visualization-item">
                      <h4>{viz.easy_viz_title}</h4>
                      {viz.easy_viz_description && (
                        <p className="viz-description">{viz.easy_viz_description}</p>
                      )}
                      {viz.easy_viz_image_path && (
                        <img
                          src={viz.easy_viz_image_path}
                          alt={viz.easy_viz_title}
                          className="viz-image"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.style.display = 'none';
                            const fallback = document.createElement('div');
                            fallback.className = 'image-fallback';
                            fallback.textContent = viz.easy_viz_title;
                            fallback.style.cssText = 'padding: 40px; text-align: center; background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; color: #6c757d;';
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
              .filter(eq => eq.math_equation_section_ref === section.easy_section_id)
              .map((equation) => (
                <div key={equation.math_equation_id} className="equation-item">
                  <div className="equation-header">
                    <div className="equation-index">{equation.math_equation_index}</div>
                    <div className="equation-title">수식 {equation.math_equation_index}</div>
                    <button
                      className="equation-toggle"
                      onClick={() => toggleEquation(equation.math_equation_id)}
                    >
                      {activeEquation === equation.math_equation_id ? '숨기기' : '설명 보기'}
                    </button>
                  </div>
                  
                  <div className="equation" ref={mathJaxRef}>
                    {`$$${equation.math_equation_latex}$$`}
                  </div>
                  
                  {activeEquation === equation.math_equation_id && (
                    <div className="equation-explanation">
                      <div dangerouslySetInnerHTML={{ __html: formatText(equation.math_equation_explanation) }} />
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
    if (!text) return '';
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  };

  const displayStats = () => {
    if (!integratedData?.paper_info) return null;

    const { paper_info } = integratedData;
    const sectionsWithEquations = integratedData.easy_sections.filter(
      section => integratedData.math_equations.some(
        eq => eq.math_equation_section_ref === section.easy_section_id
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
              <span id="paper-title-text">{integratedData.paper_info.paper_title}</span>
            </p>
            <p>
              <strong>저자:</strong> <span id="paper-authors">{integratedData.paper_info.paper_authors}</span>
            </p>
            <p>
              <strong>발표:</strong> <span id="paper-venue">{integratedData.paper_info.paper_venue}</span>
            </p>
            <p>
              <strong>논문 ID:</strong> <span id="paper-id">{integratedData.paper_info.paper_id}</span>
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

        <footer className="paper-footer">
          <p>AI 통합 분석 시스템 | YOLOv1 논문 분석 결과</p>
          {displayStats()}
        </footer>
      </div>
    </div>
  );
};

export default Result;
