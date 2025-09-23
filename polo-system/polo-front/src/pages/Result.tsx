import React, { useState, useEffect, useRef, useMemo } from "react";
import { useLocation } from "react-router-dom";
import { marked } from "marked";
import "./Result.css";
import type { FigureMeta, EasyParagraph as EasyParagraphType, IntegratedData as IntegratedDataType } from "../types";
import { loadFigureQueue, createFigureQueue, type FigureItem } from "../utils/figureMap";
import { replaceFigureTokens, analyzeFigureTokens, type TextChunk } from "../utils/figureTokens";

marked.setOptions({ gfm: true, breaks: true });

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
  paragraph_type?: string; // "math_equation" for math paragraphs
  math_equation?: any; // Math equation data
  visualization?: { image_path?: string }; // Visualization data
}

interface EasyVisualization {
  easy_viz_id: string;
  easy_viz_title: string;
  easy_viz_description?: string;
  easy_viz_image_path?: string;
  easy_viz_type: "chart" | "diagram" | "graph" | "table";
}

interface VizApiResult {
  viz_api_id: string;
  viz_api_title: string;
  viz_api_description?: string;
  viz_api_image_url?: string;
  viz_api_type: "section_visualization";
  viz_api_status: "success" | "error" | "loading";
  viz_api_error?: string;
}

interface MathEquation {
  math_equation_id: string;
  math_equation_index: string;
  math_equation_latex: string;
  math_equation_explanation: string;
  math_equation_context?: string;
  math_equation_section_ref?: string; // 어떤 섹션에 속하는지
  math_equation_env?: string; // 수식 환경 (cases, aligned 등)
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

// 파일 상단 utils 인근에 추가
const renderMarkdown = (t?: string) => {
  if (!t) return "";
  
  // 마크다운 텍스트 내의 LaTeX 수식 전처리
  let processed = t;
  
  // 인라인 수식 $...$ 전처리
  processed = processed.replace(/\$([^$]+)\$/g, (match, content) => {
    return `$${preprocessLatex(content)}$`;
  });
  
  // 블록 수식 $$...$$ 전처리
  processed = processed.replace(/\$\$([^$]+)\$\$/g, (match, content) => {
    return `$$${preprocessLatex(content)}$$`;
  });
  
  return marked.parse(processed);
};

// 역슬래시 이스케이프 복원 함수
const unescapeOnce = (s: string): string => {
  if (!s) return s;
  // "\\(" -> "\(" , "\\leq" -> "\leq", "\\phi" -> "\phi" 등
  return s.replace(/\\\\/g, '\\');
};

// LaTeX 텍스트에서 MathJax가 지원하지 않는 매크로를 지원되는 형태로 치환
const fixLatexMacros = (s: string): string => {
  if (!s) return s;
  
  // \mathlarger{...} -> \large{...}로 대체 또는 제거
  s = s.replace(/\\mathlarger\s*\{([^}]+)\}/g, '\\large{$1}');
  s = s.replace(/\\mathlarger\s+/g, '');  // 인수 없는 경우 제거
  
  // \mathbbm{1} -> \mathbf{1} (지표함수용), 다른 문자는 \mathbb로
  s = s.replace(/\\mathbbm\{1\}/g, '\\mathbf{1}');
  s = s.replace(/\\mathbbm\{([^}]+)\}/g, '\\mathbb{$1}');
  
  return s;
};

// LaTeX 수식 전처리 파이프라인 (이스케이프 복원 + 매크로 수정)
const preprocessLatex = (s: string): string => {
  if (!s) return s;
  
  // 1. 역슬래시 이스케이프 복원
  let processed = unescapeOnce(s);
  
  // 2. 매크로 수정
  processed = fixLatexMacros(processed);
  
  return processed;
};

// 설명 텍스트 정리(접두 제거 + 군더더기 제거)
const sanitizeExplain = (t?:string) =>
  (t ?? "")
    .replace(/^\s*(조수|assistant)\s*[:：\-]?\s*/i, "")
    .replace(/^\s*(조수|assistant)\s*[:：\-]?\s*/gmi, "")
    .replace(/\[?\s*수학\s*\d+\s*\]?/g, "")   // [수학0] 등 제거
    .replace(/^\s*보조\s*:?/gmi, "")          // '보조' 접두 제거
    .trim();

// 파일 상단 utils 근처에 보조 함수 2개 추가
const coalesce = <T,>(...vals: (T | undefined | null)[]) => vals.find(v => v !== undefined && v !== null);

const pickEquation = (raw: any) => {
  const id    = coalesce(raw?.math_equation_id, raw?.equation_id, raw?.id);
  const latex = coalesce(raw?.math_equation_latex, raw?.equation_latex, raw?.latex) || "";
  const env   = coalesce(raw?.math_equation_env,   raw?.equation_env,   raw?.env);
  const expl  = coalesce(raw?.math_equation_explanation, raw?.equation_explanation, raw?.explanation);
  const idx   = coalesce(raw?.math_equation_index, raw?.equation_index);
  return { id, latex, env, explanation: expl, index: idx };
};

// MathJax 준비 보장 후 typeset
const typesetNodes = async (nodes: Element[]) => {
  const w:any = window as any;
  if (!w.MathJax) return;
  // MathJax v3는 startup.promise 대기 후 typesetPromise 권장
  if (w.MathJax.startup?.promise) { try { await w.MathJax.startup.promise; } catch {} }
  if (w.MathJax.typesetPromise)  { try { await w.MathJax.typesetPromise(nodes); } catch {} }
};

// 섹션을 그룹화: 상위 section 뒤에 나오는 subsections를 묶음
function groupSections(sections: EasySection[]) {
  const groups: { parent: EasySection; children: EasySection[] }[] = [];
  let current: { parent: EasySection; children: EasySection[] } | null = null;

  for (const sec of sections.sort((a,b)=>a.easy_section_order-b.easy_section_order)) {
    if (sec.easy_section_type === "section") {
      current = { parent: sec, children: [] };
      groups.push(current);
    } else if (sec.easy_section_type === "subsection") {
      if (!current) {
        current = { parent: sec, children: [] };
        groups.push(current);
      } else {
        current.children.push(sec);
      }
    } else {
      // 기타 타입 대비
      if (!current) {
        current = { parent: sec, children: [] };
        groups.push(current);
      } else {
        current.children.push(sec);
      }
    }
  }
  return groups;
}

// easy_paragraphs가 없거나 빈 경우, easy_content를 빈줄 기준으로 문단화
function ensureParagraphs(sec: EasySection): EasyParagraph[] {
  if (sec.easy_paragraphs && sec.easy_paragraphs.length) return sec.easy_paragraphs;
  const chunks = (sec.easy_content || "").split(/\n{2,}/).map(s=>s.trim()).filter(Boolean);
  return chunks.map((t, i) => ({
    easy_paragraph_id: `${sec.easy_section_id}_p${i+1}`,
    easy_paragraph_text: t,
    easy_paragraph_order: i+1
  }));
}

const Result: React.FC<ResultProps> = ({ data, onDownload, onPreview }) => {
  const location = useLocation();
  const [integratedData, setIntegratedData] = useState<IntegratedData | null>(
    data || location.state?.data || null
  );
  const [loading, setLoading] = useState(!data && !location.state?.data);
  const [error, setError] = useState<string | null>(null);
  const [activeViz, setActiveViz] = useState<{ [key: string]: boolean }>({});
  const [activeVizApi, setActiveVizApi] = useState<{ [key: string]: boolean }>(
    {}
  );
  const [activeEquation, setActiveEquation] = useState<string | null>(null);
  const [loadingVizApi, setLoadingVizApi] = useState<{
    [key: string]: boolean;
  }>({});
  const [isDownloading, setIsDownloading] = useState(false);
  const mathJaxRef = useRef<HTMLDivElement>(null);
  // 미니-TOC만 사용 (현재/이전/다음 섹션)
  // 불필요: 검색/보통모드/펼침 상태는 제거
  const [activeTocId, setActiveTocId] = useState<string>("");
  
  const [imageModal, setImageModal] = useState<{ open: boolean; src: string; alt?: string }>({ open: false, src: "" });
  const openImage = (src: string, alt?: string) => setImageModal({ open: true, src, alt });
  const closeImage = () => setImageModal({ open: false, src: "" });
  const [dark, setDark] = useState<boolean>(false);
  useEffect(() => {
    // body에 다크모드 클래스 토글 → CSS가 전체 적용
    const cls = document.documentElement.classList;
    if (dark) cls.add("dark-mode"); else cls.remove("dark-mode");
  }, [dark]);

  // MathJax 설정: 수식만 렌더(mathjax), Easy 본문은 제외(no-mathjax)
  useEffect(() => {
    const win = window as any;
    if (win.MathJax) return;
    const config = document.createElement("script");
    config.type = "text/javascript";
    config.text = `
      window.MathJax = {
        loader: { load: ['[tex]/ams', '[tex]/mathtools', '[tex]/physics'] },
        tex: {
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
          packages: { '[+]': ['ams','mathtools','physics'] },
          processEscapes: true,
          tags: 'none',
          macros: {
            mathlarger: ['{\\\\large #1}', 1],
            mathbbm: ['{\\\\mathbb{#1}}', 1],
            wt: ['{\\\\widetilde{#1}}', 1],
            wh: ['{\\\\widehat{#1}}', 1],
            dfn: '{\\\\triangleq}',
            dB: '{\\\\mathrm{dB}}',
            snr: '{\\\\mathrm{SNR}}',
            bsnr: '{\\\\mathrm{S}\\\\widetilde{\\\\mathrm{N}}\\\\mathrm{R}}',
            dsnr: '{\\\\Delta\\\\snr}'
          }
        },
        options: {
          ignoreHtmlClass: 'no-mathjax',
          processHtmlClass: 'mathjax'
        },
        svg: { fontCache: 'global', scale: 1 }   /* 줄바꿈/축소 없음 → CSS로 스크롤 */
      };
    `;
    document.head.appendChild(config);
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js";
    script.async = true;
    script.onload = () => {
      if (win.MathJax && typeof win.MathJax.typeset === 'function') {
        win.MathJax.typeset();
      }
    };
    document.head.appendChild(script);
  }, []);

  useEffect(() => {
    if (!data && !location.state?.data) {
      loadIntegratedData();
    } else if (location.state?.data) {
      console.log(
        "✅ [Result] location.state에서 통합 데이터 받음:",
        location.state.data
      );
    }
  }, [data, location.state?.data]);

  // 데이터/토글 변화 시 수식만 다시 typeset
  useEffect(() => {
    const win = window as any;
    if (win?.MathJax?.typesetPromise) {
      const nodes = Array.from(document.querySelectorAll('.mathjax'));
      win.MathJax.typesetPromise(nodes).catch(console.warn);
    }
  }, [integratedData, activeEquation]);

  // TOC: 현재 섹션 하이라이트
  useEffect(() => {
    if (!integratedData?.easy_sections) return;
    const ids = integratedData.easy_sections.map(s=>s.easy_section_id);
    const obs = new IntersectionObserver(entries=>{
      const visible = entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);
      if (visible[0]) setActiveTocId(visible[0].target.id);
    }, { rootMargin:"0px 0px -75% 0px", threshold:0 });
    ids.forEach(id => { const el = document.getElementById(id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, [integratedData]);

  const groups = groupSections(integratedData?.easy_sections || []);
  const currentGroupIdx = Math.max(
    0,
    groups.findIndex(g =>
      g.parent.easy_section_id === activeTocId || g.children.some(s=>s.easy_section_id===activeTocId)
    )
  );
  const miniSlice = groups.slice(Math.max(0, currentGroupIdx-1), Math.min(groups.length, currentGroupIdx+2));
  const tocGroups = miniSlice; // 항상 미니-TOC만

  const loadIntegratedData = async () => {
    try {
      setLoading(true);
      // URL에서 paper_id 추출 (경로 파라미터에서)
      const pathParts = window.location.pathname.split("/");
      const paper_id = pathParts[pathParts.length - 1];

      console.log(`[Result] paper_id: ${paper_id}`);

      // 통합 결과 API 호출
      const response = await fetch(`/api/integrated-result/${paper_id}`);
      if (!response.ok) {
        console.warn(
          `[Result] 통합 결과 API 실패: ${response.status}, Easy 결과만 로드 시도`
        );

        // 통합 결과 실패 시 Easy 결과만 로드
        const easyResponse = await fetch(
          `/api/results/${paper_id}/easy_results.json`
        );
        if (easyResponse.ok) {
          const easyData = await easyResponse.json();
          const partialData = {
            paper_info: easyData.paper_info || {
              paper_id: paper_id,
              paper_title: `논문 ${paper_id}`,
              paper_authors: "Unknown",
              paper_venue: "Unknown",
              total_sections: easyData.easy_sections?.length || 0,
              total_equations: 0,
            },
            easy_sections: easyData.easy_sections || [],
            math_equations: [],
            model_errors: {
              easy_model_error: undefined,
              math_model_error: "Math 모델이 아직 처리되지 않았습니다",
              viz_api_error: "Viz API가 아직 처리되지 않았습니다",
            },
            processing_logs: [
              "✅ Easy 모델 완료 - 중학생도 이해할 수 있는 쉬운 설명 생성됨",
              "⏳ Math 모델 처리 중 - 수식 분석 및 상세 해설 생성 중",
              "⏳ Viz API 처리 중 - 섹션별 시각화 이미지 생성 중",
            ],
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
      setError(err instanceof Error ? err.message : "데이터 로드 실패");

      // 에러 시에도 기본 데이터 표시
      const pathParts = window.location.pathname.split("/");
      const paper_id = pathParts[pathParts.length - 1];
      const fallbackData = {
        paper_info: {
          paper_id: paper_id,
          paper_title: `논문 ${paper_id}`,
          paper_authors: "Unknown",
          paper_venue: "Unknown",
          total_sections: 0,
          total_equations: 0,
        },
        easy_sections: [],
        math_equations: [],
        model_errors: {
          easy_model_error: "Easy 모델 처리 실패",
          math_model_error: "Math 모델 처리 실패",
          viz_api_error: "Viz API 처리 실패",
        },
        processing_logs: ["모든 모델 처리 실패", "데이터를 불러올 수 없습니다"],
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

  // Easy 텍스트는 MathJax 대상에서 제외
  const EasyText: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <div className="easy-content no-mathjax">{children}</div>
  );

  // 현재 paperId 추출 (state → URL 순)
  const getCurrentPaperId = () => {
    return (
      integratedData?.paper_info?.paper_id ||
      (typeof window !== 'undefined' ? window.location.pathname.split('/').pop() : '') ||
      ''
    );
  };

  // 이미지 경로 정규화: 다양한 경로 형식 지원
  const getImageSrc = (raw?: string) => {
    if (!raw) return "";
    
    // 이미 완전한 URL이거나 절대 경로인 경우
    if (/^https?:\/\//.test(raw) || raw.startsWith("/")) {
      // Windows 역슬래시를 슬래시로 정규화
      return raw.replace(/\\\\/g, "/");
    }
    
    // 상대 경로인 경우 /outputs/{paperId}/ 프리픽스 부여
    const pid = getCurrentPaperId();
    const normalizedPath = raw.replace(/\\\\/g, "/"); // Windows 경로 정규화
    const web = `/outputs/${pid}/${normalizedPath}`;
    return web;
  };

  // 수식 라텍스 정규화: 부분식·정렬기호 보정
  const normalizeLatex = (latex: string, env?: string) => {
    const src = (latex || "").trim();
    if (!src) return src;
    const hasBegin = /\\begin\{[a-zA-Z*]+\}/.test(src);
    if (env === 'cases' && !/\\begin\{cases\}/.test(src)) {
      return `\\begin{cases}\n${src}\n\\end{cases}`;
    }
    // 정렬 기호 &가 있으나 정렬 환경이 없으면 aligned로 감싸기
    const hasAlignChar = /(^|[^\\])&/.test(src);
    const inAlignEnv = /(aligned|align|align\*|split)/.test(src);
    if (hasAlignChar && !hasBegin && !inAlignEnv) {
      return `\\begin{aligned}\n${src}\n\\end{aligned}`;
    }
    return src;
  };

  const toggleVisualization = (sectionId: string, paragraphId: string) => {
    const key = `${sectionId}-${paragraphId}`;
    setActiveViz((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const toggleVizApi = (sectionId: string) => {
    setActiveVizApi((prev) => ({
      ...prev,
      [sectionId]: !prev[sectionId],
    }));
  };

  const toggleEquation = (equationId: string) => {
    setActiveEquation(prev => (prev === equationId ? null : equationId));
  };

  const copyToClipboard = async (text: string) => {
    try { await navigator.clipboard.writeText(text); } catch {}
  };

  // 수식 전용: MathJax 대상으로만(typesetPromise)
  const EquationView: React.FC<{ eq: any }> = ({ eq }) => {
    const ref = useRef<HTMLDivElement>(null);     // 수식 본문
    const expRef = useRef<HTMLDivElement>(null);  // 설명 박스
    const picked = pickEquation(eq);
    const id = picked.id!;
    const latex = normalizeLatex(picked.latex, picked.env);
    const explain = sanitizeExplain(picked.explanation);
    const open = activeEquation === id;

    useEffect(() => {
      const nodes:Element[] = [];
      if (ref.current) nodes.push(ref.current);
      if (open && expRef.current) nodes.push(expRef.current);
      if (nodes.length) typesetNodes(nodes);
    }, [id, latex, open]);
    return (
      <div className={`equation-item ${open ? "open":""}`} id={id}>
        <div className="equation-toolbar">
          {/* 형광등 토글 */}
          <button
            type="button"
            aria-label="수학 설명 토글"
            className={`bulb-btn ${open ? "on" : "off"}`}
            onClick={() => toggleEquation(id)}
            title={open ? "설명 끄기" : "설명 켜기"}
          >
            <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor">
              <path d="M9 21h6a1 1 0 0 0 1-1v-1h-8v1a1 1 0 0 0 1 1zM12 2c-4.1 0-7 3.02-7 6.75 0 2.37 1.25 4.08 3.23 5.62.38.3.77.9.85 1.38h5.84c.08-.49.47-1.08.85-1.38C17.75 12.83 19 11.12 19 8.75 19 5.02 16.1 2 12 2z"/>
              <g stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round">
                <path d="M12 .8v2.4M3.6 5.6l1.7 1M20.4 5.6l-1.7 1M1.8 11.2h2.4M19.8 11.2h2.4M3.6 16.8l1.7-1M20.4 16.8l-1.7-1"/>
              </g>
            </svg>
          </button>
        </div>
        <div ref={ref} className="equation-body mathjax">
          <div dangerouslySetInnerHTML={{ __html: `$$${preprocessLatex(latex)}$$` }} />
        </div>
        {open && !!explain && (
          <div ref={expRef} className="equation-explain mathjax"
               dangerouslySetInnerHTML={{ __html: renderMarkdown(explain) }} />
        )}
      </div>
    );
  };

  // Figure 컴포넌트
  const FigureView: React.FC<{ figure: FigureMeta; openImage: (s:string,a?:string)=>void; className?: string }> = ({ figure, openImage, className = "" }) => {
    const altText = figure.caption ?? figure.label ?? 'Figure';
    
    return (
      <figure className={`figure-card ${className}`}>
        <img 
          src={figure.image_path} 
          alt={altText}
          className="figure-image"
          onClick={() => openImage(figure.image_path, altText)}
          style={{ cursor: 'zoom-in' }}
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.style.display = "none";
            console.warn(`Figure 로드 실패: ${figure.image_path}`);
          }}
        />
        {(figure.caption || figure.label) && (
          <figcaption className="figure-caption">
            {figure.label && <strong>{figure.label}</strong>}
            {figure.label && figure.caption && ': '}
            {figure.caption}
          </figcaption>
        )}
        
        {/* 멀티페이지 지원 */}
        {figure.all_pages && figure.all_pages.length > 1 && (
          <div className="figure-pages">
            <span className="pages-label">Pages: </span>
            {figure.all_pages.map((pageUrl, idx) => (
              <button
                key={idx}
                className="page-btn"
                onClick={() => openImage(pageUrl, `${altText} - Page ${idx + 1}`)}
                title={`Page ${idx + 1}`}
              >
                {idx + 1}
              </button>
            ))}
          </div>
        )}
      </figure>
    );
  };

  // 문단 렌더러 (텍스트/수식/시각화/Figure 인라인)
  const ParagraphView: React.FC<{ p: any; sectionId: string; openImage: (s:string,a?:string)=>void; getImageSrc:(s?:string)=>string; }> = ({ p, sectionId, openImage, getImageSrc }) => {
    // 수식 문단: 여러 스키마 대응
    const isEq =
      p.paragraph_type === "math_equation" ||
      !!p.math_equation ||
      !!p.equation_latex || !!p.math_equation_latex;
    if (isEq) {
      const eq = p.math_equation || p;
      return <EquationView eq={eq} />;
    }
    // 일반 텍스트 문단 + 시각화(있으면)
    const hasViz = !!p.visualization?.image_path;
    return (
      <div className="paper-paragraph">
        <div className="no-mathjax easy-md"
             dangerouslySetInnerHTML={{ __html: renderMarkdown(p.easy_paragraph_text) }} />
        
        {/* Figure (원본 PDF/이미지) 우선 표시 */}
        {p.figure && (
          <FigureView 
            figure={p.figure} 
            openImage={openImage} 
            className="paragraph-figure"
          />
        )}
        
        {/* 자동 생성 시각화 (Figure가 없을 때만) */}
        {hasViz && !p.figure && (
          <figure className="figure-card" onClick={() => openImage(getImageSrc(p.visualization.image_path), "visualization")}>
            {/* eslint-disable-next-line jsx-a11y/alt-text */}
            <img src={getImageSrc(p.visualization.image_path)} />
            <figcaption className="caption">도표: 문단 {p.easy_paragraph_order}</figcaption>
          </figure>
        )}
      </div>
    );
  };

  // Viz API 호출 함수 (임시)
  const callVizApi = async (
    sectionId: string,
    sectionTitle: string,
    sectionContent: string
  ) => {
    const key = sectionId;
    setLoadingVizApi((prev) => ({ ...prev, [key]: true }));

    try {
      // 임시 Viz API 호출 (실제 API 엔드포인트로 교체 필요)
      const response = await fetch("/api/viz-api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
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
      setIntegratedData((prev) => {
        if (!prev) return prev;

        const updatedSections = prev.easy_sections.map((section) => {
          if (section.easy_section_id === sectionId) {
            return {
              ...section,
              viz_api_result: vizResult,
            };
          }
          return section;
        });

        return {
          ...prev,
          easy_sections: updatedSections,
        };
      });
    } catch (error) {
      console.error("Viz API 호출 오류:", error);

      // 에러 상태로 Viz API 결과 설정
      const errorResult: VizApiResult = {
        viz_api_id: `${sectionId}_error`,
        viz_api_title: `${sectionTitle} 시각화`,
        viz_api_description: "시각화 생성 중 오류가 발생했습니다.",
        viz_api_type: "section_visualization",
        viz_api_status: "error",
        viz_api_error:
          error instanceof Error ? error.message : "알 수 없는 오류",
      };

      setIntegratedData((prev) => {
        if (!prev) return prev;

        const updatedSections = prev.easy_sections.map((section) => {
          if (section.easy_section_id === sectionId) {
            return {
              ...section,
              viz_api_result: errorResult,
            };
          }
          return section;
        });

        return {
          ...prev,
          easy_sections: updatedSections,
        };
      });
    } finally {
      setLoadingVizApi((prev) => ({ ...prev, [key]: false }));
    }
  };

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };


  const createSectionElement = (section: EasySection, index: number) => {
    const level = section.easy_section_level ?? (section.easy_section_type === "subsection" ? 2 : 1);
    const isSubsection = level > 1;
    const sectionClass = isSubsection ? "paper-subsection" : "paper-section";
    const headerClass = isSubsection ? "subsection-header" : "section-header";
    const titleTag = level > 1 ? "h4" : "h2";
    const displayTitle = section.easy_section_title && section.easy_section_title.trim().length > 0
      ? section.easy_section_title
      : `(제목 없음)`;
    const sectionPlainText = (section.easy_paragraphs || [])
      .map(p => (p.easy_paragraph_text || "").replace(/<[^>]+>/g, "").trim())
      .join("\n");

    return (
      <div
        key={section.easy_section_id}
        className={sectionClass}
        id={section.easy_section_id}
      >
        <div className={headerClass}>
          {React.createElement(
            titleTag as any,
            { className: "section-title", style: { margin: 0 } },
            <>
              <span className="section-order">
                {section.easy_section_order}
              </span>
              <span style={{ marginLeft: 8 }}>{displayTitle}</span>
              <button
                onClick={() => callVizApi(section.easy_section_id, displayTitle, sectionPlainText)}
                disabled={!!loadingVizApi[section.easy_section_id]}
                className="vizapi-btn"
                style={{ marginLeft: 12, padding: '6px 10px', fontSize: 12 }}
                title="이 섹션을 Viz API로 시각화"
              >
                {loadingVizApi[section.easy_section_id] ? '생성중…' : '시각화 생성'}
              </button>
            </>
          )}
        </div>

        <div className="easy-content">
          {section.easy_paragraphs.map((paragraph) => (
            <div
              key={paragraph.easy_paragraph_id}
              className="paragraph-container"
            >
              <p
                className="paragraph-text"
                dangerouslySetInnerHTML={{
                  __html: formatText(paragraph.easy_paragraph_text),
                }}
              />

              {/* 문단에 삽입된 수식 렌더링 및 토글 설명 */}
              {(paragraph as any).paragraph_type === "math_equation" &&
                (paragraph as any).math_equation && (
                  <div className="equation-item">
                    <div className="equation-header">
                      <div className="equation-index">
                        {(paragraph as any).math_equation.equation_index}
                      </div>
                      <div className="equation-title">
                        {(paragraph as any).math_equation.equation_context || "수식"}
                      </div>
                      <button
                        className="equation-toggle"
                        onClick={() =>
                          toggleEquation((paragraph as any).math_equation.equation_id)
                        }
                      >
                        {activeEquation === (paragraph as any).math_equation.equation_id
                          ? "숨기기"
                          : "설명 보기"}
                      </button>
                    </div>

                    <div
                      className={`equation mathjax ${
                        activeEquation === (paragraph as any).math_equation.equation_id
                          ? "equation-active"
                          : ""
                      }`}
                      ref={mathJaxRef}
                      onClick={() =>
                        toggleEquation((paragraph as any).math_equation.equation_id)
                      }
                      style={{ cursor: "pointer" }}
                      title="수식을 클릭하면 설명을 볼 수 있습니다"
                    >
                      {`$$${normalizeLatex((paragraph as any).math_equation.equation_latex, (paragraph as any).math_equation.equation_env)}$$`}
                    </div>

                    {activeEquation === (paragraph as any).math_equation.equation_id && (
                      <div className="equation-explanation">
                        <div className="explanation-header">
                          <span className="explanation-icon">💡</span>
                          <span className="explanation-title">수식 설명</span>
                        </div>
                        <div
                          className="explanation-content"
                          dangerouslySetInnerHTML={{
                            __html: formatText(
                              (paragraph as any).math_equation.equation_explanation || ""
                            ),
                          }}
                        />
                        {(paragraph as any).math_equation.equation_variables &&
                          (paragraph as any).math_equation.equation_variables.length > 0 && (
                            <div className="equation-variables">
                              <div className="explanation-header">
                                <span className="explanation-icon">🔠</span>
                                <span className="explanation-title">변수 설명</span>
                              </div>
                              <ul>
                                {(paragraph as any).math_equation.equation_variables.map(
                                  (v: any, idx: number) => (
                                    <li key={idx}>{typeof v === "string" ? v : JSON.stringify(v)}</li>
                                  )
                                )}
                              </ul>
                            </div>
                          )}
                      </div>
                    )}
                  </div>
                )}

              {/* 시각화 항상 표시 (존재 시) */}
              {(paragraph as any).visualization?.image_path && (
                <div className="visualization-container">
                  <img
                    src={getImageSrc((paragraph as any).visualization.image_path)}
                    alt={section.easy_section_title}
                    className="viz-image"
                    onClick={() => openImage(getImageSrc((paragraph as any).visualization.image_path), section.easy_section_title)}
                    style={{ cursor: 'zoom-in' }}
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.style.display = "none";
                      const fallback = document.createElement("div");
                      fallback.className = "image-fallback";
                      fallback.textContent = "이미지를 불러올 수 없습니다";
                      fallback.style.cssText =
                        "padding: 40px; text-align: center; background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; color: #6c757d;";
                      target.parentNode?.appendChild(fallback);
                    }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Viz API 결과 표시 영역 */}
        {!isSubsection &&
          activeVizApi[section.easy_section_id] &&
          section.viz_api_result && (
            <div className="viz-api-container">
              <div className="viz-api-header">
                <h4>🎨 {section.viz_api_result.viz_api_title}</h4>
                {section.viz_api_result.viz_api_description && (
                  <p className="viz-api-description">
                    {section.viz_api_result.viz_api_description}
                  </p>
                )}
              </div>

              {section.viz_api_result.viz_api_status === "success" &&
                section.viz_api_result.viz_api_image_url && (
                  <div className="viz-api-image-container">
                    <img
                      src={section.viz_api_result.viz_api_image_url}
                      alt={section.viz_api_result.viz_api_title}
                      className="viz-api-image"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = "none";
                        const fallback = document.createElement("div");
                        fallback.className = "viz-api-fallback";
                        fallback.textContent = "이미지를 불러올 수 없습니다";
                        fallback.style.cssText =
                          "padding: 40px; text-align: center; background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; color: #6c757d;";
                        target.parentNode?.appendChild(fallback);
                      }}
                    />
                  </div>
                )}

              {section.viz_api_result.viz_api_status === "error" && (
                <div className="viz-api-error">
                  <div className="error-icon">⚠️</div>
                  <div className="error-message">
                    <strong>시각화 생성 실패</strong>
                    <p>
                      {section.viz_api_result.viz_api_error ||
                        "알 수 없는 오류가 발생했습니다."}
                    </p>
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
        {integratedData?.math_equations && integratedData.math_equations.length > 0 && (
          <div className="math-equations">
            <h3 style={{margin:'0 0 10px 0'}}>수식</h3>
            {integratedData.math_equations
              .filter((eq) => eq.math_equation_section_ref === section.easy_section_id)
              .map((equation) => (
                <div key={equation.math_equation_id} className="equation-item">
                  <div className="equation-header">
                    <span className="equation-index">{equation.math_equation_index?.replace(/[()]/g,'') || '?'}</span>
                    <span className="equation-title">{equation.math_equation_context || '수식'}</span>
                    <button
                      className="toggle-explanation"
                      onClick={() => toggleEquation(equation.math_equation_id)}
                      style={{ marginLeft: 'auto' }}
                    >
                      {activeEquation === equation.math_equation_id ? '숨기기' : '설명 보기'}
                    </button>
                  </div>
                  <div
                    className={`equation mathjax ${activeEquation === equation.math_equation_id ? 'equation-active' : ''}`}
                    ref={mathJaxRef}
                    onClick={() => toggleEquation(equation.math_equation_id)}
                    style={{ cursor: 'pointer', fontSize: '0.9em' }}
                    title="수식을 클릭하면 설명을 볼 수 있습니다"
                  >
                    {`$$${normalizeLatex(equation.math_equation_latex, equation.math_equation_env)}$$`}
                  </div>
                  {activeEquation === equation.math_equation_id && (
                    <div className="equation-explanation">
                      <div className="explanation-header">
                        <span className="explanation-icon">💡</span>
                        <span className="explanation-title">수식 설명</span>
                      </div>
                      <div
                        className="explanation-content"
                        dangerouslySetInnerHTML={{ __html: formatText(equation.math_equation_explanation) }}
                      />
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
    // **강조**는 굵게만, ==중요문장== 은 은은한 형광펜으로
    let html = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/==([^=]+)==/g, '<mark style="background:#fff3b0; color:inherit;">$1</mark>');
    return html;
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
      body { font-family: 'Pretendard','Spoqa Han Sans Neo','Noto Sans KR','Apple SD Gothic Neo','Inter','Segoe UI',system-ui,-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif; line-height: 1.8; color: #222; background-color: #f8f9fa; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
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
      .section-title { font-size: 1.75em; color: #1f2937; margin-bottom: 10px; font-weight: 700; letter-spacing: .2px; }
      .section-order { display: inline-block; background: #f59e0b; color: white; width: 30px; height: 30px; border-radius: 50%; text-align: center; line-height: 30px; font-weight: bold; margin-right: 15px; vertical-align: middle; }
      .easy-content { margin-bottom: 30px; padding: 24px; background: #fff8e1; border-radius: 10px; border-left: 4px solid #f59e0b; }
      .easy-content p { font-size: 1.03em; line-height: 1.95; color: #222; letter-spacing: 0.1px; }
      .easy-content p + p { margin-top: 10px; }
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
                  <strong>Easy 모델:</strong>{" "}
                  {integratedData.model_errors.easy_model_error}
                </div>
              )}
              {integratedData.model_errors.math_model_error && (
                <div className="model-error-item">
                  <strong>Math 모델:</strong>{" "}
                  {integratedData.model_errors.math_model_error}
                </div>
              )}
              {integratedData.model_errors.viz_api_error && (
                <div className="model-error-item">
                  <strong>Viz API:</strong>{" "}
                  {integratedData.model_errors.viz_api_error}
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
    <div className={`container${dark ? ' dark-mode' : ''}`} style={{ fontFamily: "'Pretendard','Spoqa Han Sans Neo','Noto Sans KR','Apple SD Gothic Neo','Inter','Segoe UI',system-ui,-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif" }}>
      {dark && (
        <style>{`
          .dark-mode { background-color: #0b1220; color: #e5e7eb; }
          .dark-mode .paper-header { background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%); }
          .dark-mode .section-title { color: #e5e7eb; }
          .dark-mode .paper-section, .dark-mode .paper-subsection { background-color: #0f172a; border-color: #1f2a44; }
          .dark-mode .easy-content { background: #0c1222; border-left-color: #38bdf8; }
          .dark-mode .equation { background: #0c1222; border-color: #1f2a44; }
          .dark-mode .paper-footer { background: #0b1220; color: #94a3b8; }
          .dark-mode a { color: #93c5fd; }
          .dark-mode .table-of-contents { background: #0f172a; border-color: #1f2a44; }
          .dark-mode .toc-search { background:#0b1220; color:#e5e7eb; border:1px solid #1f2a44; }
          .dark-mode .download-btn { background:#111827; color:#e5e7eb; border:1px solid #374151; }
          .dark-mode .viz-image { box-shadow: 0 6px 18px rgba(0,0,0,.5); }
        `}</style>
      )}
      <main className="result-main">
        {imageModal.open && (
          <div className="image-modal" onClick={closeImage} style={{position:'fixed', inset:0, background:'rgba(0,0,0,0.6)', display:'flex', alignItems:'center', justifyContent:'center', zIndex:9999}}>
            <img src={imageModal.src} alt={imageModal.alt || ''} style={{maxWidth:'90vw', maxHeight:'90vh', borderRadius:8, boxShadow:'0 10px 30px rgba(0,0,0,0.4)'}} />
          </div>
        )}

        {/* 좌측 세로 목차 */}
        <aside className="sidebar">
          <div className="table-of-contents">
            <h3 className="toc-title">목차</h3>
            <ul className="toc-sections">
              {groups.map(({ parent }) => (
                <li key={parent.easy_section_id}>
                  <a href={`#${parent.easy_section_id}`}
                     className={`toc-link ellipsis-one ${activeTocId===parent.easy_section_id?'active':''}`}
                     title={parent.easy_section_title}>
                    {parent.easy_section_title}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </aside>

        {/* 우측 본문(히어로 + 섹션들) */}
        <section className="content">
          {/* 히어로 카드(제목) */}
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
          </div>
        </header>
            {groupSections(integratedData.easy_sections).map(({ parent, children }) => (
              <article key={parent.easy_section_id} id={parent.easy_section_id} className="paper-section-card">
                <header className="section-header"><h2>{parent.easy_section_title}</h2></header>
                
                {/* 섹션 레벨 Figures */}
                {(parent as any).figures?.map((figure: FigureMeta, idx: number) => (
                  <FigureView 
                    key={`section-fig-${idx}`}
                    figure={figure} 
                    openImage={openImage} 
                    className="section-figure"
                  />
                ))}
                
                {ensureParagraphs(parent).map(p => (
                  <ParagraphView key={p.easy_paragraph_id} p={p} sectionId={parent.easy_section_id} openImage={openImage} getImageSrc={getImageSrc}/>
                ))}
                {children.map(sub => (
                  <section key={sub.easy_section_id} id={sub.easy_section_id} className="paper-subsection">
                    <header className="subsection-header"><h3>{sub.easy_section_title}</h3></header>
                    
                    {/* 서브섹션 레벨 Figures */}
                    {(sub as any).figures?.map((figure: FigureMeta, idx: number) => (
                      <FigureView 
                        key={`subsection-fig-${idx}`}
                        figure={figure} 
                        openImage={openImage} 
                        className="subsection-figure"
                      />
                    ))}
                    
                    {ensureParagraphs(sub).map(p => (
                      <ParagraphView key={p.easy_paragraph_id} p={p} sectionId={sub.easy_section_id} openImage={openImage} getImageSrc={getImageSrc}/>
                    ))}
                  </section>
                ))}
              </article>
            ))}
          </section>

        {/* 오른쪽 패널 없음 — 수식은 문단 인라인만 */}
      </main>
    </div>
  );
};

export default Result;
