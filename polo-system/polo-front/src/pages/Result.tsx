import React, { useState, useEffect, useRef, useMemo } from "react";
import { useLocation } from "react-router-dom";
import { marked } from "marked";
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import "./Result.css";
import type { FigureMeta, EasyParagraph as EasyParagraphType, IntegratedData as IntegratedDataType } from "../types";
import { FIGURE_MAP, FIGURE_CAPTION } from "../figureMapTemplate";

declare global {
  interface Window { MathJax?: any; __MATHJAX_LOADING__?: Promise<void>; }
}
const MJX_SRC = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js";
const MJX_ID  = "mjx-tex-svg";

// MathJax 설정 + 스크립트 로딩 (싱글턴)
function injectMathJaxConfigOnce() {
  if (window.MathJax || window.__MATHJAX_LOADING__) return;
  const cfg = document.createElement("script");
  cfg.type = "text/javascript";
  cfg.text = `
    window.MathJax = {
      loader: { load: ['[tex]/ams','[tex]/mathtools','[tex]/physics'] },
      startup: { typeset: false },
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
        packages: { '[+]': ['ams','mathtools','physics'] },
        processEscapes: true,
        macros: { scriptsize: '' }  // \scriptsize 무시
      },
      svg: { fontCache: 'global', scale: 1 }
    };
  `;
  document.head.appendChild(cfg);
}
async function ensureMathJax(): Promise<void> {
  if (window.MathJax?.typesetPromise) return;
  if (window.__MATHJAX_LOADING__) return window.__MATHJAX_LOADING__;
  injectMathJaxConfigOnce();

  let s = document.getElementById(MJX_ID) as HTMLScriptElement | null;
  if (!s) {
    s = document.createElement("script");
    s.id = MJX_ID;
    s.src = MJX_SRC;
    s.defer = true;
    document.head.appendChild(s);
  }
  window.__MATHJAX_LOADING__ = new Promise<void>((resolve, reject) => {
    if ((s as any)._loaded) return resolve();
    s!.addEventListener("load", () => { (s as any)._loaded = true; resolve(); });
    s!.addEventListener("error", reject);
  });
  return window.__MATHJAX_LOADING__;
}

// ✔ TSX children을 그대로 렌더하고 해당 영역만 typeset 해주는 블록
function MathBlock(props: { children: string; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      await ensureMathJax();
      if (!mounted || !ref.current) return;
      const el = ref.current;
      // 렌더 반영 후 타입셋
      requestAnimationFrame(() => {
        setTimeout(() => {
          window.MathJax?.typesetClear?.([el]);
          window.MathJax?.typesetPromise?.([el]).catch(console.error);
        }, 0);
      });
    })();
    return () => { mounted = false; };
  }, [props.children]);

  return (
    <div ref={ref} className={props.className}>
      {props.children}
    </div>
  );
}

// 특정 문장 하이라이트 함수
function highlightSpecificSentences(text: string): string {
  if (!text) return text;
  
  let highlighted = text;
  
  // 지정된 두 문장에만 형광펜 적용
  const targetSentences = [
    "우리는 object detection(이미지 속에서 물체의 위치와 종류를 찾아내는 작업)을 이미지 픽셀에서 bounding box(물체를 둘러싸는 네모 상자) 좌표와 class probability(각 물체가 특정 클래스일 확률)까지의 단일 regression(연속적인 값을 예측하는 문제) 문제로 재구성합니다.",
    "현재 탐지 시스템들은 분류기를 재활용하여 탐지를 수행합니다. 물체를 탐지하기 위해 이런 시스템들은 해당 물체에 대한 분류기를 가져와서 테스트 이미지의 다양한 위치와 크기에서 평가합니다."
  ];
  
  // 각 문장에 형광펜 적용
  targetSentences.forEach(sentence => {
    if (highlighted.includes(sentence)) {
      highlighted = highlighted.replace(sentence, `<mark class="keyword-highlight">${sentence}</mark>`);
    }
  });
  
  return highlighted;
}


// [ADD] Figures sidecar optional loader (간소화)
type FigureItem = {
  order: number;
  image_path: string;
};

async function loadFigureQueue(): Promise<FigureItem[]> {
  // 1차: 메인 서버(/static)
  try {
    const r = await fetch('/static/viz/figures_map.json', { cache: 'no-store' });
    if (r.ok) {
      const ct = r.headers.get('content-type') || '';
      if (ct.includes('application/json')) {
        const data = await r.json();
        console.log('✅ [FIG] 메인 서버에서 로드:', data.figures?.length || 0);
        return data.figures ?? [];
      } else {
        console.warn('⚠️ [FIG] 메인 서버가 JSON이 아니라 다른 걸 반환:', ct);
      }
    }
  } catch (e) {
    console.warn('⚠️ [FIG] 메인 서버 실패:', e);
  }

  // 2차: 사이드카(있을 때만)
  try {
    const r2 = await fetch('http://localhost:8020/static/viz/figures_map.json', { cache: 'no-store' });
    if (r2.ok) {
      const ct2 = r2.headers.get('content-type') || '';
      if (ct2.includes('application/json')) {
        const data = await r2.json();
        console.log('✅ [FIG] 사이드카에서 로드:', data.figures?.length || 0);
        return data.figures ?? [];
      } else {
        console.warn('⚠️ [FIG] 사이드카가 JSON이 아님:', ct2);
      }
    }
  } catch (e2) {
    console.warn('⚠️ [FIG] 사이드카 실패:', e2);
  }

  console.info('ℹ️ [FIG] figures_map.json 없음 - 기존 렌더링 유지');
  return [];
}

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
  
  return marked.parse(processed) as string;
};

// 역슬래시 이스케이프 복원 함수
const unescapeOnce = (s: string): string => {
  if (!s) return s;
  // "\\(" -> "\(" , "\\leq" -> "\leq", "\\phi" -> "\phi" 등
  return s.replace(/\\\\/g, '\\');
};

// LaTeX 백슬래시 정규화 함수
const normalizeTexBackslashes = (s: string): string => {
  if (!s) return s;
  // 이중 백슬래시를 단일 백슬래시로 변환
  return s.replace(/\\\\/g, '\\');
};

// 설명 텍스트 안의 LaTeX 수식을 자동으로 $...$로 감싸는 함수
const wrapInlineMath = (input: string): string => {
  if (!input || !input.includes("\\")) return input; // 빠른 패스
  
  // 이미 감싼 수식은 보호
  const inlineAlready = [/(\$(?:\\.|[^$])+\$)/g, /(\\\((?:\\.|[^)])+\\\))/g];
  const blockAlready = [/(\$\$(?:\\.|[^$])+\$\$)/g, /(\\\[(?:\\.|[^\]])+\\\])/g];

  // 보호 토큰으로 임시 치환
  const buckets: string[] = [];
  function protect(text: string, re: RegExp) {
    return text.replace(re, (m) => {
      buckets.push(m);
      return `__MATH_BUCKET_${buckets.length - 1}__`;
    });
  }
  
  let s = input;
  [...blockAlready, ...inlineAlready].forEach((re) => (s = protect(s, re)));

  // ( \large{...} ) 같은 괄호 래핑 -> $...$
  s = s.replace(/\((\\[A-Za-z][^()]*)\)/g, (_m, g1) => `$${g1.trim()}$`);

  // LaTeX 토막 자동 래핑 (더 정확한 패턴)
  // \Pr(\textrm{Object}), \mathbb{1}_{ij}^{\text{obj}}, \sqrt{w_i} 등을 감지
  s = s.replace(
    /\\[A-Za-z]+(?:[\{\}\[\]\(\)]|\\[A-Za-z]+|[_^]|[0-9A-Za-z,+\-*/.:= ]){0,120}/g,
    (frag) => {
      const f = frag.trim();
      // 혹시 이미 $…$ 형태면 그대로
      if (f.startsWith("$") && f.endsWith("$")) return f;
      return `$${f}$`;
    }
  );

  // 보호 복원
  return s.replace(/__MATH_BUCKET_(\d+)__/g, (_m, i) => buckets[Number(i)]);
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
  
  // \large{\mathbb{1}} 같은 중첩 매크로 처리
  s = s.replace(/\\large\{\\mathbb\{1\}\}/g, '\\mathbf{1}');
  s = s.replace(/\\large\{\\mathbb\{([^}]+)\}\}/g, '\\mathbb{$1}');
  s = s.replace(/\\large\{\\mathbf\{([^}]+)\}\}/g, '\\mathbf{$1}');
  s = s.replace(/\\large\{\\text\{([^}]+)\}\}/g, '\\text{$1}');
  
  // \large 매크로가 문제를 일으키는 경우 제거
  s = s.replace(/\\large\{([^}]+)\}/g, (match, content) => {
    if (content.includes('\\mathbb') || content.includes('\\mathbf') || content.includes('\\text')) {
      return content; // \large 제거하고 내용만 유지
    }
    return match;
  });
  
  return s;
};

// build_paper_context 함수에서 추출한 매크로 처리 로직 - 제거 (문제 발생)
// const detectMacroUsage = (latex: string): string[] => {
//   const patterns = [
//     /\\mathlarger\{([^}]+)\}/g,
//     /\\mathbbm\{([^}]+)\}/g,
//     /\\Pr\(([^)]+)\)/g,
//     /\\textrm\{([^}]+)\}/g,
//     /\\lambda_\\textbf\{([^}]+)\}/g,
//     /\\lambda_\\textrm\{([^}]+)\}/g
//   ];
//   
//   const foundMacros: string[] = [];
//   patterns.forEach(pattern => {
//     const matches = latex.match(pattern);
//     if (matches) {
//       foundMacros.push(...matches);
//     }
//   });
//   
//   return foundMacros;
// };

// 매크로 정의 맵 (build_paper_context에서 추출) - 제거
// const MACRO_DEFINITIONS = {
//   "mathlarger": "\\mathlarger{#1}",
//   "mathbbm": "\\mathbbm{1}",
//   "Pr": "\\Pr",
//   "textrm": "\\textrm{#1}",
//   "textbf": "\\textbf{#1}",
//   "lambda": "\\lambda"
// };

// 향상된 매크로 처리 함수 - 제거 (문제 발생)
// const processMacros = (latex: string): string => {
//   if (!latex) return latex;
//   
//   let processed = latex;
//   
//   // 1. 매크로 사용 패턴 감지
//   const usedMacros = detectMacroUsage(latex);
//   console.log('🔍 감지된 매크로:', usedMacros);
//   
//   // 2. 각 매크로별 처리
//   usedMacros.forEach(macro => {
//     // \mathlarger 처리
//     if (macro.includes('\\mathlarger')) {
//       processed = processed.replace(/\\mathlarger\{([^}]+)\}/g, '\\large{$1}');
//     }
//     
//     // \mathbbm 처리
//     if (macro.includes('\\mathbbm')) {
//       processed = processed.replace(/\\mathbbm\{1\}/g, '\\mathbf{1}');
//       processed = processed.replace(/\\mathbbm\{([^}]+)\}/g, '\\mathbb{$1}');
//     }
//     
//     // \Pr 처리
//     if (macro.includes('\\Pr')) {
//       processed = processed.replace(/\\Pr\(/g, '\\Pr(');
//     }
//     
//     // \textrm 처리
//     if (macro.includes('\\textrm')) {
//       processed = processed.replace(/\\textrm\{([^}]+)\}/g, '\\text{$1}');
//     }
//     
//     // \lambda_\\textbf 처리
//     if (macro.includes('\\lambda_\\textbf')) {
//       processed = processed.replace(/\\lambda_\\textbf\{([^}]+)\}/g, '\\lambda_{\\textbf{$1}}');
//     }
//     
//     // \lambda_\\textrm 처리
//     if (macro.includes('\\lambda_\\textrm')) {
//       processed = processed.replace(/\\lambda_\\textrm\{([^}]+)\}/g, '\\lambda_{\\text{$1}}');
//     }
//   });
//   
//   return processed;
// };

// LaTeX 수식 전처리 파이프라인 (이스케이프 복원 + 매크로 수정)
const preprocessLatex = (s: string): string => {
  if (!s) return s;
  
  // 1. 역슬래시 이스케이프 복원
  let processed = unescapeOnce(s);
  
  // 2. 기존 매크로 수정만 적용 (안전한 방식)
  processed = fixLatexMacros(processed);
  
  // 3. 추가 LaTeX 매크로 처리 (MathJax 호환성)
  // \large{\mathbb{1}} 같은 중첩 매크로 처리
  processed = processed.replace(/\\large\{\\mathbb\{1\}\}/g, '\\mathbf{1}');
  processed = processed.replace(/\\large\{\\mathbb\{([^}]+)\}\}/g, '\\mathbb{$1}');
  processed = processed.replace(/\\large\{\\mathbf\{([^}]+)\}\}/g, '\\mathbf{$1}');
  processed = processed.replace(/\\large\{\\text\{([^}]+)\}\}/g, '\\text{$1}');
  
  // \large 매크로가 문제를 일으키는 경우 제거
  processed = processed.replace(/\\large\{([^}]+)\}/g, (match, content) => {
    if (content.includes('\\mathbb') || content.includes('\\mathbf') || content.includes('\\text')) {
      return content; // \large 제거하고 내용만 유지
    }
    return match;
  });
  
  return processed;
};

// normalize_for_mathjax 함수 (math/app.py에서 추출) - 제거
// const normalizeForMathJax = (eq: string): string => {
//   if (!eq) return eq;
//   
//   let s = eq;
//   
//   // \mathbbm{1} -> \mathbf{1} (지표함수용)
//   s = s.replace(/\\mathbbm\s*\{\s*1\s*\}/g, '\\mathbf{1}');
//   
//   // \mathbbm{...} -> \mathbb{...} (다른 문자들)
//   s = s.replace(/\\mathbbm\s*\{/g, '\\mathbb{');
//   
//   return s;
// };

// [ADD] LaTeX 수식 변환 함수 (컴포넌트 외부)
const convertLatexToMathJax = (latexText: string): string => {
  if (!latexText) return '';
  
  let converted = latexText;
  
  // 1. 깨진 LaTeX 명령어들 수정
  const latexFixes = [
    // 백슬래시 누락 수정
    { from: /Pr\(/g, to: '\\Pr(' },
    { from: /textem\{/g, to: '\\text{' },
    { from: /mathbbm/g, to: '\\mathbb' },
    { from: /mathlarger/g, to: '\\large' },
    { from: /boxed\{/g, to: '\\boxed{' },
    { from: /phi\(/g, to: '\\phi(' },
    { from: /hat/g, to: '\\hat' },
    { from: /sqrt/g, to: '\\sqrt' },
    
    // 사용자 언급 특정 문제들 해결
    { from: /Pr\(Class_i \| Object\)/g, to: '\\Pr(\\text{Class}_i | \\text{Object})' },
    { from: /\\Pr\(\\textem\{Class\}_i \|/g, to: '\\Pr(\\text{Class}_i |' },
    { from: /\\boxed\{\\Pr\(\\textrm\{Object\}\)\}/g, to: '\\boxed{\\Pr(\\text{Object})}' },
    { from: /\\mathlarger/g, to: '\\large' },
    { from: /\\mathbbm/g, to: '\\mathbb' },
    { from: /\\text\{/g, to: '\\text{' },
    { from: /\\hat\{/g, to: '\\hat{' },
    { from: /\\sqrt\{/g, to: '\\sqrt{' },
    { from: /\\boxed\{1\}/g, to: '\\boxed{1}' },
    { from: /\\phi\(x\)/g, to: '\\phi(x)' },
    
    // 중괄호 누락 수정
    { from: /\\Pr\(([^)]+)\)/g, to: '\\Pr($1)' },
    { from: /\\text\{([^}]+)\}/g, to: '\\text{$1}' },
    { from: /\\boxed\{([^}]+)\}/g, to: '\\boxed{$1}' },
    { from: /\\phi\(([^)]+)\)/g, to: '\\phi($1)' },
    { from: /\\sqrt\{([^}]+)\}/g, to: '\\sqrt{$1}' },
    
    // 특수 문자 처리
    { from: /λ/g, to: '\\lambda' },
    { from: /α/g, to: '\\alpha' },
    { from: /β/g, to: '\\beta' },
    { from: /γ/g, to: '\\gamma' },
    { from: /δ/g, to: '\\delta' },
    { from: /ε/g, to: '\\epsilon' },
    { from: /θ/g, to: '\\theta' },
    { from: /π/g, to: '\\pi' },
    { from: /σ/g, to: '\\sigma' },
    { from: /τ/g, to: '\\tau' },
    { from: /φ/g, to: '\\phi' },
    { from: /ψ/g, to: '\\psi' },
    { from: /ω/g, to: '\\omega' },
    
    // 수학 연산자
    { from: /≤/g, to: '\\leq' },
    { from: /≥/g, to: '\\geq' },
    { from: /≠/g, to: '\\neq' },
    { from: /≈/g, to: '\\approx' },
    { from: /∞/g, to: '\\infty' },
    { from: /∑/g, to: '\\sum' },
    { from: /∏/g, to: '\\prod' },
    { from: /∫/g, to: '\\int' },
    { from: /∂/g, to: '\\partial' },
    { from: /∇/g, to: '\\nabla' },
    
    // 집합 기호
    { from: /∈/g, to: '\\in' },
    { from: /∉/g, to: '\\notin' },
    { from: /⊂/g, to: '\\subset' },
    { from: /⊃/g, to: '\\supset' },
    { from: /∪/g, to: '\\cup' },
    { from: /∩/g, to: '\\cap' },
    { from: /∅/g, to: '\\emptyset' },
    
    // 논리 연산자
    { from: /∧/g, to: '\\land' },
    { from: /∨/g, to: '\\lor' },
    { from: /¬/g, to: '\\neg' },
    { from: /→/g, to: '\\rightarrow' },
    { from: /←/g, to: '\\leftarrow' },
    { from: /↔/g, to: '\\leftrightarrow' },
    { from: /∀/g, to: '\\forall' },
    { from: /∃/g, to: '\\exists' },
  ];
  
  // 변환 적용
  latexFixes.forEach(fix => {
    converted = converted.replace(fix.from, fix.to);
  });
  
  // 2. MathJax 래퍼 추가 (이미 있는지 확인)
  if (!converted.includes('$$') && !converted.includes('\\(') && !converted.includes('\\[')) {
    // 수식이 여러 줄에 걸쳐 있거나 복잡한 경우
    if (converted.includes('\\') || converted.includes('{') || converted.includes('}')) {
      converted = `$$${converted}$$`;
    }
  }
  
  return converted;
};

// 설명 텍스트 정리(접두 제거 + 군더더기 제거 + LaTeX 변환)
const sanitizeExplain = (t?:string) => {
  if (!t) return "";
  
  // 1. 기본 정리
  let cleaned = t
    .replace(/^\s*(조수|assistant)\s*[:：\-]?\s*/i, "")
    .replace(/^\s*(조수|assistant)\s*[:：\-]?\s*/gmi, "")
    .replace(/\[?\s*수학\s*\d+\s*\]?/g, "")   // [수학0] 등 제거
    .replace(/^\s*보조\s*:?/gmi, "")          // '보조' 접두 제거
    .trim();
  
  // 2. LaTeX 변환 적용
  return convertLatexToMathJax(cleaned);
};

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

// MathJax 준비 보장 후 typeset (NaN 에러 방지)
const typesetNodes = async (nodes: Element[]) => {
  const w:any = window as any;
  if (!w.MathJax) return;
  
  try {
    // MathJax v3는 startup.promise 대기 후 typesetPromise 권장
    if (w.MathJax.startup?.promise) { 
      await w.MathJax.startup.promise; 
    }
    
    // NaN 에러 방지를 위한 안전한 typeset
    if (w.MathJax.typesetPromise) {
      // 각 노드를 개별적으로 처리하여 NaN 에러 방지
      for (const node of nodes) {
        try {
          // 노드가 유효한지 확인
          if (node && node.nodeType === Node.ELEMENT_NODE) {
            await w.MathJax.typesetPromise([node]);
          }
        } catch (error) {
          console.warn('MathJax typeset error for node:', error);
          // 에러가 발생한 노드는 무시하고 계속 진행
        }
      }
    }
  } catch (error) {
    console.warn('MathJax typeset error:', error);
  }
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

// Abstract 섹션을 제외한 섹션 그룹 필터링 및 이미지 인덱스 조정
function filterNonAbstractSections(groups: { parent: EasySection; children: EasySection[] }[]) {
  return groups.filter(group => {
    const title = group.parent.easy_section_title?.toLowerCase() || '';
    return !title.includes('abstract');
  });
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
  // [ADD] 외부 API 이미지 팝업 상태
  const [externalImagePopup, setExternalImagePopup] = useState<{
    isOpen: boolean;
    imageUrl: string;
    sectionTitle: string;
  }>({
    isOpen: false,
    imageUrl: "",
    sectionTitle: ""
  });
  
  // [ADD] Figure 사이드카 상태 (옵션)
  const [figQueue, setFigQueue] = useState<FigureItem[]>([]);
  
  // [ADD] Figure 사이드카 로드
  useEffect(() => { 
    loadFigureQueue().then(setFigQueue); 
  }, []);

  // [ADD] LaTeX 변환 테스트 함수 (디버깅용)
  const testLatexConversion = (testText: string) => {
    console.log('🧪 [LaTeX 변환 테스트]');
    console.log('입력:', testText);
    const result = convertLatexToMathJax(testText);
    console.log('출력:', result);
    return result;
  };
  
  // [ADD] Figure 큐 팝 함수
  const popFig = useMemo(() => { 
    let i = 0; 
    return () => figQueue[i++] as FigureItem | undefined; 
  }, [figQueue]);
  
  // [ADD] [Figure] 토큰 주입 함수 (강화됨)
  function injectFigures(text: string): (string | FigureItem)[] {
    if (!text || figQueue.length === 0) return [text];
    
    // [Figure] 한 개씩만 치환 (없으면 그대로)
    const token = /\[Figure[^\]]*\]/i;
    if (!token.test(text)) return [text];
    
    const parts: (string | FigureItem)[] = [];
    let rest = text;
    
    while (true) {
      const m = rest.match(token);
      if (!m) { 
        if (rest) parts.push(rest); 
        break; 
      }
      
      // 토큰 이전 텍스트
      if (m.index! > 0) {
        parts.push(rest.slice(0, m.index!));
      }
      
      // Figure 또는 원본 토큰
      const fig = popFig();
      if (fig) {
        parts.push(fig);
        console.log(`🔄 [FIG] 토큰 교체: ${m[0]} → Figure ${fig.order}`);
      } else {
        parts.push(m[0]); // Figure 없으면 원문 유지
        console.warn(`⚠️ [FIG] Figure 부족: ${m[0]}`);
      }
      
      rest = rest.slice(m.index! + m[0].length);
    }
    
    return parts;
  }

  // [ADD] 남은 Figure들을 가져오는 함수
  function getRemainingFigures(): FigureItem[] {
    const remaining: FigureItem[] = [];
    let fig;
    while ((fig = popFig())) {
      remaining.push(fig);
    }
    return remaining;
  }
  
  const [imageModal, setImageModal] = useState<{ open: boolean; src: string; alt?: string }>({ open: false, src: "" });
  const openImage = (src: string, alt?: string) => setImageModal({ open: true, src, alt });
  const closeImage = () => setImageModal({ open: false, src: "" });
  const [dark, setDark] = useState<boolean>(false);
  useEffect(() => {
    // body에 다크모드 클래스 토글 → CSS가 전체 적용
    const cls = document.documentElement.classList;
    if (dark) cls.add("dark-mode"); else cls.remove("dark-mode");
  }, [dark]);

  // MathJax는 이제 싱글턴 로더로 관리됨 (ensureMathJax 함수 사용)

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
      // 안전한 typeset 처리
      typesetNodes(nodes);
    }
    
    // KaTeX 인라인 수식 처리
    const katexElements = document.querySelectorAll('.katex-inline');
    katexElements.forEach(element => {
      try {
        const math = element.textContent;
        if (math) {
          import('katex').then(katex => {
            element.innerHTML = katex.renderToString(math, { displayMode: false });
          });
        }
      } catch (error) {
        console.warn('KaTeX 렌더링 실패:', error);
      }
    });
  }, [integratedData, activeEquation]);

  // 수식 설명이 표시될 때 MathJax 재타입셋팅
  useEffect(() => {
    if (activeEquation) {
      // 약간의 지연을 두고 MathJax 실행 (DOM 업데이트 후)
      const timeoutId = setTimeout(() => {
        const win = window as any;
        if (win?.MathJax?.typesetPromise) {
          const explanationNodes = Array.from(document.querySelectorAll('.explanation-content.mathjax'));
          if (explanationNodes.length > 0) {
            win.MathJax.typesetPromise(explanationNodes).catch(console.warn);
          }
        }
      }, 100);
      
      return () => clearTimeout(timeoutId);
    }
  }, [activeEquation]);


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
          <BlockMath math={latex} />
        </div>
        {open && !!explain && (
          <div ref={expRef} className="equation-explain mathjax">
            <div dangerouslySetInnerHTML={{ 
              __html: renderMarkdown(explain)
                .replace(/\\Pr\(\\text\{Object\}\)/g, '\\Pr(\\text{Object})')
                .replace(/\\text\{IOU\}_\{\\text\{pred\}\}\^\{\\text\{truth\}\}/g, '\\text{IOU}_{\\text{pred}}^{\\text{truth}}')
                .replace(/\\rightarrow/g, '\\rightarrow')
                .replace(/\$([^$]+)\$/g, (match: string, math: string) => {
                  try {
                    return `<span class="katex-inline">${math}</span>`;
                  } catch {
                    return match;
                  }
                })
            }} />
          </div>
        )}
      </div>
    );
  };


  // 기존 Figure 컴포넌트 (FigureMeta 타입용)
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

  // 문단 렌더러 (텍스트/수식/시각화/사이드카 Figure 인라인)
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

    // [ADD] 하드코딩 매핑 기반 Figure 찾기 (최종 간소화)
    const figIdx = FIGURE_MAP[p.easy_paragraph_id];
    const fig = figIdx ? figQueue.find(f => f.order === figIdx) : undefined;
    
    // [Figure] 토큰 제거 (하드코딩 매핑 사용 시)
    const cleanText = (p.easy_paragraph_text || '').replace(/\[Figure[^\]]*\]/gi, '').trim();
    
    // 일반 텍스트 문단 + 시각화(있으면)
    const hasViz = !!p.visualization?.image_path;
    const hasExistingFigure = !!p.figure; // 기존 figure 필드 (통합 JSON 방식)
    
    return (
      <div className="paper-paragraph">
        {/* [ADD] 하드코딩 매핑 기반 텍스트 + Figure */}
        <div className="easy-md mathjax">
          <span dangerouslySetInnerHTML={{ __html: formatText(cleanText) }} />
        </div>
        
        {/* 하드코딩 매핑된 Figure (캡션 하드코딩) */}
        {fig && figIdx && (
          <figure className="my-3 mapped-figure">
            <img
              src={fig.image_path}
              alt={FIGURE_CAPTION[figIdx] ?? ''}
              onClick={() => openImage(fig.image_path, FIGURE_CAPTION[figIdx] ?? '')}
              className="cursor-zoom-in"
            />
            <figcaption className="text-sm text-gray-500 mt-1">
              {FIGURE_CAPTION[figIdx]}
            </figcaption>
          </figure>
        )}
        
        {/* 기존 Figure (통합 JSON 방식) - 호환성 유지 */}
        {hasExistingFigure && (
          <FigureView 
            figure={p.figure} 
            openImage={openImage} 
            className="paragraph-figure legacy-figure"
          />
        )}
        
        {/* 자동 생성 시각화 (Figure가 없을 때만) */}
        {hasViz && !hasExistingFigure && (
          <figure className="figure-card viz-figure" onClick={() => openImage(getImageSrc(p.visualization.image_path), "visualization")}>
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


  // [ADD] 외부 API 이미지 팝업 열기 함수
  const openExternalImage = (sectionIdx: number, sectionTitle: string) => {
    // URL에서 paper_id 직접 추출 (가장 확실한 방법)
    const pathParts = window.location.pathname.split("/");
    const paperId = pathParts[pathParts.length - 1];
    
    // paper_id가 올바른지 확인 (doc_로 시작하는지)
    if (!paperId || paperId === 'yolo_v1_analysis') {
      console.error('❌ [이미지] 잘못된 paper_id:', paperId);
      console.log('🔍 [이미지] 현재 URL:', window.location.pathname);
      console.log('🔍 [이미지] pathParts:', pathParts);
      return;
    }
    
    // 외부 API 이미지 경로 구성 (절대 URL 사용)
    // 서버에서 /outputs → server/data/outputs 매핑
    const imageUrl = `http://localhost:8000/outputs/${paperId}/api/${sectionIdx}.png`;
    // 디버깅: paper_id 값 확인
    console.log('🔍 [이미지 경로] 최종 경로:', {
      paperId: paperId,
      sectionIdx: sectionIdx,
      sectionTitle: sectionTitle,
      imageUrl: imageUrl,
      currentUrl: window.location.pathname,
      expectedServerPath: `C:\\POLO\\POLO\\polo-system\\server\\data\\outputs\\${paperId}\\api\\${sectionIdx}.png`
    });
    
    setExternalImagePopup({
      isOpen: true,
      imageUrl: imageUrl,
      sectionTitle: sectionTitle
    });
  };

  // [ADD] 외부 API 이미지 팝업 닫기 함수
  const closeExternalImage = () => {
    setExternalImagePopup({
      isOpen: false,
      imageUrl: "",
      sectionTitle: ""
    });
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

    // Abstract 섹션인지 확인 (대소문자 구분 없이)
    const isAbstractSection = displayTitle.toLowerCase().includes('abstract');

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
              {/* Abstract 섹션은 시각화 버튼 제거 */}
              {!isAbstractSection && (
                <button
                  onClick={() => callVizApi(section.easy_section_id, displayTitle, sectionPlainText)}
                  disabled={!!loadingVizApi[section.easy_section_id]}
                  className="vizapi-btn"
                  style={{ marginLeft: 12, padding: '6px 10px', fontSize: 12 }}
                  title="이 섹션을 Viz API로 시각화"
                >
                  {loadingVizApi[section.easy_section_id] ? '생성중…' : '시각화 생성'}
                </button>
              )}
            </>
          )}
        </div>

        <div className="easy-content">
          {section.easy_paragraphs.map((paragraph) => (
            <div
              key={paragraph.easy_paragraph_id}
              className="paragraph-container"
            >
              <div className="paragraph-text mathjax">
                <div dangerouslySetInnerHTML={{ __html: formatText(paragraph.easy_paragraph_text) }} />
              </div>

              {/* 문단에 삽입된 수식 렌더링 및 토글 설명 */}
              {(paragraph as any).paragraph_type === "math_equation" &&
                (paragraph as any).math_equation && (
                  <div className="equation-item">
                    <div className="equation-header">
                      <div className="equation-index">
                        {(paragraph as any).math_equation.math_equation_index}
                      </div>
                      <div className="equation-title">
                        {(paragraph as any).math_equation.math_equation_context || "수식"}
                      </div>
                      <button
                        className="equation-toggle"
                        onClick={() =>
                          toggleEquation((paragraph as any).math_equation.math_equation_id)
                        }
                      >
                        {activeEquation === (paragraph as any).math_equation.math_equation_id
                          ? "숨기기"
                          : "설명 보기"}
                      </button>
                    </div>

                    <div
                      className={`equation mathjax ${
                        activeEquation === (paragraph as any).math_equation.math_equation_id
                          ? "equation-active"
                          : ""
                      }`}
                      ref={mathJaxRef}
                      onClick={() =>
                        toggleEquation((paragraph as any).math_equation.math_equation_id)
                      }
                      style={{ cursor: "pointer" }}
                      title="수식을 클릭하면 설명을 볼 수 있습니다"
                    >
                      <BlockMath math={(paragraph as any).math_equation.math_equation_latex} />
                    </div>

                    {activeEquation === (paragraph as any).math_equation.math_equation_id && (
                      <div className="equation-explanation">
                        <div className="explanation-header">
                          <span className="explanation-icon">💡</span>
                          <span className="explanation-title">수식 설명</span>
                        </div>
                        <div className="explanation-content mathjax">
                          <div dangerouslySetInnerHTML={{ 
                            __html: formatText((paragraph as any).math_equation.math_equation_explanation || "")
                              .replace(/\$([^$]+)\$/g, (match, math) => {
                                try {
                                  return `<span class="katex-inline">${math}</span>`;
                                } catch {
                                  return match;
                                }
                              })
                          }} />
                        </div>
                        {(paragraph as any).math_equation.math_equation_variables &&
                          (paragraph as any).math_equation.math_equation_variables.length > 0 && (
                            <div className="equation-variables">
                              <div className="explanation-header">
                                <span className="explanation-icon">🔠</span>
                                <span className="explanation-title">변수 설명</span>
                              </div>
                              <ul>
                                {(paragraph as any).math_equation.math_equation_variables.map(
                                  (v: any, idx: number) => (
                                    <li key={idx} className="mathjax">
                                      <div dangerouslySetInnerHTML={{ __html: formatText(typeof v === "string" ? v : JSON.stringify(v)) }} />
                                    </li>
                                  )
                                )}
                              </ul>
                            </div>
                          )}
                      </div>
                    )}
                  </div>
                )}

              {/* 시각화 항상 표시 (존재 시) - 단일 이미지 */}
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

              {/* 시각화 항상 표시 (존재 시) - 다중 이미지 */}
              {(paragraph as any).visualizations && (paragraph as any).visualizations.length > 0 && (
                <div className="visualization-container">
                  {(paragraph as any).visualizations.map((viz: any, index: number) => (
                    <div key={index} className="viz-image-wrapper" style={{ marginBottom: '10px' }}>
                      <img
                        src={getImageSrc(viz.image_path)}
                        alt={`${section.easy_section_title} - 이미지 ${index + 1}`}
                        className="viz-image"
                        onClick={() => openImage(getImageSrc(viz.image_path), `${section.easy_section_title} - 이미지 ${index + 1}`)}
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
                    <BlockMath math={equation.math_equation_latex} />
                  </div>
                  {activeEquation === equation.math_equation_id && (
                    <div className="equation-explanation">
                      <div className="explanation-header">
                        <span className="explanation-icon">💡</span>
                        <span className="explanation-title">수식 설명</span>
                      </div>
                      <div className="explanation-content mathjax">
                        <div dangerouslySetInnerHTML={{ 
                          __html: formatText(equation.math_equation_explanation)
                            .replace(/\$([^$]+)\$/g, (match, math) => {
                              try {
                                return `<span class="katex-inline">${math}</span>`;
                              } catch {
                                return match;
                              }
                            })
                        }} />
                      </div>
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
    
    // 1. 먼저 LaTeX 수식을 자동으로 $...$로 감싸기
    let processed = wrapInlineMath(text);
    
    // 2. 이중 백슬래시 정상화 (핵심!)
    processed = normalizeTexBackslashes(processed);
    
    // 3. <code>와 <pre> 태그 제거 (MathJax가 파싱하지 않음)
    processed = processed.replace(/<code[^>]*>([^<]*)<\/code>/gi, '$1');
    processed = processed.replace(/<pre[^>]*>([^<]*)<\/pre>/gi, '$1');
    
    // 4. 특정 문장 하이라이트 (형광펜 효과)
    processed = highlightSpecificSentences(processed);
    
    // 5. LaTeX 수식 처리 (이미 감싸진 수식에 적용)
    // MathJax가 지원하지 않는 매크로를 지원되는 형태로 치환
    processed = processed.replace(/\Pr\(([^)]+)\)/g, '\\Pr($1)');
    processed = processed.replace(/\textrm\{([^}]+)\}/g, '\\text{$1}');
    processed = processed.replace(/\mathbbm\{([^}]+)\}/g, '\\mathbb{$1}');
    processed = processed.replace(/\mathlarger\{([^}]+)\}/g, '\\large{$1}');
    processed = processed.replace(/\textbf\{([^}]+)\}/g, '\\textbf{$1}');
    
    // \large{\mathbb{1}} 같은 중첩 매크로 처리 개선
    processed = processed.replace(/\\large\{\\mathbb\{1\}\}/g, '\\mathbf{1}');  // \large{\mathbb{1}} → \mathbf{1}
    processed = processed.replace(/\\large\{\\mathbb\{([^}]+)\}\}/g, '\\mathbb{$1}');  // \large{\mathbb{...}} → \mathbb{...}
    processed = processed.replace(/\\large\{\\mathbf\{([^}]+)\}\}/g, '\\mathbf{$1}');  // \large{\mathbf{...}} → \mathbf{...}
    
    // \text 매크로 처리
    processed = processed.replace(/\\text\{obj\}/g, '\\text{obj}');  // \text{obj} 유지
    processed = processed.replace(/\\text\{objects\}/g, '\\text{objects}');  // \text{objects} 유지
    processed = processed.replace(/\\text\{no objects\}/g, '\\text{no objects}');  // \text{no objects} 유지
    
    // \large 매크로가 문제를 일으키는 경우 제거 (더 정확한 처리)
    processed = processed.replace(/\\large\{([^}]+)\}/g, (match, content) => {
      // \large 매크로가 문제를 일으키는 경우 제거
      if (content.includes('\\mathbb') || content.includes('\\mathbf') || content.includes('\\text')) {
        return content; // \large 제거하고 내용만 유지
      }
      return match;
    });
    
    // 인라인 수식 $...$ 처리
    processed = processed.replace(/\$([^$]+)\$/g, (match, content) => {
      return `$${preprocessLatex(content)}$`;
    });
    
    // 블록 수식 $$...$$ 처리
    processed = processed.replace(/\$\$([^$]+)\$\$/g, (match, content) => {
      return `$$${preprocessLatex(content)}$$`;
    });
    
    // LaTeX 수식 환경 처리
    processed = processed.replace(/\\begin\{([^}]+)\}([\s\S]*?)\\end\{\1\}/g, (match, env, content) => {
      return `$$\\begin{${env}}${preprocessLatex(content)}\\end{${env}}$$`;
    });
    
    // 인라인 LaTeX 수식 \(...\) 처리
    processed = processed.replace(/\\\(([^)]+)\\\)/g, (match, content) => {
      return `$${preprocessLatex(content)}$`;
    });
    
    // 블록 LaTeX 수식 \[...\] 처리
    processed = processed.replace(/\\\[([^\]]+)\\\]/g, (match, content) => {
      return `$$${preprocessLatex(content)}$$`;
    });
    
    // **강조**는 굵게만, ==중요문장== 은 은은한 형광펜으로
    let html = processed.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/==([^=]+)==/g, '<mark style="background:#fff3b0; color:inherit;">$1</mark>');
    
    // 특정 LaTeX 패턴들을 KaTeX 인라인 수식으로 치환 (문자열 치환 방식)
    if (html.includes('\\Pr(\\text{Object})')) {
      html = html.replace(/\\Pr\\(\\text\\{Object\\}\\)/g, '<span class="katex-inline">\\Pr(\\text{Object})</span>');
      console.log('✅ [LaTeX] \\Pr(\\text{Object}) 치환됨');
    }
    if (html.includes('\\text{IOU}_{\\text{pred}}^{\\text{truth}}')) {
      html = html.replace(/\\text\\{IOU\\}_\\{\\text\\{pred\\}\\}\\^\\{\\text\\{truth\\}\\}/g, '<span class="katex-inline">\\text{IOU}_{\\text{pred}}^{\\text{truth}}</span>');
      console.log('✅ [LaTeX] \\text{IOU}_{\\text{pred}}^{\\text{truth}} 치환됨');
    }
    // \rightarrow는 단독으로 있을 때만 치환 (한국어와 섞이지 않은 경우)
    html = html.replace(/\s\\rightarrow\s/g, ' <span class="katex-inline">\\rightarrow</span> ');
    
    // # 문자를 KaTeX에서 안전하게 처리하도록 이스케이프
    html = html.replace(/#/g, '\\#');
    
    // 디버깅을 위한 로그 추가
    console.log('🔍 [LaTeX 치환] 처리된 HTML:', html.substring(0, 200));
    
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
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', margin:'12px 0'}}>
        <button
          className="download-btn"
          onClick={async () => {
            try {
              const pathParts = window.location.pathname.split("/");
              const paper_id = pathParts[pathParts.length - 1];
              const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
                const res = await fetch(`${apiBase}/api/upload/download/integrated-math-html/${paper_id}`);
              if (!res.ok) {
                console.error(`다운로드 실패: ${res.status}`, res.statusText);
                throw new Error(`다운로드 실패: ${res.status} - ${res.statusText}`);
              }
              const blob = await res.blob();
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `integrated_math_${paper_id}.html`;
              document.body.appendChild(a);
              a.click();
              window.URL.revokeObjectURL(url);
              document.body.removeChild(a);
            } catch (e) {
              console.error(e);
              alert("HTML 다운로드 중 오류가 발생했습니다.");
            }
          }}
          style={{ padding:'8px 12px', fontSize:12 }}
          title="통합 HTML 다운로드"
        >
          통합 HTML 다운로드
        </button>
      </div>
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
              <strong>출판일:</strong>{" "}
              <span id="paper-venue">
                {integratedData.paper_info.paper_venue}
              </span>
            </p>
          </div>
        </header>
            {(() => {
              const allGroups = groupSections(integratedData.easy_sections);
              const nonAbstractGroups = filterNonAbstractSections(allGroups);
              
              return allGroups.map(({ parent, children }, sectionIdx) => {
                // Abstract 섹션인지 확인
                const isAbstractSection = parent.easy_section_title?.toLowerCase().includes('abstract') || false;
                
                // Abstract가 아닌 섹션들 중에서의 인덱스 계산
                const nonAbstractIndex = nonAbstractGroups.findIndex(group => group.parent.easy_section_id === parent.easy_section_id);
                
                // 마지막 섹션인지 확인
                const isLastSection = sectionIdx === allGroups.length - 1;
              
              return (
                <article key={parent.easy_section_id} id={parent.easy_section_id} className="paper-section-card">
                  <header className="section-header">
                    <h2>{parent.easy_section_title}</h2>
                    {/* Abstract 섹션이 아닌 경우에만 요약 이미지 버튼 표시 */}
                    {!isAbstractSection && (
                      <button
                        onClick={() => {
                          console.log('🖼️ [이미지] 요약 이미지 클릭:', {
                            sectionIdx: nonAbstractIndex,
                            sectionTitle: parent.easy_section_title
                          });
                          openExternalImage(nonAbstractIndex, parent.easy_section_title);
                        }}
                        style={{
                          marginLeft: '20px',
                          padding: '8px 16px',
                          backgroundColor: '#4CAF50',
                          color: 'white',
                          border: 'none',
                          borderRadius: '6px',
                          cursor: 'pointer',
                          fontSize: '14px',
                          fontWeight: '500',
                          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                        }}
                        title="섹션별 요약 이미지"
                      >
                        요약 슬라이드
                      </button>
                    )}
                  </header>
                  
                  {/* [ADD] 하드코딩 매핑된 섹션 Figure */}
                  {(() => {
                    const sectionFigIdx = FIGURE_MAP[parent.easy_section_id];
                    const sectionFig = sectionFigIdx ? figQueue.find(f => f.order === sectionFigIdx) : undefined;
                    const sectionCaption = sectionFigIdx ? FIGURE_CAPTION[sectionFigIdx] : '';
                    return sectionFig ? (
                      <figure className="my-4 sidecar-figure mapped-figure section-mapped-figure">
                        <img 
                          src={sectionFig.image_path} 
                          alt={sectionCaption}
                          onClick={() => openImage(sectionFig.image_path, sectionCaption)}
                          style={{ cursor: 'zoom-in', maxWidth: '100%' }}
                        />
                        <figcaption className="text-sm text-gray-500 mt-1">
                          {sectionCaption}
                        </figcaption>
                      </figure>
                    ) : null;
                  })()}
                  
                  {/* 기존 섹션 레벨 Figures (호환성) */}
                  {(parent as any).figures?.map((figure: FigureMeta, idx: number) => (
                    <FigureView 
                      key={`section-fig-${idx}`}
                      figure={figure} 
                      openImage={openImage} 
                      className="section-figure legacy-figure"
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
                  
                  {/* [ADD] 마지막 섹션에 남은 figures 자동 추가 */}
                  {isLastSection && (() => {
                    const remainingFigures = getRemainingFigures();
                    if (remainingFigures.length > 0) {
                      console.log(`📊 [FIG] 남은 figures를 마지막 섹션에 추가: ${remainingFigures.length}개`);
                      return (
                        <div className="remaining-figures">
                          <h4 className="remaining-figures-title">관련 그림</h4>
                          {remainingFigures.map((fig, i) => {
                            const remainingCaption = FIGURE_CAPTION[fig.order] ?? `Figure ${fig.order}`;
                            return (
                              <figure key={`remaining-${i}`} className="my-3 sidecar-figure remaining-figure">
                                <img 
                                  src={fig.image_path} 
                                  alt={remainingCaption}
                                  onClick={() => openImage(fig.image_path, remainingCaption)}
                                  style={{ cursor: 'zoom-in', maxWidth: '100%' }}
                                />
                                <figcaption className="text-sm text-gray-500 mt-1">
                                  {remainingCaption}
                                </figcaption>
                              </figure>
                            );
                          })}
                        </div>
                      );
                    }
                    return null;
                  })()}
                </article>
              );
              });
            })()}
          </section>

        {/* 오른쪽 패널 없음 — 수식은 문단 인라인만 */}
      </main>
      
      {/* [ADD] 외부 API 이미지 팝업 */}
      {externalImagePopup.isOpen && (
        <div 
          className="external-image-popup-overlay"
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            cursor: 'pointer'
          }}
          onClick={closeExternalImage}
        >
          <div 
            className="external-image-popup-content"
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              padding: '20px',
              maxWidth: '90vw',
              maxHeight: '90vh',
              position: 'relative',
              cursor: 'default'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* 닫기 버튼 */}
            <button
              onClick={closeExternalImage}
              style={{
                position: 'absolute',
                top: '10px',
                right: '15px',
                background: 'none',
                border: 'none',
                fontSize: '24px',
                cursor: 'pointer',
                color: '#666'
              }}
            >
              ×
            </button>
            
            {/* 섹션 제목 */}
            <h3 style={{ 
              margin: '0 0 15px 0', 
              fontSize: '18px',
              color: '#333',
              textAlign: 'center'
            }}>
              {externalImagePopup.sectionTitle}
            </h3>
            
            {/* 이미지 */}
            <img
              src={externalImagePopup.imageUrl}
              alt={`${externalImagePopup.sectionTitle} 시각화`}
              style={{
                maxWidth: '100%',
                maxHeight: '70vh',
                objectFit: 'contain',
                borderRadius: '4px',
                boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
              }}
              onLoad={() => {
                console.log('✅ [이미지] 이미지 로드 성공:', externalImagePopup.imageUrl);
              }}
              onError={(e) => {
                console.warn('❌ [이미지] 외부 API 이미지 로드 실패:', externalImagePopup.imageUrl);
                console.warn('❌ [이미지] 에러 이벤트:', e);
                
                // 이미지 로딩 재시도
                const img = e.target as HTMLImageElement;
                const originalSrc = img.src;
                
                // 1초 후 재시도
                setTimeout(() => {
                  console.log('🔄 [이미지] 이미지 로딩 재시도:', originalSrc);
                  img.src = originalSrc + '?t=' + Date.now(); // 캐시 방지
                }, 1000);
                
                // 3초 후에도 실패하면 에러 메시지 표시
                setTimeout(() => {
                  if (img.complete && img.naturalHeight === 0) {
                    img.style.display = 'none';
                    const errorDiv = document.createElement('div');
                    errorDiv.innerHTML = `
                      <div style="text-align: center; padding: 40px; color: #666;">
                        <p>📊 이미지를 불러올 수 없습니다</p>
                        <p style="font-size: 14px; margin-top: 10px;">
                          외부 API로 생성된 이미지가 아직 준비되지 않았거나<br/>
                          경로를 찾을 수 없습니다.
                        </p>
                        <p style="font-size: 12px; margin-top: 10px; color: #999;">
                          경로: ${externalImagePopup.imageUrl}
                        </p>
                        <button onclick="window.open('${externalImagePopup.imageUrl}', '_blank')" 
                                style="margin-top: 10px; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                          🔗 직접 링크 열기
                        </button>
                      </div>
                    `;
                    img.parentNode?.appendChild(errorDiv);
                  }
                }, 3000);
              }}
            />
            
            {/* 이미지 정보 */}
            <div style={{
              marginTop: '10px',
              textAlign: 'center',
              fontSize: '14px',
              color: '#666'
            }}>
              외부 API로 생성된 시각화 이미지
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Result;
