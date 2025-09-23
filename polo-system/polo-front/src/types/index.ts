/**
 * POLO 시스템 타입 정의
 */

// Figure 메타데이터
export interface FigureMeta {
  image_path: string;      // "/static/viz/figures/model_p1.png?v=abcd1234"
  caption?: string;        // 캡션 텍스트
  label?: string;          // LaTeX 라벨 (fig:model 등)
  src_file?: string;       // 원본 파일 경로
  all_pages?: string[];    // 멀티페이지 PDF → PNG들
  order?: number;          // 순서
  key?: string;            // 내부 키
}

// 시각화 메타데이터 (기존)
export interface VizMeta {
  image_path: string;
  viz_type: string;
  created_at?: string;
}

// Easy 문단 (업데이트됨)
export interface EasyParagraph {
  easy_paragraph_id: string;
  easy_paragraph_text: string;
  easy_paragraph_order?: number;
  paragraph_type?: string;
  
  // 수식 관련
  math_equation?: {
    equation_latex: string;
    equation_explanation: string;
    equation_id?: string;
  };
  equation_latex?: string;
  math_equation_latex?: string;
  
  // 시각화 (자동 생성)
  visualization?: VizMeta;
  
  // Figure (원본 PDF/이미지)
  figure?: FigureMeta;
}

// Easy 섹션
export interface EasySection {
  easy_section_id: string;
  easy_section_title: string;
  easy_section_order?: number;
  easy_content?: string;
  easy_paragraphs?: EasyParagraph[];
  
  // 섹션 레벨 figures
  figures?: FigureMeta[];
}

// 통합 데이터
export interface IntegratedData {
  paper_info?: {
    paper_id: string;
    paper_title: string;
    arxiv_id?: string;
    authors?: string[];
    abstract?: string;
  };
  easy_sections: EasySection[];
  math_results?: any[];
  processing_info?: {
    created_at: string;
    processing_time?: number;
    version?: string;
  };
}

// API 응답 타입들
export interface ApiResponse<T = any> {
  ok: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface BuildFiguresResponse {
  ok: boolean;
  paper_id: string;
  figures_count: number;
  integrated_path: string;
  validation: {
    valid: boolean;
    stats?: {
      sections: number;
      paragraphs: number;
      section_figures: number;
      paragraph_figures: number;
      remaining_tokens: number;
    };
    total_figures?: number;
    error?: string;
  };
}

export interface GetFiguresResponse {
  ok: boolean;
  paper_id: string;
  figures_count: number;
  figures: Array<{
    order: number;
    label?: string;
    caption?: string;
    src_file: string;
    pngs: string[];
    rel_url: string;
    key: string;
    hash: string;
    web_url: string;
    all_web_pages: string[];
  }>;
}

// 이벤트 핸들러 타입들
export type ImageClickHandler = (src: string, alt?: string) => void;

// 컴포넌트 Props 타입들
export interface ResultProps {
  data?: IntegratedData;
  onDownload?: () => void;
  onPreview?: () => void;
}

export interface ParagraphViewProps {
  p: EasyParagraph;
  sectionId: string;
  openImage: ImageClickHandler;
  getImageSrc: (src?: string) => string;
}

export interface FigureViewProps {
  figure: FigureMeta;
  openImage: ImageClickHandler;
  className?: string;
}

// 유틸리티 타입들
export type ProcessingStatus = 'idle' | 'processing' | 'completed' | 'error';

export interface ProcessingState {
  status: ProcessingStatus;
  progress?: number;
  message?: string;
  error?: string;
}
