/**
 * Figure 사이드카 맵 관리
 * 통합 JSON 구조를 건드리지 않고 별도로 Figure 정보 제공
 */

export interface FigureItem {
  order: number;
  label?: string;
  caption?: string;
  graphics: string;
  src_file: string;
  image_path: string;   // "/static/viz/figures/model_p1.png?v=abcd1234"
  all_pages: string[];  // 멀티페이지 지원
  hash: string;
}

export interface FigureMapData {
  figures: FigureItem[];
  metadata: {
    total_count: number;
    generated_at: string;
    source_assets: string;
    source_dir: string;
    static_root: string;
  };
}

/**
 * 사이드카 맵 로드
 */
export async function loadFigureQueue(): Promise<FigureItem[]> {
  try {
    const res = await fetch('/static/viz/figures_map.json');
    
    if (!res.ok) {
      console.warn(`Figure 맵 로드 실패: ${res.status} ${res.statusText}`);
      return [];
    }
    
    const data: FigureMapData = await res.json();
    const queue: FigureItem[] = data.figures ?? [];
    
    console.log(`✅ Figure 맵 로드 완료: ${queue.length}개 figures`);
    return queue;
    
  } catch (error) {
    console.warn('Figure 맵 로드 실패:', error);
    return [];
  }
}

/**
 * 라벨 기반 Figure 찾기
 */
export function findFigureByLabel(figures: FigureItem[], label: string): FigureItem | undefined {
  if (!label) return undefined;
  
  return figures.find(fig => {
    const figLabel = fig.label || '';
    return figLabel && (label.includes(figLabel) || figLabel.includes(label));
  });
}

/**
 * 키워드 기반 Figure 찾기
 */
export function findFigureByKeywords(figures: FigureItem[], text: string): FigureItem | undefined {
  const textLower = text.toLowerCase();
  const keywords = textLower.match(/\b[a-zA-Z가-힣]{3,}\b/g) || [];
  
  if (keywords.length === 0) return undefined;
  
  let bestMatch: FigureItem | undefined;
  let bestScore = 0;
  
  for (const fig of figures) {
    const caption = (fig.caption || '').toLowerCase();
    const graphics = fig.graphics.toLowerCase();
    
    // 키워드 매칭 점수 계산
    let score = 0;
    for (const keyword of keywords) {
      if (caption.includes(keyword)) score += 2;
      if (graphics.includes(keyword)) score += 1;
    }
    
    if (score > bestScore) {
      bestScore = score;
      bestMatch = fig;
    }
  }
  
  // 최소 2점 이상일 때만 매칭
  return bestScore >= 2 ? bestMatch : undefined;
}

/**
 * 텍스트에서 Figure 참조 추출
 */
export function extractFigureReferences(text: string): string[] {
  const references: string[] = [];
  
  // LaTeX 참조 패턴: \ref{fig:...}, \label{fig:...}
  const latexRefs = text.match(/\\(?:ref|label)\{(fig:[^}]+)\}/gi);
  if (latexRefs) {
    references.push(...latexRefs.map(ref => ref.match(/\{([^}]+)\}/)?.[1] || ''));
  }
  
  // 일반적인 figure 참조: Figure 1, Fig. 2 등
  const figNums = text.match(/(?:Figure|Fig\.?)\s*(\d+)/gi);
  if (figNums) {
    references.push(...figNums.map(ref => {
      const num = ref.match(/(\d+)/)?.[1];
      return num ? `fig:${num}` : '';
    }));
  }
  
  // 직접적인 라벨 언급
  const directLabels = text.match(/(fig:[a-zA-Z0-9_-]+)/gi);
  if (directLabels) {
    references.push(...directLabels);
  }
  
  return references.filter(ref => ref.length > 0);
}

/**
 * 텍스트 내용 기반으로 최적의 Figure 찾기
 */
export function matchFigureToText(text: string, figures: FigureItem[]): FigureItem | undefined {
  // 1. 라벨 기반 매칭 시도
  const references = extractFigureReferences(text);
  for (const ref of references) {
    const matched = findFigureByLabel(figures, ref);
    if (matched) {
      console.log(`🎯 Figure 라벨 매칭: ${ref} → ${matched.graphics}`);
      return matched;
    }
  }
  
  // 2. 키워드 기반 매칭
  const keywordMatch = findFigureByKeywords(figures, text);
  if (keywordMatch) {
    console.log(`🎯 Figure 키워드 매칭: ${keywordMatch.graphics}`);
    return keywordMatch;
  }
  
  return undefined;
}

/**
 * Figure 큐 생성기
 * 순서대로 소모되는 iterator 생성
 */
export function createFigureQueue(figures: FigureItem[]) {
  let index = 0;
  const usedFigures = new Set<number>();
  
  return {
    /**
     * 다음 Figure 가져오기 (순서 기반)
     */
    next(): FigureItem | undefined {
      while (index < figures.length) {
        const figure = figures[index++];
        if (!usedFigures.has(figure.order)) {
          usedFigures.add(figure.order);
          return figure;
        }
      }
      return undefined;
    },
    
    /**
     * 특정 Figure 사용 표시 (라벨 매칭용)
     */
    markUsed(figure: FigureItem): void {
      usedFigures.add(figure.order);
    },
    
    /**
     * 남은 Figure 개수
     */
    remaining(): number {
      return figures.length - usedFigures.size;
    },
    
    /**
     * 리셋
     */
    reset(): void {
      index = 0;
      usedFigures.clear();
    }
  };
}
