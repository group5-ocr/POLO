/**
 * [Figure] 토큰 교체 유틸리티
 * 텍스트의 [Figure] 토큰을 실제 Figure 객체로 교체
 */

import type { FigureItem } from './figureMap';
import { matchFigureToText } from './figureMap';

// [Figure] 토큰 패턴 (대소문자 무관)
const FIGURE_TOKEN = /\[Figure[^\]]*\]/gi;

export type TextChunk = string | FigureItem;

/**
 * [Figure] 토큰을 Figure 객체로 교체
 * 
 * @param text 원본 텍스트
 * @param figureQueue Figure 큐 (순서 기반 소모)
 * @param availableFigures 전체 Figure 리스트 (라벨 매칭용)
 * @returns 텍스트와 Figure가 섞인 배열
 */
export function replaceFigureTokens(
  text: string,
  figureQueue: { next(): FigureItem | undefined; markUsed(fig: FigureItem): void },
  availableFigures: FigureItem[] = []
): TextChunk[] {
  if (!text) return [text];
  
  const parts: TextChunk[] = [];
  let lastIndex = 0;
  
  // 모든 [Figure] 토큰 찾기
  const matches = Array.from(text.matchAll(FIGURE_TOKEN));
  
  for (const match of matches) {
    const tokenStart = match.index!;
    const tokenEnd = tokenStart + match[0].length;
    
    // 토큰 이전 텍스트 추가
    if (tokenStart > lastIndex) {
      const beforeText = text.slice(lastIndex, tokenStart);
      if (beforeText) parts.push(beforeText);
    }
    
    // Figure 매칭 시도
    let matchedFigure: FigureItem | undefined;
    
    // 1. 라벨/키워드 기반 매칭 (주변 텍스트 분석)
    if (availableFigures.length > 0) {
      // 토큰 주변 텍스트 추출 (앞뒤 200자)
      const contextStart = Math.max(0, tokenStart - 200);
      const contextEnd = Math.min(text.length, tokenEnd + 200);
      const context = text.slice(contextStart, contextEnd);
      
      matchedFigure = matchFigureToText(context, availableFigures);
      if (matchedFigure) {
        figureQueue.markUsed(matchedFigure);
      }
    }
    
    // 2. 순서 기반 fallback
    if (!matchedFigure) {
      matchedFigure = figureQueue.next();
    }
    
    // Figure 또는 원본 토큰 추가
    if (matchedFigure) {
      parts.push(matchedFigure);
      console.log(`🔄 [Figure] 토큰 교체: ${match[0]} → ${matchedFigure.graphics}`);
    } else {
      // Figure가 없으면 원본 토큰 유지
      parts.push(match[0]);
      console.warn(`⚠️ [Figure] 토큰 교체 실패: ${match[0]} (Figure 부족)`);
    }
    
    lastIndex = tokenEnd;
  }
  
  // 마지막 텍스트 추가
  if (lastIndex < text.length) {
    const remainingText = text.slice(lastIndex);
    if (remainingText) parts.push(remainingText);
  }
  
  return parts;
}

/**
 * 단순 순서 기반 토큰 교체 (이전 버전 호환)
 */
export function replaceFigureTokensSimple(
  text: string,
  nextFigure: () => FigureItem | undefined
): TextChunk[] {
  if (!text) return [text];
  
  const parts: TextChunk[] = [];
  let rest = text;
  
  while (true) {
    const match = rest.match(FIGURE_TOKEN);
    if (!match) {
      if (rest) parts.push(rest);
      break;
    }
    
    const tokenIndex = match.index!;
    
    // 토큰 이전 텍스트
    if (tokenIndex > 0) {
      parts.push(rest.slice(0, tokenIndex));
    }
    
    // Figure 또는 원본 토큰
    const fig = nextFigure();
    if (fig) {
      parts.push(fig);
      console.log(`🔄 [Figure] 간단 교체: ${match[0]} → ${fig.graphics}`);
    } else {
      parts.push(match[0]); // Figure 없으면 원문 유지
      console.warn(`⚠️ [Figure] 간단 교체 실패: ${match[0]}`);
    }
    
    // 나머지 텍스트
    rest = rest.slice(tokenIndex + match[0].length);
  }
  
  return parts;
}

/**
 * 텍스트에서 [Figure] 토큰 개수 세기
 */
export function countFigureTokens(text: string): number {
  if (!text) return 0;
  const matches = text.match(FIGURE_TOKEN);
  return matches ? matches.length : 0;
}

/**
 * 텍스트에서 [Figure] 토큰 제거
 */
export function removeFigureTokens(text: string): string {
  if (!text) return text;
  return text.replace(FIGURE_TOKEN, '').replace(/\s+/g, ' ').trim();
}

/**
 * 전체 섹션에서 Figure 토큰 통계
 */
export function analyzeFigureTokens(sections: any[]): {
  totalTokens: number;
  sectionTokens: Array<{ sectionId: string; title: string; tokens: number }>;
  paragraphTokens: Array<{ paragraphId: string; sectionId: string; tokens: number }>;
} {
  let totalTokens = 0;
  const sectionTokens: Array<{ sectionId: string; title: string; tokens: number }> = [];
  const paragraphTokens: Array<{ paragraphId: string; sectionId: string; tokens: number }> = [];
  
  for (const section of sections) {
    let sectionCount = 0;
    
    // 섹션 콘텐츠
    const sectionContent = section.easy_content || '';
    const sectionContentTokens = countFigureTokens(sectionContent);
    sectionCount += sectionContentTokens;
    
    // 문단들
    const paragraphs = section.easy_paragraphs || [];
    for (const paragraph of paragraphs) {
      const paragraphText = paragraph.easy_paragraph_text || '';
      const paragraphTokens = countFigureTokens(paragraphText);
      
      if (paragraphTokens > 0) {
        paragraphTokens.push({
          paragraphId: paragraph.easy_paragraph_id,
          sectionId: section.easy_section_id,
          tokens: paragraphTokens
        });
      }
      
      sectionCount += paragraphTokens;
    }
    
    if (sectionCount > 0) {
      sectionTokens.push({
        sectionId: section.easy_section_id,
        title: section.easy_section_title,
        tokens: sectionCount
      });
    }
    
    totalTokens += sectionCount;
  }
  
  return {
    totalTokens,
    sectionTokens,
    paragraphTokens
  };
}
