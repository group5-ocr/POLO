# -*- coding: utf-8 -*-
"""
integrated_result.json에서 [Figure] 토큰이 들어간 문단/섹션을 찾아
FIGURE_MAP 템플릿(TypeScript)을 생성합니다.
"""

import json
from pathlib import Path

# 파일 경로 (환경에 맞춰 수정)
INTEGRATED = Path(r"C:\POLO\POLO\polo-system\server\data\outputs\doc_8837655066123463610_2767669\integrated_result.json")
OUT_TS     = Path(r"C:\POLO\POLO\polo-system\polo-front\src\figureMapTemplate.ts")

def main():
    if not INTEGRATED.exists():
        print(f"❌ integrated_result.json 없음: {INTEGRATED}")
        return
    
    print(f"📖 [START] Figure 매핑 템플릿 생성")
    print(f"  - 입력: {INTEGRATED}")
    print(f"  - 출력: {OUT_TS}")
    
    data = json.loads(INTEGRATED.read_text(encoding="utf-8"))
    mapping = {}
    order = 1
    
    # 섹션별 [Figure] 토큰 검색
    for sec in data.get("easy_sections", []):
        sec_id = sec.get("easy_section_id")
        
        # 섹션 content에서 [Figure] 검색
        sec_content = sec.get("easy_content", "")
        if isinstance(sec_content, str) and "[Figure" in sec_content:
            mapping[sec_id] = order
            print(f"🔍 [SECTION] {sec_id} → Figure {order}")
            order += 1
        
        # 문단별 [Figure] 검색
        for p in sec.get("easy_paragraphs", []):
            pid = p.get("easy_paragraph_id")
            text = p.get("easy_paragraph_text", "")
            if isinstance(text, str) and "[Figure" in text:
                mapping[pid] = order
                print(f"🔍 [PARAGRAPH] {pid} → Figure {order}")
                order += 1
    
    # TypeScript 파일 생성
    lines = [
        "// 자동 생성된 FIGURE_MAP 템플릿",
        "// integrated_result.json의 [Figure] 토큰 위치를 기반으로 생성됨",
        "",
        "export const FIGURE_MAP: Record<string, number> = {"
    ]
    
    for k, v in mapping.items():
        lines.append(f'  "{k}": {v},')
    
    lines.extend([
        "};",
        "",
        f"// 총 {len(mapping)}개 매핑 생성됨",
        f"// 생성 시각: {INTEGRATED.stat().st_mtime}",
    ])
    
    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    OUT_TS.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"\n🎉 [COMPLETE] FIGURE_MAP 생성 완료!")
    print(f"  📄 출력: {OUT_TS}")
    print(f"  📊 총 {len(mapping)}개 매핑")
    
    if mapping:
        print(f"  🔗 예시: {list(mapping.items())[:3]}")
    else:
        print("  ⚠️ [Figure] 토큰을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
