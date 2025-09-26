"""
HTML 템플릿 생성기 - Result.tsx와 동일한 구조로 HTML 다운로드 기능 구현
"""
import json
from pathlib import Path
from typing import Dict, Any

def generate_integrated_html(paper_id: str) -> str:
    """
    Result.tsx와 동일한 구조로 통합 HTML 생성
    """
    # 데이터 로드
    integrated_file = Path(f"data/outputs/{paper_id}/integrated_result.json")
    if not integrated_file.exists():
        raise FileNotFoundError(f"integrated_result.json not found for paper_id: {paper_id}")
    
    with open(integrated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # HTML 헤더
    html_head = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>POLO - Integrated Results for {paper_id}</title>
    <style>
        body {{
            font-family: 'Pretendard', 'Spoqa Han Sans Neo', 'Noto Sans KR', 'Apple SD Gothic Neo', 'Inter', 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            background: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .paragraph {{
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-left: 4px solid #3498db;
        }}
        .equation {{
            margin: 20px 0;
            padding: 15px;
            background: #f0f8ff;
            border: 1px solid #b0d4f1;
            border-radius: 5px;
            text-align: center;
        }}
        .equation-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .equation-content {{
            font-size: 1.2em;
            margin: 10px 0;
        }}
        .equation-explanation {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .image-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }}
        .mathjax {{
            font-size: 1.1em;
        }}
    </style>
    <script>
        window.MathJax = {{
            loader: {{ load: ['[tex]/ams', '[tex]/mathtools', '[tex]/physics'] }},
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                packages: {{ '[+]': ['ams', 'mathtools', 'physics'] }},
                processEscapes: true,
                tags: 'none',
                macros: {{
                    mathlarger: ['{{\\\\large #1}}', 1],
                    mathbbm: ['{{\\\\mathbb{{#1}}}}', 1],
                    wt: ['{{\\\\widetilde{{#1}}}}', 1],
                    wh: ['{{\\\\widehat{{#1}}}}', 1],
                    dfn: '{{\\\\triangleq}}',
                    dB: '{{\\\\mathrm{{dB}}}}',
                    snr: '{{\\\\mathrm{{SNR}}}}',
                    bsnr: '{{\\\\mathrm{{S}}\\\\widetilde{{\\\\mathrm{{N}}}}\\\\mathrm{{R}}}}'
                }}
            }},
            options: {{
                ignoreHtmlClass: 'no-mathjax',
                processHtmlClass: 'mathjax'
            }},
            svg: {{ 
                fontCache: 'global', 
                scale: 1,
                minScale: 0.5,
                mtextInheritFont: true,
                merrorInheritFont: true,
                mathmlSpacing: false,
                skipAttributes: {{}},
                exFactor: 0.5,
                displayAlign: 'center',
                displayIndent: '0'
            }}
        }};
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>POLO - 통합 결과 미리보기</h1>
            <div>Paper ID: {paper_id}</div>
        </div>
"""
    
    # 본문 생성
    body_content = ""
    sections = data.get('easy_sections', [])
    
    for section in sections:
        section_title = section.get('easy_section_title', '')
        body_content += f'<div class="section">\n'
        body_content += f'<h2>{section_title}</h2>\n'
        
        for paragraph in section.get('easy_paragraphs', []):
            para_id = paragraph.get('easy_paragraph_id', '')
            para_text = paragraph.get('easy_paragraph_text', '')
            para_type = paragraph.get('paragraph_type', '')
            
            if para_type == 'math_equation':
                # 수식 문단
                math_eq = paragraph.get('math_equation', {})
                eq_id = math_eq.get('equation_id', '')
                eq_index = math_eq.get('equation_index', '')
                eq_latex = math_eq.get('equation_latex', '')
                eq_explanation = math_eq.get('equation_explanation', '')
                
                body_content += f'<div class="equation">\n'
                body_content += f'<div class="equation-header">수식 {eq_index}</div>\n'
                body_content += f'<div class="equation-content mathjax">$${eq_latex}$$</div>\n'
                if eq_explanation:
                    body_content += f'<div class="equation-explanation">{eq_explanation}</div>\n'
                body_content += f'</div>\n'
            else:
                # 일반 문단
                body_content += f'<div class="paragraph">\n'
                body_content += f'<p>{para_text}</p>\n'
                
                # 이미지가 있는 경우 - 단일 이미지
                if 'visualization' in paragraph:
                    img_path = paragraph['visualization'].get('image_path', '')
                    if img_path:
                        body_content += f'<div class="image-container">\n'
                        body_content += f'<img src="{img_path}" alt="{para_id}" />\n'
                        body_content += f'<div class="image-caption">원본 이미지: {para_id}</div>\n'
                        body_content += f'</div>\n'
                
                # 이미지가 있는 경우 - 다중 이미지
                if 'visualizations' in paragraph:
                    for i, viz in enumerate(paragraph['visualizations']):
                        img_path = viz.get('image_path', '')
                        if img_path:
                            body_content += f'<div class="image-container">\n'
                            body_content += f'<img src="{img_path}" alt="{para_id} - 이미지 {i+1}" />\n'
                            body_content += f'<div class="image-caption">원본 이미지: {para_id} - 이미지 {i+1}</div>\n'
                            body_content += f'</div>\n'
                
                body_content += f'</div>\n'
        
        body_content += f'</div>\n'
    
    # HTML 푸터
    html_footer = """
    </div>
    <script>
        // MathJax 렌더링
        if (window.MathJax) {
            window.MathJax.typesetPromise();
        }
    </script>
</body>
</html>
"""
    
    return html_head + body_content + html_footer

if __name__ == "__main__":
    # 테스트
    try:
        html_content = generate_integrated_html("30")
        print("HTML 생성 성공!")
        print(f"HTML 길이: {len(html_content)} 문자")
    except Exception as e:
        print(f"HTML 생성 실패: {e}")
