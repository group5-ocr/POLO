// YOLOv1 논문 통합 분석 결과 JavaScript

class IntegratedPaperViewer {
  constructor() {
    this.integratedData = null;
    this.init();
  }

  async init() {
    try {
      await this.loadIntegratedData();
      this.hideLoading();
      this.displayIntegratedPaper();
      this.displayStats();
      this.setupDownloadButtons();
    } catch (error) {
      this.showError(
        "데이터를 불러오는 중 오류가 발생했습니다: " + error.message
      );
    }
  }

  async loadIntegratedData() {
    try {
      const response = await fetch("./data/integrated_result.jsonl");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const text = await response.text();
      const lines = text.trim().split("\n");

      const parsedLines = lines
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch (e) {
            console.error("JSON 파싱 오류:", e, "Line:", line);
            return null;
          }
        })
        .filter((item) => item !== null);

      if (parsedLines.length === 0) {
        throw new Error("통합 데이터가 비어있습니다.");
      }

      // 첫 번째 라인은 논문 정보 (paper_info가 있는지 확인)
      const firstLine = parsedLines[0];
      if (firstLine.paper_title || firstLine.paper_id) {
        // 논문 정보가 포함된 첫 번째 라인
        this.paperInfo = firstLine;
        this.integratedData = parsedLines.slice(1);
      } else {
        // 논문 정보가 없는 경우, 첫 번째 섹션에서 추출
        this.paperInfo = {
          paper_id: firstLine.paper_id || "",
          paper_title: firstLine.paper_title || "",
          paper_authors: firstLine.paper_authors || "",
          paper_date: firstLine.paper_date || "",
          paper_venue: firstLine.paper_venue || "",
          paper_doi: firstLine.paper_doi || "",
          total_sections: parsedLines.length,
          total_equations: parsedLines.reduce(
            (sum, section) =>
              sum +
              (section.math_equations ? section.math_equations.length : 0),
            0
          ),
        };
        this.integratedData = parsedLines;
      }

      console.log("논문 정보:", this.paperInfo);
      console.log(
        "통합 데이터 로드 완료:",
        this.integratedData.length,
        "개 섹션"
      );
    } catch (error) {
      console.error("데이터 로드 실패:", error);
      throw error;
    }
  }

  displayIntegratedPaper() {
    const container = document.getElementById("paper-sections");

    // 논문 정보 동적 업데이트
    this.updatePaperInfo();

    // 기존 내용 제거
    container.innerHTML = "";

    // 섹션들 표시
    this.integratedData.forEach((section, index) => {
      const sectionElement = this.createSectionElement(section, index);
      container.appendChild(sectionElement);
    });

    // 목차 생성
    this.createTableOfContents();

    // MathJax 렌더링
    this.renderMathJax();
  }

  updatePaperInfo() {
    if (!this.paperInfo) return;

    // 논문 제목 업데이트
    const paperTitle = document.getElementById("paper-title");
    const paperTitleText = document.getElementById("paper-title-text");
    const paperAuthors = document.getElementById("paper-authors");
    const paperVenue = document.getElementById("paper-venue");
    const paperId = document.getElementById("paper-id");

    if (paperTitle) {
      paperTitle.textContent = this.paperInfo.paper_title || "논문 제목 없음";
    }
    if (paperTitleText) {
      paperTitleText.textContent =
        this.paperInfo.paper_title || "논문 제목 없음";
    }
    if (paperAuthors) {
      paperAuthors.textContent =
        this.paperInfo.paper_authors || "저자 정보 없음";
    }
    if (paperVenue) {
      const venueText = this.paperInfo.paper_venue || "학회 정보 없음";
      const dateText = this.paperInfo.paper_date
        ? ` (${this.paperInfo.paper_date})`
        : "";
      paperVenue.textContent = venueText + dateText;
    }
    if (paperId) {
      paperId.textContent = this.paperInfo.paper_id || "논문 ID 없음";
    }
  }

  createSectionElement(section, index) {
    const sectionDiv = document.createElement("div");
    sectionDiv.className = "paper-section";
    sectionDiv.id = `section-${index}`;

    const sectionHeader = this.createSectionHeader(section);
    const easyContent = this.createEasyContent(section);
    const mathEquations = this.createMathEquations(section);
    const visualizations = this.createVisualizations(section);

    sectionDiv.appendChild(sectionHeader);
    sectionDiv.appendChild(easyContent);
    sectionDiv.appendChild(mathEquations);

    // 시각화가 있는 경우에만 추가
    if (visualizations) {
      sectionDiv.appendChild(visualizations);
    }

    return sectionDiv;
  }

  createSectionHeader(section) {
    const headerDiv = document.createElement("div");
    headerDiv.className = "section-header";

    const titleDiv = document.createElement("div");
    titleDiv.className = "section-title";

    const orderSpan = document.createElement("span");
    orderSpan.className = "section-order";
    orderSpan.textContent = section.section_order || "";

    const titleText = document.createElement("span");
    titleText.textContent = section.section_title || section.section_name;

    titleDiv.appendChild(orderSpan);
    titleDiv.appendChild(titleText);

    headerDiv.appendChild(titleDiv);

    return headerDiv;
  }

  createEasyContent(section) {
    const easyDiv = document.createElement("div");
    easyDiv.className = "easy-content";

    const content = document.createElement("p");
    content.innerHTML = this.formatText(section.easy_content || "");

    easyDiv.appendChild(content);

    return easyDiv;
  }

  createMathEquations(section) {
    const mathDiv = document.createElement("div");
    mathDiv.className = "math-equations";

    // 수식이 있는 경우에만 내용 표시
    if (section.math_equations && section.math_equations.length > 0) {
      section.math_equations.forEach((equation) => {
        const equationElement = this.createEquationElement(equation);
        mathDiv.appendChild(equationElement);
      });
    }

    return mathDiv;
  }

  createVisualizations(section) {
    // 시각화가 없는 경우 아무것도 반환하지 않음
    if (!section.visualizations || section.visualizations.length === 0) {
      return null;
    }

    const vizDiv = document.createElement("div");
    vizDiv.className = "visualizations";

    section.visualizations.forEach((viz) => {
      const vizElement = this.createVisualizationElement(viz);
      vizDiv.appendChild(vizElement);
    });

    return vizDiv;
  }

  createVisualizationElement(viz) {
    const vizDiv = document.createElement("div");
    vizDiv.className = "visualization-item";

    const title = document.createElement("h4");
    title.textContent = viz.title || "데이터 분석";
    vizDiv.appendChild(title);

    if (viz.description) {
      const description = document.createElement("p");
      description.textContent = viz.description;
      description.className = "viz-description";
      vizDiv.appendChild(description);
    }

    if (viz.image_path) {
      const img = document.createElement("img");
      // 상대 경로로 수정 (frontend/charts 폴더 사용)
      const imageFileName = viz.image_path.split("/").pop();
      img.src = `./charts/${imageFileName}`;
      img.alt = viz.title || "데이터 분석";
      img.className = "viz-image";

      // 디버깅을 위한 로그
      console.log("이미지 로드 시도:", img.src);

      img.onload = function () {
        console.log("이미지 로드 성공:", img.src);
      };

      img.onerror = function () {
        console.error("이미지 로드 실패:", img.src);
        // 이미지 로드 실패 시 대체 텍스트 표시
        this.style.display = "none";
        const fallback = document.createElement("div");
        fallback.className = "image-fallback";
        fallback.textContent = `${viz.title || "데이터 분석"}`;
        fallback.style.cssText =
          "padding: 40px; text-align: center; background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; color: #6c757d;";
        this.parentNode.appendChild(fallback);
      };
      vizDiv.appendChild(img);
    }

    return vizDiv;
  }

  createEquationElement(equation) {
    const equationDiv = document.createElement("div");
    equationDiv.className = "equation-item";

    const header = document.createElement("div");
    header.className = "equation-header";

    const index = document.createElement("div");
    index.className = "equation-index";
    index.textContent = equation.index || "";

    const title = document.createElement("div");
    title.className = "equation-title";
    title.textContent = `수식 ${equation.index || ""}`;

    header.appendChild(index);
    header.appendChild(title);

    const equationContent = document.createElement("div");
    equationContent.className = "equation";
    equationContent.innerHTML = `$$${equation.equation || ""}$$`;

    const explanation = document.createElement("div");
    explanation.className = "equation-explanation";
    explanation.innerHTML = this.formatText(equation.explanation || "");

    equationDiv.appendChild(header);
    equationDiv.appendChild(equationContent);
    equationDiv.appendChild(explanation);

    return equationDiv;
  }

  createTableOfContents() {
    const tocList = document.getElementById("toc-list");
    if (!tocList || !this.integratedData) return;

    tocList.innerHTML = "";

    this.integratedData.forEach((section, index) => {
      const li = document.createElement("li");
      const a = document.createElement("a");

      a.href = `#section-${index}`;
      a.textContent = section.section_title || section.section_name;
      a.addEventListener("click", (e) => {
        e.preventDefault();
        this.scrollToSection(index);
      });

      li.appendChild(a);
      tocList.appendChild(li);
    });
  }

  scrollToSection(index) {
    const section = document.getElementById(`section-${index}`);
    if (section) {
      section.scrollIntoView({ behavior: "smooth" });

      // 목차에서 활성 상태 업데이트
      this.updateActiveTocItem(index);
    }
  }

  updateActiveTocItem(activeIndex) {
    const tocLinks = document.querySelectorAll("#toc-list a");
    tocLinks.forEach((link, index) => {
      if (index === activeIndex) {
        link.classList.add("active");
      } else {
        link.classList.remove("active");
      }
    });
  }

  formatText(text) {
    if (!text) return "";

    // **텍스트**를 <strong>텍스트</strong>로 변환
    return text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  }

  displayStats() {
    const statsDiv = document.getElementById("stats");
    if (!this.paperInfo) return;

    const totalSections = this.paperInfo.total_sections || 0;
    const totalEquations = this.paperInfo.total_equations || 0;

    // 수식이 포함된 섹션 수 계산
    const sectionsWithEquations = this.integratedData
      ? this.integratedData.filter(
          (section) =>
            section.math_equations && section.math_equations.length > 0
        ).length
      : 0;

    statsDiv.innerHTML = `
            <div class="stat-item">
                <span class="stat-number">${totalSections}</span>
                <span class="stat-label">총 섹션</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">${totalEquations}</span>
                <span class="stat-label">총 수식</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">${sectionsWithEquations}</span>
                <span class="stat-label">수식 포함 섹션</span>
            </div>
        `;
  }

  hideLoading() {
    const loading = document.getElementById("loading");
    const integratedPaper = document.getElementById("integrated-paper");

    if (loading) loading.style.display = "none";
    if (integratedPaper) integratedPaper.style.display = "block";
  }

  showError(message) {
    const loading = document.getElementById("loading");
    const error = document.getElementById("error");
    const errorMessage = document.getElementById("error-message");

    if (loading) loading.style.display = "none";
    if (error) error.style.display = "block";
    if (errorMessage) errorMessage.textContent = message;

    console.error("오류:", message);
  }

  renderMathJax() {
    // MathJax가 로드되었는지 확인하고 렌더링
    if (window.MathJax && typeof window.MathJax.typeset === "function") {
      window.MathJax.typeset();
    } else if (
      window.MathJax &&
      typeof window.MathJax.typesetPromise === "function"
    ) {
      // 이전 버전 호환성
      window.MathJax.typesetPromise();
    } else {
      // MathJax가 아직 로드되지 않은 경우 잠시 후 재시도
      setTimeout(() => this.renderMathJax(), 100);
    }
  }

  setupDownloadButtons() {
    const htmlBtn = document.getElementById("download-html-btn");

    if (htmlBtn) {
      htmlBtn.addEventListener("click", () => this.downloadAsHTML());
    }
  }

  async downloadAsHTML() {
    const htmlBtn = document.getElementById("download-html-btn");
    const originalText = htmlBtn.innerHTML;

    try {
      // 버튼 비활성화 및 로딩 상태 표시
      htmlBtn.disabled = true;
      htmlBtn.innerHTML =
        '<span class="download-icon">⏳</span>HTML 생성 중...';

      // HTML 내용 가져오기
      const element = document.querySelector(".integrated-paper");
      let htmlContent = element.outerHTML;

      // 이미지를 Base64로 변환
      htmlContent = await this.convertImagesToBase64(htmlContent);

      // 전체 HTML 문서 생성
      const fullHtml = `
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv1 논문 분석 결과</title>
    <style>
        ${this.getInlineStyles()}
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
                <h1>${this.paperInfo?.paper_title || "YOLOv1 논문 분석"}</h1>
                <div class="paper-info">
                    <p><strong>논문 제목:</strong> ${
                      this.paperInfo?.paper_title || "로딩 중..."
                    }</p>
                    <p><strong>저자:</strong> ${
                      this.paperInfo?.paper_authors || "로딩 중..."
                    }</p>
                    <p><strong>발표:</strong> ${
                      this.paperInfo?.paper_venue || "로딩 중..."
                    }</p>
                    <p><strong>논문 ID:</strong> ${
                      this.paperInfo?.paper_id || "로딩 중..."
                    }</p>
                </div>
            </header>
            ${htmlContent}
            <footer class="paper-footer">
                <p>AI 통합 분석 시스템 | YOLOv1 논문 분석 결과</p>
                <div class="stats">
                    <div class="stat-item">
                        <span class="stat-number">${
                          this.paperInfo?.total_sections || 0
                        }</span>
                        <span class="stat-label">총 섹션</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${
                          this.paperInfo?.total_equations || 0
                        }</span>
                        <span class="stat-label">총 수식</span>
                    </div>
                </div>
            </footer>
        </div>
    </div>
</body>
</html>`;

      // Blob 생성 및 다운로드
      const blob = new Blob([fullHtml], { type: "text/html;charset=utf-8" });
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
      alert("HTML 다운로드 중 오류가 발생했습니다: " + error.message);
    } finally {
      // 버튼 상태 복원
      htmlBtn.disabled = false;
      htmlBtn.innerHTML = originalText;
    }
  }

  async convertImagesToBase64(htmlContent) {
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

          ctx.drawImage(img, 0, 0);
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
  }

  async waitForMathJax() {
    return new Promise((resolve) => {
      if (window.MathJax && typeof window.MathJax.typeset === "function") {
        window.MathJax.typeset();
        setTimeout(resolve, 1000); // MathJax 렌더링 완료 대기
      } else {
        setTimeout(resolve, 2000); // MathJax 로드 대기
      }
    });
  }

  getInlineStyles() {
    // CSS 스타일을 인라인으로 반환 (간단한 버전)
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
      .visualizations { margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745; }
      .visualization-item { margin-bottom: 25px; padding: 15px; background: white; border-radius: 6px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }
      .visualization-item h4 { color: #2c3e50; margin-bottom: 10px; font-size: 1.1em; font-weight: 600; }
      .viz-description { color: #6c757d; margin-bottom: 15px; font-size: 0.95em; line-height: 1.5; }
      .viz-image { width: 100%; max-width: 600px; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); display: block; margin: 0 auto; }
      .paper-footer { background: #2c3e50; color: white; padding: 30px; border-radius: 12px; margin-top: 30px; text-align: center; }
      .paper-footer p { margin-bottom: 15px; font-size: 1.1em; }
      .stats { display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; }
      .stat-item { text-align: center; padding: 15px; background: rgba(255, 255, 255, 0.1); border-radius: 8px; min-width: 100px; }
      .stat-number { font-size: 2em; font-weight: bold; color: #ffd700; display: block; margin-bottom: 5px; }
      .stat-label { font-size: 0.9em; opacity: 0.8; }
    `;
  }
}

// 페이지 로드 시 실행
document.addEventListener("DOMContentLoaded", () => {
  new IntegratedPaperViewer();
});

// MathJax 로드 완료 후 재렌더링
window.addEventListener("load", () => {
  if (window.MathJax) {
    if (typeof window.MathJax.typeset === "function") {
      window.MathJax.typeset();
    } else if (typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise();
    }
  }
});
