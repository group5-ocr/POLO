import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

export default function Home() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [isVisible, setIsVisible] = useState(false);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(true);

  // 페이지 로드 시 애니메이션 시작
  useEffect(() => {
    setIsVisible(true);
  }, []);

  // 스크롤 기반 애니메이션
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("animate-in");
          }
        });
      },
      { threshold: 0.1 }
    );

    const elements = document.querySelectorAll(".animate-on-scroll");
    elements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  // 자동 슬라이드 기능
  useEffect(() => {
    if (!isAutoPlaying) return;

    const interval = setInterval(() => {
      setCurrentSlide((prev) => {
        // 0, 1, 2 슬라이드만 반복 (총 3개)
        return (prev + 1) % 3;
      });
    }, 4000); // 4초마다 자동으로 넘어감

    return () => clearInterval(interval);
  }, [isAutoPlaying]);

  // 수동 슬라이드 조작 시 자동 재생 일시정지
  const handleSlideChange = (newSlide: number) => {
    setCurrentSlide(newSlide);
    setIsAutoPlaying(false);

    // 5초 후 자동 재생 재개
    setTimeout(() => {
      setIsAutoPlaying(true);
    }, 5000);
  };

  return (
    <>
      {/* 히어로 섹션 */}
      <section className="hero">
        {/* 추가 장식 요소들 */}
        <div className="hero-decorations">
          <div className="hero-decoration"></div>
          <div className="hero-decoration"></div>
          <div className="hero-decoration"></div>
          <div className="hero-decoration"></div>
        </div>

        <div className={`hero-content ${isVisible ? "animate-in" : ""}`}>
          <div className="hero-badge">
            <span className="hero-badge-icon">⭐</span>
            <span className="hero-badge-text">AI 논문 이해 도우미</span>
          </div>
          <h1 className="hero-title">
            복잡한 논문을 <span className="hero-highlight">쉬운 설명으로</span>
          </h1>
          <p className="hero-description">
            POLO는 어려운 학술 논문을 누구나 이해할 수 있는 간단한 설명으로
            변환해드립니다. 연구 내용을 더 많은 사람들과 쉽게 공유하세요.
          </p>
          <div className="hero-buttons">
            <button
              className="btn-primary hero-cta-btn"
              onClick={() => {
                if (!user) {
                  alert("로그인을 먼저 해주세요!");
                  navigate("/login");
                  return;
                }
                window.scrollTo(0, 0);
                navigate("/upload");
              }}
            >
              <span className="btn-text">논문 변환하기</span>
            </button>
          </div>
          <div className="hero-features">
            <div className="hero-feature-item">
              <span className="feature-dot"></span>
              <span>무료 체험 가능</span>
            </div>
            <div className="hero-feature-item">
              <span className="feature-dot"></span>
              <span>즉시 사용 가능</span>
            </div>
          </div>
        </div>

        <div className="hero-visual">
          <canvas className="hero-sine-canvas" />
        </div>
      </section>

      {/* 기능 소개 섹션 */}
      <section className="features">
        <div className="container">
          <div className="features-badge animate-on-scroll">핵심 기능</div>
          <h2 className="section-title animate-on-scroll">
            왜 POLO를 선택해야 할까요?
          </h2>
          <p className="features-subtitle animate-on-scroll">
            복잡한 학술 논문을 누구나 이해할 수 있게 만드는 혁신적인 기능들을
            만나보세요.
          </p>
          <div className="features-grid">
            <div className="feature-card animate-on-scroll">
              <div className="feature-icon-wrapper">
                <div className="feature-icon">🧠</div>
              </div>
              <h3>스마트 AI 분석</h3>
              <p>
                최신 AI 기술로 논문의 핵심 내용을 정확하게 파악하고 쉬운 언어로
                변환합니다.
              </p>
              <div className="feature-hover-effect"></div>
            </div>
            <div className="feature-card animate-on-scroll">
              <div className="feature-icon-wrapper">
                <div className="feature-icon">⚡</div>
              </div>
              <h3>빠른 처리 속도</h3>
              <p>
                몇 분 안에 복잡한 논문을 이해하기 쉬운 형태로 변환해드립니다.
              </p>
              <div className="feature-hover-effect"></div>
            </div>
            <div className="feature-card animate-on-scroll">
              <div className="feature-icon-wrapper">
                <div className="feature-icon">📊</div>
              </div>
              <h3>수학 설명서</h3>
              <p>
                어려운 수식과 알고리즘을 시각적으로 보여주고 쉽게 설명합니다.
              </p>
              <div className="feature-hover-effect"></div>
            </div>
            <div className="feature-card animate-on-scroll">
              <div className="feature-icon-wrapper">
                <div className="feature-icon">🛡️</div>
              </div>
              <h3>정확성 보장</h3>
              <p>원본 논문의 의미를 왜곡하지 않고 정확한 내용을 전달합니다.</p>
              <div className="feature-hover-effect"></div>
            </div>
          </div>
        </div>
      </section>

      {/* 사용법 섹션 */}
      <section className="how-it-works">
        <div className="container">
          <div className="how-it-works-badge animate-on-scroll">사용 방법</div>
          <h2 className="section-title animate-on-scroll">
            어떻게 작동하나요?
          </h2>
          <p className="how-it-works-subtitle animate-on-scroll">
            간단한 3단계로 복잡한 논문을 쉬운 설명으로 변환하세요.
          </p>
          <div className="steps">
            <div className="step animate-on-scroll">
              <div className="step-icon-wrapper">
                <div className="step-icon">📤</div>
              </div>
              <div className="step-badge">1단계</div>
              <div className="step-content">
                <h3>논문 업로드</h3>
                <p>PDF 파일을 드래그하거나 클릭해서 업로드하세요.</p>
              </div>
            </div>
            <div className="step-arrow animate-on-scroll">→</div>
            <div className="step animate-on-scroll">
              <div className="step-icon-wrapper">
                <div className="step-icon">✨</div>
              </div>
              <div className="step-badge">2단계</div>
              <div className="step-content">
                <h3>AI 분석</h3>
                <p>AI가 논문을 분석하고 이해하기 쉽게 변환합니다.</p>
              </div>
            </div>
            <div className="step-arrow animate-on-scroll">→</div>
            <div className="step animate-on-scroll">
              <div className="step-icon-wrapper">
                <div className="step-icon">✏️</div>
              </div>
              <div className="step-badge">3단계</div>
              <div className="step-content">
                <h3>결과 확인</h3>
                <p>이해하기 쉬운 논문과 수식 설명서를 확인할 수 있습니다.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 사용자 후기 섹션 */}
      <section className="testimonials">
        <div className="container">
          <h2 className="section-title animate-on-scroll">
            사용자들의 생생한 후기
          </h2>
          <p className="testimonials-subtitle animate-on-scroll">
            POLO를 사용하고 있는 연구자들의 실제 경험을 들어보세요.
          </p>
          <div className="testimonials-container">
            <div className="testimonials-slider">
              <div
                className="testimonials-track"
                style={{ transform: `translateX(-${currentSlide * 100}%)` }}
              >
                {/* 첫 번째 슬라이드 (카드 1-3) */}
                <div className="testimonials-slide">
                  <div className="testimonial-card animate-on-scroll">
                    <div className="testimonial-stars">⭐⭐⭐⭐⭐</div>
                    <div className="testimonial-content">
                      <div className="testimonial-header">
                        <div className="testimonial-info">
                          <h4>노현지</h4>
                          <p>대학원생</p>
                        </div>
                        <div className="testimonial-quote">"</div>
                      </div>
                      <p className="testimonial-text">
                        복잡한 논문들을 이해하는데 정말 많은 도움이 되었어요.
                        연구 효율성이 크게 향상되었습니다.
                      </p>
                    </div>
                  </div>
                  <div className="testimonial-card animate-on-scroll">
                    <div className="testimonial-stars">⭐⭐⭐⭐⭐</div>
                    <div className="testimonial-content">
                      <div className="testimonial-header">
                        <div className="testimonial-info">
                          <h4>이 훤</h4>
                          <p>연구원</p>
                        </div>
                        <div className="testimonial-quote">"</div>
                      </div>
                      <p className="testimonial-text">
                        POLO 덕분에 다른 분야의 논문도 쉽게 이해할 수 있게
                        되었습니다. 정말 혁신적인 서비스예요!
                      </p>
                    </div>
                  </div>
                  <div className="testimonial-card animate-on-scroll">
                    <div className="testimonial-stars">⭐⭐⭐⭐⭐</div>
                    <div className="testimonial-content">
                      <div className="testimonial-header">
                        <div className="testimonial-info">
                          <h4>손단하</h4>
                          <p>교수</p>
                        </div>
                        <div className="testimonial-quote">"</div>
                      </div>
                      <p className="testimonial-text">
                        학생들에게 논문을 설명할 때 POLO의 요약본을 활용하니
                        이해도가 훨씬 높아졌습니다.
                      </p>
                    </div>
                  </div>
                </div>

                {/* 두 번째 슬라이드 (카드 4-5) */}
                <div className="testimonials-slide">
                  <div className="testimonial-card animate-on-scroll">
                    <div className="testimonial-stars">⭐⭐⭐⭐⭐</div>
                    <div className="testimonial-content">
                      <div className="testimonial-header">
                        <div className="testimonial-info">
                          <h4>신현식</h4>
                          <p>박사과정</p>
                        </div>
                        <div className="testimonial-quote">"</div>
                      </div>
                      <p className="testimonial-text">
                        논문 리뷰 시간이 절반으로 줄어들었어요. POLO가 없었다면
                        어떻게 했을지 모르겠습니다.
                      </p>
                    </div>
                  </div>
                  <div className="testimonial-card animate-on-scroll">
                    <div className="testimonial-stars">⭐⭐⭐⭐⭐</div>
                    <div className="testimonial-content">
                      <div className="testimonial-header">
                        <div className="testimonial-info">
                          <h4>고현서</h4>
                          <p>연구소장</p>
                        </div>
                        <div className="testimonial-quote">"</div>
                      </div>
                      <p className="testimonial-text">
                        팀원들이 복잡한 연구 내용을 더 쉽게 이해할 수 있게 되어
                        프로젝트 진행이 훨씬 원활해졌습니다.
                      </p>
                    </div>
                  </div>
                  <div className="testimonial-card animate-on-scroll">
                    <div className="testimonial-stars">⭐⭐⭐⭐⭐</div>
                    <div className="testimonial-content">
                      <div className="testimonial-header">
                        <div className="testimonial-info">
                          <h4>김사과</h4>
                          <p>대학원생</p>
                        </div>
                        <div className="testimonial-quote">"</div>
                      </div>
                      <p className="testimonial-text">
                        복잡한 논문들을 이해하는데 정말 많은 도움이 되었어요.
                        연구 효율성이 크게 향상되었습니다.
                      </p>
                    </div>
                  </div>
                </div>

                {/* 세 번째 슬라이드 (카드 1-3 반복) */}
                <div className="testimonials-slide">
                  <div className="testimonial-card animate-on-scroll">
                    <div className="testimonial-stars">⭐⭐⭐⭐⭐</div>
                    <div className="testimonial-content">
                      <div className="testimonial-header">
                        <div className="testimonial-info">
                          <h4>반하나</h4>
                          <p>대학원생</p>
                        </div>
                        <div className="testimonial-quote">"</div>
                      </div>
                      <p className="testimonial-text">
                        복잡한 논문들을 이해하는데 정말 많은 도움이 되었어요.
                        연구 효율성이 크게 향상되었습니다.
                      </p>
                    </div>
                  </div>
                  <div className="testimonial-card animate-on-scroll">
                    <div className="testimonial-stars">⭐⭐⭐⭐⭐</div>
                    <div className="testimonial-content">
                      <div className="testimonial-header">
                        <div className="testimonial-info">
                          <h4>채애리</h4>
                          <p>연구원</p>
                        </div>
                        <div className="testimonial-quote">"</div>
                      </div>
                      <p className="testimonial-text">
                        POLO 덕분에 다른 분야의 논문도 쉽게 이해할 수 있게
                        되었습니다. 정말 혁신적인 서비스예요!
                      </p>
                    </div>
                  </div>
                  <div className="testimonial-card animate-on-scroll">
                    <div className="testimonial-stars">⭐⭐⭐⭐⭐</div>
                    <div className="testimonial-content">
                      <div className="testimonial-header">
                        <div className="testimonial-info">
                          <h4>이메론</h4>
                          <p>교수</p>
                        </div>
                        <div className="testimonial-quote">"</div>
                      </div>
                      <p className="testimonial-text">
                        학생들에게 논문을 설명할 때 POLO의 요약본을 활용하니
                        이해도가 훨씬 높아졌습니다.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="testimonials-controls">
              <button
                className="testimonial-btn prev-btn"
                onClick={() => handleSlideChange(Math.max(0, currentSlide - 1))}
                disabled={currentSlide === 0}
              >
                ‹
              </button>
              <div className="testimonials-dots">
                {[0, 1, 2].map((index) => (
                  <button
                    key={index}
                    className={`testimonial-dot ${
                      currentSlide === index ? "active" : ""
                    }`}
                    onClick={() => handleSlideChange(index)}
                  />
                ))}
              </div>
              <button
                className="testimonial-btn next-btn"
                onClick={() => handleSlideChange(Math.min(2, currentSlide + 1))}
                disabled={currentSlide === 2}
              >
                ›
              </button>
              <button
                className={`testimonial-btn play-pause-btn ${
                  isAutoPlaying ? "playing" : "paused"
                }`}
                onClick={() => setIsAutoPlaying(!isAutoPlaying)}
                title={isAutoPlaying ? "자동 재생 일시정지" : "자동 재생 시작"}
              >
                {isAutoPlaying ? "⏸" : "▶"}
              </button>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
