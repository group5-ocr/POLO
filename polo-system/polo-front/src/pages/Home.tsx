import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();
  const graphCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // 페이지 로드 시 애니메이션 시작
  useEffect(() => {
    setIsVisible(true);
  }, []);

  // 마우스 위치 추적
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
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

  // 애니메이션된 사인함수 그래프 그리기
  useEffect(() => {
    const canvas = graphCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let time = 0;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };

    const animate = () => {
      time += 0.08;
      drawAnimatedGraph(ctx, canvas.width, canvas.height, time);
      animationId = requestAnimationFrame(animate);
    };

    resizeCanvas();
    animate();
    window.addEventListener("resize", resizeCanvas);

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, []);

  return (
    <>
      {/* 히어로 섹션 */}
      <section className="hero">
        {/* 파티클 배경 */}
        <div className="particles">
          {[...Array(50)].map((_, i) => (
            <div
              key={i}
              className="particle"
              style={{
                left: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 10}s`,
                animationDuration: `${5 + Math.random() * 10}s`,
              }}
            />
          ))}
        </div>

        {/* 마우스 추적 그라데이션 */}
        <div
          className="mouse-gradient"
          style={{
            background: `radial-gradient(600px circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(167, 243, 208, 0.15), transparent 40%)`,
          }}
        />

        <div className={`hero-content ${isVisible ? "animate-in" : ""}`}>
          <div className="hero-logo">
            <img
              src="/img/head_logo.png"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = "none";
              }}
              alt="POLO 로고"
            />
          </div>
          <p className="hero-description">
            <span className="hero-text-line">
              복잡한 AI 논문을 누구나 이해할 수 있도록
            </span>
            <span className="hero-text-line">
              시각적이고 쉬운 언어로 변환해주는 AI 서비스
            </span>
            <span className="hero-cta-text">
              지금 바로 시작해보세요!
              <br />
              복잡한 AI 논문도 POLO와 함께라면 쉽게 이해할 수 있어요
            </span>
          </p>
          <div className="hero-buttons">
            <button
              className="btn-primary hero-cta-btn"
              onClick={() => {
                window.scrollTo(0, 0);
                navigate("/upload");
              }}
            >
              <span className="btn-text">논문 변환하기</span>
            </button>
          </div>
        </div>

        <div className="hero-visual">
          <canvas className="hero-sine-canvas" ref={graphCanvasRef} />
          <div className="floating-card card-1">
            <div className="card-icon">📊</div>
            <div className="card-text">데이터 분석</div>
            <div className="card-glow"></div>
          </div>
          <div className="floating-card card-2">
            <div className="card-icon">⚡</div>
            <div className="card-text">AI 처리</div>
            <div className="card-glow"></div>
          </div>
          <div className="floating-card card-3">
            <div className="card-icon">💡</div>
            <div className="card-text">쉬운 설명</div>
            <div className="card-glow"></div>
          </div>
          <div className="floating-card card-4">
            <div className="card-icon">🎯</div>
            <div className="card-text">수학 시각화</div>
            <div className="card-glow"></div>
          </div>
        </div>
      </section>

      {/* Vector DB 수집 논문 흐름 섹션 */}
      <section className="graphs">
        <h2>Vector DB: 2023.01 ~ 2024.12 수집 논문 흐름</h2>
        <div className="graph-stage">
          <canvas className="graph-canvas" ref={graphCanvasRef} />
          <div className="graph-overlay" ref={overlayRef}></div>
        </div>
      </section>

      {/* 기능 소개 섹션 */}
      <section className="features">
        <div className="container">
          <h2 className="section-title animate-on-scroll">
            POLO의 특별한 기능들
          </h2>
          <div className="features-grid">
            <div className="feature-card animate-on-scroll">
              <div className="feature-icon">📖</div>
              <h3>쉬운 설명서</h3>
              <p>
                복잡한 AI 논문을 누구나 이해할 수 있도록 일상 언어로
                변환해드려요!
              </p>
              <div className="feature-hover-effect"></div>
            </div>
            <div className="feature-card animate-on-scroll">
              <div className="feature-icon">📊</div>
              <h3>수학 설명서</h3>
              <p>
                어려운 수식과 알고리즘을 시각적으로 보여주고 쉽게 설명해드려요!
              </p>
              <div className="feature-hover-effect"></div>
            </div>
            <div className="feature-card animate-on-scroll">
              <div className="feature-icon">🚀</div>
              <h3>빠른 처리</h3>
              <p>PDF 업로드만 하면 몇 초 만에 결과를 받아볼 수 있어요!</p>
              <div className="feature-hover-effect"></div>
            </div>
          </div>
        </div>
      </section>

      {/* 사용법 섹션 */}
      <section className="how-it-works">
        <div className="container">
          <h2 className="section-title animate-on-scroll">
            간단한 3단계 프로세스
          </h2>
          <div className="steps">
            <div className="step animate-on-scroll">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>논문 업로드</h3>
                <p>PDF 파일을 드래그하거나 클릭해서 업로드하세요</p>
              </div>
              <div className="step-glow"></div>
            </div>
            <div className="step-arrow animate-on-scroll">→</div>
            <div className="step animate-on-scroll">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>AI 분석</h3>
                <p>POLO가 논문을 분석하고 이해하기 쉽게 변환해드려요</p>
              </div>
              <div className="step-glow"></div>
            </div>
            <div className="step-arrow animate-on-scroll">→</div>
            <div className="step animate-on-scroll">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>결과 확인</h3>
                <p>쉬운 설명서와 수학 설명서를 받아보세요!</p>
              </div>
              <div className="step-glow"></div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

// 애니메이션된 사인함수 그래프 그리기 함수
function drawAnimatedGraph(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  time: number
) {
  // 캔버스 초기화
  ctx.clearRect(0, 0, width, height);

  // 배경 그라데이션
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "rgba(167, 243, 208, 0.1)");
  gradient.addColorStop(0.5, "rgba(191, 219, 254, 0.15)");
  gradient.addColorStop(1, "rgba(224, 231, 255, 0.1)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  // 격자 그리기 (더 연하게)
  ctx.strokeStyle = "rgba(167, 243, 208, 0.05)";
  ctx.lineWidth = 1;

  // 세로 격자
  for (let i = 0; i <= 20; i++) {
    const x = (width / 20) * i;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  // 가로 격자
  for (let i = 0; i <= 10; i++) {
    const y = (height / 10) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // 애니메이션된 사인함수 그리기
  const centerY = height / 2;
  const amplitude = height * 0.25;
  const frequency = 0.015;
  const waveSpeed = 0.08;

  // 메인 사인파
  ctx.strokeStyle = "#a7f3d0";
  ctx.lineWidth = 4;
  ctx.globalAlpha = 0.8;
  ctx.beginPath();

  for (let x = 0; x < width; x += 1) {
    const y = centerY + Math.sin(x * frequency + time * waveSpeed) * amplitude;
    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  // 두 번째 사인파 (더 작은 진폭)
  ctx.strokeStyle = "#bfdbfe";
  ctx.lineWidth = 3;
  ctx.globalAlpha = 0.6;
  ctx.beginPath();

  for (let x = 0; x < width; x += 1) {
    const y =
      centerY +
      Math.sin(x * frequency * 1.3 + time * waveSpeed * 1.5) *
        (amplitude * 0.7);
    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  // 세 번째 사인파 (가장 작은 진폭)
  ctx.strokeStyle = "#e0e7ff";
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.4;
  ctx.beginPath();

  for (let x = 0; x < width; x += 1) {
    const y =
      centerY +
      Math.sin(x * frequency * 0.7 + time * waveSpeed * 0.8) *
        (amplitude * 0.5);
    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  ctx.globalAlpha = 1;
}
