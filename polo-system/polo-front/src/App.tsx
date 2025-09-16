import { Routes, Route, Link, useLocation } from "react-router-dom";
import { useEffect } from "react";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Upload from "./pages/Upload";

function AppContent() {
  const location = useLocation();
  const { user, logout } = useAuth();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location]);

  return (
    <div className="app-root">
      <header className="header">
        <div className="header-inner small">
          <div className="brand-wrap">
            <Link to="/" className="brand-link">
              <img
                className="brand-logo-main"
                src="/img/head_logo.png"
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).style.display = "none";
                }}
                alt="POLO"
              />
              <img
                className="brand-logo"
                src="/img/logo.png"
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).style.display = "none";
                }}
                alt="A!POLO"
              />
            </Link>
          </div>
          <div className="brand-center">
            <span className="brand-text">A!POLO</span>
          </div>
          <nav className="nav-auth">
            {user ? (
              <>
                <span className="user-greeting">
                  안녕하세요! {user.nickname}님
                </span>
                <button className="btn-secondary" onClick={logout}>
                  로그아웃
                </button>
              </>
            ) : (
              <>
                <Link className="link" to="/login">
                  로그인
                </Link>
                <Link className="btn-primary" to="/signup">
                  회원가입
                </Link>
              </>
            )}
          </nav>
        </div>
      </header>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/upload" element={<Upload />} />
      </Routes>

      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-content">
            <div className="footer-brand">
              <div className="footer-logo">A!POLO</div>
              <p className="footer-description">
                복잡한 학술 논문을 누구나 이해할 수 있게 만드는 AI 기반 플랫폼
              </p>
            </div>
            <div className="footer-links">
              <div className="footer-column">
                <h4>서비스</h4>
                <ul>
                  <li>
                    <a href="#features">기능 소개</a>
                  </li>
                  <li>
                    <a href="#how-it-works">사용 방법</a>
                  </li>
                  <li>
                    <a href="#testimonials">사용자 후기</a>
                  </li>
                </ul>
              </div>
              <div className="footer-column">
                <h4>지원</h4>
                <ul>
                  <li>
                    <a href="#help">도움말</a>
                  </li>
                  <li>
                    <a href="#contact">문의하기</a>
                  </li>
                  <li>
                    <a href="#faq">자주 묻는 질문</a>
                  </li>
                </ul>
              </div>
              <div className="footer-column">
                <h4>회사</h4>
                <ul>
                  <li>
                    <a href="#about">회사 소개</a>
                  </li>
                  <li>
                    <a href="#privacy">개인정보처리방침</a>
                  </li>
                  <li>
                    <a href="#terms">이용약관</a>
                  </li>
                </ul>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <div className="footer-copy">
              © {new Date().getFullYear()} A!POLO. All Rights Reserved.
            </div>
            <div className="footer-social">
              <a href="#" className="social-link">
                📧
              </a>
              <a href="#" className="social-link">
                🐦
              </a>
              <a href="#" className="social-link">
                📘
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

function line(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number
) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

// 유틸: 랜덤 날짜 문자열 (2023.01 ~ 2024.12)
function rndDate() {
  const start = new Date(2023, 0, 1).getTime();
  const end = new Date(2024, 11, 31).getTime();
  const d = new Date(start + Math.random() * (end - start));
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  return `${y}.${m}`;
}
