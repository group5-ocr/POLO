import { Routes, Route, Link, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Upload from "./pages/Upload";
import Result from "./pages/Result";

function AppContent() {
  const location = useLocation();
  const { user, logout } = useAuth();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location]);

  // 모바일 메뉴 토글
  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  // 메뉴 클릭 시 자동으로 닫기
  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  // 배경 클릭 시 메뉴 닫기
  const handleMobileMenuBackdrop = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      closeMobileMenu();
    }
  };

  // ESC 키로 메뉴 닫기
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isMobileMenuOpen) {
        closeMobileMenu();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [isMobileMenuOpen]);

  // 모바일 메뉴 열릴 때 body 스크롤 방지
  useEffect(() => {
    if (isMobileMenuOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }

    // 컴포넌트 언마운트 시 스크롤 복원
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isMobileMenuOpen]);

  return (
    <div className="app-root">
      <header className="header">
        <div className="header-inner">
          <div className="brand-wrap">
            <Link to="/" className="brand-link" onClick={closeMobileMenu}>
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

          {/* 데스크톱 네비게이션 */}
          <nav className="nav-auth desktop-nav">
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

          {/* 모바일 햄버거 메뉴 버튼 */}
          <button
            className="mobile-menu-toggle"
            onClick={toggleMobileMenu}
            aria-label="메뉴 열기/닫기"
          >
            <span className={`hamburger ${isMobileMenuOpen ? "active" : ""}`}>
              <span></span>
              <span></span>
              <span></span>
            </span>
          </button>
        </div>

        {/* 모바일 메뉴 */}
        <div
          className={`mobile-menu ${isMobileMenuOpen ? "open" : ""}`}
          onClick={handleMobileMenuBackdrop}
        >
          <div className="mobile-menu-content">
            {user ? (
              <>
                <div className="mobile-user-info">
                  <span className="mobile-user-greeting">
                    안녕하세요! {user.nickname}님
                  </span>
                </div>
                <div className="mobile-menu-actions">
                  <Link
                    className="mobile-menu-link"
                    to="/upload"
                    onClick={closeMobileMenu}
                  >
                    논문 변환하기
                  </Link>
                  <button
                    className="mobile-menu-button logout"
                    onClick={() => {
                      logout();
                      closeMobileMenu();
                    }}
                  >
                    로그아웃
                  </button>
                </div>
              </>
            ) : (
              <div className="mobile-menu-actions">
                <Link
                  className="mobile-menu-link"
                  to="/login"
                  onClick={closeMobileMenu}
                >
                  로그인
                </Link>
                <Link
                  className="mobile-menu-button primary"
                  to="/signup"
                  onClick={closeMobileMenu}
                >
                  회원가입
                </Link>
              </div>
            )}
          </div>
        </div>
      </header>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/result" element={<Result />} />
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
