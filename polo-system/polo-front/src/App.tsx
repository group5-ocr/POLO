import { Routes, Route, Link } from 'react-router-dom'
import Home from './pages/Home'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Upload from './pages/Upload'

export default function App() {
  return (
    <div className="app-root">
      <header className="header">
        <div className="header-inner small">
          <div className="brand-wrap">
            <Link to="/" className="brand-link">
              <img className="brand-logo" src="/img/head_logo.png" onError={(e) => { (e.currentTarget as HTMLImageElement).src = '/img/logo.png' }} alt="A!POLO" />
            </Link>
          </div>
          <div className="brand-center">
            <span className="brand-text">A!POLO</span>
          </div>
          <nav className="nav-auth">
            <Link className="link" to="/login">로그인</Link>
            <Link className="btn-primary" to="/signup">회원가입</Link>
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
          <div className="logo">A!POLO</div>
          <div className="copy">© {new Date().getFullYear()} A!POLO. All Rights Reserved.</div>
        </div>
      </footer>
    </div>
  )
}

function line(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x2, y2)
  ctx.stroke()
}

// 유틸: 랜덤 날짜 문자열 (2023.01 ~ 2024.12)
function rndDate() {
  const start = new Date(2023, 0, 1).getTime()
  const end = new Date(2024, 11, 31).getTime()
  const d = new Date(start + Math.random() * (end - start))
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  return `${y}.${m}`
}


