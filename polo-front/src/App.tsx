import { useEffect, useRef } from 'react'

function App() {
  // 히어로 섹션 돋보기(캔버스)
  const magnifierCanvasRef = useRef<HTMLCanvasElement | null>(null)
  // 그래프 섹션 캔버스
  const graphCanvasRef = useRef<HTMLCanvasElement | null>(null)

  // 히어로 섹션: 돋보기 애니메이션
  useEffect(() => {
    const canvas = magnifierCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resize = () => {
      canvas.width = canvas.clientWidth
      canvas.height = canvas.clientHeight
    }
    resize()
    window.addEventListener('resize', resize)

    let t = 0
    function draw() {
      const width = canvas.width
      const height = canvas.height
      ctx.clearRect(0, 0, width, height)

      // 문서 영역 비주얼(가상 좌표)
      const docWidth = Math.min(260, width * 0.35)
      const docHeight = Math.min(340, height * 0.5)
      const docX = width / 2 - docWidth / 2
      const docY = height / 2 - docHeight / 2

      // 내부 라인
      ctx.strokeStyle = '#e5e7eb'
      ctx.lineWidth = 3
      for (let i = 0; i < 7; i++) {
        const lx = docX + 20
        const ly = docY + 40 + i * 28
        const lw = docWidth - 40 - (i % 3) * 20
        ctx.beginPath()
        ctx.moveTo(lx, ly)
        ctx.lineTo(lx + lw, ly)
        ctx.stroke()
      }

      // 돋보기 궤적
      const radius = Math.max(70, Math.min(width, height) * 0.12)
      const cx = docX + docWidth / 2 + Math.sin(t * 0.004) * (docWidth * 0.35)
      const cy = docY + docHeight / 2 + Math.cos(t * 0.0035) * (docHeight * 0.3)

      // 확대 효과: 라인을 조금 더 진하게 그리기
      ctx.save()
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, Math.PI * 2)
      ctx.clip()
      ctx.strokeStyle = '#94a3b8'
      ctx.lineWidth = 3.2
      for (let i = 0; i < 7; i++) {
        const lx = docX + 20
        const ly = docY + 40 + i * 28
        const lw = docWidth - 40 - (i % 3) * 20
        ctx.beginPath()
        ctx.moveTo(lx, ly)
        ctx.lineTo(lx + lw, ly)
        ctx.stroke()
      }
      ctx.restore()

      // 돋보기 테두리 + 손잡이
      ctx.strokeStyle = '#60a5fa'
      ctx.lineWidth = 4
      ctx.beginPath()
      ctx.arc(cx, cy, radius, 0, Math.PI * 2)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(cx + radius * 0.7, cy + radius * 0.7)
      ctx.lineTo(cx + radius * 1.2, cy + radius * 1.2)
      ctx.stroke()

      // 문서 바운딩 박스
      const pulse = (Math.sin(t * 0.01) + 1) / 2
      ctx.strokeStyle = `rgba(34,197,94,${0.5 + pulse * 0.5})`
      ctx.lineWidth = 3
      dashedRect(ctx, docX - 8, docY - 8, docWidth + 16, docHeight + 16, 10, 12)

      t += 1
      requestAnimationFrame(draw)
    }
    draw()

    return () => window.removeEventListener('resize', resize)
  }, [])

  // 그래프 섹션: 간단한 애니메이션 라인 그래프
  useEffect(() => {
    const canvas = graphCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const resize = () => {
      canvas.width = canvas.clientWidth
      canvas.height = 320
    }
    resize()
    window.addEventListener('resize', resize)
    let t = 0
    function draw() {
      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)
      ctx.strokeStyle = '#60a5fa'
      ctx.lineWidth = 3
      ctx.beginPath()
      for (let x = 0; x < w; x++) {
        const y = h / 2 + Math.sin((x + t) * 0.02) * 40 + Math.cos((x + t) * 0.01) * 20
        if (x === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
      t += 2
      requestAnimationFrame(draw)
    }
    draw()
    return () => window.removeEventListener('resize', resize)
  }, [])

  return (
    <div className="app-root">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-inner">
          <h1 className="brand">A!POLO</h1>
          <p className="tagline">논문을 시각적·쉬운 언어로 풀어주는 AI</p>

          <div className="hero-stage">
            {/* 배경 이미지 4종 */}
            <img className="hero-img i1" src="/img/main1.png" alt="visual 1" />
            <img className="hero-img i2" src="/img/main2.png" alt="visual 2" />
            <img className="hero-img i3" src="/img/main3.png" alt="visual 3" />
            <img className="hero-img i4" src="/img/main4.png" alt="visual 4" />

            {/* 돋보기 캔버스 */}
            <canvas className="magnifier" ref={magnifierCanvasRef} />
          </div>
        </div>
        <div className="scroll-cue">스크롤</div>
      </section>

      {/* Graph Section */}
      <section className="graphs">
        <h2>설명 결과 미리보기 그래프</h2>
        <canvas className="graph-canvas" ref={graphCanvasRef} />
      </section>

      {/* Upload-like Section */}
      <section className="upload-mimic">
        <h2>이미지/논문 업로드</h2>
        <div className="upload-box">
          <div className="upload-illustration">Drag & Drop or Click</div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="logo">A!POLO</div>
          <div className="copy">© {new Date().getFullYear()} A!POLO. All Rights Reserved.</div>
        </div>
      </footer>
    </div>
  )
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  const rr = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.arcTo(x + w, y, x + w, y + h, rr)
  ctx.arcTo(x + w, y + h, x, y + h, rr)
  ctx.arcTo(x, y + h, x, y, rr)
  ctx.arcTo(x, y, x + w, y, rr)
  ctx.closePath()
}

function dashedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  dash: number,
  gap: number
) {
  ctx.setLineDash([dash, gap])
  ctx.strokeRect(x, y, w, h)
  ctx.setLineDash([])
}

export default App


