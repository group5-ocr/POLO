import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'

export default function Home() {
  const heroStageRef = useRef<HTMLDivElement | null>(null)
  const scannerCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const graphCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const overlayRef = useRef<HTMLDivElement | null>(null)

  // 히어로 스캐너(바운딩 박스)
  useEffect(() => {
    const canvas = scannerCanvasRef.current
    const stage = heroStageRef.current
    if (!canvas || !stage) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const scanner = { x: 80, y: 60, w: 220, h: 150, vx: 2.8, vy: 2.2, margin: 16 }
    const resize = () => {
      const { clientWidth: w, clientHeight: h } = stage
      canvas.width = w
      canvas.height = h
      scanner.w = Math.min(300, w * 0.28)
      scanner.h = Math.min(210, h * 0.35)
      scanner.x = Math.min(Math.max(scanner.margin, scanner.x), w - scanner.w - scanner.margin)
      scanner.y = Math.min(Math.max(scanner.margin, scanner.y), h - scanner.h - scanner.margin)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(stage)

    let t = 0
    let raf = 0
    const draw = () => {
      const w = canvas.width
      const h = canvas.height
      const ctx2 = ctx
      ctx2.clearRect(0, 0, w, h)
      scanner.x += scanner.vx
      scanner.y += scanner.vy
      if (scanner.x <= scanner.margin || scanner.x + scanner.w >= w - scanner.margin) scanner.vx *= -1
      if (scanner.y <= scanner.margin || scanner.y + scanner.h >= h - scanner.margin) scanner.vy *= -1
      const pulse = (Math.sin(t * 0.06) + 1) / 2
      ctx2.lineWidth = 3
      ctx2.setLineDash([10, 10])
      ctx2.strokeStyle = `rgba(34,197,94,${0.55 + pulse * 0.35})`
      ctx2.strokeRect(scanner.x, scanner.y, scanner.w, scanner.h)
      ctx2.setLineDash([])
      const len = 22
      ctx2.lineWidth = 4
      ctx2.strokeStyle = '#22c55e'
      line(ctx2, scanner.x, scanner.y, scanner.x + len, scanner.y)
      line(ctx2, scanner.x, scanner.y, scanner.x, scanner.y + len)
      line(ctx2, scanner.x + scanner.w, scanner.y, scanner.x + scanner.w - len, scanner.y)
      line(ctx2, scanner.x + scanner.w, scanner.y, scanner.x + scanner.w, scanner.y + len)
      line(ctx2, scanner.x, scanner.y + scanner.h, scanner.x + len, scanner.y + scanner.h)
      line(ctx2, scanner.x, scanner.y + scanner.h, scanner.x, scanner.y + scanner.h - len)
      line(ctx2, scanner.x + scanner.w, scanner.y + scanner.h, scanner.x + scanner.w - len, scanner.y + scanner.h)
      line(ctx2, scanner.x + scanner.w, scanner.y + scanner.h, scanner.x + scanner.w, scanner.y + scanner.h - len)
      t += 1
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => { cancelAnimationFrame(raf); ro.disconnect() }
  }, [])

  // 그래프(동적 라인 + 토스트)
  useEffect(() => {
    const canvas = graphCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const overlay = overlayRef.current
    const sampleTitles = [
      'Contrastive Vision-Language Pretraining for...' ,
      'Retrieval-Augmented Generation in Practice',
      'Efficient QLoRA Fine-tuning for 7B Models',
      'Visualizing Transformers: Attention Graphs',
      'Mathematical Reasoning with LLMs',
      'Graph-based Vector Indexing 2024',
    ]
    const resize = () => {
      canvas.width = canvas.clientWidth
      canvas.height = 360
    }
    resize()
    const onResize = () => resize()
    window.addEventListener('resize', onResize)

    let t = 0
    let lastTextTime = 0
    let raf = 0
    const draw = () => {
      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)
      ctx.strokeStyle = '#2563eb'
      ctx.lineWidth = 3
      ctx.beginPath()
      for (let x = 0; x < w; x++) {
        const y = h / 2 + Math.sin((x + t) * 0.02) * 46 + Math.cos((x + t) * 0.01) * 26
        if (x === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
      t += 2

      const now = performance.now()
      if (overlay && now - lastTextTime > 1200) {
        lastTextTime = now
        const burstCount = 2 + Math.floor(Math.random() * 3)
        for (let i = 0; i < burstCount; i++) {
          const el = document.createElement('div')
          el.className = 'graph-toast color-' + ((Math.floor(Math.random() * 6) % 6) + 1)
          el.textContent = `${rndDate()} · ${sampleTitles[Math.floor(Math.random() * sampleTitles.length)]}`
          overlay.appendChild(el)
          el.style.left = Math.floor(Math.random() * 80 + 10) + '%'
          el.style.top = Math.floor(Math.random() * 60 + 20) + '%'
          setTimeout(() => {
            el.classList.add('hide')
            setTimeout(() => el.remove(), 600)
          }, 1600 + Math.random() * 500)
        }
        const items = overlay.querySelectorAll('.graph-toast')
        if (items.length > 6) {
          for (let i = 0; i < items.length - 6; i++) {
            items[i].classList.add('hide')
            setTimeout(() => items[i].remove(), 400)
          }
        }
      }

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)

    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', onResize) }
  }, [])

  const navigate = useNavigate()

  return (
    <>
      <section className="hero">
        <div className="hero-inner">
          <div className="slogan">
            <div className="slogan-logo">
              <img src="/img/logo.png" onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none' }} alt="logo" />
            </div>
            <p className="tagline strong">논문을 시각적·쉬운 언어로 바꿔주는 AI</p>
          </div>
          <div className="hero-stage" ref={heroStageRef}>
            <div className="carousel-track">
              <img className="slide" src="/img/main1.png" alt="visual 1" />
              <img className="slide" src="/img/main2.png" alt="visual 2" />
              <img className="slide" src="/img/main3.png" alt="visual 3" />
              <img className="slide" src="/img/main4.png" alt="visual 4" />
              <img className="slide" src="/img/main1.png" alt="visual 1 duplicate" />
              <img className="slide" src="/img/main2.png" alt="visual 2 duplicate" />
            </div>
            <canvas className="scanner" ref={scannerCanvasRef} />
          </div>
        </div>
        <div className="scroll-cue">스크롤</div>
      </section>

      <section className="graphs">
        <h2>Vector DB: 2023.01 ~ 2024.12 수집 논문 흐름</h2>
        <div className="graph-stage">
          <canvas className="graph-canvas" ref={graphCanvasRef} />
          <div className="graph-overlay" ref={overlayRef}></div>
        </div>
      </section>

      <section className="cta-pipeline">
        <div className="cta-inner">
          <div className="pipeline-demo">
            <div className="demo-left">
              <div className="upload-demo">
                <div className="upload-area">
                  <img className="upload-icon" src="/img/upload.png" alt="upload" />
                  <div className="pdf-icon">📄</div>
                </div>
                <div className="result-pages">
                  <img className="result-page" src="/img/easy_page.png" alt="easy version" />
                  <img className="result-page" src="/img/math_page.png" alt="math guide" />
                </div>
              </div>
            </div>
            <div className="demo-right">
              <h3>수학 보조 설명서입니다</h3>
              <p>논문의 어려운 수학 개념과 공식을<br />직관적으로 이해할 수 있도록<br />시각화와 쉬운 설명을 제공합니다.</p>
              <button className="cta-button" onClick={() => navigate('/upload')}>논문 변환하러 가기</button>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}

function line(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x2, y2)
  ctx.stroke()
}

function rndDate() {
  const start = new Date(2023, 0, 1).getTime()
  const end = new Date(2024, 11, 31).getTime()
  const d = new Date(start + Math.random() * (end - start))
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  return `${y}.${m}`
}
