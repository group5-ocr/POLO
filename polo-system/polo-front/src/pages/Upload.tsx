export default function Upload() {
  const onChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    if (f.type !== 'application/pdf') {
      alert('PDF 파일만 업로드할 수 있습니다.')
      e.currentTarget.value = ''
      return
    }
    if (f.size > 50 * 1024 * 1024) {
      alert('파일은 50MB 이하만 가능합니다.')
      e.currentTarget.value = ''
      return
    }
    alert('업로드 준비 완료: ' + f.name)
  }

  return (
    <div className="page upload">
      <h2>논문 업로드 (PDF 전용)</h2>
      <label className="upload-box">
        <input type="file" accept="application/pdf" onChange={onChange} />
        <div>여기를 클릭하거나 파일을 드래그하여 PDF를 업로드하세요.</div>
      </label>
    </div>
  )
}
