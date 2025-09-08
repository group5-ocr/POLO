export default function Login() {
  return (
    <div className="page auth">
      <h2>로그인</h2>
      <form className="form" onSubmit={(e)=>e.preventDefault()}>
        <label>아이디<input type="text" required /></label>
        <label>비밀번호<input type="password" required /></label>
        <button className="btn-primary" type="submit">로그인</button>
      </form>
    </div>
  )
}
