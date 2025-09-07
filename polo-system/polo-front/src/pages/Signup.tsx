export default function Signup() {
  return (
    <div className="page auth">
      <h2>회원가입</h2>
      <form className="form" onSubmit={(e)=>e.preventDefault()}>
        <label>이름<input type="text" required /></label>
        <label>직종<input type="text" required /></label>
        <label>휴대폰 번호<input type="tel" pattern="^\\d{3}-?\\d{3,4}-?\\d{4}$" placeholder="010-1234-5678" required /></label>
        <label>아이디<input type="text" required /></label>
        <label>비밀번호<input type="password" required /></label>
        <button className="btn-primary" type="submit">가입하기</button>
      </form>
    </div>
  )
}
