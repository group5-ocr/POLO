import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Signup() {
  const [formData, setFormData] = useState({
    nickname: "",
    job: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrors({});

    // 비밀번호 확인
    if (formData.password !== formData.confirmPassword) {
      setErrors({ confirmPassword: "비밀번호가 일치하지 않습니다." });
      setIsLoading(false);
      return;
    }

    try {
      // 백엔드 API로 회원가입 요청
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE ?? "http://localhost:8000"}/db/users`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            nickname: formData.nickname,
            email: formData.email,
            password: formData.password,
            job: formData.job,
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        alert("회원가입이 완료되었습니다!");
        navigate("/login");
      } else {
        const errorData = await response.json();
        console.error("회원가입 오류 상세:", errorData);

        // 오류 메시지를 더 명확하게 표시
        let errorMessage = "알 수 없는 오류";
        if (errorData.detail) {
          if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail
              .map((err: any) =>
                typeof err === "string" ? err : err.msg || JSON.stringify(err)
              )
              .join(", ");
          } else if (typeof errorData.detail === "string") {
            errorMessage = errorData.detail;
          } else {
            errorMessage = JSON.stringify(errorData.detail);
          }
        }

        alert(`회원가입 실패: ${errorMessage}`);
      }
    } catch (error) {
      console.error("회원가입 오류:", error);
      alert("회원가입 중 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    // 에러 메시지 초기화
    if (errors[e.target.name]) {
      setErrors({
        ...errors,
        [e.target.name]: "",
      });
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-particles">
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
      </div>
      <div className="auth-decorations">
        <div className="decoration-circle"></div>
        <div className="decoration-circle"></div>
        <div className="decoration-circle"></div>
      </div>
      <div className="auth-container">
        <div className="auth-header">
          <img
            src="/img/logo.png"
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.display = "none";
            }}
            alt="POLO 로고"
            className="auth-logo"
          />
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="nickname">닉네임</label>
            <input
              type="text"
              id="nickname"
              name="nickname"
              value={formData.nickname}
              onChange={handleChange}
              placeholder="사용할 닉네임을 입력하세요"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="job">직종</label>
            <select
              id="job"
              name="job"
              value={formData.job}
              onChange={handleChange}
              required
            >
              <option value="">직종을 선택하세요</option>
              <option value="학생">학생</option>
              <option value="연구원">연구원</option>
              <option value="교수">교수</option>
              <option value="개발자">개발자</option>
              <option value="데이터 사이언티스트">데이터 사이언티스트</option>
              <option value="AI 엔지니어">AI 엔지니어</option>
              <option value="기업인">기업인</option>
              <option value="정부기관">정부기관</option>
              <option value="의료진">의료진</option>
              <option value="법무">법무</option>
              <option value="금융">금융</option>
              <option value="교육">교육</option>
              <option value="기타">기타</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="email">이메일</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="example@email.com"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">비밀번호</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="비밀번호를 입력하세요"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="confirmPassword">비밀번호 확인</label>
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
              placeholder="비밀번호를 다시 입력하세요"
              required
            />
            {errors.confirmPassword && (
              <span className="error-message">{errors.confirmPassword}</span>
            )}
          </div>

          <button
            type="submit"
            className="btn-primary btn-full"
            disabled={isLoading}
          >
            {isLoading ? "가입 중..." : "회원가입하기"}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            이미 계정이 있으신가요?{" "}
            <button className="link-button" onClick={() => navigate("/login")}>
              로그인하기
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
