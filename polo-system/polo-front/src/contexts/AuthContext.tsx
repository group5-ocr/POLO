import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

interface User {
  user_id: number;
  email: string;
  nickname: string;
  job: string;
  created_at: string;
}

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // 앱 시작 시 로그인 상태 확인
  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const token = localStorage.getItem("authToken");
      if (token) {
        // 토큰이 있으면 사용자 정보 조회
        const response = await fetch(
          `${
            import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
          }/db/users`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (response.ok) {
          const data = await response.json();
          // 첫 번째 사용자를 현재 사용자로 설정 (실제로는 토큰에서 사용자 ID를 추출해야 함)
          if (data.users && data.users.length > 0) {
            setUser(data.users[0]);
          }
        }
      }
    } catch (error) {
      console.error("인증 상태 확인 실패:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      setIsLoading(true);

      // 로그인 API 호출
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE ?? "http://localhost:8000"}/db/login`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            email: email,
            password: password,
          }),
        }
      );

      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
        localStorage.setItem("authToken", "dummy-token"); // 실제로는 JWT 토큰
        return true;
      } else {
        const errorData = await response.json();
        console.error("로그인 실패:", errorData.detail);
        return false;
      }
    } catch (error) {
      console.error("로그인 실패:", error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem("authToken");
  };

  const value: AuthContextType = {
    user,
    login,
    logout,
    isLoading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
