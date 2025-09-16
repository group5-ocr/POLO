import React, { useState, useEffect } from "react";

interface DatabaseStatusProps {
  className?: string;
}

interface DatabaseStats {
  users: number;
  documents: number;
  processing_jobs: number;
  model_results: number;
  jobs_by_status: Record<string, number>;
}

export default function DatabaseStatus({
  className = "",
}: DatabaseStatusProps) {
  const [stats, setStats] = useState<DatabaseStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDatabaseStats = async () => {
    setLoading(true);
    setError(null);

    try {
      const apiBase = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
      console.log("[DatabaseStatus] API Base URL:", apiBase);
      
      const response = await fetch(`${apiBase}/db/stats`);

      if (response.ok) {
        const data = await response.json();
        setStats(data.database_stats);
      } else {
        setError("데이터베이스 통계를 가져올 수 없습니다.");
      }
    } catch (err) {
      setError("데이터베이스 연결에 실패했습니다.");
      console.error("Database stats error:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatabaseStats();
  }, []);

  if (loading) {
    return (
      <div className={`database-status ${className}`}>
        <div className="status-loading">데이터베이스 상태 확인 중...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`database-status ${className}`}>
        <div className="status-error">❌ {error}</div>
        <button onClick={fetchDatabaseStats} className="retry-btn">
          다시 시도
        </button>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className={`database-status ${className}`}>
        <div className="status-no-data">데이터를 불러올 수 없습니다.</div>
      </div>
    );
  }

  return (
    <div className={`database-status ${className}`}>
      <h3>📊 데이터베이스 상태</h3>
      <div className="stats-grid">
        <div className="stat-item">
          <span className="stat-label">사용자</span>
          <span className="stat-value">{stats.users}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">문서</span>
          <span className="stat-value">{stats.documents}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">처리 작업</span>
          <span className="stat-value">{stats.processing_jobs}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">모델 결과</span>
          <span className="stat-value">{stats.model_results}</span>
        </div>
      </div>

      {Object.keys(stats.jobs_by_status).length > 0 && (
        <div className="status-breakdown">
          <h4>작업 상태별 분포</h4>
          <div className="status-list">
            {Object.entries(stats.jobs_by_status).map(([status, count]) => (
              <div key={status} className="status-item">
                <span className="status-name">{status}</span>
                <span className="status-count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <button onClick={fetchDatabaseStats} className="refresh-btn">
        🔄 새로고침
      </button>
    </div>
  );
}



