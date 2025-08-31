-- Gerrit開発者定着予測システム - データベース初期化スクリプト

-- 拡張機能の有効化
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 開発者テーブル
CREATE TABLE IF NOT EXISTS developers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    expertise_level FLOAT DEFAULT 0.0,
    stress_level FLOAT DEFAULT 0.0,
    activity_pattern JSONB,
    collaboration_quality FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- レビューテーブル
CREATE TABLE IF NOT EXISTS reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    change_id VARCHAR(255) NOT NULL,
    reviewer_id UUID REFERENCES developers(id),
    author_id UUID REFERENCES developers(id),
    score INTEGER,
    response_time_hours FLOAT,
    complexity_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 予測結果テーブル
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    developer_id UUID REFERENCES developers(id),
    prediction_type VARCHAR(50) NOT NULL, -- 'retention', 'stress', 'boiling_point'
    prediction_value FLOAT NOT NULL,
    confidence_score FLOAT,
    model_version VARCHAR(50),
    features JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- メトリクステーブル
CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20),
    tags JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- アラートテーブル
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    developer_id UUID REFERENCES developers(id),
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'resolved', 'suppressed'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- 設定変更履歴テーブル
CREATE TABLE IF NOT EXISTS config_changes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(255) NOT NULL,
    old_value JSONB,
    new_value JSONB,
    change_type VARCHAR(20) NOT NULL, -- 'added', 'modified', 'deleted'
    impact_level VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    description TEXT,
    changed_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- インデックスの作成
CREATE INDEX IF NOT EXISTS idx_developers_email ON developers(email);
CREATE INDEX IF NOT EXISTS idx_developers_updated_at ON developers(updated_at);

CREATE INDEX IF NOT EXISTS idx_reviews_change_id ON reviews(change_id);
CREATE INDEX IF NOT EXISTS idx_reviews_reviewer_id ON reviews(reviewer_id);
CREATE INDEX IF NOT EXISTS idx_reviews_author_id ON reviews(author_id);
CREATE INDEX IF NOT EXISTS idx_reviews_created_at ON reviews(created_at);

CREATE INDEX IF NOT EXISTS idx_predictions_developer_id ON predictions(developer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);

CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_tags ON metrics USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_alerts_developer_id ON alerts(developer_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

CREATE INDEX IF NOT EXISTS idx_config_changes_key ON config_changes(config_key);
CREATE INDEX IF NOT EXISTS idx_config_changes_created_at ON config_changes(created_at);

-- トリガー関数：updated_atの自動更新
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- トリガーの作成
CREATE TRIGGER update_developers_updated_at 
    BEFORE UPDATE ON developers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_reviews_updated_at 
    BEFORE UPDATE ON reviews 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- パーティショニング（大量データ対応）
-- メトリクステーブルの月次パーティショニング
CREATE TABLE IF NOT EXISTS metrics_y2024m01 PARTITION OF metrics
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE IF NOT EXISTS metrics_y2024m02 PARTITION OF metrics
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 以下、必要に応じて月次パーティションを追加

-- データ保持ポリシー（古いデータの自動削除）
CREATE OR REPLACE FUNCTION cleanup_old_metrics()
RETURNS void AS $$
BEGIN
    -- 90日より古いメトリクスを削除
    DELETE FROM metrics WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- 30日より古い解決済みアラートを削除
    DELETE FROM alerts 
    WHERE status = 'resolved' 
    AND resolved_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- 定期実行用の関数（cron拡張が利用可能な場合）
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_metrics();');

-- 初期データの挿入（開発・テスト用）
INSERT INTO developers (email, name, expertise_level) VALUES
    ('dev1@example.com', 'Developer One', 0.8),
    ('dev2@example.com', 'Developer Two', 0.6),
    ('dev3@example.com', 'Developer Three', 0.9)
ON CONFLICT (email) DO NOTHING;

-- 権限設定
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO gerrit_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO gerrit_user;

-- 統計情報の更新
ANALYZE;