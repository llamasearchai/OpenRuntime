-- =============================================================================
-- OpenRuntime Enhanced - Database Initialization Script
-- PostgreSQL database schema for production deployment
-- Author: Nik Jois <nikjois@llamasearch.ai>
-- =============================================================================

-- Create database if it doesn't exist
-- Note: This is typically handled by Docker/environment setup

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =============================================================================
-- Core Tables
-- =============================================================================

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    salt VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    api_key VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- GPU devices table
CREATE TABLE IF NOT EXISTS gpu_devices (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    device_type VARCHAR(50) NOT NULL,
    memory_total BIGINT NOT NULL,
    memory_available BIGINT NOT NULL,
    compute_units INTEGER NOT NULL,
    is_available BOOLEAN DEFAULT true,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- AI agents table
CREATE TABLE IF NOT EXISTS ai_agents (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(100) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    model VARCHAR(255) NOT NULL,
    temperature DECIMAL(3,2) DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 2048,
    is_active BOOLEAN DEFAULT true,
    capabilities JSONB DEFAULT '[]',
    system_prompt TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table for GPU and AI tasks
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    operation VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    device_id VARCHAR(255) REFERENCES gpu_devices(id) ON DELETE SET NULL,
    agent_id VARCHAR(255) REFERENCES ai_agents(id) ON DELETE SET NULL,
    input_data JSONB DEFAULT '{}',
    result JSONB,
    error_message TEXT,
    execution_time DECIMAL(10,3),
    priority INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id VARCHAR(255) REFERENCES gpu_devices(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    memory_usage DECIMAL(5,2),
    gpu_utilization DECIMAL(5,2),
    temperature DECIMAL(5,2),
    power_usage DECIMAL(8,2),
    throughput DECIMAL(12,2),
    metadata JSONB DEFAULT '{}'
);

-- AI insights table
CREATE TABLE IF NOT EXISTS ai_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) REFERENCES tasks(task_id) ON DELETE CASCADE,
    agent_id VARCHAR(255) REFERENCES ai_agents(id) ON DELETE SET NULL,
    insight_type VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    confidence DECIMAL(3,2),
    tokens_used INTEGER,
    execution_time DECIMAL(10,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Security events table
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    source_ip INET NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    threat_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    blocked BOOLEAN DEFAULT false,
    additional_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time DECIMAL(10,3),
    request_size INTEGER,
    response_size INTEGER,
    user_agent TEXT,
    source_ip INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Configuration settings
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    description TEXT,
    is_sensitive BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Tasks indexes
CREATE INDEX IF NOT EXISTS idx_tasks_task_id ON tasks(task_id);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_operation ON tasks(operation);
CREATE INDEX IF NOT EXISTS idx_tasks_device_id ON tasks(device_id);
CREATE INDEX IF NOT EXISTS idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_metrics_device_id ON performance_metrics(device_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_device_timestamp ON performance_metrics(device_id, timestamp);

-- AI insights indexes
CREATE INDEX IF NOT EXISTS idx_insights_task_id ON ai_insights(task_id);
CREATE INDEX IF NOT EXISTS idx_insights_agent_id ON ai_insights(agent_id);
CREATE INDEX IF NOT EXISTS idx_insights_type ON ai_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_insights_created_at ON ai_insights(created_at);

-- Security events indexes
CREATE INDEX IF NOT EXISTS idx_security_event_id ON security_events(event_id);
CREATE INDEX IF NOT EXISTS idx_security_source_ip ON security_events(source_ip);
CREATE INDEX IF NOT EXISTS idx_security_threat_type ON security_events(threat_type);
CREATE INDEX IF NOT EXISTS idx_security_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_created_at ON security_events(created_at);

-- API usage indexes
CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON api_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_source_ip ON api_usage(source_ip);

-- System config indexes
CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(key);

-- =============================================================================
-- Triggers for Automatic Updates
-- =============================================================================

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_gpu_devices_updated_at BEFORE UPDATE ON gpu_devices
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_agents_updated_at BEFORE UPDATE ON ai_agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- Active tasks view
CREATE OR REPLACE VIEW active_tasks AS
SELECT 
    t.*,
    u.username,
    d.name as device_name,
    a.name as agent_name
FROM tasks t
LEFT JOIN users u ON t.user_id = u.id
LEFT JOIN gpu_devices d ON t.device_id = d.id
LEFT JOIN ai_agents a ON t.agent_id = a.id
WHERE t.status IN ('pending', 'running');

-- Recent performance metrics view
CREATE OR REPLACE VIEW recent_metrics AS
SELECT 
    pm.*,
    gd.name as device_name,
    gd.device_type
FROM performance_metrics pm
JOIN gpu_devices gd ON pm.device_id = gd.id
WHERE pm.timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
ORDER BY pm.timestamp DESC;

-- Security summary view
CREATE OR REPLACE VIEW security_summary AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    threat_type,
    severity,
    COUNT(*) as event_count,
    COUNT(CASE WHEN blocked = true THEN 1 END) as blocked_count
FROM security_events
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at), threat_type, severity
ORDER BY hour DESC;

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Insert default AI agents
INSERT INTO ai_agents (id, name, role, provider, model, system_prompt, capabilities) VALUES
('perf_optimizer', 'Performance Optimizer', 'performance_optimizer', 'openai', 'gpt-4o-mini', 
 'You are a GPU performance optimization expert. Analyze system metrics and provide optimization recommendations for GPU workloads. Focus on memory usage, compute efficiency, and throughput optimization.',
 '["performance_analysis", "optimization_recommendations", "benchmarking"]'),
('system_analyst', 'System Analyst', 'system_analyst', 'openai', 'gpt-4o-mini',
 'You are a system analysis expert specializing in GPU computing systems. Analyze system behavior, identify bottlenecks, and provide diagnostic insights.',
 '["system_diagnostics", "bottleneck_analysis", "resource_monitoring"]'),
('code_generator', 'Code Generator', 'code_generator', 'openai', 'gpt-4o-mini',
 'You are an expert programmer specializing in GPU computing and parallel algorithms. Generate optimized code for various programming languages with focus on performance.',
 '["code_generation", "optimization", "testing", "documentation"]'),
('shell_executor', 'Shell Executor', 'shell_executor', 'openai', 'gpt-4o-mini',
 'You are a shell command expert. Generate safe and efficient shell commands for system administration and automation tasks. Always prioritize security.',
 '["shell_commands", "automation", "system_administration"]')
ON CONFLICT (id) DO NOTHING;

-- Insert default system configuration
INSERT INTO system_config (key, value, description) VALUES
('max_concurrent_tasks', '20', 'Maximum number of concurrent tasks'),
('cache_max_size', '1000', 'Maximum cache size for results'),
('gpu_fallback_to_cpu', 'true', 'Enable CPU fallback when GPU unavailable'),
('rate_limit_requests_per_minute', '100', 'API rate limit per minute per user'),
('security_threat_detection_enabled', 'true', 'Enable threat detection system'),
('metrics_retention_days', '30', 'Number of days to retain performance metrics'),
('log_level', '"info"', 'System log level'),
('prometheus_enabled', 'true', 'Enable Prometheus metrics export')
ON CONFLICT (key) DO NOTHING;

-- Create admin user (password: admin123, should be changed in production)
INSERT INTO users (username, email, password_hash, salt, is_admin, api_key) VALUES
('admin', 'admin@openruntime.local', 
 '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp3fVA0Lm', 
 'admin_salt_change_in_production',
 true,
 'openruntime_admin_api_key_change_in_production')
ON CONFLICT (username) DO NOTHING;

-- =============================================================================
-- Functions for Common Operations
-- =============================================================================

-- Function to clean old metrics
CREATE OR REPLACE FUNCTION cleanup_old_metrics(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM performance_metrics 
    WHERE timestamp < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get device utilization
CREATE OR REPLACE FUNCTION get_device_utilization(device_id_param VARCHAR(255), hours_back INTEGER DEFAULT 1)
RETURNS TABLE(
    avg_utilization DECIMAL,
    max_utilization DECIMAL,
    avg_memory_usage DECIMAL,
    max_memory_usage DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(pm.gpu_utilization)::DECIMAL,
        MAX(pm.gpu_utilization)::DECIMAL,
        AVG(pm.memory_usage)::DECIMAL,
        MAX(pm.memory_usage)::DECIMAL
    FROM performance_metrics pm
    WHERE pm.device_id = device_id_param
    AND pm.timestamp >= CURRENT_TIMESTAMP - (hours_back || ' hours')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- Function to get task statistics
CREATE OR REPLACE FUNCTION get_task_stats(hours_back INTEGER DEFAULT 24)
RETURNS TABLE(
    total_tasks BIGINT,
    completed_tasks BIGINT,
    failed_tasks BIGINT,
    avg_execution_time DECIMAL,
    success_rate DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT,
        COUNT(CASE WHEN status = 'completed' THEN 1 END)::BIGINT,
        COUNT(CASE WHEN status = 'failed' THEN 1 END)::BIGINT,
        AVG(execution_time)::DECIMAL,
        (COUNT(CASE WHEN status = 'completed' THEN 1 END)::DECIMAL / COUNT(*)::DECIMAL * 100)::DECIMAL
    FROM tasks
    WHERE created_at >= CURRENT_TIMESTAMP - (hours_back || ' hours')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Permissions and Security
-- =============================================================================

-- Create read-only user for monitoring
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'openruntime_readonly') THEN
        CREATE ROLE openruntime_readonly LOGIN PASSWORD 'readonly_password_change_in_production';
    END IF;
END
$$;

-- Grant read-only permissions
GRANT CONNECT ON DATABASE openruntime TO openruntime_readonly;
GRANT USAGE ON SCHEMA public TO openruntime_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO openruntime_readonly;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO openruntime_readonly;

-- Grant permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO openruntime_readonly;

-- =============================================================================
-- Maintenance
-- =============================================================================

-- Schedule cleanup job (this would typically be done via cron or pg_cron)
-- SELECT cron.schedule('cleanup-old-metrics', '0 2 * * *', 'SELECT cleanup_old_metrics(30);');

COMMIT;

-- =============================================================================
-- Database Setup Complete
-- 
-- Next Steps:
-- 1. Change default passwords and API keys
-- 2. Set up regular backup schedule
-- 3. Configure monitoring and alerting
-- 4. Set up connection pooling (PgBouncer)
-- 5. Configure SSL/TLS for connections
-- 6. Set up read replicas for scaling
-- ============================================================================= 