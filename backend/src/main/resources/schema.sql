CREATE TABLE IF NOT EXISTS experiments (
    run_id CHAR(36) PRIMARY KEY,
    run_status VARCHAR(12) NOT NULL,
    config_json TEXT NOT NULL,
    round_count INT NOT NULL DEFAULT 0,
    result_json TEXT,
    error_code VARCHAR(40),
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    INDEX experiment_queue (run_status, created_at, run_id),
    CHECK (run_status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED'))
);

CREATE TABLE IF NOT EXISTS experiment_rounds (
    run_id CHAR(36) NOT NULL,
    round_number INT NOT NULL,
    metrics_json TEXT NOT NULL,
    PRIMARY KEY (run_id, round_number),
    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
);
