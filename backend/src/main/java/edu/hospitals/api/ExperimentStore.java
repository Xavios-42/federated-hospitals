package edu.hospitals.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;
import java.util.UUID;

@Repository
class ExperimentStore {
    enum Status { QUEUED, RUNNING, COMPLETED, FAILED }

    record Experiment(UUID runId, Status status, ExperimentConfig config, int roundCount, String errorCode) {}

    private final JdbcTemplate jdbc;
    private final ObjectMapper json;

    ExperimentStore(JdbcTemplate jdbc, ObjectMapper json) {
        this.jdbc = jdbc;
        this.json = json;
    }

    UUID create(ExperimentConfig config) {
        UUID id = UUID.randomUUID();
        jdbc.update("INSERT INTO experiments (run_id, run_status, config_json) VALUES (?, 'QUEUED', ?)",
                id.toString(), encode(config));
        return id;
    }

    Optional<Experiment> find(UUID id) {
        return jdbc.query("""
                SELECT run_status, config_json, round_count, error_code FROM experiments WHERE run_id = ?
                """, (row, index) -> new Experiment(
                id, Status.valueOf(row.getString("run_status")), decode(row.getString("config_json"), ExperimentConfig.class),
                row.getInt("round_count"), row.getString("error_code")), id.toString()).stream().findFirst();
    }

    @Transactional
    public Optional<Experiment> claimNext() {
        var pending = jdbc.query("""
                SELECT run_id FROM experiments WHERE run_status = 'QUEUED' ORDER BY created_at, run_id LIMIT 1
                """, (row, index) -> UUID.fromString(row.getString(1)));
        if (pending.isEmpty()) {
            return Optional.empty();
        }
        UUID id = pending.get(0);
        int updated = jdbc.update("""
                UPDATE experiments SET run_status = 'RUNNING' WHERE run_id = ? AND run_status = 'QUEUED'
                """, id.toString());
        return updated == 1 ? find(id) : Optional.empty();
    }

    @Transactional
    public void recordRound(UUID id, Metrics.Round round) {
        int updated = jdbc.update("""
                UPDATE experiments SET round_count = ? WHERE run_id = ? AND run_status = 'RUNNING' AND round_count = ?
                """, round.round(), id.toString(), round.round() - 1);
        if (updated != 1) {
            throw new IllegalStateException("Unexpected round sequence");
        }
        jdbc.update("INSERT INTO experiment_rounds (run_id, round_number, metrics_json) VALUES (?, ?, ?)",
                id.toString(), round.round(), encode(round));
    }

    void complete(UUID id, int rounds, Metrics.Result result) {
        int updated = jdbc.update("""
                UPDATE experiments SET run_status = 'COMPLETED', result_json = ?
                WHERE run_id = ? AND run_status = 'RUNNING' AND round_count = ?
                """, encode(result), id.toString(), rounds);
        if (updated != 1) {
            throw new IllegalStateException("Experiment has incomplete rounds");
        }
    }

    void fail(UUID id, String code) {
        jdbc.update("""
                UPDATE experiments SET run_status = 'FAILED', error_code = ?
                WHERE run_id = ? AND run_status = 'RUNNING'
                """, code, id.toString());
    }

    void recoverInterrupted() {
        jdbc.update("""
                UPDATE experiments SET run_status = 'FAILED', error_code = 'INTERRUPTED'
                WHERE run_status = 'RUNNING'
                """);
    }

    @Transactional(readOnly = true)
    public Metrics.Report metrics(UUID id) {
        String result = jdbc.queryForObject("SELECT result_json FROM experiments WHERE run_id = ?", String.class, id.toString());
        var history = jdbc.query("""
                SELECT metrics_json FROM experiment_rounds WHERE run_id = ? ORDER BY round_number
                """, (row, index) -> decode(row.getString(1), Metrics.Round.class), id.toString());
        return new Metrics.Report("malignant", history, result == null ? null : decode(result, Metrics.Result.class));
    }

    boolean ready() {
        return Integer.valueOf(1).equals(jdbc.queryForObject("SELECT 1", Integer.class));
    }

    private String encode(Object data) {
        try {
            return json.writeValueAsString(data);
        } catch (JsonProcessingException error) {
            throw new IllegalStateException("Cannot encode experiment metadata", error);
        }
    }

    private <T> T decode(String data, Class<T> type) {
        try {
            return json.readValue(data, type);
        } catch (JsonProcessingException error) {
            throw new IllegalStateException("Cannot decode experiment metadata", error);
        }
    }
}
