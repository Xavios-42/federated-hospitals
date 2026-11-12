package edu.hospitals.api;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.time.Duration;
import java.util.UUID;
import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest(properties = {
        "spring.datasource.url=jdbc:h2:mem:experiments;MODE=MySQL;DB_CLOSE_DELAY=-1",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "fl.poll-delay-ms=50"
})
@AutoConfigureMockMvc
class ApiIntegrationTest {
    private static final String CONFIG = """
            {"algorithm":"fedavg","mode":"iid","seed":42,"rounds":1,"epochs":1,"batchSize":32,"learningRate":0.01}
            """;

    @Autowired MockMvc api;
    @Autowired ObjectMapper json;
    @Autowired JdbcTemplate jdbc;
    @MockitoBean PythonEngine engine;

    @Test
    void submitsRunsPersistsAndExportsAggregateMetrics() throws Exception {
        var validation = new Metrics.Scores(93, 0.4, 1, 1, 1, 1, new Metrics.Confusion(35, 58, 0, 0));
        var test = new Metrics.Scores(114, 0.5, 1, 1, 1, 1, new Metrics.Confusion(42, 72, 0, 0));
        when(engine.execute(argThat(config -> config != null && config.seed() == 42), any()))
                .thenAnswer(call -> {
                    Consumer<Metrics.Round> callback = call.getArgument(1);
                    callback.accept(new Metrics.Round(1, validation, 5679, 5679));
                    return new Metrics.Result(test, new Metrics.Communication(5679, 5679, 11358, 17037),
                            new Metrics.HospitalCounts(120, 121, 121));
                });
        String id = submit(CONFIG);
        awaitStatus(id, "COMPLETED");
        api.perform(get("/api/experiments/{id}", id))
                .andExpect(jsonPath("$.roundCount").value(1))
                .andExpect(jsonPath("$.config.model").value("logistic"));
        var metrics = api.perform(get("/api/experiments/{id}/metrics", id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.history[0].validation.samples").value(93))
                .andExpect(jsonPath("$.finalMetrics.test.samples").value(114))
                .andExpect(jsonPath("$.finalMetrics.communication.total_model_bytes").value(17037))
                .andReturn().getResponse().getContentAsString();
        assertThat(metrics).doesNotContain("model_state", "features", "labels");
        assertThat(jdbc.queryForObject("SELECT metrics_json FROM experiment_rounds WHERE run_id = ?", String.class, id))
                .doesNotContain("model_state", "features", "labels");
        api.perform(get("/api/experiments/{id}/export", id))
                .andExpect(status().isOk())
                .andExpect(header().string("Content-Disposition", "attachment; filename=\"experiment-" + id + ".json\""))
                .andExpect(jsonPath("$.metrics.finalMetrics.test.samples").value(114));
        api.perform(get("/api/experiments/{id}/export?format=csv", id))
                .andExpect(status().isOk())
                .andExpect(content().contentTypeCompatibleWith("text/csv"))
                .andExpect(content().string(org.hamcrest.Matchers.containsString("test,1,114,")));
    }

    @Test
    void failedWorkerLeavesPartialMetricsVisibleAndBlocksExport() throws Exception {
        var scores = new Metrics.Scores(93, 0.4, 1, 1, 1, 1, new Metrics.Confusion(35, 58, 0, 0));
        when(engine.execute(argThat(config -> config != null && config.seed() == 43), any()))
                .thenAnswer(call -> {
                    Consumer<Metrics.Round> callback = call.getArgument(1);
                    callback.accept(new Metrics.Round(1, scores, 5679, 5679));
                    throw new PythonEngine.WorkerFailure("TRAINING_FAILED");
                });
        String id = submit(CONFIG.replace("\"seed\":42", "\"seed\":43").replace("\"rounds\":1", "\"rounds\":2"));
        awaitStatus(id, "FAILED");
        api.perform(get("/api/experiments/{id}", id))
                .andExpect(jsonPath("$.roundCount").value(1))
                .andExpect(jsonPath("$.errorCode").value("TRAINING_FAILED"));
        api.perform(get("/api/experiments/{id}/metrics", id))
                .andExpect(jsonPath("$.history[0].round").value(1))
                .andExpect(jsonPath("$.finalMetrics").doesNotExist());
        api.perform(get("/api/experiments/{id}/export", id)).andExpect(status().isConflict());
    }

    @Test
    void validatesRequestsAndReportsDependencyHealth() throws Exception {
        for (String invalid : new String[]{
                CONFIG.replace("\"rounds\":1", "\"rounds\":0"),
                CONFIG.replace("\"rounds\":1", "\"rounds\":\"1\""),
                CONFIG.replace("\"rounds\":1", "\"rounds\":1.5"),
                CONFIG.replace("\"seed\":42", "\"seed\":42,\"features\":[1]"),
                CONFIG.replace("\"seed\":42", "\"seed\":42,\"mu\":0.1"),
                CONFIG.replace("\"seed\":42", "\"seed\":42,\"model\":\"mlp\"")
        }) {
            api.perform(post("/api/experiments").contentType("application/json").content(invalid))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.code").value("INVALID_CONFIG"));
        }
        api.perform(get("/api/experiments/{id}", UUID.randomUUID()))
                .andExpect(status().isNotFound());
        when(engine.ready()).thenReturn(true, false);
        api.perform(get("/api/health")).andExpect(status().isOk())
                .andExpect(jsonPath("$.database").value("UP"))
                .andExpect(jsonPath("$.mlEngine").value("UP"));
        api.perform(get("/api/health")).andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.mlEngine").value("DOWN"));
    }

    private String submit(String configuration) throws Exception {
        String response = api.perform(post("/api/experiments").contentType("application/json").content(configuration))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.status").value("QUEUED"))
                .andReturn().getResponse().getContentAsString();
        return json.readTree(response).get("runId").asText();
    }

    private void awaitStatus(String id, String status) {
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() ->
                api.perform(get("/api/experiments/{id}", id)).andExpect(jsonPath("$.status").value(status)));
    }
}
