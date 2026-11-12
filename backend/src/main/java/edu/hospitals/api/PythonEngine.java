package edu.hospitals.api;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

@Component
class PythonEngine {
    private final ObjectMapper json;
    private final Path executable;
    private final Path directory;
    private final long timeoutSeconds;
    private final java.util.concurrent.ScheduledExecutorService timer = Executors.newSingleThreadScheduledExecutor();

    PythonEngine(ObjectMapper json, @Value("${fl.python}") String python,
                 @Value("${fl.engine-directory}") String directory,
                 @Value("${fl.job-timeout-seconds}") long timeoutSeconds) {
        this.json = json;
        this.executable = Path.of(python).toAbsolutePath().normalize();
        this.directory = Path.of(directory).toAbsolutePath().normalize();
        if (timeoutSeconds < 1) {
            throw new IllegalArgumentException("Worker timeout must be positive");
        }
        this.timeoutSeconds = timeoutSeconds;
    }

    Metrics.Result execute(ExperimentConfig config, Consumer<Metrics.Round> onRound)
            throws IOException, InterruptedException {
        Process worker = command().start();
        var expired = new AtomicBoolean();
        var deadline = timer.schedule(() -> {
            if (worker.isAlive()) {
                expired.set(true);
                worker.destroyForcibly();
            }
        }, timeoutSeconds, TimeUnit.SECONDS);
        try {
            try (var input = worker.getOutputStream()) {
                input.write(json.writeValueAsBytes(config));
            }
            Metrics.Result result = null;
            int roundNumber = 0;
            long uploads = 0;
            long downloads = 0;
            try (var output = new BufferedReader(new InputStreamReader(worker.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = output.readLine()) != null) {
                    JsonNode event = json.readTree(line);
                    if (result != null || event == null || !event.isObject() || event.size() != 2
                            || !event.has("type") || !event.hasNonNull("metrics")) {
                        throw new WorkerFailure("INVALID_WORKER_OUTPUT");
                    }
                    switch (event.get("type").asText()) {
                        case "round" -> {
                            var round = decode(event.get("metrics"), Metrics.Round.class);
                            if (round.round() != ++roundNumber || roundNumber > config.rounds()) {
                                throw new WorkerFailure("INVALID_WORKER_OUTPUT");
                            }
                            uploads = Math.addExact(uploads, round.uploadModelBytes());
                            downloads = Math.addExact(downloads, round.downloadModelBytes());
                            onRound.accept(round);
                        }
                        case "complete" -> result = decode(event.get("metrics"), Metrics.Result.class);
                        default -> throw new WorkerFailure("INVALID_WORKER_OUTPUT");
                    }
                }
            }
            int exit = worker.waitFor();
            if (expired.get() || exit != 0) {
                throw new WorkerFailure(expired.get() ? "WORKER_TIMEOUT" : "TRAINING_FAILED");
            }
            if (result == null || roundNumber != config.rounds()
                    || uploads != result.communication().uploadModelBytes()
                    || downloads != result.communication().downloadModelBytes() - result.communication().initialDownloadModelBytes()) {
                throw new WorkerFailure("INVALID_WORKER_OUTPUT");
            }
            return result;
        } catch (IOException failure) {
            if (expired.get()) {
                throw new WorkerFailure("WORKER_TIMEOUT");
            }
            throw failure;
        } finally {
            deadline.cancel(false);
            if (worker.isAlive()) {
                worker.destroyForcibly();
            }
        }
    }

    boolean ready() {
        Process probe = null;
        try {
            ProcessBuilder command = command();
            command.command().add("--health");
            probe = command.redirectOutput(ProcessBuilder.Redirect.DISCARD).start();
            return probe.waitFor(3, TimeUnit.SECONDS) && probe.exitValue() == 0;
        } catch (IOException error) {
            return false;
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return false;
        } finally {
            if (probe != null && probe.isAlive()) {
                probe.destroyForcibly();
            }
        }
    }

    private ProcessBuilder command() {
        ProcessBuilder command = new ProcessBuilder(executable.toString(), "-m", "app.worker")
                .directory(directory.toFile())
                .redirectError(ProcessBuilder.Redirect.DISCARD);
        command.environment().keySet().removeIf(name -> name.startsWith("FL_DB_")
                || name.startsWith("SPRING_DATASOURCE_") || name.equals("PYTHONPATH") || name.equals("PYTHONHOME"));
        return command;
    }

    private <T> T decode(JsonNode data, Class<T> type) throws IOException {
        return json.readerFor(type)
                .with(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES)
                .with(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
                .readValue(data);
    }

    @PreDestroy
    void stopTimer() {
        timer.shutdownNow();
    }

    static final class WorkerFailure extends IOException {
        WorkerFailure(String code) {
            super(code);
        }
    }
}
