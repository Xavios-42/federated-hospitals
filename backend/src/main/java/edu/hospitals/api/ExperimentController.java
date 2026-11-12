package edu.hospitals.api;

import org.springframework.dao.DataAccessException;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.net.URI;
import java.util.Map;
import java.util.StringJoiner;
import java.util.UUID;

@RestController
@RequestMapping("/api")
class ExperimentController {
    record Accepted(UUID runId, ExperimentStore.Status status) {}
    record Export(ExperimentStore.Experiment experiment, Metrics.Report metrics) {}
    record ApiError(String code, String message) {}

    private final ExperimentStore store;
    private final PythonEngine python;

    ExperimentController(ExperimentStore store, PythonEngine python) {
        this.store = store;
        this.python = python;
    }

    @PostMapping("/experiments")
    ResponseEntity<Accepted> create(@RequestBody ExperimentConfig config) {
        UUID id = store.create(config);
        return ResponseEntity.accepted().location(URI.create("/api/experiments/" + id))
                .body(new Accepted(id, ExperimentStore.Status.QUEUED));
    }

    @GetMapping("/experiments/{id}")
    ExperimentStore.Experiment status(@PathVariable UUID id) {
        return find(id);
    }

    @GetMapping("/experiments/{id}/metrics")
    Metrics.Report metrics(@PathVariable UUID id) {
        find(id);
        return store.metrics(id);
    }

    @GetMapping("/experiments/{id}/export")
    ResponseEntity<?> export(@PathVariable UUID id, @RequestParam(defaultValue = "json") String format) {
        if (!format.equals("json") && !format.equals("csv")) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Export format must be json or csv");
        }
        var experiment = find(id);
        if (experiment.status() != ExperimentStore.Status.COMPLETED) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Export requires a completed experiment");
        }
        var report = store.metrics(id);
        var response = ResponseEntity.ok()
                .header("Content-Disposition", "attachment; filename=\"experiment-" + id + "." + format + "\"");
        if (format.equals("json")) {
            return response.contentType(MediaType.APPLICATION_JSON).body(new Export(experiment, report));
        }
        var csv = new StringJoiner("\r\n", "", "\r\n");
        csv.add("split,round,samples,loss,accuracy,precision,recall,f1,tp,tn,fp,fn,upload_model_bytes,download_model_bytes");
        for (var round : report.history()) {
            csv.add(csvRow("validation", round.round(), round.validation(),
                    round.uploadModelBytes(), round.downloadModelBytes()));
        }
        var finalMetrics = report.finalMetrics();
        csv.add(csvRow("test", experiment.roundCount(), finalMetrics.test(),
                finalMetrics.communication().uploadModelBytes(), finalMetrics.communication().downloadModelBytes()));
        return response.contentType(MediaType.parseMediaType("text/csv")).body(csv.toString());
    }

    @GetMapping("/health")
    ResponseEntity<Map<String, String>> health() {
        boolean databaseUp;
        try {
            databaseUp = store.ready();
        } catch (DataAccessException error) {
            databaseUp = false;
        }
        boolean pythonUp = python.ready();
        var status = databaseUp && pythonUp ? HttpStatus.OK : HttpStatus.SERVICE_UNAVAILABLE;
        return ResponseEntity.status(status).body(Map.of(
                "api", "UP", "database", databaseUp ? "UP" : "DOWN", "mlEngine", pythonUp ? "UP" : "DOWN"
        ));
    }

    private ExperimentStore.Experiment find(UUID id) {
        return store.find(id).orElseThrow(
                () -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Experiment does not exist"));
    }

    private static String csvRow(String split, int round, Metrics.Scores scores, long uploads, long downloads) {
        var cm = scores.confusionMatrix();
        return String.join(",", split, String.valueOf(round), String.valueOf(scores.samples()),
                String.valueOf(scores.loss()), String.valueOf(scores.accuracy()), String.valueOf(scores.precision()),
                String.valueOf(scores.recall()), String.valueOf(scores.f1()), String.valueOf(cm.tp()),
                String.valueOf(cm.tn()), String.valueOf(cm.fp()), String.valueOf(cm.fn()),
                String.valueOf(uploads), String.valueOf(downloads));
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    ResponseEntity<ApiError> invalidRequest(HttpMessageNotReadableException error) {
        Throwable root = error.getMostSpecificCause();
        String message = root instanceof ExperimentConfig.InvalidExperimentConfig
                ? root.getMessage() : "Expected a JSON experiment configuration with supported fields and types";
        return ResponseEntity.badRequest().body(new ApiError("INVALID_CONFIG", message));
    }

    @ExceptionHandler(ResponseStatusException.class)
    ResponseEntity<ApiError> requestError(ResponseStatusException error) {
        return ResponseEntity.status(error.getStatusCode())
                .body(new ApiError("REQUEST_ERROR", error.getReason()));
    }
}
