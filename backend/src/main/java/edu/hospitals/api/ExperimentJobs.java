package edu.hospitals.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.sql.init.dependency.DependsOnDatabaseInitialization;
import org.springframework.dao.DataAccessException;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.UUID;

@Component
@DependsOnDatabaseInitialization
class ExperimentJobs {
    private static final Logger LOG = LoggerFactory.getLogger(ExperimentJobs.class);

    private final ExperimentStore store;
    private final PythonEngine python;
    private Failure pendingFailure;

    private record Failure(UUID id, String code) {}

    ExperimentJobs(ExperimentStore store, PythonEngine python) {
        this.store = store;
        this.python = python;
    }

    @PostConstruct
    void recover() {
        store.recoverInterrupted();
    }

    @Scheduled(fixedDelayString = "${fl.poll-delay-ms}")
    void dispatch() {
        try {
            if (pendingFailure != null) {
                store.fail(pendingFailure.id(), pendingFailure.code());
                pendingFailure = null;
            }
            store.claimNext().ifPresent(this::execute);
        } catch (DataAccessException error) {
            LOG.warn("Experiment queue is waiting for the metadata database");
        }
    }

    private void execute(ExperimentStore.Experiment experiment) {
        var id = experiment.runId();
        try {
            var result = python.execute(experiment.config(), round -> store.recordRound(id, round));
            store.complete(id, experiment.config().rounds(), result);
        } catch (PythonEngine.WorkerFailure failure) {
            fail(experiment, failure.getMessage());
        } catch (JsonProcessingException error) {
            fail(experiment, "INVALID_WORKER_OUTPUT");
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            fail(experiment, "INTERRUPTED");
        } catch (IOException error) {
            fail(experiment, "WORKER_IO_ERROR");
        } catch (RuntimeException error) {
            fail(experiment, "INTERNAL_ERROR");
        }
    }

    private void fail(ExperimentStore.Experiment experiment, String code) {
        LOG.warn("Experiment {} failed ({})", experiment.runId(), code);
        pendingFailure = new Failure(experiment.runId(), code);
        store.fail(experiment.runId(), code);
        pendingFailure = null;
    }
}
