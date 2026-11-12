package edu.hospitals.api;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

final class Metrics {
    private Metrics() {}

    record Confusion(int tp, int tn, int fp, int fn) {
        Confusion {
            if (tp < 0 || tn < 0 || fp < 0 || fn < 0) {
                throw new IllegalArgumentException("Negative confusion count");
            }
        }
    }

    record Scores(
            int samples, double loss, double accuracy, double precision, double recall, double f1,
            @JsonProperty("confusion_matrix") Confusion confusionMatrix
    ) {
        Scores {
            if (samples < 1 || samples > 569 || confusionMatrix == null
                    || (long) confusionMatrix.tp() + confusionMatrix.tn() + confusionMatrix.fp() + confusionMatrix.fn() != samples
                    || !Double.isFinite(loss) || loss < 0
                    || !fraction(accuracy) || !fraction(precision) || !fraction(recall) || !fraction(f1)) {
                throw new IllegalArgumentException("Invalid aggregate scores");
            }
        }

        private static boolean fraction(double value) {
            return value >= 0 && value <= 1;
        }
    }

    record Round(
            int round, Scores validation,
            @JsonProperty("upload_model_bytes") long uploadModelBytes,
            @JsonProperty("download_model_bytes") long downloadModelBytes
    ) {
        Round {
            if (round < 1 || validation == null || uploadModelBytes < 1 || downloadModelBytes < 1) {
                throw new IllegalArgumentException("Invalid round metrics");
            }
        }
    }

    record Communication(
            @JsonProperty("initial_download_model_bytes") long initialDownloadModelBytes,
            @JsonProperty("upload_model_bytes") long uploadModelBytes,
            @JsonProperty("download_model_bytes") long downloadModelBytes,
            @JsonProperty("total_model_bytes") long totalModelBytes
    ) {
        Communication {
            if (initialDownloadModelBytes < 1 || uploadModelBytes < 1
                    || downloadModelBytes <= initialDownloadModelBytes
                    || Math.addExact(uploadModelBytes, downloadModelBytes) != totalModelBytes) {
                throw new IllegalArgumentException("Invalid communication metrics");
            }
        }
    }

    record HospitalCounts(@JsonProperty("A") int a, @JsonProperty("B") int b, @JsonProperty("C") int c) {
        HospitalCounts {
            if (a < 1 || b < 1 || c < 1 || (long) a + b + c > 455) {
                throw new IllegalArgumentException("Invalid hospital sample counts");
            }
        }
    }

    record Result(
            Scores test, Communication communication,
            @JsonProperty("hospital_training_samples") HospitalCounts hospitalTrainingSamples
    ) {
        Result {
            if (test == null || test.samples() != 114 || communication == null || hospitalTrainingSamples == null) {
                throw new IllegalArgumentException("Invalid final metrics");
            }
        }
    }

    record Report(String positiveClass, List<Round> history, Result finalMetrics) {}
}
