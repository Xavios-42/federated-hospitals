package edu.hospitals.api;

import java.util.Set;

record ExperimentConfig(
        String algorithm, String mode, String model, Long seed, Integer rounds,
        Integer epochs, Integer batchSize, Double learningRate, Double mu, Double beta,
        Double serverLearningRate
) {
    ExperimentConfig {
        model = model == null ? "logistic" : model;
        mu = mu == null ? 0.0 : mu;
        beta = beta == null ? 0.0 : beta;
        serverLearningRate = serverLearningRate == null ? 1.0 : serverLearningRate;
        require(algorithm != null && Set.of("fedavg", "fedprox", "fedavgm").contains(algorithm),
                "algorithm must be fedavg, fedprox, or fedavgm");
        require("iid".equals(mode) || "non-iid".equals(mode), "mode must be iid or non-iid");
        require("logistic".equals(model), "model must be logistic");
        require(seed != null && seed >= 0 && seed <= 4_294_967_295L, "seed must be an unsigned 32-bit integer");
        require(rounds != null && rounds >= 1 && rounds <= 100, "rounds must be between 1 and 100");
        require(epochs != null && epochs >= 1 && epochs <= 10, "epochs must be between 1 and 10");
        require(batchSize != null && batchSize >= 1 && batchSize <= 512, "batchSize must be between 1 and 512");
        require(learningRate != null && Double.isFinite(learningRate) && learningRate > 0 && learningRate <= 1,
                "learningRate must be greater than 0 and at most 1");
        require(Double.isFinite(mu) && mu >= 0 && mu <= 10, "mu must be between 0 and 10");
        require(Double.isFinite(beta) && beta >= 0 && beta < 1, "beta must be between 0 (inclusive) and 1 (exclusive)");
        require(Double.isFinite(serverLearningRate) && serverLearningRate > 0 && serverLearningRate <= 10,
                "serverLearningRate must be greater than 0 and at most 10");
        require(algorithm.equals("fedprox") || mu == 0.0, "Only FedProx accepts nonzero mu");
        require(algorithm.equals("fedavgm") || (beta == 0.0 && serverLearningRate == 1.0),
                "Only FedAvgM accepts custom server momentum settings");
    }

    private static void require(boolean valid, String message) {
        if (!valid) {
            throw new InvalidExperimentConfig(message);
        }
    }

    static final class InvalidExperimentConfig extends IllegalArgumentException {
        InvalidExperimentConfig(String message) {
            super(message);
        }
    }
}
