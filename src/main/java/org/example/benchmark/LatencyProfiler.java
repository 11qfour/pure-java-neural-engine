package org.example.benchmark;

import org.example.data.DatasetFactory;
import org.example.model.NeuralNetwork;

public  class LatencyProfiler {
    public static void benchmarkInference(NeuralNetwork nn) {
        double[] sampleInput = DatasetFactory.CIRCLE.inputs();

        // Warm-up JVM / JIT-compiler (unstable influence of the start)
        for (int i = 0; i < 50_000; i++) {
            nn.predict(sampleInput);
        }


        int iterations = 200_000;
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            nn.predict(sampleInput);
        }
        long totalTimeNs = System.nanoTime() - start;

        double avgNs = (double) totalTimeNs / iterations;
        double avgUs = avgNs / 1000.0;

        System.out.println("\n=== INFERENCE BENCHMARK RESULTS ===");
        System.out.printf("Iterations: %d%n", iterations);
        System.out.printf("Average time of one inference: %.2f ns (%.4f mcs)%n", avgNs, avgUs);
        System.out.printf("Throughput): %,d ops/sec%n", (long)(1_000_000_000.0 / avgNs));
    }

    public static void printMemoryAnalysis(int inputs, int hidden, int outputs) {
        System.out.println("\n=== Memory Usage Analysis ===");

        // W1: inputs * hidden
        // W2: hidden * outputs
        int w1Count = inputs * hidden;
        int w2Count = hidden * outputs;
        int totalWeights = w1Count + w2Count;

        int totalBiases = hidden + outputs;

        int totalActivations = hidden + outputs;

        int totalDeltas = hidden + outputs;
        //activation buffer
        int bytesPerDouble = 8;

        long inferenceBytes = (long) (totalWeights + totalBiases + totalActivations) * bytesPerDouble;


        long trainingBytes = inferenceBytes + (long)(totalWeights + totalDeltas) * bytesPerDouble;

        System.out.printf("Network topology: [%d -> %d -> %d]%n", inputs, hidden, outputs);
        System.out.printf("1. Number of parameters (weights + biases): %d%n", totalWeights + totalBiases);
        System.out.printf("2. Memory for INFERENCE (raw data): %d bytes (~%.2f KB)%n",
                inferenceBytes, inferenceBytes / 1024.0);
        System.out.printf("3. Memory for TRAINING (including gradients): %d bytes (~%.2f KB)%n",
                trainingBytes, trainingBytes / 1024.0);

        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("4. Current JVM heap memory in use: %d bytes (~%.2f MB)%n",
                usedMemory, usedMemory / (1024.0 * 1024.0));
    }
}