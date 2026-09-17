package org.example;

import org.example.benchmark.LatencyProfiler;
import org.example.data.DatasetFactory;
import org.example.data.ShapeSample;
import org.example.model.NeuralNetwork;

import java.util.List;

public class Main {
    private static final double CONFIDENCE_THRESHOLD = 0.90;
    public static void main(String[] args) {
        compareStructures();

        System.out.println("INITIALIZING NETWORK FOR RECOGNITION SHAPE...");
        NeuralNetwork nn = new NeuralNetwork(49,16,3);

        List<ShapeSample> trainSet = DatasetFactory.getTrainingSet();
        double learningRate = 0.3; //recommend
        int maxEpochs = 20_000;
        double targetError = 0.001;

        System.out.println("START LEARNING...");
        long startTime = System.currentTimeMillis();
        int epoch = 0;
        double epochError = 1.0;

        for (epoch = 1; epoch <= maxEpochs; epoch++) {
            epochError = 0.0;
            for (ShapeSample sample : trainSet) {
                epochError += nn.trainSample(sample.inputs(), sample.targets(), learningRate);
            }

            if (epoch % 500 == 0 || epochError < targetError) {
                System.out.printf("Epoch: %5d | Error: %.6f%n", epoch, epochError);
            }

            if (epochError < targetError) {
                break;
            }
        }

        long trainingDurationMs = System.currentTimeMillis() - startTime;
        System.out.println("--------------------------------------------------");
        System.out.printf("Learning ended for : %d мс%n", trainingDurationMs);
        System.out.printf("Amount epoches: %d%n", Math.min(epoch, maxEpochs));
        System.out.printf("Final error: %.6f%n", epochError);

        System.out.println("\n=== CHECK BASE SHAPE ===");
        for (ShapeSample sample : trainSet) {
            testSample(nn, sample);
        }

        System.out.println("\n=== CHECK BAD SHAPE ===");


        System.out.println(">>> TEST 1: some noise (2 dead pixels) - recognition:");
        ShapeSample noisyCircle = DatasetFactory.createDistortedSample(DatasetFactory.CIRCLE, 2, 42);
        testSample(nn, noisyCircle);

        ShapeSample noisySquare = DatasetFactory.createDistortedSample(DatasetFactory.SQUARE, 2, 99);
        testSample(nn, noisySquare);

        System.out.println(">>> TEST 2: Loud noise (18 dead pixels) - no recognition:");
        ShapeSample heavyNoisyTriangle = DatasetFactory.createDistortedSample(DatasetFactory.TRIANGLE, 18, 777);
        testSample(nn, heavyNoisyTriangle);
        System.out.println(">>> TEST 3: Some noise (2 dead pixels) - no recognition - cross:");
        ShapeSample originalUnknownCross = DatasetFactory.createDistortedSample(DatasetFactory.UNKNOWN_CROSS, 2, 42);
        testSample(nn, originalUnknownCross);

        LatencyProfiler.benchmarkInference(nn);
        LatencyProfiler.printMemoryAnalysis(49, 16, 3);
    }

    public static void compareStructures() {
        System.out.println("\n=== Comparison of network structures ===");
        int[][] topologies = {
                {49, 8, 3},
                {49, 16, 3},
                {49, 32, 3}
        };

        System.out.printf("%-15s | %-12s | %-10s | %-12s%n",
                "Topology", "Epochs", "Memory (KB)", "Confidence (2 noises)");
        System.out.println("-----------------------------------------------------------------");

        for (int[] topo : topologies) {
            NeuralNetwork net = new NeuralNetwork(topo);
            int epochs = 0;
            double err = 1.0;

            while (err > 0.001 && epochs < 20_000) {
                err = 0.0;
                for (ShapeSample s : DatasetFactory.getTrainingSet()) {
                    err += net.trainSample(s.inputs(), s.targets(), 0.3);
                }
                epochs++;
            }

            // Testing resilience against 2 dead pixels.
            ShapeSample testCircle = DatasetFactory.createDistortedSample(DatasetFactory.CIRCLE, 2, 42);
            double confidence = net.predict(testCircle.inputs())[0];

            int weights = (topo[0] * topo[1]) + (topo[1] * topo[2]) + topo[1] + topo[2];
            double memoryKb = (weights * 8) / 1024.0;

            System.out.printf("[%d -> %2d -> %d] | %-12d | %-10.2f | %-12.1f%%%n",
                    topo[0], topo[1], topo[2], epochs, memoryKb, confidence * 100);
        }
    }

    private static void testSample(NeuralNetwork nn, ShapeSample sample) {
        double[] output = nn.predict(sample.inputs());

        // argmax
        String[] classNames = {"Circle", "Square", "Triangle"};
        int bestClass = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[bestClass]) {
                bestClass = i;
            }
        }
        double confidence = output[bestClass];
        String resultStatus;

        if (confidence >= CONFIDENCE_THRESHOLD) {
            resultStatus = String.format("RECOGNITION: %-11s (confidence: %.1f%%)",
                    classNames[bestClass], confidence * 100);
        } else {
            resultStatus = String.format("NO RECOGNITION (all max confidence %.1f%% < %.0f%%)",
                    confidence * 100, CONFIDENCE_THRESHOLD * 100);
        }

        System.out.printf("Shape: %-15s -> Recognition as: %-11s | Probabilities: [Circle: %.3f, Square: %.3f, Triangle: %.3f]%n",
                sample.name(), resultStatus, output[0], output[1], output[2]);
    }
}