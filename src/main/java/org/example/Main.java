package org.example;

import org.example.model.NeuralNetwork;

public class Main {
    public static void main(String[] args) {
        double[][] xorInputs = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };

        double[][] xorTargets = {
                {0.0},
                {1.0},
                {1.0},
                {0.0}
        };

        NeuralNetwork nn = new NeuralNetwork(2, 4, 1);

        double learningRate = 0.5;
        int maxEpochs = 10_000;
        double targetError = 0.001;

        System.out.println("Start learning XOR...");

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            double epochError = 0.0;
            for (int i = 0; i < xorInputs.length; i++) {
                epochError += nn.trainSample(xorInputs[i], xorTargets[i], learningRate);
            }

            if (epoch % 1000 == 0 || epochError < targetError) {
                System.out.printf("Epoch %5d | Summary Error: %.6f%n", epoch, epochError);
            }

            if (epochError < targetError) {
                System.out.printf("Network close to %d epoch!%n", epoch);
                break;
            }
        }

        System.out.println("\nResults checking XOR:");
        for (int i = 0; i < xorInputs.length; i++) {
            double[] pred = nn.predict(xorInputs[i]);
            System.out.printf("Input: [%.0f, %.0f] -> Waiting: %.0f | Output network: %.4f%n",
                    xorInputs[i][0], xorInputs[i][1], xorTargets[i][0], pred[0]);
        }
    }
}