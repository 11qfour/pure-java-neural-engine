package org.example.model;

import java.security.PublicKey;
import java.util.concurrent.ThreadLocalRandom;

public class Layer{
    public final int numInputs;
    public final int numNeurons;

    // weights[i][j] - weight from j input to i neuron
    public final double[][] weights;
    public final double[] biases;

    public final double[] outputs; // y_i^k
    public final double[] deltas;  // delta_i^k

    public Layer(int numInputs, int numNeurons) {
        this.numInputs = numInputs;
        this.numNeurons = numNeurons;
        this.weights = new double[numNeurons][numInputs];
        this.biases = new double[numNeurons];
        this.outputs= new double[numNeurons];
        this.deltas= new double[numNeurons];

        //Xavier-like range
        double range = Math.sqrt(2.0/(numNeurons*numInputs));
        for (int i = 0; i < numNeurons; i++) {
            for (int j = 0; j < numInputs; j++) {
                weights[i][j] = ThreadLocalRandom.current().nextDouble(-range, range);
            }
            biases[i]= ThreadLocalRandom.current().nextDouble(-range, range);
        }

    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double[] forward(double[] inputs){
        for (int i = 0; i < numNeurons; i++) {
            double sum = biases[i];
            for (int j = 0; j < numInputs; j++) {
                sum += weights[i][j]*inputs[j];
            }
            outputs[i] = sigmoid(sum);
        }
        return outputs;
    }
}