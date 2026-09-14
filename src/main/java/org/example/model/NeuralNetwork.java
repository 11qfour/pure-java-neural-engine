package org.example.model;

public class NeuralNetwork {
    private final Layer[] layers;


    public NeuralNetwork(int... topology) {
        if (topology.length < 2){
            throw new IllegalStateException("В НС должно быть как минимум 2 слоя");
        }
        this.layers = new Layer[topology.length-1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(topology[i], topology[i + 1]);
        }
    }

    public double[] predict(double[] inputs){
        double[] current = inputs;
        for (Layer l : layers){
            current = l.forward(current);
        }
        return current;
    }

    public double trainSample(double[] inputs, double[] targets, double alpha){
        double[] outputs = predict(inputs);
        double totalError = 0.0;
        for (int i = 0; i < targets.length; i++) {
            double diff = targets[i] - outputs[i];
            totalError += 0.5 * diff * diff;//MSE
        }
        //delta output
        Layer outputLayer = layers[layers.length - 1];
        for (int i = 0; i < outputLayer.numNeurons; i++) {
            double y = outputLayer.outputs[i]; //output
            outputLayer.deltas[i] = y * (1.0 - y) * (targets[i] - y); //f'(IN) * OUTactual
        }
        //backpropagation + Hij output
        for (int l = layers.length-2; l >= 0; l--) {
            Layer currentLayer = layers[l];
            Layer nextLayer = layers[l+1];

            for (int i = 0; i < currentLayer.numNeurons; i++) {
                double sum = 0.0;
                for (int j = 0; j < nextLayer.numNeurons; j++) {
                    sum += nextLayer.deltas[j] * nextLayer.weights[j][i];
                }
                double y = currentLayer.outputs[i];
                currentLayer.deltas[i] = y * (1.0 - y) * sum;
            }
        }
        //forward + bias
        for (int l = 0; l < layers.length; l++) {
            Layer layer = layers[l];
            double[] prevInputs = (l == 0) ? inputs : layers[l - 1].outputs;

            for (int i = 0; i < layer.numNeurons; i++) {
                for (int j = 0; j < layer.numInputs; j++) {
                    layer.weights[i][j] += alpha * layer.deltas[i] * prevInputs[j];
                }
                layer.biases[i] += alpha * layer.deltas[i];
            }
        }

        return totalError;
    }
}