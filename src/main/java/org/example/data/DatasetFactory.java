package org.example.data;

import java.util.List;
import java.util.Random;

public class DatasetFactory {
    public static final ShapeSample CIRCLE = new ShapeSample(
            "Circle",
            ShapeSample.fromAscii("""
                        ..###..
                        .#...#.
                        #.....#
                        #.....#
                        #.....#
                        .#...#.
                        ..###..
                    """),
            new double[]{1.0, 0.0, 0.0}
    );
    public static final ShapeSample SQUARE = new ShapeSample(
            "Square",
            ShapeSample.fromAscii("""
                        .#####.
                        .#...#.
                        .#...#.
                        .#...#.
                        .#...#.
                        .#...#.
                        .#####.
                    """),
            new double[]{0.0, 1.0, 0.0}
    );

    public static final ShapeSample TRIANGLE = new ShapeSample(
            "Triangle",
            ShapeSample.fromAscii("""
                        .......
                        ...#...
                        ..#.#..
                        .#...#.
                        #.....#
                        #######
                        .......
                    """),
            new double[]{0.0, 0.0, 1.0}
    );

    public static List<ShapeSample> getTrainingSet() {
        return List.of(CIRCLE, SQUARE, TRIANGLE);
    }

    /**
     * Creates a distorted copy of the shape
     * @param original The original shape
     * @param flippedPixelsCount The number of inverted pixels - noise
     * @param seed For result reproducibility
     */
    public static ShapeSample createDistortedSample(ShapeSample original, int flippedPixelsCount, long seed) {
        double[] noisyInputs = original.inputs().clone();
        Random random = new Random(seed);

        for (int i = 0; i < flippedPixelsCount; i++) {
            int pixelIndex = random.nextInt(noisyInputs.length);
            // Inversion: 1.0 -> 0.0 || 0.0 -> 1.0
            noisyInputs[pixelIndex] = (noisyInputs[pixelIndex] == 1.0) ? 0.0 : 1.0;
        }

        return new ShapeSample(original.name() + "_noisy_" + flippedPixelsCount, noisyInputs, original.targets());
    }
}
