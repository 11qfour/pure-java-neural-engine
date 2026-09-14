package org.example.data;

public record ShapeSample(String name, double[] inputs, double[] targets) {

    public static double[] fromAscii(String asciiArt) {
        return asciiArt.lines()
                .map(String::trim)
                .filter(line -> !line.isEmpty())
                .flatMapToInt(String::chars)
                .filter(c -> c == '#' || c == '.')
                .mapToDouble(c -> c == '#' ? 1.0 : 0.0)
                .toArray();
    }
}