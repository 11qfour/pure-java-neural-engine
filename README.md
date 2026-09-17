# Pure Java Neural Engine

A lightweight, high-performance, **zero-dependency** Multilayer Perceptron (MLP) implementation from scratch in **pure Java 21**. 

Designed with a systems engineering focus: cache-friendly data layouts, explicit forward/backward propagation math, low-latency inference, and minimal memory footprint.

[![Java](https://img.shields.io/badge/Java-21-orange.svg)](https://openjdk.org/projects/jdk/21/)
[![Build Tool](https://img.shields.io/badge/Gradle-8.10-blue.svg)](https://gradle.org)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architectural Highlights

* **Zero External Dependencies:** Built strictly using `java.base` without third-party linear algebra or deep learning libraries (no DL4J, ND4J, or PyTorch JNI bindings).
* **L1 Cache Residency:** Model parameters for inference require only **~6.80 KB**, ensuring 100% residency inside the CPU **L1 Data Cache (L1D)** for maximum throughput without memory bus stalls.
* **Low-Latency Inference:** Sub-microsecond execution time (**~1.14 µs** per sample), achieving **>875,000 classifications/sec** on a single CPU core.
* **Arbitrary Deep Topology:** Configurable layers via dynamic topology varargs: `new NeuralNetwork(49, 16, 3)` or multi-layer architectures.
* **Production-Grade OOD Handling:** Gated inference with a configurable **Confidence Threshold (90%)** for reliable rejection of heavy noise and Out-of-Distribution (OOD) samples.
* **Dual-Domain Verification:** Validated on non-linear **XOR** convergence and **7x7 Bitmap Geometric Shape Recognition**.

---

## Mathematical Formulation

The engine implements standard stochastic gradient descent with Backpropagation:

1. **Forward Propagation:**
   $$y_i^k = \sigma \left( \sum_{j} w_{ij}^k \cdot y_j^{k-1} + b_i^k \right)$$
   Where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic sigmoid activation function.

2. **Output Layer Error Gradient:**
   $$\delta_i^N = y_i^N (1 - y_i^N)(t_i - y_i^N)$$

3. **Hidden Layer Error Propagation:**
   $$\delta_i^k = y_i^k (1 - y_i^k) \sum_{j} \delta_j^{k+1} \cdot w_{ji}^{k+1}$$

4. **Weight & Bias Updates:**
   $$w_{ij}^k \leftarrow w_{ij}^k + \alpha \cdot \delta_i^k \cdot y_j^{k-1}$$
   $$b_i^k \leftarrow b_i^k + \alpha \cdot \delta_i^k$$

---

## Benchmark & Systems Performance

Evaluated on OpenJDK 21 (x86_64 architecture):

| Metric | Result | System Significance |
| :--- | :--- | :--- |
| **Average Inference Latency** | **1.14 µs** (1,142 ns) | Pure Java forward pass without framework overhead |
| **Throughput (Single Core)** | **~875,600 ops/sec** | Measured over 200,000 iterations after JIT warm-up |
| **Inference Memory Footprint** | **6.80 KB** (851 `double` params) | Fits entirely within L1D Cache (typically 32–48 KB) |
| **Training Memory Footprint** | **13.45 KB** (weights + deltas) | Zero intermediate allocations in heap during forward/backward steps |
| **Training Convergence Time** | **93 ms** (~1,850 epochs) | To target error threshold $E \le 0.001$ |

### Topology Comparison (Ablation Study)

Evaluation of hidden layer capacities under identical training conditions ($\alpha = 0.3, E < 0.001$):

| Topology | Parameters | Memory (KB) | Epochs to Converge | Generalization Confidence (Noise = 2) | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `[49 -> 8 -> 3]` | 427 | 3.34 KB | 3,143 | 97.4% | Slow convergence; narrow bottleneck |
| **`[49 -> 16 -> 3]`** | **851** | **6.65 KB** | **1,849** | **97.1%** | **Optimal Pareto balance (Selected)** |
| `[49 -> 32 -> 3]` | 1,699 | 13.27 KB | 1,239 | 96.6% | Parameter redundancy; slight overfitting risk |

---

## 7x7 Shape Classification & Robustness

The engine accepts $7 \times 7$ binary bitmap images representing geometric shapes:

```text
    CIRCLE                 SQUARE                TRIANGLE
  ..###..                .#####.                .......
  .#...#.                .#...#.                ...#...
  #.....#                .#...#.                ..#.#..
  #.....#                .#...#.                .#...#.
  #.....#                .#...#.                #.....#
  .#...#.                .#...#.                #######
  ..###..                .#####.                .......
```

### Noise Resistance & Out-of-Distribution Rejection

With a **90% confidence threshold gate**, the model correctly identifies valid shapes while rejecting severe distortions and unfamiliar inputs:

```text
>>> Clean Shapes:
  - Circle             -> RECOGNITION: Circle     (confidence: 98.2%)
  - Square             -> RECOGNITION: Square     (confidence: 98.3%)
  - Triangle           -> RECOGNITION: Triangle   (confidence: 98.4%)

>>> Distorted Shapes (2 dead pixels - ~4% noise):
  - Circle_noisy_2     -> RECOGNITION: Circle     (confidence: 97.3%)
  - Square_noisy_2     -> RECOGNITION: Square     (confidence: 97.6%)

>>> Critical Distortion & Anomalies (Failure Cases):
  - Triangle_noisy_18  -> NO RECOGNITION (Max confidence: 89.1% < 90%)
  - Unknown_Cross      -> NO RECOGNITION (Max confidence: 30.7% < 90%)
```

---

## Project Structure

```text
pure-java-neural-engine/
├── build.gradle.kts                   # Gradle build script (Kotlin DSL)
├── gradle.properties                  # JVM environment configuration
└── src/
    ├── main/java/org/example/
    │   ├── Main.java                  # CLI demo & benchmarking execution
    │   ├── benchmark/
    │   │   └── LatencyProfiler.java   # JIT warm-up, latency & RAM profiling
    │   ├── data/
    │   │   ├── DatasetFactory.java    # Dataset definitions & noise generator
    │   │   └── ShapeSample.java       # Record definition with ASCII parser
    │   └── model/
    │       ├── Layer.java             # Weights, biases, activations, deltas
    │       └── NeuralNetwork.java     # Forward/backward pass orchestration
    └── test/java/org/example/
        ├── NeuralNetworkTest.java     # Unit tests (XOR, gradient decrease)
        └── ShapeSampleTest.java       # ASCII parsing & shape dimension tests
```

---

## Quick Start

### Prerequisites
* **Java Development Kit (JDK):** Version 21 (or compatible 21+)
* **Gradle:** 8.10+ (Wrapper included)

### Build & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/11qfour/pure-java-neural-engine.git
   cd pure-java-neural-engine
   ```

2. **Compile:**
   ```bash
   ./gradlew compileJava
   ```

3. **Run the Full Suite (Training, Evaluation, Benchmarking):**
   ```bash
   ./gradlew run
   ```

4. **Run Unit Tests:**
   ```bash
   ./gradlew test
   ```

---

## Quick Code Example

```java
// 1. Initialize network: 49 inputs -> 16 hidden neurons -> 3 outputs
NeuralNetwork net = new NeuralNetwork(49, 16, 3);

// 2. Train on sample
double error = net.trainSample(sampleInputs, targetOneHot, 0.3);

// 3. Predict output
double[] probabilities = net.predict(sampleInputs);
```

---

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
```