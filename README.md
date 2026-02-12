# Inverse MST Benchmark

This project provides a comprehensive benchmarking suite for the Inverse Minimum Spanning Tree (IMST) problem under the $L_1$ norm. It compares an **Optimal Algorithm** based on Min-Cost Max-Flow (MCMF) against a **Greedy Heuristic**.

The benchmark focuses on two key metrics:
1. **Optimality Gap**: Measuring the cost difference (total modification cost) between the optimal solution and the heuristic.
2. **Computational Complexity**: Analyzing the runtime scalability of both approaches across different problem sizes.

## Algorithms Benchmark

- **MCMF (Min-Cost Max-Flow)**:
  - Theoretical Complexity: $O(k \cdot M \cdot N)$, where $k$ is the number of augmentation steps.
  - Guaranteed to find the optimal solution with minimum cost.
  - Implemenation uses the SPFA (Shortest Path Faster Algorithm) for augmenting paths.
  
- **Greedy Heuristic**:
  - Theoretical Complexity: $O(M \cdot N)$ (dominated by path finding on the tree).
  - Iterates through non-tree edges and attempts to resolve conflicts locally.
  - **Note**: This approach is suboptimal and often overestimates the cost significantly as it fails to coordinate changes across shared tree edges.

## Prerequisites

- **C++ Compiler**: Must support **C++17** standard (e.g., `g++` 7.0+).
- **Python 3.x**: Recommended 3.8+.
- **Python Libraries**:
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `numpy`

## Usage

### 1. Compilation

Compile the C++ benchmark source code using `g++`:

```bash
g++ -O3 -std=c++17 -o bench benchmark_pro.cpp
```

### 2. Running Benchmarks

Execute the compiled binary to run the experiments. This will process test cases from the `MSTtests/` directory (ensure this directory exists and contains `.in` files).

```bash
./bench
```

The program will output progress to the console and save the results in `advanced_benchmark.csv`.

### 3. Visualization

Run the Python script to generate the analysis report. This script reads `advanced_benchmark.csv` and produces high-quality plots.

```bash
python3 visualize.py
```

The script will generate:
- **`Analysis_Report.png`**: A visualization figure containing:
    - **(a) Optimality Analysis**: Bar chart comparing the $L_1$ modification cost.
    - **(b) Runtime Scalability**: Log-log plot of execution time vs. problem size ($N \times M$), with fitted complexity curves.

## File Structure

- `benchmark_pro.cpp`: Main C++ source file implementing the algorithms and benchmarking logic.
- `visualize.py`: Python script for data analysis and visualization (IEEE/ACM style plots).
- `MSTtests/`: Directory containing input test cases (`*.in`).
- `advanced_benchmark.csv`: Output data file (generated).
- `Analysis_Report.png`: Output visualization (generated).

## Author

**Lizhan Hong**  
Email: lizhan.hong@polytechnique.edu
