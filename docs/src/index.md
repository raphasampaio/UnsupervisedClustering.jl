# UnsupervisedClustering.jl

A Julia package providing a unified interface for unsupervised clustering algorithms with advanced optimization techniques to escape local optima and reduce overfitting.

## Key Features

- **Unified Interface**: Almost all clustering algorithms use the same `fit(algorithm, data, k)` pattern (the kmedoids algorithm is an exception, requiring a distance matrix)
- **Advanced Optimization**: Metaheuristic approaches including genetic algorithms and multi-start strategies
- **Type-Stable**: Modern Julia design with parameterized types for zero runtime overhead
- **Composable**: Mix and match algorithms through chaining and metaheuristic composition

## Quick Start

```julia
using UnsupervisedClustering
using RegularizedCovarianceMatrices

# Generate sample data
n = 100
d = 2
k = 3
data = rand(n, d)

# Local search algorithms
kmeans = Kmeans()
gmm = GMM(estimator = EmpiricalCovarianceMatrix(n, d))

# Metaheuristic algorithms
genetic = GeneticAlgorithm(local_search = kmeans)
multi_start = MultiStart(local_search = kmeans)
random_swap = RandomSwap(local_search = kmeans)

# All use the same fit function!
result1 = fit(kmeans, data, k)
result2 = fit(genetic, data, k)
result3 = fit(multi_start, data, k)

# The unified interface enables some compositions:
chain = ClusteringChain(kmeans, gmm)
result4 = fit(chain, data, k)
```

## Algorithm Categories

### Local Search Algorithms
- **[K-means](local_search/kmeans.md)**: Classic centroid-based clustering with Lloyd's algorithm
- **[K-means++](local_search/kmeanspp.md)**: Improved initialization for better clustering quality
- **[K-medoids](local_search/kmedoids.md)**: Robust clustering using actual data points as centers
- **[GMM](local_search/gmm.md)**: Gaussian Mixture Models with EM algorithm

### Metaheuristic Algorithms
- **[Genetic Algorithm](metaheuristic/genetic_algorithm.md)**: Evolutionary approach for global optimization
- **[Multi-Start](metaheuristic/multi_start.md)**: Multiple random initializations with best result selection
- **[Random Swap](metaheuristic/random_swap.md)**: Perturbation-based local search escape

### Ensemble Methods
- **[Clustering Chain](ensemble/chain.md)**: Sequential algorithm composition for refined results

## Installation

```julia
using Pkg
Pkg.add("UnsupervisedClustering")
```

## Related Packages

This package integrates with:
- **[RegularizedCovarianceMatrices.jl](https://github.com/raphasampaio/RegularizedCovarianceMatrices.jl)**: For advanced GMM regularization techniques
- **[Distances.jl](https://github.com/JuliaStats/Distances.jl)**: For distance metrics in clustering algorithms

## Citing

If you find UnsupervisedClustering useful in your work, we kindly request that you cite the following [paper](https://www.sciencedirect.com/science/article/abs/pii/S003132032400061X):

```bibtex
@article{sampaio2024regularization,
  title={Regularization and optimization in model-based clustering},
  author={Sampaio, Raphael Araujo and Garcia, Joaquim Dias and Poggi, Marcus and Vidal, Thibaut},
  journal={Pattern Recognition},
  pages={110310},
  year={2024},
  publisher={Elsevier}
}
```
