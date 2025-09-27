[ANN] UnsupervisedClustering.jl – A unified interface for clustering with optimization techniques

I'm happy to announce [UnsupervisedClustering.jl](https://github.com/raphasampaio/UnsupervisedClustering.jl), a Julia package that provides a consistent interface for unsupervised clustering algorithms, along with strategies to escape local optima and reduce overfitting.

UnsupervisedClustering.jl is not a new package, but it was not previously announced here. It was developed during my master's thesis research, where I explored [regularization](https://github.com/raphasampaio/RegularizedCovarianceMatrices.jl) and optimization techniques in model-based clustering. The study resulted in a [published paper](https://arxiv.org/abs/2302.02450) that introduces novel approaches to improve clustering quality and robustness.
 
It has also been used in production at PSR (https://github.com/psrenergy), an energy company and contributor to the JuMP ecosystem.

One of the main advantages of UnsupervisedClustering.jl is its consistent interface across all clustering methods. Every algorithm follows the same simple pattern:

```julia
using UnsupervisedClustering

# All algorithms use the same interface
result = fit(algorithm, data, k)
```

This applies to all implemented methods:

```julia
# Local search algorithms
kmeans = Kmeans()

# Metaheuristic algorithms
genetic = GeneticAlgorithm(local_search = kmeans)
multi_start = MultiStart(local_search = kmeans)
random_swap = RandomSwap(local_search = kmeans)

# All use the same fit function!
result1 = fit(kmeans, data, k)
result2 = fit(genetic, data, k)
result3 = fit(multi_start, data, k)
```

The unified interface enables some compositions:

```julia
kmeans = Kmeans()

estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)
gmm = GMM(estimator = estimator)

# Chain algorithms together
chain = ClusteringChain(kmeans, gmm)
result = fit(chain, data, k)
```


