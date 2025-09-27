# Gaussian Mixture Models (GMM)

Gaussian Mixture Models represent data as a mixture of Gaussian distributions, providing probabilistic cluster assignments and soft clustering capabilities. Unlike k-means, GMM can model elliptical clusters and provides uncertainty estimates.

## Algorithm Overview

GMM uses the Expectation-Maximization (EM) algorithm:
1. **E-step**: Calculate probabilities of each point belonging to each Gaussian component
2. **M-step**: Update Gaussian parameters (means, covariances, mixing weights) based on probabilities
3. **Repeat**: Continue until convergence of log-likelihood

## Basic Usage

```julia
using UnsupervisedClustering

# Generate sample data
data = rand(100, 2)
k = 3

# Create covariance estimator (required for GMM)
n, d = size(data)
estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)

# Create and run GMM
gmm = GMM(estimator = estimator)
result = fit(gmm, data, k)

println("Log-likelihood: $(result.objective)")
println("Converged: $(result.converged)")
```

## Covariance Matrix Estimation

GMM requires a covariance estimator to handle numerical stability:

```julia
# Different estimator types
n, d = size(data)

# Empirical covariance (default)
emp_estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)

# For integration with RegularizedCovarianceMatrices.jl
# regularized_estimator = SomeRegularizedEstimator(n, d)

gmm_empirical = GMM(estimator = emp_estimator)
result = fit(gmm_empirical, data, k)
```

## Comparison with K-means

```julia
using UnsupervisedClustering, Random

# Create elliptical clusters (GMM should perform better)
Random.seed!(42)
cluster1 = [randn(50) randn(50) * 0.3] .+ [2, 2]   # Elongated cluster
cluster2 = [randn(50) * 0.3 randn(50)] .+ [-2, -2]  # Elongated cluster
cluster3 = randn(50, 2) .+ [0, 3]                   # Circular cluster
data = vcat(cluster1, cluster2, cluster3)

n, d = size(data)
estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)

# Compare algorithms
algorithms = [
    ("K-means", Kmeans()),
    ("GMM", GMM(estimator = estimator))
]

println("Algorithm Comparison on Elliptical Data:")
for (name, alg) in algorithms
    result = fit(alg, data, 3)
    println("$name: objective = $(round(result.objective, digits=3))")
end
```

## Configuration Options

```julia
# Customize GMM parameters
n, d = size(data)
estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)

gmm = GMM(
    estimator = estimator,
    tolerance = 1e-6,           # Convergence threshold
    max_iterations = 1000,      # Maximum EM iterations
    verbose = false             # Print progress
)

result = fit(gmm, data, k)
```

## Probabilistic Clustering

GMM provides soft cluster assignments (probabilities):

```julia
# Get clustering result
result = fit(GMM(estimator = estimator), data, 3)

# Hard assignments (like k-means)
hard_assignments = result.assignments

# For soft assignments, you would typically access the posterior probabilities
# This requires running the E-step separately or modifying the algorithm
println("Hard cluster assignments: $(hard_assignments[1:10])")
```

## Advanced Example with Model Selection

```julia
using UnsupervisedClustering, Random

# Create complex dataset
Random.seed!(42)
cluster1 = randn(40, 3) .+ [3, 3, 3]
cluster2 = randn(40, 3) .+ [-3, -3, -3]
cluster3 = randn(40, 3) .+ [3, -3, 0]
data = vcat(cluster1, cluster2, cluster3)

# Try different numbers of components
k_values = 2:6
log_likelihoods = Float64[]

n, d = size(data)
estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)

println("Model Selection for GMM:")
for k in k_values
    gmm = GMM(estimator = estimator, max_iterations = 200)
    result = fit(gmm, data, k)
    push!(log_likelihoods, result.objective)
    println("k=$k: log-likelihood = $(round(result.objective, digits=3))")
end

# Higher log-likelihood indicates better fit
best_k = k_values[argmax(log_likelihoods)]
println("Best k based on log-likelihood: $best_k")
```

## Integration with Other Algorithms

```julia
# Use GMM in algorithm chains
n, d = size(data)
estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)

# Initialize with k-means, refine with GMM
chain = ClusteringChain(
    Kmeans(max_iterations = 50),
    GMM(estimator = estimator, max_iterations = 100)
)

result_chain = fit(chain, data, 3)
println("Chained k-means → GMM: $(result_chain.objective)")

# Use GMM with metaheuristics
genetic_gmm = GeneticAlgorithm(
    local_search = GMM(estimator = estimator, max_iterations = 50),
    max_iterations = 50
)

result_genetic = fit(genetic_gmm, data, 3)
println("Genetic GMM: $(result_genetic.objective)")
```

## Handling High-Dimensional Data

```julia
# For high-dimensional data, consider regularization
high_dim_data = rand(100, 20)  # 20-dimensional data
n, d = size(high_dim_data)

# Use empirical estimator
estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)

# Reduce iterations for efficiency
gmm_hd = GMM(
    estimator = estimator,
    max_iterations = 100,
    tolerance = 1e-4
)

result = fit(gmm_hd, high_dim_data, 5)
println("High-dimensional GMM: $(result.objective)")
```

## Performance Comparison

```julia
using Statistics

# Compare convergence behavior
objectives_gmm = Float64[]
objectives_kmeans = Float64[]

for trial in 1:10
    # K-means
    result_km = fit(Kmeans(), data, 3)
    push!(objectives_kmeans, result_km.objective)

    # GMM
    n, d = size(data)
    estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)
    result_gmm = fit(GMM(estimator = estimator), data, 3)
    push!(objectives_gmm, result_gmm.objective)
end

println("K-means objective statistics:")
println("  Mean: $(round(mean(objectives_kmeans), digits=3))")
println("  Std:  $(round(std(objectives_kmeans), digits=3))")

println("GMM log-likelihood statistics:")
println("  Mean: $(round(mean(objectives_gmm), digits=3))")
println("  Std:  $(round(std(objectives_gmm), digits=3))")
```

## When to Use GMM

GMM is preferred when:
- **Probabilistic clusters**: Need soft assignments or uncertainty estimates
- **Non-spherical clusters**: Data has elliptical or complex cluster shapes
- **Overlapping clusters**: Clusters have significant overlap
- **Model-based approach**: Want principled statistical foundation
- **Density estimation**: Need to model data distribution, not just clustering

## Limitations

- **Computational cost**: More expensive than k-means
- **Covariance estimation**: Requires careful handling in high dimensions
- **Local optima**: Can get stuck like other EM-based methods
- **Model assumptions**: Assumes Gaussian distributions

## API Reference

```@autodocs
Modules = [UnsupervisedClustering]
Pages   = ["local_search/gmm.jl"]
```
