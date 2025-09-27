# Getting Started

This guide will walk you through the basics of using UnsupervisedClustering.jl, from installation to advanced usage patterns.

## Installation

```julia
using Pkg
Pkg.add("UnsupervisedClustering")
```

## Basic Usage

### Your First Clustering

```julia
using UnsupervisedClustering
using Random

# Set seed for reproducibility
Random.seed!(42)

# Generate sample data: 100 points in 2D space
data = rand(100, 2)

# Create a k-means algorithm instance
kmeans = Kmeans()

# Perform clustering with 3 clusters
result = fit(kmeans, data, 3)

println("Clustering completed!")
println("Number of clusters: $(result.k)")
println("Objective value: $(result.objective)")
println("Converged: $(result.converged)")
println("Iterations: $(result.iterations)")
```

### Understanding Results

All algorithms return result objects with a consistent interface:

```julia
# Access cluster assignments
assignments = result.assignments  # Vector indicating cluster for each point
println("First 10 assignments: $(assignments[1:10])")

# Access clustering objective (lower is usually better)
objective = result.objective

# Check convergence status
if result.converged
    println("Algorithm converged in $(result.iterations) iterations")
else
    println("Algorithm did not converge")
end

# Execution time
println("Clustering took $(result.elapsed) seconds")
```

## Trying Different Algorithms

The beauty of UnsupervisedClustering.jl is the unified interface. Let's compare multiple algorithms:

```julia
using UnsupervisedClustering
Random.seed!(42)

# Create a more interesting dataset
n, d, k = 150, 2, 3
data = vcat(
    randn(50, 2) .+ [2, 2],   # Cluster 1
    randn(50, 2) .+ [-2, 2],  # Cluster 2
    randn(50, 2) .+ [0, -2]   # Cluster 3
)

# Try different algorithms
algorithms = [
    ("K-means", Kmeans()),
    ("K-means++", KmeansPlusPlus()),
    ("K-medoids", Kmedoids()),
]

println("Algorithm Comparison:")
println("=" ^ 50)

for (name, algorithm) in algorithms
    result = fit(algorithm, data, k)
    println("$name:")
    println("  Objective: $(round(result.objective, digits=3))")
    println("  Iterations: $(result.iterations)")
    println("  Time: $(round(result.elapsed, digits=4))s")
    println()
end
```

## Advanced Algorithms

### Metaheuristic Approaches

For challenging datasets where local optima are a concern:

```julia
# Genetic Algorithm for global optimization
genetic = GeneticAlgorithm(
    local_search = Kmeans(),
    max_iterations = 100,
    π_max = 50,  # Population size
    verbose = false
)

result_genetic = fit(genetic, data, k)
println("Genetic Algorithm objective: $(result_genetic.objective)")

# Multi-Start for robust results
multi_start = MultiStart(
    local_search = KmeansPlusPlus(),
    max_iterations = 20
)

result_multi = fit(multi_start, data, k)
println("Multi-Start objective: $(result_multi.objective)")
```

### Gaussian Mixture Models

For probabilistic clustering:

```julia
# GMM requires an estimator
n, d = size(data)
estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d)

gmm = GMM(
    estimator = estimator,
    max_iterations = 100,
    tolerance = 1e-6
)

result_gmm = fit(gmm, data, k)
println("GMM log-likelihood: $(result_gmm.objective)")
```

## Algorithm Composition

### Chaining Algorithms

Combine algorithms for improved results:

```julia
# First apply k-means, then refine with GMM
chain = ClusteringChain(
    Kmeans(max_iterations = 10),
    GMM(estimator = estimator, max_iterations = 50)
)

result_chain = fit(chain, data, k)
println("Chained algorithm objective: $(result_chain.objective)")
```

### Nested Metaheuristics

Use one algorithm to improve another:

```julia
# Use genetic algorithm with k-means++ as local search
sophisticated = GeneticAlgorithm(
    local_search = KmeansPlusPlus(max_iterations = 20),
    max_iterations = 50
)

result_sophisticated = fit(sophisticated, data, k)
```

## Working with Real Data

### Loading and Preprocessing

```julia
using DelimitedFiles

# Example: Load data from CSV (replace with your data)
# data = readdlm("your_data.csv", ',', Float64)

# For this example, we'll create realistic data
Random.seed!(123)
data = vcat(
    randn(40, 3) .+ [1, 1, 1],    # Cluster 1: high values
    randn(40, 3) .+ [-1, -1, -1], # Cluster 2: low values
    randn(40, 3) .+ [1, -1, 0]    # Cluster 3: mixed
)

# Standardize features (recommended for k-means)
using Statistics
data_std = (data .- mean(data, dims=1)) ./ std(data, dims=1)

println("Original data range: $(minimum(data)) to $(maximum(data))")
println("Standardized data range: $(minimum(data_std)) to $(maximum(data_std))")
```

### Choosing the Number of Clusters

```julia
# Try different numbers of clusters
k_values = 2:6
objectives = Float64[]

for k in k_values
    result = fit(Kmeans(), data_std, k)
    push!(objectives, result.objective)
end

println("Clustering objectives for different k:")
for (i, k) in enumerate(k_values)
    println("k=$k: objective=$(round(objectives[i], digits=3))")
end

# Choose k with steepest decrease (elbow method)
best_k = k_values[argmin(objectives)]
println("Suggested k: $best_k")
```

## Performance Tips

### 1. Algorithm Selection
```julia
# For speed: K-means
fast_result = fit(Kmeans(max_iterations = 50), data, k)

# For quality: K-means++
quality_result = fit(KmeansPlusPlus(), data, k)

# For robustness: Genetic Algorithm
robust_result = fit(GeneticAlgorithm(local_search = Kmeans()), data, k)
```

### 2. Parameter Tuning
```julia
# Fine-tune convergence
precise_kmeans = Kmeans(
    tolerance = 1e-8,        # Stricter convergence
    max_iterations = 1000    # More iterations allowed
)

# Quick and dirty clustering
fast_kmeans = Kmeans(
    tolerance = 1e-2,        # Looser convergence
    max_iterations = 50      # Fewer iterations
)
```

### 3. Reproducibility
```julia
using Random, StableRNGs

# Use StableRNG for cross-platform reproducibility
rng = StableRNG(42)
kmeans_reproducible = Kmeans(rng = rng)
result = fit(kmeans_reproducible, data, k)
```

## Common Patterns

### Batch Processing
```julia
# Process multiple datasets
datasets = [rand(100, 2), rand(150, 3), rand(200, 4)]
k_values = [3, 4, 5]

results = []
for (data, k) in zip(datasets, k_values)
    algorithm = KmeansPlusPlus(max_iterations = 100)
    result = fit(algorithm, data, k)
    push!(results, result)
    println("Dataset $(length(results)): objective = $(result.objective)")
end
```

### Error Handling
```julia
function safe_clustering(data, k, algorithm)
    try
        if size(data, 1) < k
            error("Number of data points ($(size(data, 1))) must be ≥ k ($k)")
        end

        result = fit(algorithm, data, k)

        if !result.converged
            @warn "Algorithm did not converge"
        end

        return result
    catch e
        @error "Clustering failed: $e"
        return nothing
    end
end

# Usage
result = safe_clustering(data, 3, Kmeans())
if result !== nothing
    println("Clustering successful: $(result.objective)")
end
```

## Next Steps

- Explore specific algorithms in the [Local Search](local_search/kmeans.md) and [Metaheuristic](metaheuristic/genetic_algorithm.md) sections
- Learn about [algorithm composition](ensemble/chain.md) for advanced workflows
- Check the API documentation for detailed parameter descriptions
- See the research paper for theoretical background on the optimization techniques