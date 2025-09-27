# K-means Clustering

K-means is one of the most popular clustering algorithms. It partitions data into k clusters by minimizing the within-cluster sum of squared distances to cluster centroids.

## Algorithm Overview

The k-means algorithm iteratively:
1. Assigns each point to the nearest cluster centroid
2. Updates centroids to the mean of assigned points
3. Repeats until convergence or maximum iterations

## Basic Usage

```julia
using UnsupervisedClustering

# Generate sample data
data = rand(100, 2)
k = 3

# Create and run k-means
kmeans = Kmeans()
result = fit(kmeans, data, k)

println("Objective: $(result.objective)")
println("Converged: $(result.converged)")
```

## Variants

UnsupervisedClustering.jl provides two k-means variants:

### Standard K-means
```julia
kmeans = Kmeans(
    metric = SqEuclidean(),      # Distance metric
    tolerance = 1e-3,            # Convergence threshold
    max_iterations = 1000,       # Maximum iterations
    verbose = false              # Print progress
)
```

### Balanced K-means
Forces approximately equal cluster sizes:

```julia
balanced_kmeans = BalancedKmeans(
    metric = SqEuclidean(),
    tolerance = 1e-3,
    max_iterations = 1000
)

result = fit(balanced_kmeans, data, k)
```

## Advanced Example

```julia
using UnsupervisedClustering, Random, Statistics

# Create structured data with known clusters
Random.seed!(42)
cluster1 = randn(50, 2) .+ [2, 2]
cluster2 = randn(50, 2) .+ [-2, -2]
cluster3 = randn(50, 2) .+ [2, -2]
data = vcat(cluster1, cluster2, cluster3)

# Standardize data (recommended)
data_std = (data .- mean(data, dims=1)) ./ std(data, dims=1)

# Configure k-means with custom parameters
kmeans = Kmeans(
    tolerance = 1e-6,     # High precision
    max_iterations = 500,
    verbose = true        # Show progress
)

result = fit(kmeans, data_std, 3)

# Analyze results
println("Clusters found: $(result.k)")
println("Final objective: $(result.objective)")
println("Iterations needed: $(result.iterations)")

# Check cluster assignments
for i in 1:3
    cluster_points = sum(result.assignments .== i)
    println("Cluster $i: $cluster_points points")
end
```

## Parameter Selection

### Choosing the Number of Clusters (k)

Use the elbow method to find optimal k:

```julia
function elbow_method(data, max_k=10)
    objectives = Float64[]

    for k in 1:max_k
        if k > size(data, 1)
            break
        end

        result = fit(Kmeans(), data, k)
        push!(objectives, result.objective)
    end

    return objectives
end

# Find elbow point
objectives = elbow_method(data_std, 8)

println("Objectives by k:")
for (k, obj) in enumerate(objectives)
    println("k=$k: $(round(obj, digits=3))")
end
```

### Distance Metrics

K-means supports different distance metrics:

```julia
using Distances

# Euclidean distance (default)
kmeans_euclidean = Kmeans(metric = Euclidean())

# Squared Euclidean (faster, same results)
kmeans_sq_euclidean = Kmeans(metric = SqEuclidean())

# Manhattan distance
kmeans_manhattan = Kmeans(metric = Cityblock())

# Compare results
algorithms = [
    ("Euclidean", kmeans_euclidean),
    ("Squared Euclidean", kmeans_sq_euclidean),
    ("Manhattan", kmeans_manhattan)
]

for (name, alg) in algorithms
    result = fit(alg, data, 3)
    println("$name: objective = $(round(result.objective, digits=3))")
end
```

## Integration with Metaheuristics

K-means works as a local search component in metaheuristic algorithms:

```julia
# Use k-means in genetic algorithm
genetic_kmeans = GeneticAlgorithm(
    local_search = Kmeans(max_iterations = 50),
    max_iterations = 100
)

# Use k-means in multi-start
multi_start_kmeans = MultiStart(
    local_search = Kmeans(),
    max_iterations = 20
)

# Compare with standard k-means
standard_result = fit(Kmeans(), data, 3)
genetic_result = fit(genetic_kmeans, data, 3)
multi_result = fit(multi_start_kmeans, data, 3)

println("Standard k-means: $(standard_result.objective)")
println("Genetic k-means: $(genetic_result.objective)")
println("Multi-start k-means: $(multi_result.objective)")
```

## Performance Considerations

### Large Datasets
```julia
# For large datasets, reduce iterations
fast_kmeans = Kmeans(
    max_iterations = 50,
    tolerance = 1e-2
)

# Or use with multi-start for quality
fast_multi = MultiStart(
    local_search = fast_kmeans,
    max_iterations = 5
)
```

### Reproducibility
```julia
using StableRNGs

# Deterministic results across platforms
rng = StableRNG(42)
kmeans_reproducible = Kmeans(rng = rng)
result = fit(kmeans_reproducible, data, 3)
```

## API Reference

```@autodocs
Modules = [UnsupervisedClustering]
Pages   = ["local_search/kmeans.jl"]
```
