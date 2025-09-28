# K-medoids Clustering

K-medoids is a robust clustering algorithm that uses actual data points (medoids) as cluster centers instead of computed centroids. This makes it more resistant to outliers and suitable for non-Euclidean distance metrics.

## Algorithm Overview

K-medoids iteratively:
1. **Initialization**: Select k data points as initial medoids
2. **Assignment**: Assign each point to the nearest medoid
3. **Update**: Find the optimal medoid for each cluster (point that minimizes total distance)
4. **Repeat**: Continue until no improvement in objective

## Basic Usage

```julia
using UnsupervisedClustering, Distances

# Generate sample data
data = rand(100, 2)
k = 3

# Compute pairwise distances
distances = pairwise(SqEuclidean(), data, dims = 1)

# Create and run k-medoids
kmedoids = Kmedoids()
result = fit(kmedoids, distances, k)

println("Objective: $(result.objective)")
println("Medoids: $(result.centroids)")
```

## Comparison with K-means

```julia
using UnsupervisedClustering, Random, Distances

# Create data with outliers
Random.seed!(42)
normal_data = randn(90, 2)
outliers = rand(10, 2) .* 10  # Add extreme outliers
data = vcat(normal_data, outliers)

# Compute pairwise distances for k-medoids
distances = pairwise(SqEuclidean(), data, dims = 1)

# Compare robustness
algorithms = [
    ("K-means", Kmeans(), data),
    ("K-medoids", Kmedoids(), distances)
]

println("Robustness Comparison:")
for (name, alg, input_data) in algorithms
    result = fit(alg, input_data, 3)
    println("$name: objective = $(round(result.objective, digits=3))")
end
```

## Configuration Options

```julia
# Customize k-medoids parameters
kmedoids = Kmedoids(
    metric = Euclidean(),        # Distance metric
    max_iterations = 1000,       # Maximum iterations
    verbose = false              # Print progress
)

# Compute distances with the specified metric
distances = pairwise(kmedoids.metric, data, dims = 1)
result = fit(kmedoids, distances, k)
```

## Working with Different Distance Metrics

K-medoids works well with various distance metrics:

```julia
using Distances

# Try different distance metrics
metrics = [
    ("Euclidean", Euclidean()),
    ("Manhattan", Cityblock()),
    ("Chebyshev", Chebyshev())
]

println("Distance Metric Comparison:")
for (name, metric) in metrics
    distances = pairwise(metric, data, dims = 1)
    kmedoids = Kmedoids(metric = metric)
    result = fit(kmedoids, distances, 3)
    println("$name: objective = $(round(result.objective, digits=3))")
end
```

## Categorical Data Example

K-medoids excels with categorical or mixed data:

```julia
# Simulate categorical data (using Hamming distance)
Random.seed!(42)
categorical_data = rand(0:3, 100, 5)  # 5 categorical features with 4 levels each

# Compute Hamming distances for categorical data
distances = pairwise(Hamming(), categorical_data, dims = 1)

# Use Hamming distance for categorical data
hamming_kmedoids = Kmedoids(metric = Hamming())
result = fit(hamming_kmedoids, distances, 3)

println("Categorical clustering objective: $(result.objective)")
```

## Advanced Example with Preprocessing

```julia
using UnsupervisedClustering, Random, Statistics, Distances

# Create complex dataset
Random.seed!(42)
cluster1 = randn(40, 3) .+ [2, 2, 2]
cluster2 = randn(40, 3) .+ [-2, -2, -2]
cluster3 = randn(40, 3) .+ [2, -2, 0]
data = vcat(cluster1, cluster2, cluster3)

# Add some outliers
outliers = randn(20, 3) .* 5
data_with_outliers = vcat(data, outliers)

# Compute pairwise distances
distances = pairwise(Euclidean(), data_with_outliers, dims = 1)

# Configure robust k-medoids
kmedoids = Kmedoids(
    metric = Euclidean(),
    max_iterations = 500,
    verbose = true
)

result = fit(kmedoids, distances, 3)

println("Robust clustering completed:")
println("  Objective: $(result.objective)")
println("  Iterations: $(result.iterations)")
println("  Medoids are actual data points: $(all(in.(eachrow(result.centroids), [eachrow(data_with_outliers)])))")
```

## Integration with Metaheuristics

```julia
# Compute distances for all algorithms
distances = pairwise(SqEuclidean(), data, dims = 1)

# Use k-medoids with genetic algorithm
genetic_kmedoids = GeneticAlgorithm(
    local_search = Kmedoids(max_iterations = 50),
    max_iterations = 100
)

# Multi-start k-medoids
multi_kmedoids = MultiStart(
    local_search = Kmedoids(),
    max_iterations = 15
)

# Compare approaches
standard_result = fit(Kmedoids(), distances, 3)
genetic_result = fit(genetic_kmedoids, distances, 3)
multi_result = fit(multi_kmedoids, distances, 3)

println("Standard k-medoids: $(standard_result.objective)")
println("Genetic k-medoids: $(genetic_result.objective)")
println("Multi-start k-medoids: $(multi_result.objective)")
```

## Performance Considerations

```julia
# For large datasets, limit iterations
fast_kmedoids = Kmedoids(max_iterations = 50)

# Compare timing
large_data = rand(1000, 10)
large_distances = pairwise(SqEuclidean(), large_data, dims = 1)

# Time standard vs fast versions
@time result1 = fit(Kmedoids(), large_distances, 5)
@time result2 = fit(fast_kmedoids, large_distances, 5)

println("Standard: $(result1.objective) in $(result1.elapsed)s")
println("Fast: $(result2.objective) in $(result2.elapsed)s")
```

## When to Use K-medoids

K-medoids is preferred when:
- **Outliers present**: More robust than k-means to extreme values
- **Non-Euclidean metrics**: Works with any distance metric
- **Categorical data**: Natural choice for discrete features
- **Interpretable centers**: Medoids are actual data points
- **Small-medium datasets**: Computationally more expensive than k-means

## API Reference

```@autodocs
Modules = [UnsupervisedClustering]
Pages   = ["local_search/kmedoids.jl"]
```
