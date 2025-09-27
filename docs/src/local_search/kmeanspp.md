# K-means++ Clustering

K-means++ is an improved version of k-means that uses a smarter initialization strategy. It selects initial centroids with probability proportional to their squared distance from existing centroids, leading to better clustering quality and faster convergence.

## Algorithm Overview

K-means++ uses the same iterative process as standard k-means but with enhanced initialization:
1. **Smart Initialization**: Select centroids using distance-proportional probability
2. **Assignment**: Assign points to nearest centroids
3. **Update**: Recalculate centroids as cluster means
4. **Repeat**: Continue until convergence

## Basic Usage

```julia
using UnsupervisedClustering

# Generate sample data
data = rand(100, 2)
k = 3

# Create and run K-means++
kmeanspp = KmeansPlusPlus()
result = fit(kmeanspp, data, k)

println("Objective: $(result.objective)")
println("Converged: $(result.converged)")
```

## Comparison with Standard K-means

```julia
using UnsupervisedClustering, Random

# Create structured data with clear clusters
Random.seed!(42)
cluster1 = randn(50, 2) .+ [3, 3]
cluster2 = randn(50, 2) .+ [-3, -3]
cluster3 = randn(50, 2) .+ [3, -3]
data = vcat(cluster1, cluster2, cluster3)

# Compare algorithms
algorithms = [
    ("Standard K-means", Kmeans()),
    ("K-means++", KmeansPlusPlus())
]

println("Algorithm Comparison:")
for (name, alg) in algorithms
    result = fit(alg, data, 3)
    println("$name: objective = $(round(result.objective, digits=3))")
end
```

## Configuration Options

```julia
# Customize K-means++ parameters
kmeanspp = KmeansPlusPlus(
    metric = SqEuclidean(),      # Distance metric
    tolerance = 1e-6,            # Convergence threshold
    max_iterations = 1000,       # Maximum iterations
    verbose = false              # Print progress
)

result = fit(kmeanspp, data, k)
```

## Multiple Runs for Robustness

K-means++ provides better initialization, but multiple runs can still improve results:

```julia
# Run multiple times and select best result
best_objective = Inf
best_result = nothing

for i in 1:10
    result = fit(KmeansPlusPlus(), data, 3)
    if result.objective < best_objective
        best_objective = result.objective
        best_result = result
    end
end

println("Best objective after 10 runs: $(best_objective)")
```

## Integration with Metaheuristics

Use K-means++ as a local search component:

```julia
# Genetic algorithm with K-means++ local search
genetic_kmeanspp = GeneticAlgorithm(
    local_search = KmeansPlusPlus(max_iterations = 50),
    max_iterations = 100
)

# Multi-start with K-means++
multi_kmeanspp = MultiStart(
    local_search = KmeansPlusPlus(),
    max_iterations = 20
)

# Compare results
standard_result = fit(KmeansPlusPlus(), data, 3)
genetic_result = fit(genetic_kmeanspp, data, 3)
multi_result = fit(multi_kmeanspp, data, 3)

println("Standard K-means++: $(standard_result.objective)")
println("Genetic K-means++: $(genetic_result.objective)")
println("Multi-start K-means++: $(multi_result.objective)")
```

## Performance Benefits

```julia
using Statistics

# Measure consistency across multiple runs
objectives_kmeans = Float64[]
objectives_kmeanspp = Float64[]

for i in 1:20
    result1 = fit(Kmeans(), data, 3)
    result2 = fit(KmeansPlusPlus(), data, 3)

    push!(objectives_kmeans, result1.objective)
    push!(objectives_kmeanspp, result2.objective)
end

println("Standard K-means:")
println("  Mean: $(round(mean(objectives_kmeans), digits=3))")
println("  Std:  $(round(std(objectives_kmeans), digits=3))")

println("K-means++:")
println("  Mean: $(round(mean(objectives_kmeanspp), digits=3))")
println("  Std:  $(round(std(objectives_kmeanspp), digits=3))")
```

## When to Use K-means++

K-means++ is recommended when:
- You need consistent, high-quality results
- Standard k-means shows high variance across runs
- Initialization quality significantly impacts your application
- You want faster convergence than standard k-means

## API Reference

```@docs
KmeansPlusPlus
```