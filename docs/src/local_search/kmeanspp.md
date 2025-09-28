# K-means++ Clustering

K-means++ is an improved version of k-means that uses a smarter initialization strategy. It selects initial centroids with probability proportional to their squared distance from existing centroids, leading to better clustering quality and faster convergence.

## Algorithm Overview

K-means++ uses the same iterative process as standard k-means but with enhanced initialization:
1. **Smart Initialization**: Select centroids using distance-proportional probability
2. **Assignment**: Assign points to nearest centroids
3. **Update**: Recalculate centroids as cluster means
4. **Repeat**: Continue until convergence

## Usage

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

## API Reference

```@docs
KmeansPlusPlus
```