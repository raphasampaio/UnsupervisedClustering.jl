# K-medoids Clustering

K-medoids is a robust clustering algorithm that uses actual data points (medoids) as cluster centers instead of computed centroids. This makes it more resistant to outliers and suitable for non-Euclidean distance metrics.

## Algorithm Overview

K-medoids iteratively:
1. **Initialization**: Select k data points as initial medoids
2. **Assignment**: Assign each point to the nearest medoid
3. **Update**: Find the optimal medoid for each cluster (point that minimizes total distance)
4. **Repeat**: Continue until no improvement in objective

## Basic Usage

```jldoctest
using UnsupervisedClustering, Distances, Random

# Generate sample data
Random.seed!(42);
data = rand(100, 2);
k = 3;

# Compute pairwise distances
distances = pairwise(SqEuclidean(), data, dims = 1);

# Create and run k-medoids
kmedoids = Kmedoids();
result = fit(kmedoids, distances, k);

result.objective

# output

7.22
```

## API Reference

```@autodocs
Modules = [UnsupervisedClustering]
Pages   = ["local_search/kmedoids.jl"]
```
