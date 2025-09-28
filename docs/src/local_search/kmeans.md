# K-means Clustering

K-means is one of the most popular clustering algorithms. It partitions data into k clusters by minimizing the within-cluster sum of squared distances to cluster centroids.

## Algorithm Overview

The k-means algorithm iteratively:
1. Assigns each point to the nearest cluster centroid
2. Updates centroids to the mean of assigned points
3. Repeats until convergence or maximum iterations

## Usage

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

## API Reference

```@autodocs
Modules = [UnsupervisedClustering]
Pages   = ["local_search/kmeans.jl"]
```
