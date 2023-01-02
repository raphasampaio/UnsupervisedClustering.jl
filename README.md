# UnsupervisedClustering.jl 🟢🔴🟣

## Introduction
UnsupervisedClustering.jl is a Julia package that implements several unsupervised clustering algorithms.

## Getting Started

### Installation

```julia
julia> ] add UnsupervisedClustering
```

### Example
```julia
using UnsupervisedClustering

n = 100
d = 2
k = 2

data = rand(n, d)

kmeans = Kmeans()
result = fit(kmeans, data, k)

```