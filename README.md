# UnsupervisedClustering.jl 🟢🔴🟣

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://raphasampaio.github.io/UnsupervisedClustering.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://raphasampaio.github.io/UnsupervisedClustering.jl/dev/)
[![Build Status](https://github.com/raphasampaio/UnsupervisedClustering.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/raphasampaio/UnsupervisedClustering.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/raphasampaio/UnsupervisedClustering.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/raphasampaio/UnsupervisedClustering.jl)

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