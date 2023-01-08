# UnsupervisedClustering.jl

<div align="center">
    <a href="/docs/src/assets/">
        <img src="/docs/src/assets/logo.svg" width=300px alt="UnsupervisedClustering.jl" />
    </a>
    <br>
    <br>
    <a href="https://raphasampaio.github.io/UnsupervisedClustering.jl/stable">
        <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable">
    </a>
    <a href="https://raphasampaio.github.io/UnsupervisedClustering.jl/dev">
        <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev">
    </a>
    <a href="https://github.com/raphasampaio/UnsupervisedClustering.jl/actions/workflows/CI.yml?query=branch%3Amain">
        <img src="https://github.com/raphasampaio/UnsupervisedClustering.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="CI"/>
    </a>
    <a href="https://codecov.io/gh/raphasampaio/UnsupervisedClustering.jl">
        <img src="https://codecov.io/gh/raphasampaio/UnsupervisedClustering.jl/branch/main/graph/badge.svg" alt="Coverage"/>
    </a>
</div>

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