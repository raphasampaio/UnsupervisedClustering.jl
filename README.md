# UnsupervisedClustering.jl

<div align="center">
    <a href="/docs/src/assets/">
        <img src="/docs/src/assets/logo.svg" width=250px alt="UnsupervisedClustering.jl" />
    </a>
    <br>
    <br>
    <a href="https://raphasampaio.github.io/UnsupervisedClustering.jl/stable">
        <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable">
    </a>
    <a href="https://raphasampaio.github.io/UnsupervisedClustering.jl/dev">
        <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev">
    </a>
    <a href="https://pkgs.genieframework.com?packages=UnsupervisedClustering">
        <img src="https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/UnsupervisedClustering/label:-sep:">
    </a>
    <br>
    <a href="https://juliahub.com/ui/Packages/UnsupervisedClustering/sHGR0">
        <img src="https://juliahub.com/docs/UnsupervisedClustering/version.svg" alt="Version"/>
    </a>
    <a href="https://github.com/raphasampaio/UnsupervisedClustering.jl/actions/workflows/CI.yml">
        <img src="https://github.com/raphasampaio/UnsupervisedClustering.jl/actions/workflows/CI.yml/badge.svg" alt="CI"/>
    </a>
    <a href="https://codecov.io/gh/raphasampaio/UnsupervisedClustering.jl">
        <img src="https://codecov.io/gh/raphasampaio/UnsupervisedClustering.jl/branch/main/graph/badge.svg" alt="Coverage"/>
    </a>
    <a href="https://github.com/JuliaTesting/Aqua.jl">
        <img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg" alt="Coverage"/>
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
