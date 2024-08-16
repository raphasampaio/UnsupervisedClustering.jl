<div align="center"><img src="/docs/src/assets/logo.svg" width=250px alt="UnsupervisedClustering.jl"></img></div>

# UnsupervisedClustering.jl

[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://raphasampaio.github.io/UnsupervisedClustering.jl.jl/stable)
[![CI](https://github.com/raphasampaio/UnsupervisedClustering.jl.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/raphasampaio/UnsupervisedClustering.jl.jl/actions/workflows/CI.yml)
[![Codecov](https://codecov.io/gh/raphasampaio/UnsupervisedClustering.jl.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/raphasampaio/UnsupervisedClustering.jl.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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

### Cite

```bibtex
@article{sampaio2024regularization,
  title={Regularization and optimization in model-based clustering},
  author={Sampaio, Raphael Araujo and Garcia, Joaquim Dias and Poggi, Marcus and Vidal, Thibaut},
  journal={Pattern Recognition},
  pages={110310},
  year={2024},
  publisher={Elsevier}
}
```
