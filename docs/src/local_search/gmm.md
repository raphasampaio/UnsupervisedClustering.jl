# Gaussian Mixture Models (GMM)

Gaussian Mixture Models represent data as a mixture of Gaussian distributions, providing probabilistic cluster assignments and soft clustering capabilities. Unlike k-means, GMM can model elliptical clusters and provides uncertainty estimates.

## Overview

GMM uses the Expectation-Maximization (EM) algorithm:

1. **E-step**: Calculate probabilities of each point belonging to each Gaussian component
2. **M-step**: Update Gaussian parameters (means, covariances, mixing weights) based on probabilities
3. **Repeat**: Continue until convergence of log-likelihood

## Usage

```jldoctest
using UnsupervisedClustering, Random

# Generate sample data
Random.seed!(42);
data = rand(100, 2);
k = 3;

# Create covariance estimator (required for GMM)
n, d = size(data);
estimator = UnsupervisedClustering.EmpiricalCovarianceMatrix(n, d);

# Create and run GMM
gmm = GMM(estimator = estimator);
result = fit(gmm, data, k);

result.objective

# output
-0.43900384415707366
```

## API Reference

```@autodocs
Modules = [UnsupervisedClustering]
Pages   = ["local_search/gmm.jl"]
```
