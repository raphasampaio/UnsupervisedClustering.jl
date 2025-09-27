# KmeansPlusPlus

```@docs
KmeansPlusPlus
```

## Example

```julia
using UnsupervisedClustering

n = 100
d = 2
k = 2

data = rand(n, d)

kmeans_pp = KmeansPlusPlus()
result = fit(kmeans_pp, data, k)
```

K-means++ provides better initialization compared to standard K-means by selecting initial centroids with probability proportional to their squared distance from existing centroids. This typically leads to:

- Better clustering quality
- Faster convergence
- More consistent results across runs

The algorithm follows the same interface as regular K-means but uses the improved initialization strategy described in Arthur & Vassilvitskii (2007).