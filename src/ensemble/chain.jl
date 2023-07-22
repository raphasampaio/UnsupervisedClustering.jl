@doc """
    ClusteringChain(algorithms::Algorithm...)
    
ClusteringChain represents a chain of clustering algorithms that are executed sequentially. It allows for applying multiple clustering algorithms in a specific order to refine and improve the clustering results.

# Fields
- `algorithms`: the vector of clustering algorithms that will be executed in sequence.
"""
Base.@kwdef struct ClusteringChain <: Algorithm
    algorithms::AbstractVector{<:Algorithm}

    function ClusteringChain(algorithms::Algorithm...)
        return new(collect(algorithms))
    end
end

@doc """
    fit(chain::ClusteringChain, data::AbstractMatrix{<:Real}, k::Integer)

The `fit` function applies a sequence of clustering algorithms and returns a result object representing the clustering outcome.

# Parameters:
- `meta`: an instance representing the clustering settings and parameters.
- `data`: a floating-point matrix, where each row represents a data point, and each column represents a feature.
- `k`: an integer representing the number of clusters.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)

kmeans = Kmeans()
gmm = GMM(estimator = EmpiricalCovarianceMatrix(n, d))

chain = ClusteringChain(kmeans, gmm)
result = fit(chain, data, k)
```
"""
function fit(chain::ClusteringChain, data::AbstractMatrix{<:Real}, k::Integer)
    size = length(chain.algorithms)
    @assert size > 0

    result = fit(chain.algorithms[1], data, k)
    for i in 2:size
        result = fit(chain.algorithms[i], data, result)
    end
    return result
end
