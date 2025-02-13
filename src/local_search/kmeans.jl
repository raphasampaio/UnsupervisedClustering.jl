@doc """
    Kmeans(
        metric::SemiMetric = SqEuclidean()
        verbose::Bool = DEFAULT_VERBOSE
        rng::AbstractRNG = Random.GLOBAL_RNG
        tolerance::Real = DEFAULT_TOLERANCE
        max_iterations::Integer = DEFAULT_MAX_ITERATIONS
    )

The k-means is a clustering algorithm that aims to partition data into clusters by minimizing the distances between data points and their cluster centroids.

# Fields
- `metric`: defines the distance metric used to compute the distances between data points and cluster centroids.
- `verbose`: controls whether the algorithm should display additional information during execution.
- `rng`: represents the random number generator to be used by the algorithm.
- `tolerance`: represents the convergence criterion for the algorithm. It determines the maximum change allowed in the centroid positions between consecutive iterations.
- `max_iterations`: represents the maximum number of iterations the algorithm will perform before stopping, even if convergence has not been reached.

# References
* Hartigan, John A., and Manchek A. Wong.
  Algorithm AS 136: A k-means clustering algorithm.
  Journal of the royal statistical society. series c (applied statistics) 28.1 (1979): 100-108.
* Lloyd, Stuart.
  Least squares quantization in PCM.
  IEEE transactions on information theory 28.2 (1982): 129-137.
"""
Base.@kwdef mutable struct Kmeans <: AbstractKmeans
    metric::SemiMetric = SqEuclidean()
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    tolerance::Real = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

@doc """
    BalancedKmeans(
        metric::SemiMetric = SqEuclidean()
        verbose::Bool = DEFAULT_VERBOSE
        rng::AbstractRNG = Random.GLOBAL_RNG
        tolerance::Real = DEFAULT_TOLERANCE
        max_iterations::Integer = DEFAULT_MAX_ITERATIONS
    )
    
The balanced kmeans is a variation of the k-means clustering algorithm that balances the number of data points assigned to each cluster.

# Fields
- `metric`: defines the distance metric used to compute the distances between data points and cluster centroids.
- `verbose`: controls whether the algorithm should display additional information during execution.
- `rng`: represents the random number generator to be used by the algorithm.
- `tolerance`: represents the convergence criterion for the algorithm. It determines the maximum change allowed in the centroid positions between consecutive iterations.
- `max_iterations`: represents the maximum number of iterations the algorithm will perform before stopping, even if convergence has not been reached.
"""
Base.@kwdef mutable struct BalancedKmeans <: AbstractKmeans
    metric::SemiMetric = SqEuclidean()
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    tolerance::Real = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

@doc """
    KmeansResult(
        assignments::AbstractVector{<:Integer}
        clusters::AbstractMatrix{<:Real}
        objective::Real
        objective_per_cluster::AbstractVector{<:Real}
        iterations::Integer
        elapsed::Real
        converged::Bool
        k::Integer
    )

KmeansResult struct represents the result of the k-means clustering algorithm.

# Fields
- `assignments`: an integer vector that stores the cluster assignment for each data point.
- `clusters`: a floating-point matrix representing the cluster's centroid.
- `objective`: a floating-point number representing the objective function after running the algorithm. The objective function measures the quality of the clustering solution.
- `objective_per_cluster`: a floating-point vector that stores the objective function of each cluster
- `iterations`: an integer value indicating the number of iterations performed until the algorithm has converged or reached the maximum number of iterations
- `elapsed`: a floating-point number representing the time in seconds for the algorithm to complete.
- `converged`: indicates whether the algorithm has converged to a solution.
- `k`: the number of clusters.
"""
mutable struct KmeansResult{I <: Integer, R <: Real} <: AbstractResult
    assignments::Vector{I}
    clusters::Matrix{R}
    objective::R
    objective_per_cluster::Vector{R}
    iterations::I
    elapsed::R
    converged::Bool
    k::I

    function KmeansResult(
        assignments::AbstractVector{I},
        clusters::AbstractMatrix{R},
        objective::R = Inf,
        objective_per_cluster::AbstractVector{R} = Inf * ones(size(clusters, 2)),
        iterations::I = 0,
        elapsed::R = 0.0,
        converged::Bool = false,
    ) where {I <: Integer, R <: Real}
        return new{I, R}(
            assignments,
            clusters,
            objective,
            objective_per_cluster,
            iterations,
            elapsed,
            converged,
            size(clusters, 2),
        )
    end
end

function KmeansResult(d::Integer, n::Integer, k::Integer)
    return KmeansResult(zeros(Int, n), zeros(d, k))
end

function KmeansResult(n::Integer, clusters::AbstractMatrix{<:Real})
    d, k = size(clusters)
    result = KmeansResult(d, n, k)
    result.clusters = copy(clusters)
    return result
end

function initialize!(result::KmeansResult, data::AbstractMatrix{<:Real}, indices::AbstractVector{<:Integer}; verbose::Bool = false)
    n, d = size(data)
    k = length(indices)

    for i in 1:d
        for j in 1:k
            result.clusters[i, j] = data[indices[j], i]
        end
    end

    if verbose
        print_initial_clusters(indices)
    end

    return nothing
end

@doc """
    fit!(
        kmeans::AbstractKmeans,
        data::AbstractMatrix{<:Real},
        result::KmeansResult
    )

The `fit!` function performs the k-means clustering algorithm on the given result as the initial point and updates the provided object with the clustering result.

# Parameters:
- `kmeans`: an instance representing the clustering settings and parameters.
- `data`: a floating-point matrix, where each row represents a data point, and each column represents a feature.
- `result`: a result object that will be updated with the clustering result.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)

kmeans = Kmeans()
result = KmeansResult(n, [1.0 2.0; 1.0 2.0])
fit!(kmeans, data, result)
```
"""
function fit!(kmeans::AbstractKmeans, data::AbstractMatrix{<:Real}, result::KmeansResult)
    t = time()

    n, d = size(data)
    k = result.k

    previous_objective = -Inf
    reset_objective!(result)

    result.iterations = kmeans.max_iterations
    result.converged = false

    clusters_size = zeros(Int, k)
    distances = zeros(k, n)
    is_empty = trues(k)

    for iteration in 1:kmeans.max_iterations
        previous_objective = result.objective

        result.objective = 0.0
        for i in 1:k
            result.objective_per_cluster[i] = 0.0
            is_empty[i] = true
        end

        pairwise!(kmeans.metric, distances, result.clusters, data', dims = 2)

        assignment_step!(
            kmeans,
            result = result,
            distances = distances,
            is_empty = is_empty,
        )

        fill!(result.objective_per_cluster, 0.0)
        for i in 1:n
            cluster = result.assignments[i]
            result.objective_per_cluster[cluster] += distances[cluster, i]
        end
        result.objective = sum(result.objective_per_cluster)

        change = abs(result.objective - previous_objective)

        if kmeans.verbose
            print_iteration(iteration)
            print_objective(result)
            print_change(change)
            print_newline()
        end

        # stopping condition
        if change < kmeans.tolerance || n == k
            result.converged = true
            result.iterations = iteration
            break
        end

        # update step
        for i in 1:k
            clusters_size[i] = 0
            for j in 1:d
                result.clusters[j, i] = 0
            end
        end

        for i in 1:n
            assignment = result.assignments[i]
            for j in 1:d
                result.clusters[j, assignment] += data[i, j]
            end
            clusters_size[assignment] += 1
        end

        for i in 1:k
            cluster_size = max(1, clusters_size[i])
            for j in 1:d
                result.clusters[j, i] = result.clusters[j, i] / cluster_size
            end
        end
    end

    result.elapsed = time() - t

    return nothing
end

@doc """
    fit(
        kmeans::AbstractKmeans,
        data::AbstractMatrix{<:Real},
        initial_clusters::AbstractVector{<:Integer}
    )

The `fit` function performs the k-means clustering algorithm on the given data points as the initial point and returns a result object representing the clustering result.

# Parameters:
- `kmeans`: an instance representing the clustering settings and parameters.
- `data`: a floating-point matrix, where each row represents a data point, and each column represents a feature.
- `initial_clusters`: an integer vector where each element is the initial data point for each cluster.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)

kmeans = Kmeans()
result = fit(kmeans, data, [4, 12])
```
"""
function fit(kmeans::AbstractKmeans, data::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})::KmeansResult
    n, d = size(data)
    k = length(initial_clusters)

    result = KmeansResult(d, n, k)
    if n == 0
        return result
    end

    @assert d > 0
    @assert k > 0
    @assert n >= k

    initialize!(result, data, initial_clusters, verbose = kmeans.verbose)

    fit!(kmeans, data, result)

    return result
end

@doc """
    fit(
        kmeans::AbstractKmeans,
        data::AbstractMatrix{<:Real},
        k::Integer
    )

The `fit` function performs the k-means clustering algorithm and returns a result object representing the clustering result.

# Parameters:
- `kmeans`: an instance representing the clustering settings and parameters.
- `data`: a floating-point matrix, where each row represents a data point, and each column represents a feature.
- `k`: an integer representing the number of clusters.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)

kmeans = Kmeans()
result = fit(kmeans, data, k)
```
"""
function fit(kmeans::AbstractKmeans, data::AbstractMatrix{<:Real}, k::Integer)::KmeansResult
    n, d = size(data)

    result = KmeansResult(d, n, k)
    if n == 0
        return result
    end

    @assert d > 0
    @assert k > 0
    @assert n >= k

    unique_data, indices = try_sampling_unique_data(kmeans.rng, data, k)
    initialize!(result, unique_data, indices, verbose = kmeans.verbose)

    fit!(kmeans, data, result)

    return result
end
