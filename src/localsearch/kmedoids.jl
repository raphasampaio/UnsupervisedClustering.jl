@doc """
    Kmedoids(
        verbose::Bool = DEFAULT_VERBOSE
        rng::AbstractRNG = Random.GLOBAL_RNG
        tolerance::Real = DEFAULT_TOLERANCE
        max_iterations::Integer = DEFAULT_MAX_ITERATIONS
    )

The k-medoids is a variation of k-means clustering algorithm that uses actual data points (medoids) as representatives of each cluster instead of the mean.

# Fields
- `verbose`: controls whether the algorithm should display additional information during execution.
- `rng`: represents the random number generator to be used by the algorithm.
- `tolerance`: represents the convergence criterion for the algorithm. It determines the maximum change allowed in the centroid positions between consecutive iterations.
- `max_iterations`: represents the maximum number of iterations the algorithm will perform before stopping, even if convergence has not been reached.

# References
"""
Base.@kwdef mutable struct Kmedoids <: AbstractKmedoids
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    tolerance::Real = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

Base.@kwdef mutable struct BalancedKmedoids <: AbstractKmedoids
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    tolerance::Real = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

@doc """
    KmedoidsResult(
        assignments::AbstractVector{<:Integer}
        clusters::AbstractVector{<:Integer}
        objective::Real
        objective_per_cluster::AbstractVector{<:Real}
        iterations::Integer
        elapsed::Real
        converged::Bool
        k::Integer
    )

KmedoidsResult struct represents the result of the k-medoids clustering algorithm.

# Fields
- `assignments`: an integer vector that stores the cluster assignment for each data point.
- `clusters`: an integer vector representing each cluster's centroid.
- `objective`: a floating-point number representing the objective function after running the algorithm. The objective function measures the quality of the clustering solution.
- `objective_per_cluster`: a floating-point vector that stores the objective function of each cluster
- `iterations`: an integer value indicating the number of iterations performed until the algorithm has converged or reached the maximum number of iterations
- `elapsed`: a floating-point number representing the time in seconds for the algorithm to complete.
- `converged`: indicates whether the algorithm has converged to a solution.
- `k`: the number of clusters.
"""
mutable struct KmedoidsResult{I <: Integer, R <: Real} <: AbstractResult
    assignments::Vector{I}
    clusters::Vector{I}
    objective::R
    objective_per_cluster::Vector{R}
    iterations::I
    elapsed::R
    converged::Bool
    k::I

    function KmedoidsResult(
        assignments::AbstractVector{I},
        clusters::AbstractVector{I},
        objective::R = Inf,
        objective_per_cluster::AbstractVector{R} = Inf * ones(length(clusters)),
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
            length(clusters),
        )
    end
end

function KmedoidsResult(n::Integer, k::Integer)
    return KmedoidsResult(zeros(Int, n), zeros(Int, k))
end

function KmedoidsResult(n::Integer, clusters::AbstractVector{<:Integer})
    k = length(clusters)
    result = KmedoidsResult(n, k)
    result.clusters = copy(clusters)
    return result
end

function initialize!(result::KmedoidsResult, indices::AbstractVector{<:Integer}; verbose::Bool = false)
    k = length(indices)

    for i in 1:k
        result.clusters[i] = indices[i]
    end

    if verbose
        print_initial_clusters(indices)
    end

    return nothing
end

@doc """
    fit!(
        kmedoids::AbstractKmedoids,
        distances::AbstractMatrix{<:Real},
        result::KmedoidsResult
    )

The `fit!` function performs the k-medoids clustering algorithm on the given result as the initial point and updates the provided object with the clustering result.

# Parameters:
- `kmedoids`: an instance representing the clustering settings and parameters.
- `distances`: a floating-point matrix representing the pairwise distances between the data points.
- `result`: a result object that will be updated with the clustering result.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)
distances = pairwise(SqEuclidean(), data, dims = 1)

kmedoids = Kmedoids()
result = KmedoidsResult(n, [1.0 2.0; 1.0 2.0])
fit!(kmedoids, distances, result)
```
"""
function fit!(kmedoids::AbstractKmedoids, distances::AbstractMatrix{<:Real}, result::KmedoidsResult)
    t = time()

    n = size(distances, 1)
    k = length(result.clusters)

    medoids = [Vector{Int}() for _ in 1:k]
    clusters_size = zeros(Int, k)

    previous_objective = -Inf
    reset_objective!(result)

    result.iterations = kmedoids.max_iterations
    result.converged = false

    for iteration in 1:kmedoids.max_iterations
        previous_objective = result.objective

        # assignment step
        result.objective = 0
        for i in 1:k
            empty!(medoids[i])
            clusters_size[i] = 0
            result.objective_per_cluster[i] = 0
        end

        assignment_step!(kmedoids; result, distances, medoids)

        for (cluster, medoid) in enumerate(medoids)
            for point in medoid
                distance = distances[point, cluster]

                clusters_size[cluster] += 1
                result.objective += distance
                result.objective_per_cluster[cluster] += distance
            end
        end

        change = abs(result.objective - previous_objective)

        if kmedoids.verbose
            print_iteration(iteration)
            print_objective(result)
            print_change(change)
            print_newline()
        end

        # stopping condition
        if change < kmedoids.tolerance || n == k
            result.converged = true
            result.iterations = iteration
            break
        end

        # update step
        for (i, medoid) in enumerate(medoids)
            min_j = 0
            min_distance = Inf
            for j in eachindex(medoid)
                distances_sum = 0
                for l in eachindex(medoid)
                    distances_sum += distances[medoid[l], medoid[j]]
                end

                if distances_sum < min_distance
                    min_j = j
                    min_distance = distances_sum
                end
            end
            result.clusters[i] = medoid[min_j]
        end
    end

    for (i, medoid) in enumerate(medoids)
        for j in medoid
            result.assignments[j] = i
        end
    end

    result.elapsed = time() - t

    return nothing
end

@doc """
    fit(
        kmedoids::AbstractKmedoids,
        distances::AbstractMatrix{<:Real},
        initial_clusters::AbstractVector{<:Integer}
    )

The `fit` function performs the k-medoids clustering algorithm on the given data points as the initial point and returns a result object representing the clustering result.

# Parameters:
- `kmedoids`: an instance representing the clustering settings and parameters.
- `distances`: a floating-point matrix representing the pairwise distances between the data points.
- `initial_clusters`: an integer vector where each element is the initial data point for each cluster.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)
distances = pairwise(SqEuclidean(), data, dims = 1)

kmedoids = Kmedoids()
result = fit(kmedoids, distances, [4, 12])
```
"""
function fit(kmedoids::AbstractKmedoids, distances::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})::KmedoidsResult
    n = size(distances, 1)
    k = length(initial_clusters)

    result = KmedoidsResult(n, k)
    if n == 0
        return result
    end

    @assert k > 0
    @assert n >= k

    initialize!(result, initial_clusters, verbose = kmedoids.verbose)

    fit!(kmedoids, distances, result)

    return result
end

@doc """
    fit(
        kmedoids::AbstractKmedoids,
        distances::AbstractMatrix{<:Real},
        k::Integer
    )

The `fit` function performs the k-medoids clustering algorithm and returns a result object representing the clustering result.

# Parameters:
- `kmedoids`: an instance representing the clustering settings and parameters.
- `distances`: a floating-point matrix representing the pairwise distances between the data points.
- `k`: an integer representing the number of clusters.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)
distances = pairwise(SqEuclidean(), data, dims = 1)

kmedoids = Kmedoids()
result = fit(kmedoids, distances, k)
```
"""
function fit(kmedoids::AbstractKmedoids, distances::AbstractMatrix{<:Real}, k::Integer)::KmedoidsResult
    n = size(distances, 1)

    result = KmedoidsResult(n, k)
    if n == 0
        return result
    end

    @assert k > 0
    @assert n >= k

    initial_clusters = StatsBase.sample(kmedoids.rng, 1:n, k, replace = false)
    initialize!(result, initial_clusters, verbose = kmedoids.verbose)

    fit!(kmedoids, distances, result)

    return result
end
