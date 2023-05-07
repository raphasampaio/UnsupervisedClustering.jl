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
Base.@kwdef mutable struct Kmedoids <: ClusteringAlgorithm
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
mutable struct KmedoidsResult <: ClusteringResult
    assignments::AbstractVector{<:Integer}
    clusters::AbstractVector{<:Integer}
    objective::Real
    objective_per_cluster::AbstractVector{<:Real}
    iterations::Integer
    elapsed::Real
    converged::Bool
    k::Integer

    function KmedoidsResult(
        assignments::AbstractVector{<:Integer},
        clusters::AbstractVector{<:Integer},
        objective::Real = Inf,
        objective_per_cluster::AbstractVector{<:Real} = Inf * ones(length(clusters)),
        iterations::Integer = 0,
        elapsed::Real = 0.0,
        converged::Bool = false,
    )
        return new(
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

@doc """
    fit!(
        kmedoids::Kmedoids,
        distances::AbstractMatrix{<:Real},
        result::KmedoidsResult
    )

TODO: Documentation
"""
function fit!(kmedoids::Kmedoids, distances::AbstractMatrix{<:Real}, result::KmedoidsResult)
    t = time()

    n = size(distances, 1)
    k = length(result.clusters)

    medoids = [Vector{Int}() for _ in 1:k]
    count = zeros(Int, k)

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
            count[i] = 0
            result.objective_per_cluster[i] = 0
        end

        for i in 1:n
            cluster, distance = assign(i, result.clusters, distances)

            count[cluster] += 1
            push!(medoids[cluster], i)

            result.objective += distance
            result.objective_per_cluster[cluster] += distance
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
                    distances_sum += distances[medoid[j], medoid[l]]
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
        kmedoids::Kmedoids,
        distances::AbstractMatrix{<:Real},
        initial_clusters::AbstractVector{<:Integer}
    )

TODO: Documentation
"""
function fit(kmedoids::Kmedoids, distances::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})::KmedoidsResult
    n = size(distances, 1)
    k = length(initial_clusters)

    result = KmedoidsResult(n, k)
    if n == 0
        return result
    end

    @assert n >= k

    for i in 1:k
        result.clusters[i] = initial_clusters[i]
    end

    if kmedoids.verbose
        print_initial_clusters(initial_clusters)
    end

    fit!(kmedoids, distances, result)

    return result
end

@doc """
    fit(
        kmedoids::Kmedoids,
        distances::AbstractMatrix{<:Real},
        k::Integer
    )

TODO: Documentation

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)

kmedoids = Kmedoids()
result = fit(kmedoids, data, k)
```
"""
function fit(kmedoids::Kmedoids, distances::AbstractMatrix{<:Real}, k::Integer)::KmedoidsResult
    n = size(distances, 1)

    if n == 0
        return KmedoidsResult(n, k)
    end

    @assert n >= k

    initial_clusters = StatsBase.sample(kmedoids.rng, 1:n, k, replace = false)
    return fit(kmedoids, distances, initial_clusters)
end
