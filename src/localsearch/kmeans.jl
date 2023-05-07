@doc raw"""
    Kmeans(
        metric::SemiMetric = SqEuclidean()
        verbose::Bool = DEFAULT_VERBOSE
        rng::AbstractRNG = Random.GLOBAL_RNG
        tolerance::Float64 = DEFAULT_TOLERANCE
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
"""
Base.@kwdef mutable struct Kmeans <: ClusteringAlgorithm
    metric::SemiMetric = SqEuclidean()
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    tolerance::Float64 = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

@doc raw"""
    KmeansResult(
        assignments::Vector{Int}
        clusters::Matrix{Float64}
        objective::Float64
        objective_per_cluster::Vector{Float64}
        iterations::Int
        elapsed::Float64
        converged::Bool
        k::Int
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
mutable struct KmeansResult <: ClusteringResult
    assignments::Vector{Int}
    clusters::Matrix{Float64}

    objective::Float64
    objective_per_cluster::Vector{Float64}
    iterations::Int
    elapsed::Float64
    converged::Bool

    k::Int

    function KmeansResult(
        assignments::AbstractVector{<:Integer},
        clusters::AbstractMatrix{<:Real},
        objective::Real = Inf,
        objective_per_cluster::AbstractVector{<:Real} = Inf * ones(size(clusters, 2)),
        iterations::Int = 0,
        elapsed::Float64 = 0.0,
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
            size(clusters, 2),
        )
    end
end

function KmeansResult(d::Integer, n::Integer, k::Integer)
    return KmeansResult(zeros(Int, n), zeros(Float64, d, k))
end

@doc raw"""
    fit!(kmeans::Kmeans, data::AbstractMatrix{<:Real}, result::KmeansResult)

TODO: Documentation
"""
function fit!(kmeans::Kmeans, data::AbstractMatrix{<:Real}, result::KmeansResult)
    t = time()

    n, d = size(data)
    k = size(result.clusters, 2)

    previous_objective = -Inf
    reset_objective!(result)

    result.iterations = kmeans.max_iterations
    result.converged = false

    count = zeros(Int, k)
    distances = zeros(k, n)

    for iteration in 1:kmeans.max_iterations
        previous_objective = result.objective

        # assignment step
        result.objective = 0.0
        for i in 1:k
            result.objective_per_cluster[i] = 0.0
        end

        pairwise!(distances, kmeans.metric, result.clusters, data', dims = 2)
        for i in 1:n
            cluster, distance = assign(i, distances)

            result.assignments[i] = cluster
            result.objective += distance
            result.objective_per_cluster[cluster] += distance
        end

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
            count[i] = 0
            for j in 1:d
                result.clusters[j, i] = 0
            end
        end

        for i in 1:n
            assignment = result.assignments[i]
            for j in 1:d
                result.clusters[j, assignment] += data[i, j]
            end
            count[assignment] += 1
        end

        for i in 1:k
            cluster_size = max(1, count[i])
            for j in 1:d
                result.clusters[j, i] = result.clusters[j, i] / cluster_size
            end
        end
    end

    result.elapsed = time() - t

    return nothing
end

@doc raw"""
    fit(kmeans::Kmeans, data::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})

TODO: Documentation
"""
function fit(kmeans::Kmeans, data::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})::KmeansResult
    n, d = size(data)
    k = length(initial_clusters)

    result = KmeansResult(d, n, k)
    if n == 0
        return result
    end

    @assert d > 0
    @assert n >= k

    for i in 1:d
        for j in 1:k
            result.clusters[i, j] = data[initial_clusters[j], i]
        end
    end

    if kmeans.verbose
        print_initial_clusters(initial_clusters)
    end

    fit!(kmeans, data, result)

    return result
end

@doc raw"""
    fit(kmeans::Kmeans, data::AbstractMatrix{<:Real}, k::Integer)

TODO: Documentation
"""
function fit(kmeans::Kmeans, data::AbstractMatrix{<:Real}, k::Integer)::KmeansResult
    n, d = size(data)

    if n == 0
        return KmeansResult(d, n, k)
    end

    @assert n >= k

    initial_clusters = StatsBase.sample(kmeans.rng, 1:n, k, replace = false)
    return fit(kmeans, data, initial_clusters)
end
