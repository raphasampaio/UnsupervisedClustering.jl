@doc """
    KmeansPlusPlus{RNG}(
        metric::SemiMetric = SqEuclidean()
        verbose::Bool = DEFAULT_VERBOSE
        rng::RNG = Random.GLOBAL_RNG
        tolerance::Float64 = DEFAULT_TOLERANCE
        max_iterations::Int = DEFAULT_MAX_ITERATIONS
    ) where {RNG <: AbstractRNG}

K-means++ is an improvement over the standard K-means algorithm that provides better initialization by selecting initial centroids with probability proportional to their squared distance from existing centroids. This typically leads to better clustering results and faster convergence.

# Type Parameters
- `RNG`: the specific type of the random number generator

# Fields
- `metric`: defines the distance metric used to compute the distances between data points and cluster centroids.
- `verbose`: controls whether the algorithm should display additional information during execution.
- `rng`: represents the random number generator to be used by the algorithm.
- `tolerance`: represents the convergence criterion for the algorithm. It determines the maximum change allowed in the centroid positions between consecutive iterations.
- `max_iterations`: represents the maximum number of iterations the algorithm will perform before stopping, even if convergence has not been reached.

# References
* Arthur, David, and Sergei Vassilvitskii.
  k-means++: The advantages of careful seeding.
  Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms. 2007.
"""
mutable struct KmeansPlusPlus{RNG <: AbstractRNG} <: AbstractKmeans
    metric::SemiMetric
    verbose::Bool
    rng::RNG
    tolerance::Float64
    max_iterations::Int

    function KmeansPlusPlus{RNG}(
        metric::SemiMetric,
        verbose::Bool,
        rng::RNG,
        tolerance::Float64,
        max_iterations::Int,
    ) where {RNG <: AbstractRNG}
        return new{RNG}(metric, verbose, rng, tolerance, max_iterations)
    end
end

# Convenience constructor
function KmeansPlusPlus(; metric::SemiMetric = SqEuclidean(), verbose::Bool = DEFAULT_VERBOSE,
    rng::AbstractRNG = Random.GLOBAL_RNG, tolerance::Float64 = DEFAULT_TOLERANCE,
    max_iterations::Int = DEFAULT_MAX_ITERATIONS)
    return KmeansPlusPlus{typeof(rng)}(metric, verbose, rng, tolerance, max_iterations)
end

# Assignment step method for KmeansPlusPlus (same as regular Kmeans)
function assignment_step!(
    kmeans_pp::KmeansPlusPlus,
    assignments::AbstractVector{<:Integer};
    distances::AbstractMatrix{<:Real},
    is_empty::AbstractVector{<:Bool},
)
    n = length(assignments)

    for i in 1:n
        cluster, _ = kmeans_assign(i, distances, is_empty)  # Extract only the cluster index
        assignments[i] = cluster
    end

    return nothing
end

function kmeans_plus_plus_initialization!(
    result::KmeansResult,
    data::AbstractMatrix{<:Real},
    metric::SemiMetric,
    rng::AbstractRNG;
    verbose::Bool = false,
)
    n, d = size(data)
    k = result.k

    # Choose first center randomly
    first_idx = rand(rng, 1:n)
    for i in 1:d
        result.clusters[i, 1] = data[first_idx, i]
    end

    if verbose
        println("K-means++ initialization: Selected center 1 at index $first_idx")
    end

    # Choose remaining centers using K-means++ algorithm
    for center_idx in 2:k
        # Compute squared distances from each point to nearest existing center
        squared_distances = zeros(n)

        for point_idx in 1:n
            min_dist_sq = Inf

            for existing_center in 1:(center_idx-1)
                dist_sq = evaluate(
                    metric,
                    view(data, point_idx, :),
                    view(result.clusters, :, existing_center),
                )
                min_dist_sq = min(min_dist_sq, dist_sq)
            end

            squared_distances[point_idx] = min_dist_sq
        end

        # Choose next center with probability proportional to squared distance
        total_weight = sum(squared_distances)

        if total_weight ≈ 0.0
            # All remaining points are at existing centers, choose randomly
            remaining_indices = setdiff(
                1:n,
                [
                    findfirst(
                        ==(data[i, :]),
                        eachcol(result.clusters)) for i in 1:n if any(==(data[i, :]),
                        eachcol(result.clusters),
                    )
                ],
            )
            if !isempty(remaining_indices)
                chosen_idx = rand(rng, remaining_indices)
            else
                chosen_idx = rand(rng, 1:n)
            end
        else
            # Weighted random selection
            cumulative_prob = cumsum(squared_distances ./ total_weight)
            random_val = rand(rng)
            chosen_idx = findfirst(>=(random_val), cumulative_prob)
            if chosen_idx === nothing
                chosen_idx = n
            end
        end

        # Set the new center
        for i in 1:d
            result.clusters[i, center_idx] = data[chosen_idx, i]
        end

        if verbose
            println("K-means++ initialization: Selected center $center_idx at index $chosen_idx")
        end
    end

    return nothing
end

function fit(kmeans_pp::KmeansPlusPlus, data::AbstractMatrix{<:Real}, k::Integer)::KmeansResult
    n, d = size(data)

    result = KmeansResult(d, n, k)
    if n == 0
        return result
    end

    @assert d > 0
    @assert k > 0
    @assert n >= k

    # Use K-means++ initialization instead of random sampling
    kmeans_plus_plus_initialization!(result, data, kmeans_pp.metric, kmeans_pp.rng,
        verbose = kmeans_pp.verbose)

    fit!(kmeans_pp, data, result)

    return result
end
