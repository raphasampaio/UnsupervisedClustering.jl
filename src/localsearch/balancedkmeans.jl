Base.@kwdef mutable struct BalancedKmeans <: AbstractAlgorithm
    metric::SemiMetric = SqEuclidean()
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    tolerance::Real = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

function fit!(Balanced::BalancedKmeans, data::AbstractMatrix{<:Real}, result::KmeansResult)
    t = time()

    n, d = size(data)
    k = result.k

    cluster_capacity = div(n, k)

    previous_objective = -Inf
    reset_objective!(result)

    result.iterations = kmeans.max_iterations
    result.converged = false

    clusters_size = zeros(Int, k)
    distances = zeros(k, n)

    for iteration in 1:kmeans.max_iterations
        previous_objective = result.objective

        pairwise!(kmeans.metric, distances, result.clusters, data', dims = 2)

        assignment_step!(result = result, distances = distances, cluster_capacity = cluster_capacity)

        change = abs(result.objective - previous_objective)

        # if kmeans.verbose
        #     print_iteration(iteration)
        #     print_objective(result)
        #     print_change(change)
        #     print_newline()
        # end

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

function fit(kmeans::BalancedKmeans, data::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})::KmeansResult
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

function fit(kmeans::BalancedKmeans, data::AbstractMatrix{<:Real}, k::Integer)::KmeansResult
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
