Base.@kwdef mutable struct Ksegments <: Algorithm
end

mutable struct KsegmentsResult{I <: Integer, R <: Real} <: Result
    assignments::Vector{I}
    clusters::Matrix{R}
    objective::R
    objective_per_cluster::Vector{R}
    iterations::I
    elapsed::R
    converged::Bool
    k::I

    function KsegmentsResult(
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

function KsegmentsResult(d::Integer, n::Integer, k::Integer)
    return KsegmentsResult(zeros(Int, n), zeros(d, k))
end

function KsegmentsResult(n::Integer, clusters::AbstractMatrix{<:Real})
    d, k = size(clusters)
    result = KsegmentsResult(d, n, k)
    result.clusters = copy(clusters)
    return result
end

function fit!(ksegments::Ksegments, data::AbstractMatrix{<:Real}, result::KsegmentsResult)
    t = time()

    n, d = size(data)
    k = result.k

    means = zeros(n, n, d)
    for i in 1:n
        for j in i:n
            if i == j
                means[i, j, :] = data[i, :]
            else
                means[i, j, :] = (means[i, j-1, :] * (j - i) + data[j, :]) / (j - i + 1)
            end
        end
    end

    distances = zeros(n, n)
    for i in 1:n
        for j in i:n
            for l in i:j
                distances[i, j] += sum((data[l, :] - means[i, j, :]) .^ 2)
            end
        end
    end

    segments_costs = zeros(k, n + 1)
    segments_costs[1, 2:end] = distances[1, :]

    segments_path = zeros(Int, k, n)
    segments_path[1, :] .= 0

    for i in 1:k
        segments_path[Base._sub2ind(size(segments_path), i, i)] = i - 1
    end

    for i in 2:k
        for j in i:n
            choices = segments_costs[i-1, 1:j] + distances[1:j, j]

            value, index = findmin(choices)

            segments_path[i, j] = index - 1
            segments_costs[i, j + 1] = value
        end
    end

    rhs = n
    for cluster in k:-1:1
        lhs = segments_path[cluster, rhs]

        # update assignments
        for i in (lhs + 1):rhs
            result.assignments[i] = cluster
        end

        # update clusters
        for j in 1:d
            result.clusters[j, cluster] = means[lhs + 1, rhs, j]
        end

        # update objective per cluster
        result.objective_per_cluster[cluster] = 0.0
        for i in (lhs + 1):rhs
            result.objective_per_cluster[cluster] += distances[i, rhs]
        end

        rhs = lhs
    end

    result.objective = sum(result.objective_per_cluster)
    result.iterations = 1
    result.converged = true
    result.elapsed = time() - t

    return nothing
end

function fit(ksegments::Ksegments, data::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})::KsegmentsResult
    n, d = size(data)
    k = length(initial_clusters)

    result = KsegmentsResult(d, n, k)
    if n == 0
        return result
    end

    @assert d > 0
    @assert k > 0
    @assert n >= k

    for i in 1:d
        for j in 1:k
            result.clusters[i, j] = data[initial_clusters[j], i]
        end
    end

    if ksegments.verbose
        print_initial_clusters(initial_clusters)
    end

    fit!(ksegments, data, result)

    return result
end

function fit(ksegments::Ksegments, data::AbstractMatrix{<:Real}, k::Integer)::KsegmentsResult
    n, d = size(data)

    if n == 0
        return KsegmentsResult(d, n, k)
    end

    @assert k > 0
    @assert n >= k

    initial_clusters = StatsBase.sample(ksegments.rng, 1:n, k, replace = false)
    return fit(ksegments, data, initial_clusters)
end
