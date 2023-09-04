Base.@kwdef mutable struct Ksegmentation <: Algorithm
end

const KsegmentationResult{I <: Integer, R <: Real} = KmeansResult{I, R}

function fit!(ksegmentation::Ksegmentation, data::AbstractMatrix{<:Real}, result::KsegmentationResult)::Nothing
    t = time()

    n, d = size(data)
    k = result.k

    @assert d > 0
    @assert k > 0
    @assert n >= k

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
            segments_costs[i, j+1] = value
        end
    end

    rhs = n
    for cluster in k:-1:1
        lhs = segments_path[cluster, rhs]

        # update assignments
        for i in (lhs+1):rhs
            result.assignments[i] = cluster
        end

        # update clusters
        for j in 1:d
            result.clusters[j, cluster] = means[lhs+1, rhs, j]
        end

        # update objective per cluster
        result.objective_per_cluster[cluster] = 0.0
        for i in (lhs+1):rhs
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

function fit(ksegmentation::Ksegmentation, data::AbstractMatrix{<:Real}, k::Integer)::KsegmentationResult
    n, d = size(data)

    if n == 0
        return KsegmentationResult(d, n, k)
    end

    result = KsegmentationResult(d, n, k)
    fit!(ksegmentation, data, result)
    return result
end
