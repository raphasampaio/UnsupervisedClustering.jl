function random_swap!(result::KmeansResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    n, d = size(data)
    k = size(result.clusters, 2)

    if n > 0 && k > 0
        to = rand(rng, 1:k)
        from = rand(rng, 1:n)
        for i in 1:d
            result.clusters[i, to] = data[from, i]
        end
        reset_objective!(result)
    end

    return nothing
end

function random_swap!(result::KmedoidsResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    n, d = size(data)
    k = length(result.clusters)

    if n > 0 && k > 0
        weights = ones(n)
        for i in result.clusters
            weights[i] = 0.0
        end

        to = rand(rng, 1:k)
        weights[result.clusters[to]] = 1.0
        from = sample(rng, aweights(weights))
        result.clusters[to] = from

        reset_objective!(result)
    end

    return nothing
end

function random_swap!(result::GMMResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    k = result.k
    n, d = size(data)

    if n > 0 && k > 0
        to = rand(rng, 1:k)
        from = rand(rng, 1:n)
        result.clusters[to] = copy(data[from, :])

        m = mean([det(result.covariances[j]) for j in 1:k])
        value = (m > 0 ? m : 1.0)^(1 / d)
        result.covariances[to] = Symmetric(value .* identity_matrix(d))

        reset_objective!(result)
    end

    return nothing
end
