function Base.convert(::Type{KmeansResult}, x::GMMResult)
    assignments = copy(x.assignments)
    clusters = hcat(x.clusters...)
    return KmeansResult(assignments, clusters)
end

function Base.convert(::Type{GMMResult}, x::KmeansResult)
    d, k = size(x.clusters)

    assignments = copy(x.assignments)
    weights = ones(k) ./ k
    clusters = [x.clusters[:, i] for i in 1:k]
    covariances = [identity_matrix(d) for _ in 1:k]
    return GMMResult(assignments, weights, clusters, covariances)
end
