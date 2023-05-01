function Base.convert(::Type{GMMResult}, x::KmeansResult)
    k = x.k
    d, k = size(x.clusters)

    assignments = copy(x.assignments)
    weights = ones(k) ./ k
    clusters = [x.clusters[:, i] for i in 1:k]
    covariances = [identity_matrix(d) for _ in 1:k]
    return GMMResult(k, assignments, weights, clusters, covariances, -Inf, 0, 0, false)
end

function fit(algorithm::GMM, data::AbstractMatrix{<:Real}, result_kmeans::KmeansResult)
    result_gmm = convert(GMMResult, result_kmeans)
    fit!(algorithm, data, result_gmm)
    return result_gmm
end
