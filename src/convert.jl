function Base.convert(::Type{GMMResult}, x::KmeansResult)
    d, k = size(x.clusters)

    assignments = copy(x.assignments)
    weights = ones(k) ./ k
    clusters = [x.clusters[:, i] for i in 1:k]
    covariances = [identity_matrix(d) for _ in 1:k]
    return GMMResult(assignments, weights, clusters, covariances)
end

function fit(algorithm::GMM, data::AbstractMatrix{<:Real}, result_kmeans::KmeansResult)
    result_gmm = convert(GMMResult, result_kmeans)
    fit!(algorithm, data, result_gmm)
    return result_gmm
end

function Base.convert(::Type{KmeansResult}, x::GMMResult)
    assignments = copy(x.assignments)
    clusters = hcat(x.clusters...)
    return KmeansResult(assignments, clusters)
end

function fit(algorithm::Kmeans, data::AbstractMatrix{<:Real}, result_gmm::GMMResult)
    result_kmeans = convert(Kmeans, result_gmm)
    fit!(algorithm, data, result_kmeans)
    return result_kmeans
end
