function pairwise_distances!(
    kmeans::AbstractKmeans;
    result::KmeansResult,
    distances::AbstractMatrix{<:Real},
    data::AbstractMatrix{<:Real},
)
    pairwise!(kmeans.metric, distances, result.clusters, data', dims = 2)
    return nothing
end

function pairwise_distances(
    kmeans::AbstractKmeans;
    result::KmeansResult,
    data::AbstractMatrix{<:Real},
)
    n, _ = size(data)
    k = result.k
    distances = zeros(k, n)
    pairwise_distances!(kmeans; result, distances, data)
    return distances
end