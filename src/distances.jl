function pairwise_distances!(
    metric::SemiMetric;
    distances::AbstractMatrix{<:Real},
    clusters::Matrix{<:Real},
    data::AbstractMatrix{<:Real},
)
    pairwise!(metric, distances, clusters, data', dims = 2)
    return nothing
end

function pairwise_distances!(
    kmeans::AbstractKmeans;
    distances::AbstractMatrix{<:Real},
    result::KmeansResult,
    data::AbstractMatrix{<:Real},
)
    pairwise_distances!(
        kmeans.metric;
        distances = distances,
        clusters = result.clusters,
        data = data,
    )
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
