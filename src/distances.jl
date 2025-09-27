function compute_distances!(
    metric::SemiMetric;
    distances::AbstractMatrix{<:Real},
    clusters::Matrix{<:Real},
    data::AbstractMatrix{<:Real},
)
    pairwise!(metric, distances, clusters, data', dims = 2)
    return nothing
end

function compute_distances(
    metric::SemiMetric;
    clusters::Matrix{<:Real},
    data::AbstractMatrix{<:Real},
)
    return pairwise(metric, clusters, data', dims = 2)
end

function compute_distances!(
    kmeans::AbstractKmeans;
    distances::AbstractMatrix{<:Real},
    result::KmeansResult,
    data::AbstractMatrix{<:Real},
)
    compute_distances!(
        kmeans.metric;
        distances = distances,
        clusters = result.clusters,
        data = data,
    )
    return nothing
end
