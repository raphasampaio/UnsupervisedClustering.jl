function silhouette_score(; data::AbstractMatrix{<:Real}, assignments::AbstractVector{<:Integer}, metric::SemiMetric = SqEuclidean())
    n, _ = size(data)
    unique_assignments = unique(assignments)
    k = length(unique_assignments)

    if k == 1
        throw(ArgumentError("Number of clusters must be greater than 1"))
    end

    distances = pairwise(metric, data, dims = 1)

    silhouette_scores = zeros(n)

    for i in 1:n
        cluster_i = assignments[i]
        in_cluster = findall(x -> x == cluster_i, assignments)

        # average intra-cluster distance
        cohesion = if length(in_cluster) > 1
            mean(distances[i, in_cluster[in_cluster.!=i]]) # exclude self
        else
            0
        end

        # minimum average distance to other clusters
        separation = Inf
        for cluster in unique_assignments
            if cluster != cluster_i
                out_cluster = findall(x -> x == cluster, assignments)
                if !isempty(out_cluster)
                    mean_distance = mean(distances[i, out_cluster])
                    separation = min(separation, mean_distance)
                end
            end
        end

        # silhouette score for point i
        silhouette_scores[i] = (separation - cohesion) / max(cohesion, separation)
    end

    return mean(silhouette_scores)
end
