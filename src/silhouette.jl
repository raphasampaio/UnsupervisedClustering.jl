function silhouette_score(X::Matrix{Float64}, labels::Vector{Int})
    n = size(X, 1)  # Number of data points
    unique_labels = unique(labels)
    k = length(unique_labels)  # Number of clusters

    if k == 1
        error("Silhouette score is not defined for a single cluster.")
    end

    # Compute pairwise distances
    D = pairwise(Euclidean(), X, dims=1)

    # Initialize silhouette values
    s = zeros(n)

    for i in 1:n
        cluster_i = labels[i]
        in_cluster = findall(x -> x == cluster_i, labels)

        # Compute a(i): Average intra-cluster distance
        if length(in_cluster) > 1
            a_i = mean(D[i, in_cluster[in_cluster .!= i]])  # Exclude self
        else
            a_i = 0
        end

        # Compute b(i): Minimum average distance to other clusters
        b_i = Inf
        for cluster in unique_labels
            if cluster != cluster_i
                out_cluster = findall(x -> x == cluster, labels)
                if !isempty(out_cluster)
                    mean_dist = mean(D[i, out_cluster])
                    b_i = min(b_i, mean_dist)
                end
            end
        end

        # Compute silhouette score for point i
        s[i] = (b_i - a_i) / max(a_i, b_i)
    end

    # Return mean silhouette score
    return mean(s)
end
