function assign(point::Int, distances::Matrix{Float64})::Tuple{Int, Float64}
    k, n = size(distances)

    min_cluster = 0
    min_distance = Inf

    for cluster in 1:k
        distance = distances[cluster, point]
        if distance < min_distance
            min_cluster = cluster
            min_distance = distance
        end
    end

    return min_cluster, min_distance
end

function assign(point::Int, clusters::Vector{Int}, distances::Matrix{Float64})::Tuple{Int, Float64}
    n = size(distances, 1)
    k = length(clusters)

    min_cluster = 0
    min_distance = Inf

    for j in 1:k
        center = clusters[j]
        if point == center
            return j, 0.0
        end

        distance = distances[point, center]
        if distance < min_distance
            min_cluster = j
            min_distance = distance
        end
    end

    return min_cluster, min_distance
end
