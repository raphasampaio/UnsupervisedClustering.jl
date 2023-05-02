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
        cluster = clusters[j]
        if point == cluster
            return j, 0.0
        end

        distance = distances[point, cluster]
        if distance < min_distance
            min_cluster = j
            min_distance = distance
        end
    end

    return min_cluster, min_distance
end

function identity_matrix(d::Int)
    return Symmetric(Matrix{Float64}(I, d, d))
end
