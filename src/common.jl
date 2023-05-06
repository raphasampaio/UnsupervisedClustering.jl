function assign(point::Integer, distances::AbstractMatrix{<:Real})::Tuple{Integer, <:Real}
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

function assign(point::Integer, clusters::AbstractVector{<:Integer}, distances::AbstractMatrix{<:Real})::Tuple{<:Integer, <:Real}
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

function identity_matrix(d::Integer)
    return Symmetric(Matrix{Float64}(I, d, d))
end
