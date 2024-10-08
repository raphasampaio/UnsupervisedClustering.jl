function assign(::Kmeans, point::Integer, distances::AbstractMatrix{<:Real}, is_empty::AbstractVector{<:Bool})
    k, n = size(distances)

    min_cluster = 0
    min_distance = Inf

    for cluster in 1:k
        distance = distances[cluster, point]
        if distance < min_distance
            min_cluster = cluster
            min_distance = distance
        elseif distance == min_distance
            if is_empty[cluster] && !is_empty[min_cluster]
                min_cluster = cluster
                min_distance = distance
            end
        end
    end

    return min_cluster, min_distance
end

function assign(point::Integer, clusters::AbstractVector{<:Integer}, distances::AbstractMatrix{<:Real})
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

function assign(::GMM, point::Integer, probabilities::AbstractMatrix{<:Real}, is_empty::AbstractVector{<:Bool})
    n, k = size(probabilities)

    max_cluster = 0
    max_probability = -Inf

    for cluster in 1:k
        probability = probabilities[point, cluster]
        if probability > max_probability
            max_cluster = cluster
            max_probability = probability
        elseif probability == max_probability
            if is_empty[cluster] && !is_empty[max_cluster]
                max_cluster = cluster
                max_probability = probability
            end
        end
    end

    return max_cluster, max_probability
end
