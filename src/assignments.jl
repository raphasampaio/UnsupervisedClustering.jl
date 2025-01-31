function kmeans_assign(point::Integer, distances::AbstractMatrix{<:Real}, is_empty::AbstractVector{<:Bool})
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

function assignment_step!(::Kmeans; result::KmeansResult, distances::AbstractMatrix{<:Real}, is_empty::AbstractVector{<:Bool})
    k, n = size(distances)

    objective = 0.0

    for i in 1:n
        cluster, distance = kmeans_assign(i, distances, is_empty)

        is_empty[cluster] = false
        result.assignments[i] = cluster
        objective += distance
    end

    return objective
end

function assignment_step!(::BalancedKmeans; result::KmeansResult, distances::AbstractMatrix{<:Real}, is_empty::AbstractVector{<:Bool})
    k, n = size(distances)
    capacity = div(n, k)

    candidates = Vector{Tuple{Float64, Int, Int}}()
    sizehint!(candidates, k * n)

    for cluster in 1:k
        for point in 1:n
            push!(candidates, (distances[cluster, point], point, cluster))
        end
    end
    sort!(candidates, by = x -> x[1])

    fill!(result.assignments, 0)
    load = zeros(Int, k)
    objective = 0.0
    assigned_count = 0

    for (distance, point, cluster) in candidates
        if result.assignments[point] == 0 && load[cluster] < capacity
            result.assignments[point] = cluster
            load[cluster] += 1
            objective += distance
            assigned_count += 1

            if assigned_count == n
                break
            end
        end
    end

    return objective
end
