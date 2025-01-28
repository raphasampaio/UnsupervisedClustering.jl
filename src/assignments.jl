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

function assignment_step!(; result::KmeansResult, distances::AbstractMatrix{<:Real}, cluster_capacity::Integer)
    k, n = size(distances)

    # Flatten all (cluster, point) pairs into a single list
    # We'll store tuples of the form (distance, point, cluster)
    assignment_candidates = Vector{Tuple{Float64, Int, Int}}(undef, n*k)
    index = 1
    for cluster in 1:k
        for point in 1:n
            assignment_candidates[index] = (distances[cluster, point], point, cluster)
            index += 1
        end
    end

    # Sort all pairs by ascending distance
    sort!(assignment_candidates, by = x -> x[1])

    # Prepare for assignment
    fill!(assignments, 0)        # 0 => unassigned
    cluster_load = fill(0, k)    # how many points in each cluster
    total_objective = 0.0
    assigned_count = 0

    # Greedily assign
    for (dist, point, cluster) in assignment_candidates
        if assignments[point] == 0 && cluster_load[cluster] < cluster_capacity
            # Assign this point to this cluster
            assignments[point] = cluster
            cluster_load[cluster] += 1
            total_objective += dist
            assigned_count += 1
            # if all points assigned, we can stop early
            if assigned_count == n
                break
            end
        end
    end

    result.objective = total_objective
    return nothing
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
