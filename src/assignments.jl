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

function kmedoids_assign(point::Integer, clusters::AbstractVector{<:Integer}, distances::AbstractMatrix{<:Real})
    k = length(clusters)

    min_cluster = 0
    min_distance = Inf

    for j in 1:k
        cluster = clusters[j]
        if point == cluster
            return j
        end

        distance = distances[point, cluster]
        if distance < min_distance
            min_cluster = j
            min_distance = distance
        end
    end

    return min_cluster
end

function gmm_assign(point::Integer, probabilities::AbstractMatrix{<:Real}, is_empty::AbstractVector{<:Bool})
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

    return max_cluster
end

function assignment_step!(::Kmeans; result::KmeansResult, distances::AbstractMatrix{<:Real}, is_empty::AbstractVector{<:Bool})
    k, n = size(distances)

    for i in 1:n
        cluster, distance = kmeans_assign(i, distances, is_empty)

        is_empty[cluster] = false
        result.assignments[i] = cluster
    end

    return nothing
end

function assignment_step!(::BalancedKmeans; result::KmeansResult, distances::AbstractMatrix{<:Real}, is_empty::AbstractVector{<:Bool})
    k, n = size(distances)
    capacities = build_capacities_vector(n, k)
    total_candidates = k * n

    candidates_distances = Vector{Float64}(undef, total_candidates)
    candidates_point = Vector{Int}(undef, total_candidates)
    candidates_cluster = Vector{Int}(undef, total_candidates)

    index = 1
    @inbounds for cluster in 1:k
        for point in 1:n
            candidates_distances[index] = distances[cluster, point]
            candidates_point[index] = point
            candidates_cluster[index] = cluster
            index += 1
        end
    end

    permutation = sortperm(candidates_distances)

    fill!(result.assignments, 0)
    load = zeros(Int, k)
    assigned_count = 0

    @inbounds for i in permutation
        point = candidates_point[i]
        cluster = candidates_cluster[i]

        if result.assignments[point] == 0 && load[cluster] < capacities[cluster]
            result.assignments[point] = cluster
            load[cluster] += 1

            assigned_count += 1
            if assigned_count == n
                break
            end
        end
    end

    return nothing
end

function assignment_step!(::Kmedoids; result::KmedoidsResult, distances::AbstractMatrix{<:Real}, medoids::AbstractVector{<:Vector{Int}})
    k = length(result.clusters)
    n = size(distances, 1)

    for i in 1:k
        empty!(medoids[i])
    end

    for i in 1:n
        cluster = kmedoids_assign(i, result.clusters, distances)
        push!(medoids[cluster], i)
    end

    return nothing
end

function assignment_step!(::BalancedKmedoids; result::KmedoidsResult, distances::AbstractMatrix{<:Real}, medoids::AbstractVector{<:Vector{Int}})
    k = length(result.clusters)
    n = size(distances, 1)

    capacities = build_capacities_vector(n, k)
    total_candidates = k * n

    candidates_distances = Vector{Float64}(undef, total_candidates)
    candidates_point = Vector{Int}(undef, total_candidates)
    candidates_cluster = Vector{Int}(undef, total_candidates)

    index = 1
    for i in 1:k
        medoid = result.clusters[i]
        for point in 1:n
            candidates_distances[index] = distances[point, medoid]
            candidates_point[index] = point
            candidates_cluster[index] = i
            index += 1
        end
    end

    permutation = sortperm(candidates_distances)

    for j in 1:k
        empty!(medoids[j])
    end

    assigned_count = 0
    assigned = fill(false, n)

    for i in permutation
        point = candidates_point[i]
        cluster = candidates_cluster[i]

        if !assigned[point] && length(medoids[cluster]) < capacities[cluster]
            push!(medoids[cluster], point)
            assigned[point] = true

            assigned_count += 1
            if assigned_count == n
                break
            end
        end
    end

    return nothing
end
