@doc raw"""
    KmedoidsResult(
        k::Int
        assignments::Vector{Int}
        clusters::Vector{Int}
        objective::Float64
        objective_per_cluster::Vector{Float64}
        iterations::Int
        elapsed::Float64
        converged::Bool
    )

TODO: Documentation
"""
mutable struct KmedoidsResult <: ClusteringResult
    k::Int
    assignments::Vector{Int}
    clusters::Vector{Int}

    objective::Float64
    objective_per_cluster::Vector{Float64}
    iterations::Int
    elapsed::Float64
    converged::Bool
end

@doc raw"""
    KmedoidsResult(assignments::AbstractVector{<:Integer}, clusters::AbstractMatrix{<:Integer})

TODO: Documentation
"""
function KmedoidsResult(assignments::AbstractVector{<:Integer}, clusters::AbstractMatrix{<:Integer})
    d, k = size(clusters)
    return KmedoidsResult(k, assignments, clusters, Inf, Inf * zeros(k), 0, 0, false)
end

@doc raw"""
    KmedoidsResult(n::Integer, k::Integer)

TODO: Documentation
"""
function KmedoidsResult(n::Integer, k::Integer)
    return KmedoidsResult(zeros(Int, n), zeros(Int, k))
end

function isbetter(a::KmedoidsResult, b::KmedoidsResult)
    return isless(a.objective, b.objective)
end

function reset_objective!(result::KmedoidsResult)
    result.objective = Inf
    for i in 1:result.k
        result.objective_per_cluster[i] = Inf
    end
    return nothing
end

function fit!(kmedoids::Kmedoids, distances::AbstractMatrix{<:Real}, result::KmedoidsResult)
    t = time()

    n = size(distances, 1)
    k = length(result.clusters)

    medoids = [Vector{Int}() for _ in 1:k]
    count = zeros(Int, k)

    previous_objective = -Inf
    reset_objective!(result)

    result.iterations = kmedoids.max_iterations
    result.converged = false

    for iteration in 1:kmedoids.max_iterations
        previous_objective = result.objective

        # assignment step
        result.objective = 0
        for i in 1:k
            empty!(medoids[i])
            count[i] = 0
            result.objective_per_cluster[i] = 0
        end

        for i in 1:n
            cluster, distance = assign(i, result.clusters, distances)

            count[cluster] += 1
            push!(medoids[cluster], i)

            result.objective += distance
            result.objective_per_cluster[cluster] += distance
        end

        change = abs(result.objective - previous_objective)

        if kmedoids.verbose
            print_iteration(iteration)
            print_objective(result)
            print_change(change)
            print_newline()
        end

        # stopping condition
        if change < kmedoids.tolerance || n == k
            result.converged = true
            result.iterations = iteration
            break
        end

        # update step
        for (i, medoid) in enumerate(medoids)
            min_j = 0
            min_distance = Inf
            for j in eachindex(medoid)
                distances_sum = 0
                for l in eachindex(medoid)
                    distances_sum += distances[medoid[j], medoid[l]]
                end

                if distances_sum < min_distance
                    min_j = j
                    min_distance = distances_sum
                end
            end
            result.clusters[i] = medoid[min_j]
        end
    end

    for (i, medoid) in enumerate(medoids)
        for j in medoid
            result.assignments[j] = i
        end
    end

    result.elapsed = time() - t

    return nothing
end

function fit(kmedoids::Kmedoids, distances::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})::KmedoidsResult
    n = size(distances, 1)
    k = length(initial_clusters)

    result = KmedoidsResult(n, k)
    if n == 0
        return result
    end

    @assert n >= k

    for i in 1:k
        result.clusters[i] = initial_clusters[i]
    end

    if kmedoids.verbose
        print_initial_clusters(initial_clusters)
    end

    fit!(kmedoids, distances, result)

    return result
end

function fit(kmedoids::Kmedoids, distances::AbstractMatrix{<:Real}, k::Integer)::KmedoidsResult
    n = size(distances, 1)

    if n == 0
        return KmedoidsResult(n, k)
    end

    @assert n >= k

    initial_clusters = StatsBase.sample(kmedoids.rng, 1:n, k, replace = false)
    return fit(kmedoids, distances, initial_clusters)
end
