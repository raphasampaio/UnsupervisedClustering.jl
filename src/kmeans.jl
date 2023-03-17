mutable struct KmeansResult <: ClusteringResult
    k::Int
    assignments::Vector{Int}
    centers::Matrix{Float64}

    objective::Float64
    iterations::Int
    elapsed::Float64
    converged::Bool
end

function KmeansResult(d::Integer, n::Integer, k::Integer)
    return KmeansResult(k, zeros(Int, n), zeros(Float64, d, k), Inf, 0, 0, false)
end

function isbetter(a::KmeansResult, b::KmeansResult)
    return isless(a.objective, b.objective)
end

function reset_objective!(result::KmeansResult)
    result.objective = Inf
    return nothing
end

function fit!(algorithm::Kmeans, data::AbstractMatrix{<:Real}, result::KmeansResult)
    t = time()

    n, d = size(data)
    k = size(result.centers, 2)

    previous_objective = -Inf
    result.objective = Inf
    result.iterations = algorithm.max_iterations
    result.converged = false

    count = zeros(Int, k)
    distances = zeros(k, n)

    for iteration in 1:algorithm.max_iterations
        previous_objective = result.objective

        # assignment step
        result.objective = 0
        pairwise!(distances, algorithm.metric, result.centers, data', dims = 2)
        for i in 1:n
            min_distance = Inf
            min_center = 0
            for center in 1:k
                distance = distances[center, i]
                if distance < min_distance
                    min_distance = distance
                    min_center = center
                end
            end
            result.assignments[i] = min_center
            result.objective += distances[min_center, i]
        end

        change = abs(result.objective - previous_objective)

        if algorithm.verbose
            print_iteration(iteration)
            print_objective(result)
            print_change(change)
            print_newline()
        end

        # stopping condition
        if change < algorithm.tolerance || n == k
            result.converged = true
            result.iterations = iteration
            break
        end

        # update step
        for i in 1:k
            count[i] = 0
            for j in 1:d
                result.centers[j, i] = 0
            end
        end

        for i in 1:n
            assignment = result.assignments[i]
            for j in 1:d
                result.centers[j, assignment] += data[i, j]
            end
            count[assignment] += 1
        end

        for i in 1:k
            cluster_size = max(1, count[i])
            for j in 1:d
                result.centers[j, i] = result.centers[j, i] / cluster_size
            end
        end
    end

    result.elapsed = time() - t

    return nothing
end

function fit(algorithm::Kmeans, data::AbstractMatrix{<:Real}, initial_centers::Vector{<:Integer})::KmeansResult
    n, d = size(data)
    k = length(initial_centers)

    result = KmeansResult(d, n, k)
    if n == 0
        return result
    end

    @assert d > 0
    @assert n >= k

    for i in 1:d
        for j in 1:k
            result.centers[i, j] = data[initial_centers[j], i]
        end
    end

    if algorithm.verbose
        print_initial_centers(initial_centers)
    end

    fit!(algorithm, data, result)

    return result
end

function fit(algorithm::Kmeans, data::AbstractMatrix{<:Real}, k::Integer)::KmeansResult
    n, d = size(data)

    if n == 0
        return KmeansResult(d, n, k)
    end

    @assert n >= k

    initial_centers = StatsBase.sample(algorithm.rng, 1:n, k, replace = false)
    return fit(algorithm, data, initial_centers)
end
