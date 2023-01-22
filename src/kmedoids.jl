mutable struct KmedoidsResult <: ClusteringResult
    const k::Int
    assignments::Vector{Int}
    centers::Vector{Int}
    count::Vector{Int}

    objective::Float64
    iterations::Int
    elapsed::Float64
    converged::Bool
end

function KmedoidsResult(d::Integer, n::Integer, k::Integer)
    return KmedoidsResult(k, zeros(Int, n), zeros(Int, k), zeros(Int, k), Inf, 0, 0, false)
end

function Base.copy(a::KmedoidsResult)
    return KmedoidsResult(
        a.k,
        copy(a.assignments),
        copy(a.centers),
        copy(a.count),
        a.objective,
        a.iterations,
        a.elapsed,
        a.converged
    )
end

function isbetter(a::KmedoidsResult, b::KmedoidsResult)
    return isless(a.objective, b.objective)
end

function reset_objective!(result::KmedoidsResult)
    result.objective = Inf
    return nothing
end

function random_swap!(result::KmedoidsResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    n, d = size(data)
    k = length(result.centers)

    to = rand(rng, 1:k)
    from = rand(rng, 1:n)

    result.centers[to] = from

    reset_objective!(result)

    return nothing
end

function fit!(algorithm::Kmedoids, data::AbstractMatrix{<:Real}, result::KmedoidsResult)
    t = time()

    n, d = size(data)
    k = length(result.centers)

    distances = pairwise(algorithm.metric, data', dims = 2)

    previous_objective = -Inf
    result.objective = Inf
    result.iterations = algorithm.max_iterations
    result.converged = false

    for iteration in 1:algorithm.max_iterations
        previous_objective = result.objective

        # assignment step
        result.objective = 0
        for i in 1:n
            min_distance = Inf
            min_j = 0
            for j in 1:k
                distance = distances[i, result.centers[j]]
                if distance < min_distance
                    min_distance = distance
                    min_j = j
                end
            end
            result.assignments[i] = min_j
            result.objective += distances[i, result.centers[min_j]]
        end

        change = abs(result.objective - previous_objective)

        if algorithm.verbose
            print_iteration(iteration)
            print_objective(result)
            print_change(change)
            print_newline()
        end

        # stopping condition
        if change < algorithm.tolerance
            result.converged = true
            result.iterations = iteration
            break
        end

        # update step
        min_distance = Inf * ones(k)
        for i in 1:n
            distance = 0
            assignment = result.assignments[i]

            for j in 1:n
                if assignment == result.assignments[j]
                    distance += distances[i, j]
                end
            end

            if distance < min_distance[assignment]
                min_distance[assignment] = distance
                result.centers[assignment] = i
            end
        end
    end

    result.elapsed = time() - t

    return nothing
end

function fit(algorithm::Kmedoids, data::AbstractMatrix{<:Real}, initial_centers::Vector{<:Integer})::KmedoidsResult
    n, d = size(data)
    k = length(initial_centers)

    result = KmedoidsResult(d, n, k)
    for i in 1:k
        result.centers[i] = initial_centers[i]
    end

    if algorithm.verbose
        print_initial_centers(initial_centers)
    end

    fit!(algorithm, data, result)

    return result
end

function fit(algorithm::Kmedoids, data::AbstractMatrix{<:Real}, k::Integer)::KmedoidsResult
    n, d = size(data)
    initial_centers = StatsBase.sample(algorithm.rng, 1:n, k, replace = false)
    return fit(algorithm, data, initial_centers)
end
