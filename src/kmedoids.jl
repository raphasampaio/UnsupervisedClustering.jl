Base.@kwdef mutable struct Kmedoids <: Algorithm
    verbose::Bool = false
    rng::AbstractRNG = Random.GLOBAL_RNG
    metric::SemiMetric = SqEuclidean()
    tolerance::Float64 = 1e-3
    max_iterations::Integer = 1000
end

function seed!(algorithm::Kmedoids, seed::Integer)
    Random.seed!(algorithm.rng, seed)
    return
end

mutable struct KmedoidsResult <: Result
    k::Int
    assignments::Vector{Int}
    centers::Vector{Int}
    count::Vector{Int}

    objective::Float64
    iterations::Int
    elapsed::Float64
    converged::Bool

    function KmedoidsResult(d::Integer, n::Integer, k::Integer)
        return new(k, zeros(Int, n), zeros(Int, k), zeros(Int, k), Inf, 0, 0, false)
    end

    function KmedoidsResult(
        k::Int,
        assignments::Vector{Int},
        centers::Vector{Int},
        count::Vector{Int},
        objective::Float64,
        iterations::Int,
        elapsed::Float64,
        converged::Bool,
    )
        return new(k, assignments, centers, count, objective, iterations, elapsed, converged)
    end
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
    return
end

function random_swap!(result::KmedoidsResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    n, d = size(data)
    k = length(result.centers)

    to = rand(rng, 1:k)
    from = rand(rng, 1:n)

    result.centers[to] = from

    reset_objective!(result)

    return
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

    return
end

function fit(algorithm::Kmedoids, data::AbstractMatrix{<:Real}, k::Integer)::KmedoidsResult
    n, d = size(data)

    result = KmedoidsResult(d, n, k)
    permutation = randperm(algorithm.rng, n)
    for i in 1:k
        result.centers[i] = permutation[i]
    end

    fit!(algorithm, data, result)

    return result
end
