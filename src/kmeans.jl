mutable struct KmeansResult <: ClusteringResult
    k::Int
    assignments::Vector{Int}
    centers::Matrix{Float64}
    count::Vector{Int}

    objective::Float64
    iterations::Int
    elapsed::Float64
    converged::Bool
end

function KmeansResult(d::Integer, n::Integer, k::Integer)
    return KmeansResult(k, zeros(Int, n), zeros(Float64, d, k), zeros(Int, k), Inf, 0, 0, false)
end

function Base.copy(a::KmeansResult)
    return KmeansResult(
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

function isbetter(a::KmeansResult, b::KmeansResult)
    return isless(a.objective, b.objective)
end

function reset_objective!(result::KmeansResult)
    result.objective = Inf
    return nothing
end

function random_swap!(result::KmeansResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    n, d = size(data)
    k = size(result.centers, 2)

    to = rand(rng, 1:k)
    from = rand(rng, 1:n)

    result.centers[:, to] = copy(data[from, :])
    reset_objective!(result)
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

    for iteration in 1:algorithm.max_iterations
        previous_objective = result.objective

        # assignment step
        result.objective = 0
        distances = pairwise(algorithm.metric, result.centers, data', dims = 2)
        for i in 1:n
            assignment = argmin(distances[:, i])

            result.assignments[i] = assignment
            result.objective += distances[assignment, i]
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
        for i in 1:k
            result.count[i] = 0
            result.centers[:, i] .= 0
        end

        for i in 1:n
            assignment = result.assignments[i]
            result.centers[:, assignment] += data[i, :]
            result.count[assignment] += 1
        end

        for i in 1:k
            result.centers[:, i] ./= max(1, result.count[i])
        end
    end

    result.elapsed = time() - t

    return nothing
end

function fit(algorithm::Kmeans, data::AbstractMatrix{<:Real}, initial_centers::Vector{<:Integer})::KmeansResult
    n, d = size(data)
    k = length(initial_centers)

    result = KmeansResult(d, n, k)
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
    initial_centers = StatsBase.sample(algorithm.rng, 1:n, k, replace = false)
    return fit(algorithm, data, initial_centers)
end