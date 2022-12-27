Base.@kwdef struct Kmeans <: Algorithm
    verbose::Bool = false
    rng::AbstractRNG = Random.GLOBAL_RNG
    metric::SemiMetric = SqEuclidean()
    tolerance::Float64 = 1e-3
    max_iterations::Integer = 1000
end

mutable struct KmeansResult <: Result
    k::Int
    assignments::Vector{Int} # n
    centers::Matrix{Float64} # d x k
    count::Vector{Int} # k
    objective::Float64
    iterations::Int
    converged::Bool

    function KmeansResult(d::Integer, n::Integer, k::Integer)
        return new(k, zeros(Int, n), zeros(Float64, d, k), zeros(Int, k), Inf, 0, false)
    end

    function KmeansResult(
        k::Int,
        assignments::Vector{Int},
        centers::Matrix{Float64},
        count::Vector{Int},
        objective::Float64,
        iterations::Int,
        converged::Bool,
    )
        return new(k, assignments, centers, count, objective, iterations, converged)
    end
end

function Base.copy(a::KmeansResult)
    return KmeansResult(a.k, copy(a.assignments), copy(a.centers), copy(a.count), a.objective, a.iterations, a.converged)
end

function isbetter(a::KmeansResult, b::KmeansResult)
    return isless(a.objective, b.objective)
end

function reset_objective!(result::KmeansResult)
    result.objective = Inf
    return
end

function random_swap!(result::KmeansResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    d, n = size(data)
    k = size(result.centers, 2)

    to = rand(rng, 1:k)
    from = rand(rng, 1:n)

    result.centers[:, to] = copy(data[:, from])
    reset_objective!(result)
    return
end

function train!(parameters::Kmeans, data::AbstractMatrix{<:Real}, result::KmeansResult)
    d, n = size(data)
    k = size(result.centers, 2)

    previous_objective = -Inf
    result.objective = Inf
    result.iterations = parameters.max_iterations
    result.converged = false

    for iteration in 1:parameters.max_iterations
        previous_objective = result.objective

        # assignment step
        result.objective = 0
        distances = pairwise(parameters.metric, result.centers, data, dims = 2)
        for i in 1:n
            assignment = argmin(distances[:, i])

            result.assignments[i] = assignment
            result.objective += distances[assignment, i]
        end

        if parameters.verbose
            println("$iteration - $(result.objective)")
        end

        # stopping condition
        if abs(result.objective - previous_objective) < parameters.tolerance
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
            result.centers[:, assignment] += data[:, i]
            result.count[assignment] += 1
        end

        for i in 1:k
            result.centers[:, i] ./= max(1, result.count[i])
        end
    end

    return
end

# """
#     kmeans(data::Matrix{Float64}, k::Integer)::KmeansResult

# TODO.
# # Keyword Arguments
# - `rng`
# - `verbose` 
# - `metric`
# - `tolerance`
# - `max_iterations`

# # Example

# ```jldoctest; setup=:(using UnsupervisedClustering)
# julia> data = rand(10, 2);
# julia> result = kmeans(data, 3);
# ```
# """
function train(parameters::Kmeans, data::AbstractMatrix{<:Real}, k::Integer)::KmeansResult
    d, n = size(data)

    result = KmeansResult(d, n, k)
    permutation = randperm(parameters.rng, n)
    for i in 1:d
        for j in 1:k
            result.centers[i, j] = data[i, permutation[j]]
        end
    end
    train!(parameters, data, result)

    return result
end
