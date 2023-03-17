mutable struct KmedoidsResult <: ClusteringResult
    k::Int
    assignments::Vector{Int}
    centers::Vector{Int}

    objective::Float64
    iterations::Int
    elapsed::Float64
    converged::Bool
end

function KmedoidsResult(d::Integer, n::Integer, k::Integer)
    return KmedoidsResult(k, zeros(Int, n), zeros(Int, k), Inf, 0, 0, false)
end

function isbetter(a::KmedoidsResult, b::KmedoidsResult)
    return isless(a.objective, b.objective)
end

function reset_objective!(result::KmedoidsResult)
    result.objective = Inf
    return nothing
end

function fit!(algorithm::Kmedoids, data::AbstractMatrix{<:Real}, result::KmedoidsResult)
    t = time()

    n, d = size(data)
    k = length(result.centers)

    distances = pairwise(algorithm.metric, data, dims = 1)
    medoids = [Vector{Int}() for _ in 1:k]
    count = zeros(Int, k)

    previous_objective = -Inf
    result.objective = Inf
    result.iterations = algorithm.max_iterations
    result.converged = false

    for iteration in 1:algorithm.max_iterations
        previous_objective = result.objective

        # assignment step
        result.objective = 0
        for i in 1:k
            empty!(medoids[i])
            count[i] = 0
        end

        for i in 1:n
            min_distance = Inf
            min_center = 0

            for j in 1:k
                center = result.centers[j]
                if i == center
                    min_distance = 0
                    min_center = j
                    break
                end

                distance = distances[i, center]
                if distance < min_distance
                    min_distance = distance
                    min_center = j
                end
            end

            count[min_center] += 1
            push!(medoids[min_center], i)
            result.objective += distances[i, result.centers[min_center]]
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
        for (i, medoid) in enumerate(medoids)
            j = argmin(sum(distances[medoid, medoid], dims = 2))[1]
            result.centers[i] = medoid[j]
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

function fit(algorithm::Kmedoids, data::AbstractMatrix{<:Real}, initial_centers::Vector{<:Integer})::KmedoidsResult
    n, d = size(data)
    k = length(initial_centers)

    result = KmedoidsResult(d, n, k)
    if n == 0
        return result
    end

    @assert d > 0
    @assert n >= k

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

    if n == 0
        return KmedoidsResult(d, n, k)
    end

    @assert n >= k

    initial_centers = StatsBase.sample(algorithm.rng, 1:n, k, replace = false)
    return fit(algorithm, data, initial_centers)
end
