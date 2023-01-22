Base.@kwdef mutable struct GMM <: ClusteringAlgorithm
    verbose::Bool = false
    rng::AbstractRNG = Random.GLOBAL_RNG
    estimator::RegularizedCovarianceMatrices.CovarianceMatrixEstimator
    tolerance::Float64 = 1e-3
    max_iterations::Integer = 1000
end

function seed!(algorithm::GMM, seed::Integer)
    Random.seed!(algorithm.rng, seed)
    return
end

mutable struct GMMResult <: ClusteringResult
    k::Int
    assignments::Vector{Int}
    weights::Vector{Float64}
    centers::Vector{Vector{Float64}}
    covariances::Vector{Symmetric{Float64}}

    objective::Float64
    iterations::Int
    elapsed::Float64
    converged::Bool

    function GMMResult(d::Integer, n::Integer, k::Integer)
        return new(
            k,
            zeros(Int, n),
            ones(k) ./ k,
            [zeros(d) for _ in 1:k],
            [Symmetric(Matrix{Float64}(I, d, d)) for _ in 1:k],
            -Inf,
            0,
            0,
            false
        )
    end

    function GMMResult(
        k::Int,
        assignments::Vector{Int},
        weights::Vector{Float64},
        centers::Vector{Vector{Float64}},
        covariances::Vector{Symmetric{Float64}},
        objective::Float64,
        iterations::Int,
        elapsed::Float64,
        converged::Bool,
    )
        return new(
            k, 
            assignments, 
            weights, 
            centers, 
            covariances, 
            objective, 
            iterations, 
            elapsed, 
            converged
        )
    end
end

function Base.copy(a::GMMResult)
    return GMMResult(
        a.k,
        copy(a.assignments),
        copy(a.weights),
        deepcopy(a.centers),
        deepcopy(a.covariances),
        a.objective,
        a.iterations,
        a.elapsed,
        a.converged
    )
end

function isbetter(a::GMMResult, b::GMMResult)
    return isless(b.objective, a.objective)
end

function reset_objective!(result::GMMResult)
    result.objective = -Inf
    return
end

function random_swap!(result::GMMResult, data::AbstractMatrix{<:Real}, rng::AbstractRNG)
    k = result.k
    n, d = size(data)

    to = rand(rng, 1:k)
    from = rand(rng, 1:n)

    result.centers[to] = copy(data[from, :])

    m = mean([det(result.covariances[j]) for j in 1:k])
    value = (m > 0 ? m : 1.0)^(1 / d)
    result.covariances[to] = Symmetric(value .* Matrix{Float64}(I, d, d))

    reset_objective!(result)

    return
end

function estimate_gaussian_parameters(
    algorithm::GMM,
    data::AbstractMatrix{<:Real},
    k::Int,
    responsibilities::Matrix{Float64},
)
    n, d = size(data)

    weights = ones(k) * 10 * eps(Float64)
    for i in 1:k
        if sum(responsibilities[:, i]) < 1e-32
            for j in 1:n
                responsibilities[j, i] = 1.0 / n
            end
        end

        for j in 1:n
            weights[i] += responsibilities[j, i]
        end
    end
    weights = weights / sum(weights)

    centers = [zeros(d) for _ in 1:k]
    covariances = [Symmetric(Matrix{Float64}(I, d, d)) for _ in 1:k]

    for i in 1:k
        covariances_i, centers[i] = RegularizedCovarianceMatrices.fit(algorithm.estimator, data, responsibilities[:, i])
        covariances[i] = Symmetric(covariances_i)
    end
    return weights, centers, covariances
end

function compute_precision_cholesky!(result::GMMResult, precisions_cholesky::Vector{Matrix{Float64}})
    k = length(result.covariances)
    d = size(result.covariances[1], 1)

    for i in 1:k
        try
            covariances_cholesky = cholesky(result.covariances[i])
            precisions_cholesky[i] = covariances_cholesky.U \ Matrix{Float64}(I, d, d)
        catch
            decomposition = eigen(result.covariances[i], sortby = nothing)
            result.covariances[i] = Symmetric(decomposition.vectors * Matrix(Diagonal(max.(decomposition.values, 1e-6))) * decomposition.vectors')

            covariances_cholesky = cholesky(result.covariances[i])
            precisions_cholesky[i] = covariances_cholesky.U \ Matrix{Float64}(I, d, d)
        end
    end

    return
end

function estimate_weighted_log_probabilities(
    data::AbstractMatrix{<:Real}, 
    k::Int, 
    result::GMMResult, 
    precisions_cholesky::Vector{Matrix{Float64}}
)
    n, d = size(data)

    log_det_cholesky = zeros(k)
    for i in 1:k
        for j in 1:d
            log_det_cholesky[i] += log(precisions_cholesky[i][j, j])
        end
    end

    log_probabilities = zeros(n, k)
    for i in 1:k
        y = data * precisions_cholesky[i] .- (result.centers[i]' * precisions_cholesky[i])
        log_probabilities[:, i] = sum(y .^ 2, dims = 2)
    end

    return -0.5 * (d * log(2 * pi) .+ log_probabilities) .+ (log_det_cholesky') .+ log.(result.weights)'
end

function expectation_step(
    data::AbstractMatrix{<:Real},
    k::Int,
    result::GMMResult,
    precisions_cholesky::Vector{Matrix{Float64}},
)
    n, d = size(data)

    weighted_log_probabilities = estimate_weighted_log_probabilities(data, k, result, precisions_cholesky)

    log_probabilities_norm = zeros(n)
    for i in 1:n
        log_probabilities_norm[i] += LogExpFunctions.logsumexp(weighted_log_probabilities[i, :])
    end

    log_responsibilities = weighted_log_probabilities .- log_probabilities_norm
    return mean(log_probabilities_norm), log_responsibilities
end

function maximization_step!(
    algorithm::GMM,
    data::AbstractMatrix{<:Real},
    k::Int,
    result::GMMResult,
    log_responsibilities::Matrix{Float64},
    precisions_cholesky::Vector{Matrix{Float64}},
)
    responsibilities = exp.(log_responsibilities)

    result.weights, result.centers, result.covariances = estimate_gaussian_parameters(algorithm, data, k, responsibilities)
    compute_precision_cholesky!(result, precisions_cholesky)

    return
end

function fit!(algorithm::GMM, data::AbstractMatrix{<:Real}, result::GMMResult)
    t = time()

    n, d = size(data)
    k = length(result.centers)

    previous_objective = Inf
    result.objective = -Inf

    result.iterations = algorithm.max_iterations
    result.converged = false

    log_responsibilities = zeros(n, k)

    precisions_cholesky = [zeros(d, d) for _ in 1:k]
    compute_precision_cholesky!(result, precisions_cholesky)

    for iteration in 1:algorithm.max_iterations
        previous_objective = result.objective

        t1 = @elapsed result.objective, log_responsibilities = expectation_step(data, k, result, precisions_cholesky)

        t2 = @elapsed maximization_step!(algorithm, data, k, result, log_responsibilities, precisions_cholesky)

        change = abs(result.objective - previous_objective)

        if algorithm.verbose
            print_iteration(iteration)
            print_objective(result)
            print_change(change)
            print_elapsed(t1 + t2)
            print_newline()
        end

        if change < algorithm.tolerance
            result.converged = true
            result.iterations = iteration
            break
        end
    end

    weighted_log_probabilities = estimate_weighted_log_probabilities(data, k, result, precisions_cholesky)
    for i in 1:n
        result.assignments[i] = argmax(weighted_log_probabilities[i, :])
    end

    result.elapsed = time() - t

    return
end

function fit(algorithm::GMM, data::AbstractMatrix{<:Real}, initial_centers::Vector{<:Integer})::GMMResult
    n, d = size(data)
    k = length(initial_centers)

    result = GMMResult(d, n, k)
    for i in 1:k
        for j in 1:d
            result.centers[i][j] = data[initial_centers[i], j]
        end
    end

    if algorithm.verbose
        print_string("Initial centers = [")
        for i in 1:k
            print_string("$(initial_centers[i]),")
        end
        print_string("]")
        print_newline()
    end

    fit!(algorithm, data, result)

    return result
end

function fit(algorithm::GMM, data::AbstractMatrix{<:Real}, k::Integer)::GMMResult
    n, d = size(data)
    initial_centers = StatsBase.sample(algorithm.rng, 1:n, k, replace = false)
    return fit(algorithm, data, initial_centers)
end
