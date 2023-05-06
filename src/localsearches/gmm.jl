mutable struct GMMResult <: ClusteringResult
    k::Int
    assignments::Vector{Int}
    weights::Vector{Float64}
    clusters::Vector{Vector{Float64}}
    covariances::Vector{Symmetric{Float64}}

    objective::Float64
    iterations::Int
    elapsed::Float64
    converged::Bool
end

function GMMResult(
    assignments::AbstractVector{<:Integer},
    weights::AbstractVector{<:Real},
    clusters::AbstractVector{<:AbstractVector{<:Real}},
    covariances::AbstractVector{<:Symmetric{<:Real}},
)
    k = length(weights)
    return GMMResult(k, assignments, weights, clusters, covariances, -Inf, 0, 0, false)
end

function GMMResult(d::Integer, n::Integer, k::Integer)
    assignments = zeros(Int, n)
    weights = ones(k) ./ k
    clusters = [zeros(d) for _ in 1:k]
    covariances = [identity_matrix(d) for _ in 1:k]
    return GMMResult(assignments, weights, clusters, covariances)
end

function isbetter(a::GMMResult, b::GMMResult)
    return isless(b.objective, a.objective)
end

function reset_objective!(result::GMMResult)
    result.objective = -Inf
    return nothing
end

function estimate_gaussian_parameters(
    gmm::GMM,
    data::AbstractMatrix{<:Real},
    k::Integer,
    responsibilities::AbstractMatrix{<:Real},
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

    clusters = [zeros(d) for _ in 1:k]
    covariances = [identity_matrix(d) for _ in 1:k]

    for i in 1:k
        covariances_i, clusters[i] = RegularizedCovarianceMatrices.fit(gmm.estimator, data, responsibilities[:, i])
        covariances[i] = Symmetric(covariances_i)
    end
    return weights, clusters, covariances
end

function compute_precision_cholesky!(
    gmm::GMM,
    result::GMMResult,
    precisions_cholesky::AbstractVector{<:AbstractMatrix{<:Real}},
)
    k = length(result.covariances)
    d = size(result.covariances[1], 1)

    for i in 1:k
        try
            covariances_cholesky = cholesky(result.covariances[i])
            precisions_cholesky[i] = covariances_cholesky.U \ Matrix{Float64}(I, d, d)
        catch e
            if gmm.decompose_if_fails
                decomposition = eigen(result.covariances[i], sortby = nothing)
                result.covariances[i] = Symmetric(decomposition.vectors * Matrix(Diagonal(max.(decomposition.values, 1e-6))) * decomposition.vectors')
                covariances_cholesky = cholesky(result.covariances[i])
                precisions_cholesky[i] = covariances_cholesky.U \ Matrix{Float64}(I, d, d)
            else
                error("GMM Failed: $e")
            end
        end
    end

    return nothing
end

function estimate_weighted_log_probabilities(
    data::AbstractMatrix{<:Real},
    k::Integer,
    result::GMMResult,
    precisions_cholesky::AbstractVector{<:AbstractMatrix{<:Real}},
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
        y = data * precisions_cholesky[i] .- (result.clusters[i]' * precisions_cholesky[i])
        log_probabilities[:, i] = sum(y .^ 2, dims = 2)
    end

    return -0.5 * (d * log(2 * pi) .+ log_probabilities) .+ (log_det_cholesky') .+ log.(result.weights)'
end

function expectation_step(
    data::AbstractMatrix{<:Real},
    k::Integer,
    result::GMMResult,
    precisions_cholesky::AbstractVector{<:AbstractMatrix{<:Real}},
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
    gmm::GMM,
    data::AbstractMatrix{<:Real},
    k::Integer,
    result::GMMResult,
    log_responsibilities::AbstractMatrix{<:Real},
    precisions_cholesky::AbstractVector{<:AbstractMatrix{<:Real}},
)
    responsibilities = exp.(log_responsibilities)

    result.weights, result.clusters, result.covariances = estimate_gaussian_parameters(gmm, data, k, responsibilities)
    compute_precision_cholesky!(gmm, result, precisions_cholesky)

    return nothing
end

function fit!(gmm::GMM, data::AbstractMatrix{<:Real}, result::GMMResult)
    t = time()

    n, d = size(data)
    k = length(result.clusters)

    previous_objective = Inf
    result.objective = -Inf

    result.iterations = gmm.max_iterations
    result.converged = false

    log_responsibilities = zeros(n, k)

    precisions_cholesky = [zeros(d, d) for _ in 1:k]
    compute_precision_cholesky!(gmm, result, precisions_cholesky)

    for iteration in 1:gmm.max_iterations
        previous_objective = result.objective

        t1 = @elapsed result.objective, log_responsibilities = expectation_step(data, k, result, precisions_cholesky)

        t2 = @elapsed maximization_step!(gmm, data, k, result, log_responsibilities, precisions_cholesky)

        change = abs(result.objective - previous_objective)

        if gmm.verbose
            print_iteration(iteration)
            print_objective(result)
            print_change(change)
            print_elapsed(t1 + t2)
            print_newline()
        end

        if change < gmm.tolerance || n == k
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

    return nothing
end

function fit(gmm::GMM, data::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})::GMMResult
    n, d = size(data)
    k = length(initial_clusters)

    result = GMMResult(d, n, k)
    if n == 0
        return result
    end

    @assert d > 0
    @assert n >= k

    for i in 1:k
        for j in 1:d
            result.clusters[i][j] = data[initial_clusters[i], j]
        end
    end

    if gmm.verbose
        print_initial_clusters(initial_clusters)
    end

    fit!(gmm, data, result)

    return result
end

function fit(gmm::GMM, data::AbstractMatrix{<:Real}, k::Integer)::GMMResult
    n, d = size(data)

    if n == 0
        return GMMResult(d, n, k)
    end

    @assert n >= k

    initial_clusters = StatsBase.sample(gmm.rng, 1:n, k, replace = false)
    return fit(gmm, data, initial_clusters)
end
