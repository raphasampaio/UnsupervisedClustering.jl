Base.@kwdef struct GMM <: Algorithm
    verbose::Bool = false
    rng::AbstractRNG = Random.GLOBAL_RNG
    estimator::RegularizedCovarianceMatrices.CovarianceMatrixEstimator
    tolerance::Float64 = 1e-3
    max_iterations::Integer = 1000
end

mutable struct GMMResult <: Result
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

function fix(matrix::AbstractMatrix{<:Real}, eps::Float64)
    eigen_matrix = eigen(matrix)
    new_matrix = eigen_matrix.vectors * Matrix(Diagonal(max.(eigen_matrix.values, eps))) * eigen_matrix.vectors'
    return Symmetric(new_matrix)
end

function estimate_gaussian_parameters(
    parameters::GMM,
    data::AbstractMatrix{<:Real},
    k::Int,
    responsibilities::Matrix{Float64},
)
    n, d = size(data)

    weights = ones(k) * 10 * eps(Float64)
    for i in 1:k
        sum_weights = sum(responsibilities[:, i])
        if sum_weights < 1e-32
            for j in 1:n
                responsibilities[j, i] = 1.0 / n
            end
        end

        for j in 1:n
            weights[i] += responsibilities[j, i]
        end
        # weights[i] /= n
    end
    weights = weights / sum(weights)

    centers = [zeros(d) for _ in 1:k]
    covariances = [Symmetric(Matrix{Float64}(I, d, d)) for _ in 1:k]

    for i in 1:k
        covariances_i, centers[i] = RegularizedCovarianceMatrices.fit(parameters.estimator, data, responsibilities[:, i])
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
            result.covariances[i] = fix(result.covariances[i], 1e-6)
            covariances_cholesky = cholesky(result.covariances[i])
            precisions_cholesky[i] = covariances_cholesky.U \ Matrix{Float64}(I, d, d)
        end
    end

    return
end

# https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
function compute_log_det_cholesky(precisions_cholesky::Vector{Matrix{Float64}})
    k = length(precisions_cholesky)
    d = size(precisions_cholesky[1], 1)

    log_det_cholesky = zeros(k)
    for i in 1:k
        for j in 1:d
            log_det_cholesky[i] += log(precisions_cholesky[i][j, j])
        end
    end
    return log_det_cholesky
end

function estimate_weighted_log_probability(
    data::AbstractMatrix{<:Real}, 
    k::Int, 
    result::GMMResult, 
    precisions_cholesky::Vector{Matrix{Float64}}
)
    n, d = size(data)
    
    log_det = compute_log_det_cholesky(precisions_cholesky)

    log_prob = zeros(n, k)
    for i in 1:k
        y = data * precisions_cholesky[i] .- (result.centers[i]' * precisions_cholesky[i])
        log_prob[:, i] = sum(y .^ 2, dims = 2)
    end

    return -0.5 * (d * log(2 * pi) .+ log_prob) .+ (log_det') .+ log.(result.weights)'
end

function estimate_log_prob_responsibilities(
    data::AbstractMatrix{<:Real},
    k::Int,
    result::GMMResult,
    precisions_cholesky::Vector{Matrix{Float64}},
)
    n, d = size(data)

    weighted_log_prob = estimate_weighted_log_probability(data, k, result, precisions_cholesky)

    log_prob_norm = zeros(n)
    for i in 1:n
        log_prob_norm[i] += LogExpFunctions.logsumexp(weighted_log_prob[i, :])
    end

    log_resp = weighted_log_prob .- log_prob_norm
    return log_prob_norm, log_resp
end

function expectation_step(
    data::AbstractMatrix{<:Real},
    k::Int,
    result::GMMResult,
    precisions_cholesky::Vector{Matrix{Float64}},
)
    log_prob_norm, log_resp = estimate_log_prob_responsibilities(data, k, result, precisions_cholesky)
    return mean(log_prob_norm), log_resp
end

function maximization_step!(
    parameters::GMM, 
    data::AbstractMatrix{<:Real},
    k::Int,
    result::GMMResult,
    log_resp::Matrix{Float64},
    precisions_cholesky::Vector{Matrix{Float64}},
)
    responsibilities = exp.(log_resp)

    result.weights, result.centers, result.covariances = estimate_gaussian_parameters(parameters, data, k, responsibilities)
    compute_precision_cholesky!(result, precisions_cholesky)

    return
end

function train!(parameters::GMM, data::AbstractMatrix{<:Real}, result::GMMResult)
    t = time()

    n, d = size(data)
    k = length(result.centers)

    best_objective = -Inf
    previous_objective = Inf
    result.objective = -Inf
    
    result.iterations = parameters.max_iterations
    result.converged = false

    log_resp = zeros(n, k)

    precisions_cholesky = [zeros(d, d) for _ in 1:k]
    compute_precision_cholesky!(result, precisions_cholesky)

    for iteration in 1:parameters.max_iterations
        # if result.objective < best_objective && result.objective > previous_objective
        #     best_objective = result.objective
        # else
        #     best_objective = max(best_objective, result.objective)
        # end

        previous_objective = result.objective

        t1 = @elapsed result.objective, log_resp = expectation_step(data, k, result, precisions_cholesky)

        t2 = @elapsed maximization_step!(parameters, data, k, result, log_resp, precisions_cholesky)

        # change = abs(result.objective - previous_objective)
        change = abs((previous_objective - result.objective) / previous_objective)

        if parameters.verbose
            print_iteration(iteration)
            print_objective(result)
            print_objective(best_objective)
            print_change(change)
            print_elapsed(t1 + t2)
            print_newline()
        end

        if change < parameters.tolerance || best_objective ≈ result.objective
            result.converged = true
            result.iterations = iteration
            break
        end
    end

    weighted_log_prob = estimate_weighted_log_probability(data, k, result, precisions_cholesky)
    for i in 1:n
        result.assignments[i] = argmax(weighted_log_prob[i, :])
    end

    result.elapsed = time() - t

    return
end

function train(parameters::GMM, data::AbstractMatrix{<:Real}, k::Integer)::GMMResult
    n, d = size(data)

    result = GMMResult(d, n, k)
    permutation = randperm(parameters.rng, n)
    for i in 1:k
        for j in 1:d
            result.centers[i][j] = data[permutation[i], j]
        end
    end

    # responsibilities = zeros(n, k)
    # for i in 1:n
    #     min_distance = Inf
    #     min_index = -1
        
    #     for j in 1:k
    #         distance = euclidean(data[i, :], result.centers[j])
    #         if distance < min_distance
    #             min_distance = distance
    #             min_index = j
    #         end
    #     end
    #     responsibilities[i, min_index] = 1.0
    # end
    # result.weights, result.centers, result.covariances = estimate_gaussian_parameters(parameters, data, k, responsibilities)

    train!(parameters, data, result)

    return result
end