function estimate_gaussian_parameters!(data::ClusteringData, result::SoftResult, responsibilities::Matrix{Float64}, method::Function, cache::CovarianceCache) where T
    n = data.n
    d = data.d
    k = data.k

    for i in 1:k
        result.weights[i] = 10 * eps(Float64)

        sum_weights = sum(responsibilities[:, i])
        if sum_weights < 1e-32
            for j in 1:n
                responsibilities[j, i] = 1.0 / n
            end
        end

        for j in 1:n
            result.weights[i] += responsibilities[j, i]
        end
        result.weights[i] /= n
    end

    # nk = sum(responsibilities, dims=1) .+ 10 * eps(Float64)
    for i in 1:k
        method(data.X, responsibilities[:, i], result.covariances[i], result.centers[i], cache)
        # result.covariances[i] += 1e-6 * Matrix{Float64}(I, d, d)
    end
    return nothing
end

function compute_precision_cholesky!(result::SoftResult, precisions_cholesky::Vector{Matrix{Float64}})
    k = length(result.covariances)
    d = size(result.covariances[1], 1)

	for i in 1:k
        try
            covariances_cholesky = cholesky(Symmetric(result.covariances[i]))
            result.L[i] = covariances_cholesky.L
            precisions_cholesky[i] = covariances_cholesky.U \ Matrix{Float64}(I, d, d)
        catch
            # a = max(0.4 * tr(result.covariances[i])/d, 1e-6)
            # result.covariances[i] = reset_model(result.model, result.covariances[i], a)

            result.covariances[i] = fix(result.covariances[i], 1e-6)

            # result.covariances[i] = solve_model(result.model)
            # result.covariances[i] = reset_model(result.model, result.covariances[i])
            
            # model = init_model(result.covariances[i], 1e-3)
            # result.covariances[i] = solve_model(model)

            covariances_cholesky = cholesky(Symmetric(result.covariances[i]))
            result.L[i] = covariances_cholesky.L
            precisions_cholesky[i] = covariances_cholesky.U \ Matrix{Float64}(I, d, d)
		end
    end

    return nothing
end

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

# function _estimate_log_gaussian_prob(data::ClusteringData, result::SoftResult, precisions_cholesky::Vector{Matrix{Float64}})
#     centers = result.centers
    
#     n = data.n
#     d = data.d
#     k = data.k

#     log_det = compute_log_det_cholesky(precisions_cholesky)

#     log_prob = zeros(n, k) # (n_samples, n_components)
#     for i in 1:k
#         y = data.X * precisions_cholesky[i] .- (centers[i]' * precisions_cholesky[i])
#         log_prob[:, i] = sum(y.^2, dims=2)
#     end

#     return -0.5 * (d * log(2 * pi) .+ log_prob) .+ (log_det')
# end

function estimate_weighted_log_prob(data::ClusteringData, result::SoftResult, precisions_cholesky::Vector{Matrix{Float64}})
    centers = result.centers
    
    n = data.n
    d = data.d
    k = data.k

    log_det = compute_log_det_cholesky(precisions_cholesky)

    log_prob = zeros(n, k) # (n_samples, n_components)
    for i in 1:k
        y = data.X * precisions_cholesky[i] .- (centers[i]' * precisions_cholesky[i])
        log_prob[:, i] = sum(y.^2, dims=2)
    end

    return -0.5 * (d * log(2 * pi) .+ log_prob) .+ (log_det') .+ log.(result.weights)'
end

function logsumexp2(X::Matrix{Float64})
    array = zeros(size(X, 1))
    for i in 1:size(X, 1)
        array[i] += logsumexp(X[i, :])
    end
    return array
end

function estimate_log_prob_responsibilities!(data::ClusteringData, result::SoftResult, precisions_cholesky::Vector{Matrix{Float64}}, log_resp::Matrix{Float64})
    weighted_log_prob = estimate_weighted_log_prob(data, result, precisions_cholesky)
    log_prob_norm = logsumexp2(weighted_log_prob)
    log_resp .= weighted_log_prob .- log_prob_norm
    return log_prob_norm
end

function expectation_step!(data::ClusteringData, result::SoftResult, precisions_cholesky::Vector{Matrix{Float64}}, log_resp::Matrix{Float64})
    log_prob_norm = estimate_log_prob_responsibilities!(data, result, precisions_cholesky, log_resp)
    return mean(log_prob_norm)
end

function maximization_step!(data::ClusteringData, result::SoftResult, log_resp::Matrix{Float64}, precisions_cholesky, method::Function, cache::CovarianceCache)
    n = data.n
    d = data.d
    k = data.k

    responsibilities = exp.(log_resp)

    estimate_gaussian_parameters!(data, result, responsibilities, method, cache)
    compute_precision_cholesky!(result, precisions_cholesky)
    
    return nothing
end

function initialize_responsibilities(data::ClusteringData, result::SoftResult) where T
    centers = result.centers

    n = data.n
    k = data.k

    responsibilities = zeros(n, k)
    for i in 1:n
        min_distance = Inf
        min_index = -1
        
        for j in 1:k
            distance = euclidean(data.X[i, :], centers[j])
            if distance < min_distance
                min_distance = distance
                min_index = j
            end
        end

        responsibilities[i, min_index] = 1.0
    end

    return responsibilities
end

gmm(X::Matrix{T}, k::Int) where T = _gmm!(ClusteringData(X, k))
_gmm!(data::ClusteringData) = _gmm!(data, empirical_covariance!)
_gmm!(data::ClusteringData, result::SoftResult) = _gmm!(data, result, empirical_covariance!)

gmm_shrunk(X::Matrix{T}, k::Int) where T = _gmm_shrunk!(ClusteringData(X, k))
_gmm_shrunk!(data::ClusteringData) = _gmm!(data, shrunk_covariance!)
_gmm_shrunk!(data::ClusteringData, result::SoftResult) = _gmm!(data, result, shrunk_covariance!)

gmm_oas(X::Matrix{T}, k::Int) where T = _gmm_oas!(ClusteringData(X, k))
_gmm_oas!(data::ClusteringData) = _gmm!(data, oas_covariance!)
_gmm_oas!(data::ClusteringData, result::SoftResult) = _gmm!(data, result, oas_covariance!)

gmm_ledoitwolf(X::Matrix{T}, k::Int) where T = _gmm_ledoitwolf!(ClusteringData(X, k))
_gmm_ledoitwolf!(data::ClusteringData) = _gmm!(data, ledoitwolf_covariance!)
_gmm_ledoitwolf!(data::ClusteringData, result::SoftResult) = _gmm!(data, result, ledoitwolf_covariance!)

function _gmm!(data::ClusteringData, method::Function)
    result = SoftResult(data)

    cache = CovarianceCache(data.n, data.d)

    initialize_centers!(data, result)
    responsibilities = initialize_responsibilities(data, result)
    estimate_gaussian_parameters!(data, result, responsibilities, method, cache)

    # @show sum(result.weights)
    # result.weights ./= sum(result.weights) # TODO RETI

    _gmm!(data, result, method)

    return result
end

function _gmm!(data::ClusteringData, result::SoftResult, method::Function)
    n = data.n
    d = data.d
    k = data.k

    cache = CovarianceCache(n, d)
    
    lowerbound = -Inf
    previous_lowerbound = Inf

    log_resp = zeros(n, k)

    precisions_cholesky = [zeros(d, d) for i in 1:k]
    compute_precision_cholesky!(result, precisions_cholesky)

  	for i in 1:MAX_ITERATIONS
        previous_lowerbound = lowerbound

        lowerbound = expectation_step!(data, result, precisions_cholesky, log_resp)

        maximization_step!(data, result, log_resp, precisions_cholesky, method, cache)
        
		if abs(lowerbound - previous_lowerbound) < 1e-3
        	break
    	end
	end

    weighted_log_prob = estimate_weighted_log_prob(data, result, precisions_cholesky)
    result.assignments = zeros(Int, n)
	for i in 1:n
    	result.assignments[i] = argmax(weighted_log_prob[i, :])
    end
    result.totalcost = lowerbound

    return nothing
end
