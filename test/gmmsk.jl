Base.@kwdef struct GMMSK <: UnsupervisedClustering.Algorithm
    verbose::Bool = false
    rng::AbstractRNG = Random.GLOBAL_RNG
    tolerance::Float64 = 1e-3
    max_iterations::Integer = 1000
end

function UnsupervisedClustering.train!(parameters::GMMSK, data::AbstractMatrix{<:Real}, result::GMMResult)
    n, d = size(data)
    k = length(result.centers)

    μ = zeros(k, d)
    for i in 1:k
        for j in 1:d
            μ[i, j] = result.centers[i][j]
        end
    end
    
    Σ = zeros(k, d, d)
    for i in 1:k
        for j in 1:d
            for l in 1:d
                Σ[i, j, l] = result.covariances[i][j, l]
            end
        end
    end
    
    gmm = GaussianMixture(
        n_components = k,
        covariance_type = "full",
        tol = parameters.tolerance,
        reg_covar = 0.0,
        max_iter = parameters.max_iterations,
        n_init = 1,
        weights_init = result.weights,
        means_init = μ,
        precisions_init = Σ,
        verbose = parameters.verbose ? 2 : 0,
        verbose_interval = 1,
    )

    result.assignments = fit_predict!(gmm, data) .+ 1
    result.objective = gmm.lower_bound_
    result.iterations = gmm.n_iter_

    for i in 1:k
        for j in 1:d
            result.centers[i][j] = gmm.means_[i, j]
        end
    end
    
    for i in 1:k
        matrix = zeros(d, d)
        for j in 1:d
            for l in 1:d
                matrix[j, l] = gmm.covariances_[i, j, l]
            end
        end
        result.covariances[i] = Hermitian(matrix)
    end

    result.weights = copy(gmm.weights_)

    if parameters.verbose
        println("\t$(result.iterations) - $(result.objective)")
    end

    return
end

function UnsupervisedClustering.train(parameters::GMMSK, data::AbstractMatrix{<:Real}, k::Integer)::GMMResult
    n, d = size(data)

    result = GMMResult(d, n, k)
    permutation = randperm(parameters.rng, n)
    for i in 1:k
        for j in 1:d
            result.centers[i][j] = data[permutation[i], j]
        end
    end

    train!(parameters, data, result)

    return result
end