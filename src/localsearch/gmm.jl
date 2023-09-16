@doc """
    GMM(
        estimator::CovarianceMatrixEstimator
        verbose::Bool = DEFAULT_VERBOSE
        rng::AbstractRNG = Random.GLOBAL_RNG
        tolerance::Real = DEFAULT_TOLERANCE
        max_iterations::Integer = DEFAULT_MAX_ITERATIONS
        decompose_if_fails::Bool = true
    )

The GMM is a clustering algorithm that models the underlying data distribution as a mixture of Gaussian distributions.

# Fields
- `estimator`: represents the method or algorithm used to estimate the covariance matrices in the GMM. 
- `verbose`: controls whether the algorithm should display additional information during execution.
- `rng`: represents the random number generator to be used by the algorithm.
- `tolerance`: represents the convergence criterion for the algorithm. It determines the maximum change allowed in the model's log-likelihood between consecutive iterations before considering convergence.
- `max_iterations`: represents the maximum number of iterations the algorithm will perform before stopping, even if convergence has not been reached.
- `decompose_if_fails`: determines whether the algorithm should attempt to decompose the covariance matrix of a component and fix its eigenvalues if the decomposition fails due to numerical issues.       

# References
* Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin.
  Maximum likelihood from incomplete data via the EM algorithm.
  Journal of the royal statistical society: series B (methodological) 39.1 (1977): 1-22.
"""
Base.@kwdef mutable struct GMM <: Algorithm
    estimator::CovarianceMatrixEstimator
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    tolerance::Real = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
    decompose_if_fails::Bool = true
end

@doc """
    GMMResult(
        assignments::AbstractVector{<:Integer}
        weights::AbstractVector{<:Real}
        clusters::AbstractVector{<:AbstractVector{<:Real}}
        covariances::AbstractVector{<:Symmetric{<:Real}}
        objective::Real
        iterations::Integer
        elapsed::Real
        converged::Bool
        k::Integer
    )

GMMResult struct represents the result of the GMM clustering algorithm.

# Fields
- `assignments`: an integer vector that stores the cluster assignment for each data point.
- `weights`: a vector of floating-point numbers representing the weights associated with each cluster. The weight indicates the probability of a data point belonging to its respective cluster.
- `clusters`: a vector of floating-point vectors representing the cluster's centroid.
- `covariances`: a vector of symmetric matrices, where each matrix represents the covariance matrix of a cluster in the GMM model. The covariance matrix describes the shape and orientation of the data distribution within each cluster.
- `objective`: a floating-point number representing the objective function after running the algorithm. The objective function measures the quality of the clustering solution.
- `iterations`: an integer value indicating the number of iterations performed until the algorithm has converged or reached the maximum number of iterations
- `elapsed`: a floating-point number representing the time in seconds for the algorithm to complete.
- `converged`: indicates whether the algorithm has converged to a solution.
- `k`: the number of clusters.
"""
mutable struct GMMResult{I <: Integer, R <: Real} <: Result
    assignments::Vector{I}
    weights::Vector{R}
    clusters::Vector{Vector{R}}
    covariances::Vector{<:Symmetric{R}}
    objective::R
    iterations::I
    elapsed::R
    converged::Bool
    k::I

    function GMMResult(
        assignments::AbstractVector{I},
        weights::AbstractVector{R},
        clusters::AbstractVector{<:AbstractVector{R}},
        covariances::AbstractVector{<:Symmetric{R}},
        objective::R = -Inf,
        iterations::I = 0,
        elapsed::R = 0.0,
        converged::Bool = false,
    ) where {I <: Integer, R <: Real}
        return new{I, R}(
            assignments,
            weights,
            clusters,
            covariances,
            objective,
            iterations,
            elapsed,
            converged,
            length(clusters),
        )
    end
end

function GMMResult(d::Integer, n::Integer, k::Integer)
    assignments = zeros(Int, n)
    weights = ones(k) ./ k
    clusters = [zeros(d) for _ in 1:k]
    covariances = [identity_matrix(d) for _ in 1:k]
    return GMMResult(assignments, weights, clusters, covariances)
end

function GMMResult(n::Integer, clusters::AbstractVector{<:AbstractVector{<:Real}})
    k = length(clusters)
    @assert k > 0
    d = length(clusters[1])

    result = GMMResult(d, n, k)
    result.clusters = deepcopy(result.clusters)
    return result
end

mutable struct GMMCache
    log_det_cholesky::Vector{Float64}
    log_probabilities::Matrix{Float64}
    log_probabilities_norm::Vector{Float64}
    weighted_log_probabilities::Vector{Vector{Float64}}
    responsibilities::Vector{Vector{Float64}}
    log_responsibilities::Vector{Vector{Float64}}
    precisions_cholesky::Vector{Matrix{Float64}}

    function GMMCache(d::Integer, n::Integer, k::Integer)
        return new(
            zeros(k),
            zeros(n, k),
            zeros(n),
            [zeros(n) for _ in 1:k],
            [zeros(n) for _ in 1:k],
            [zeros(n) for _ in 1:k],
            [zeros(d, d) for _ in 1:k],
        )
    end
end

function estimate_gaussian_parameters!(gmm::GMM, data::AbstractMatrix{<:Real}, result::GMMResult, cache::GMMCache)::Nothing
    n, d = size(data)
    k = result.k

    for i in 1:k
        responsibilities_sum = 0.0
        for j in 1:n
            responsibilities_sum += cache.responsibilities[i][j]
        end

        if responsibilities_sum < 1e-32
            for j in 1:n
                cache.responsibilities[i][j] = 1.0 / n
            end
        end

        result.weights[i] = 10 * eps(Float64) 
        for j in 1:n
            result.weights[i] += cache.responsibilities[i][j]
        end
    end

    weights_sum = sum(result.weights)
    for i in 1:k
        result.weights[i] /= weights_sum
    end

    for i in 1:k
        covariances_i, result.clusters[i] = RegularizedCovarianceMatrices.fit(gmm.estimator, data, cache.responsibilities[i])
        result.covariances[i] = Symmetric(covariances_i)
    end

    return nothing
end

function compute_precision_cholesky!(gmm::GMM, result::GMMResult, cache::GMMCache)
    k = result.k

    for i in 1:k
        d = size(result.covariances[i], 1)

        try
            covariances_cholesky = cholesky(result.covariances[i])
            cache.precisions_cholesky[i] = covariances_cholesky.U \ identity_matrix(d)
        catch e
            if gmm.decompose_if_fails
                eig = eigen(result.covariances[i], sortby = nothing)
                result.covariances[i] = Symmetric(eig.vectors * Matrix(Diagonal(max.(eig.values, 1e-6))) * eig.vectors')
                covariances_cholesky = cholesky(result.covariances[i])
                cache.precisions_cholesky[i] = covariances_cholesky.U \ identity_matrix(d)
            else
                error("GMM Failed: $e")
            end
        end
    end

    return nothing
end

function estimate_weighted_log_probabilities!(data::AbstractMatrix{<:Real}, result::GMMResult, cache::GMMCache)::Nothing
    n, d = size(data)
    k = result.k

    for i in 1:k
        cache.log_det_cholesky[i] = 0.0
        for j in 1:d
            cache.log_det_cholesky[i] += log(cache.precisions_cholesky[i][j, j])
        end
    end

    for i in 1:k
        y = data * cache.precisions_cholesky[i] .- (result.clusters[i]' * cache.precisions_cholesky[i])

        for l in 1:n
            cache.log_probabilities[l, i] = 0.0
            for j in 1:d
                cache.log_probabilities[l, i] += y[l, j] ^ 2
            end
        end
    end

    for i in 1:k
        for j in 1:n
            cache.weighted_log_probabilities[i][j] = log(result.weights[i]) - 0.5 * (d * log(2 * pi) + cache.log_probabilities[j, i]) + cache.log_det_cholesky[i]
        end
    end

    return nothing
end

function log_sum_exp(probabilities::AbstractVector{<:AbstractVector{<:Real}}, j::Integer)
    max = -Inf
    sum = 0.0

    k = length(probabilities)
    for i in 1:k
        p = probabilities[i][j]

        if isnan(p) || isnan(max)
            max, sum = NaN, sum + exp(NaN)
        else
            if p > max
                max, sum = p, (sum + one(sum)) * exp(max - p)
            elseif p < max
                max, sum = max, sum + exp(p - max)
            else
                max, sum = max, sum + exp(zero(p - max))
            end
        end
    end

    return max + log1p(sum)
end

function expectation_step!(data::AbstractMatrix{<:Real}, result::GMMResult, cache::GMMCache)
    n, d = size(data)
    k = result.k

    estimate_weighted_log_probabilities!(data, result, cache)

    for j in 1:n
        cache.log_probabilities_norm[j] = log_sum_exp(cache.weighted_log_probabilities, j)
    end

    for i in 1:k
        for j in 1:n
            cache.log_responsibilities[i][j] = cache.weighted_log_probabilities[i][j] - cache.log_probabilities_norm[j]
        end
    end

    return mean(cache.log_probabilities_norm)
end

function maximization_step!(gmm::GMM, data::AbstractMatrix{<:Real}, result::GMMResult, cache::GMMCache)
    n, d = size(data)
    k = result.k

    for i in 1:k
        for j in 1:n
            cache.responsibilities[i][j] = exp(cache.log_responsibilities[i][j])
        end
    end
    
    estimate_gaussian_parameters!(gmm, data, result, cache)
    compute_precision_cholesky!(gmm, result, cache)

    return nothing
end

@doc """
    fit!(
        gmm::GMM,
        data::AbstractMatrix{<:Real},
        result::GMMResult
    )

The `fit!` function performs the GMM clustering algorithm on the given result as the initial point and updates the provided object with the clustering result.

# Parameters:
- `gmm`: an instance representing the clustering settings and parameters.
- `data`: a floating-point matrix, where each row represents a data point, and each column represents a feature.
- `result`: a result object that will be updated with the clustering result.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)

gmm = GMM(estimator = EmpiricalCovarianceMatrix(n, d))
result = GMMResult(n, [[1.0, 1.0], [2.0, 2.0]])
fit!(gmm, data, result)
```
"""
function fit!(gmm::GMM, data::AbstractMatrix{<:Real}, result::GMMResult)
    t = time()

    n, d = size(data)
    k = result.k

    previous_objective = Inf
    result.objective = -Inf

    result.iterations = gmm.max_iterations
    result.converged = false

    cache = GMMCache(d, n, k)

    compute_precision_cholesky!(gmm, result, cache)

    for iteration in 1:gmm.max_iterations
        previous_objective = result.objective

        t1 = @elapsed result.objective = expectation_step!(data, result, cache)

        t2 = @elapsed maximization_step!(gmm, data, result, cache)

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

    estimate_weighted_log_probabilities!(data, result, cache)
    for j in 1:n
        max = -Inf
        for i in 1:k
            probability = cache.weighted_log_probabilities[i][j]
            if probability > max
                max = probability
                result.assignments[j] = i
            end
        end
    end

    result.elapsed = time() - t

    return nothing
end

@doc """
    fit(
        gmm::GMM,
        data::AbstractMatrix{<:Real},
        initial_clusters::AbstractVector{<:Integer}
    )

The `fit` function performs the GMM clustering algorithm on the given data points as the initial point and returns a result object representing the clustering result.

# Parameters:
- `kmeans`: an instance representing the clustering settings and parameters.
- `data`: a floating-point matrix, where each row represents a data point, and each column represents a feature.
- `initial_clusters`: an integer vector where each element is the initial data point for each cluster.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)

gmm = GMM(estimator = EmpiricalCovarianceMatrix(n, d))
result = fit(gmm, data, [4, 12])
```
"""
function fit(gmm::GMM, data::AbstractMatrix{<:Real}, initial_clusters::AbstractVector{<:Integer})::GMMResult
    n, d = size(data)
    k = length(initial_clusters)

    result = GMMResult(d, n, k)
    if n == 0
        return result
    end

    @assert d > 0
    @assert k > 0
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

@doc """
    fit(
        gmm::GMM,
        data::AbstractMatrix{<:Real},
        k::Integer
    ) 

The `fit` function performs the GMM clustering algorithm and returns a result object representing the clustering result.

# Parameters:
- `gmm`: an instance representing the clustering settings and parameters.
- `data`: a floating-point matrix, where each row represents a data point, and each column represents a feature.
- `k`: an integer representing the number of clusters.

# Example

```julia
n = 100
d = 2
k = 2

data = rand(n, d)

gmm = GMM(estimator = EmpiricalCovarianceMatrix(n, d))
result = fit(gmm, data, k)
```
"""
function fit(gmm::GMM, data::AbstractMatrix{<:Real}, k::Integer)::GMMResult
    n, d = size(data)

    if n == 0
        return GMMResult(d, n, k)
    end

    @assert k > 0
    @assert n >= k

    initial_clusters = StatsBase.sample(gmm.rng, 1:n, k, replace = false)
    return fit(gmm, data, initial_clusters)
end
