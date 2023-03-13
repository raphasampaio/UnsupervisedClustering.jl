abstract type ClusteringAlgorithm end

Base.@kwdef mutable struct Kmeans <: ClusteringAlgorithm
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    metric::SemiMetric = SqEuclidean()
    tolerance::Float64 = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

Base.@kwdef mutable struct Kmedoids <: ClusteringAlgorithm
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    metric::SemiMetric = SqEuclidean()
    tolerance::Float64 = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

Base.@kwdef mutable struct GMM <: ClusteringAlgorithm
    verbose::Bool = DEFAULT_VERBOSE
    rng::AbstractRNG = Random.GLOBAL_RNG
    estimator::RegularizedCovarianceMatrices.CovarianceMatrixEstimator
    tolerance::Float64 = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
    decompose_if_fails::Bool = true
end

Base.@kwdef mutable struct MultiStart <: ClusteringAlgorithm
    local_search::ClusteringAlgorithm
    verbose::Bool = DEFAULT_VERBOSE
    max_iterations::Integer = 200
end

Base.@kwdef mutable struct RandomSwap <: ClusteringAlgorithm
    local_search::ClusteringAlgorithm
    verbose::Bool = DEFAULT_VERBOSE
    max_iterations::Integer = 200
    max_iterations_without_improvement::Integer = 150
end

Base.@kwdef mutable struct GeneticAlgorithm <: ClusteringAlgorithm
    local_search::ClusteringAlgorithm
    verbose::Bool = DEFAULT_VERBOSE
    max_iterations::Integer = 200
    max_iterations_without_improvement::Integer = 150
    π_max::Integer = 50
    π_min::Integer = 40
end
