abstract type ClusteringAlgorithm end

Base.@kwdef mutable struct Kmeans <: ClusteringAlgorithm
    verbose::Bool = false
    rng::AbstractRNG = Random.GLOBAL_RNG
    metric::SemiMetric = SqEuclidean()
    tolerance::Float64 = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

Base.@kwdef mutable struct Kmedoids <: ClusteringAlgorithm
    verbose::Bool = false
    rng::AbstractRNG = Random.GLOBAL_RNG
    metric::SemiMetric = SqEuclidean()
    tolerance::Float64 = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

Base.@kwdef mutable struct GMM <: ClusteringAlgorithm
    verbose::Bool = false
    rng::AbstractRNG = Random.GLOBAL_RNG
    estimator::RegularizedCovarianceMatrices.CovarianceMatrixEstimator
    tolerance::Float64 = DEFAULT_TOLERANCE
    max_iterations::Integer = DEFAULT_MAX_ITERATIONS
end

function seed!(algorithm::ClusteringAlgorithm, seed::Integer)
    Random.seed!(algorithm.rng, seed)
    return nothing
end
