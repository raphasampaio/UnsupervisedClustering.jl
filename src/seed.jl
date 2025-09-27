function seed!(algorithm::AbstractAlgorithm, seed::Integer)
    Random.seed!(algorithm.rng, seed)
    return nothing
end

function seed!(algorithm::Ksegmentation, seed::Integer)
    return nothing
end

function seed!(algorithm::MultiStart{LS}, seed::Integer) where {LS <: AbstractAlgorithm}
    Random.seed!(algorithm.local_search.rng, seed)
    return nothing
end

function seed!(algorithm::RandomSwap{LS}, seed::Integer) where {LS <: AbstractAlgorithm}
    Random.seed!(algorithm.local_search.rng, seed)
    return nothing
end

function seed!(algorithm::GeneticAlgorithm{LS}, seed::Integer) where {LS <: AbstractAlgorithm}
    Random.seed!(algorithm.local_search.rng, seed)
    return nothing
end

function seed!(algorithm::ClusteringChain{T}, seed::Integer) where {T<:AbstractAlgorithm}
    for algorithm in algorithm.algorithms
        seed!(algorithm, seed)
    end
    return nothing
end
