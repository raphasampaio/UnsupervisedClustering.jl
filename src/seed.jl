function seed!(algorithm::UnsupervisedClusteringAlgorithm, seed::Integer)
    Random.seed!(algorithm.rng, seed)
    return nothing
end

function seed!(algorithm::MultiStart, seed::Integer)
    Random.seed!(algorithm.local_search.rng, seed)
    return nothing
end

function seed!(algorithm::RandomSwap, seed::Integer)
    Random.seed!(algorithm.local_search.rng, seed)
    return nothing
end

function seed!(algorithm::GeneticAlgorithm, seed::Integer)
    Random.seed!(algorithm.local_search.rng, seed)
    return nothing
end

function seed!(algorithm::ClusteringChain, seed::Integer)
    for algorithm in algorithm.algorithms
        seed!(algorithm, seed)
    end
    return nothing
end
