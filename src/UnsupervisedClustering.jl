module UnsupervisedClustering

using Distances
using Hungarian
using LinearAlgebra
using LogExpFunctions
using Printf
using Random
using RegularizedCovarianceMatrices
using Statistics
using StatsBase

const DEFAULT_VERBOSE = false
const DEFAULT_TOLERANCE = 1e-3
const DEFAULT_MAX_ITERATIONS = 1000

export fit,
    fit!,
    counts,
    ClusteringResult,
    KmeansResult,
    KmedoidsResult,
    GMMResult,
    ClusteringAlgorithm,
    Kmeans,
    Kmedoids,
    GMM,
    RandomSwap,
    MultiStart,
    GeneticAlgorithm,
    ClusteringChain

abstract type ClusteringAlgorithm end
abstract type ClusteringResult end

include("result.jl")
include("algorithm.jl")

include("localsearches/kmeans.jl")
include("localsearches/gmm.jl")
include("localsearches/kmedoids.jl")

include("metaheuristics/multistart.jl")
include("metaheuristics/randomswap.jl")
include("metaheuristics/generation.jl")
include("metaheuristics/geneticalgorithm.jl")

include("chain.jl")

include("results/convert.jl")
include("results/copy.jl")
include("results/counts.jl")
include("results/isbetter.jl")
include("results/randomswap.jl")
include("results/resetobjective.jl")
include("results/sort.jl")

include("common.jl")
include("concatenate.jl")

include("distances.jl")
include("print.jl")
include("seed.jl")

end
