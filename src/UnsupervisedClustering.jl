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
    UnsupervisedClusteringResult,
    KmeansResult,
    KmedoidsResult,
    GMMResult,
    UnsupervisedClusteringAlgorithm,
    Kmeans,
    Kmedoids,
    GMM,
    RandomSwap,
    MultiStart,
    GeneticAlgorithm,
    ClusteringChain

abstract type UnsupervisedClusteringAlgorithm end
abstract type UnsupervisedClusteringResult end

include("localsearch/kmeans.jl")
include("localsearch/kmedoids.jl")
include("localsearch/gmm.jl")

include("metaheuristic/multistart.jl")
include("metaheuristic/randomswap.jl")
include("metaheuristic/generation.jl")
include("metaheuristic/geneticalgorithm.jl")

include("ensemble/chain.jl")
include("ensemble/kmeans.jl")
include("ensemble/gmm.jl")

include("results/convert.jl")
include("results/copy.jl")
include("results/counts.jl")
include("results/equal.jl")
include("results/isbetter.jl")
include("results/randomswap.jl")
include("results/resetobjective.jl")
include("results/sort.jl")

include("assignments.jl")
include("concatenate.jl")
include("distances.jl")
include("linearalgebra.jl")
include("print.jl")
include("seed.jl")

end
