module UnsupervisedClustering

using Distances
using Hungarian
using LinearAlgebra
using Printf
using Random
using RegularizedCovarianceMatrices
using Statistics
using StatsBase

const DEFAULT_VERBOSE = false
const DEFAULT_TOLERANCE = 1e-3
const DEFAULT_MAX_ITERATIONS = 1000

export concatenate,
    counts,
    fit,
    fit!,
    seed!,
    sort!,
    Kmeans,
    Kmedoids,
    GMM,
    RandomSwap,
    MultiStart,
    GeneticAlgorithm,
    ClusteringChain

abstract type Algorithm end
abstract type Result end

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

include("result/convert.jl")
include("result/copy.jl")
include("result/counts.jl")
include("result/equal.jl")
include("result/isbetter.jl")
include("result/randomswap.jl")
include("result/resetobjective.jl")
include("result/sort.jl")

include("assignments.jl")
include("concatenate.jl")
include("distances.jl")
include("linearalgebra.jl")
include("print.jl")
include("seed.jl")

end
