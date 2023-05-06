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
    EnsembleClustering

include("result.jl")
include("algorithm.jl")

include("localsearch/kmeans.jl")
include("localsearch/gmm.jl")
include("localsearch/kmedoids.jl")

include("ensemble.jl")
include("multistart.jl")
include("randomswap.jl")
include("generation.jl")
include("geneticalgorithm.jl")

include("common.jl")
include("concatenate.jl")
include("convert.jl")
include("copy.jl")
include("distances.jl")
include("print.jl")
include("seed.jl")
include("sort.jl")

end
