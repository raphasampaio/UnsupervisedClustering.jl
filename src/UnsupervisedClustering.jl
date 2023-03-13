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

export 
    fit,
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
    GeneticAlgorithm

include("algorithm.jl")
include("result.jl")
include("seed.jl")
include("kmeans.jl")
include("gmm.jl")
include("kmedoids.jl")
include("multistart.jl")
include("randomswap.jl")
include("generation.jl")
include("geneticalgorithm.jl")
include("print.jl")
include("concatenate.jl")
include("copy.jl")
include("convert.jl")

end
