module UnsupervisedClustering

using MKL

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
    KmeansResult,
    KmedoidsResult,
    GMMResult,
    Kmeans,
    Kmedoids,
    GMM,
    RandomSwap,
    MultiStart,
    GeneticAlgorithm

include("abstract.jl")
include("algorithm.jl")
include("seeding.jl")
include("kmeans.jl")
include("gmm.jl")
include("kmedoids.jl")
include("multistart.jl")
include("randomswap.jl")
include("generation.jl")
include("geneticalgorithm.jl")
include("result.jl")
include("print.jl")

end
