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

const DEFAULT_TOLERANCE = 1e-3
const DEFAULT_MAX_ITERATIONS = 1000

export fit, 
    fit!, 
    Kmeans, 
    KmeansResult,
    GMM, 
    GMMResult,
    Kmedoids, 
    KmedoidsResult,
    RandomSwap, 
    MultiStart,
    GeneticAlgorithm

include("algorithm.jl")
include("result.jl")
include("kmeans.jl")
include("gmm.jl")
include("kmedoids.jl")
include("multistart.jl")
include("randomswap.jl")
include("generation.jl")
include("geneticalgorithm.jl")
include("print.jl")

end
