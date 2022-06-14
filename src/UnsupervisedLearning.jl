module UnsupervisedLearning

using Clustering
using Distances
using Hungarian
using LinearAlgebra
using Random
using RegularizedCovariances
using Statistics
using StatsBase
using StatsFuns

export kmeans,
    kmeans_ms,
    kmeans_rs,
    kmeans_hg,
    gmm,
    gmm_shrunk,
    gmm_oas,
    gmm_ledoitwolf,
    gmm_ms,
    gmm_ms_shrunk,
    gmm_ms_oas,
    gmm_ms_ledoitwolf,
    gmm_rs,
    gmm_rs_shrunk,
    gmm_rs_oas,
    gmm_rs_ledoitwolf,
    gmm_hg,
    gmm_hg_shrunk,
    gmm_hg_oas,
    gmm_hg_ledoitwolf

const DEFAULT_LOCAL_ITERATIONS = 10000
const DEFAULT_GLOBAL_ITERATIONS = 100

include("common.jl")
include("results.jl")
include("kmeans.jl")
include("gmm.jl")
include("multistart.jl")
include("randomswap.jl")
include("generation.jl")
include("genetic_algorithm.jl")

end
