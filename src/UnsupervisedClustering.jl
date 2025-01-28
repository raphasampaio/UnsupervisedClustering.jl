module UnsupervisedClustering

using Distances
using Hungarian
using LinearAlgebra
using PrecompileTools
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
    stochastic_matrix,
    GMM,
    Kmeans,
    BalancedKmeans,
    Kmedoids,
    Ksegmentation,
    RandomSwap,
    MultiStart,
    GeneticAlgorithm,
    ClusteringChain

include("abstract.jl")

include("localsearch/gmm.jl")
include("localsearch/kmeans.jl")
include("localsearch/balanced_kmeans.jl")
include("localsearch/kmedoids.jl")
include("localsearch/ksegmentation.jl")

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
include("markov.jl")
include("print.jl")
include("seed.jl")
include("random.jl")

@setup_workload begin
    n, d, k = 100, 2, 2

    Random.seed!(1)
    data = rand(n, d)
    distances = pairwise(SqEuclidean(), data, dims = 1)

    @compile_workload begin
        kmeans = Kmeans(rng = MersenneTwister(1))
        fit(kmeans, data, k)

        ksegmentation = Ksegmentation()
        fit(ksegmentation, data, k)

        kmedoids = Kmedoids(rng = MersenneTwister(1))
        fit(kmedoids, distances, k)

        gmm = GMM(rng = MersenneTwister(1), estimator = EmpiricalCovarianceMatrix(n, d))
        fit(gmm, data, k)

        multi_start = MultiStart(local_search = kmeans)
        fit(multi_start, data, k)

        random_swap = RandomSwap(local_search = kmeans)
        fit(random_swap, data, k)

        genetic_algorithm = GeneticAlgorithm(local_search = kmeans)
        fit(genetic_algorithm, data, k)
    end
end

end
