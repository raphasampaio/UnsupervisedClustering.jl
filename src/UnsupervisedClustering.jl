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
    KmeansPlusPlus,
    Kmedoids,
    BalancedKmedoids,
    Ksegmentation,
    RandomSwap,
    MultiStart,
    GeneticAlgorithm,
    ClusteringChain

include("abstract.jl")

include("local_search/gmm.jl")
include("local_search/kmeans.jl")
include("local_search/kmeanspp.jl")
include("local_search/kmedoids.jl")
include("local_search/ksegmentation.jl")

include("result_type.jl")

include("metaheuristic/multi_start.jl")
include("metaheuristic/random_swap.jl")
include("metaheuristic/generation.jl")
include("metaheuristic/genetic_algorithm.jl")

include("ensemble/chain.jl")
include("ensemble/kmeans.jl")
include("ensemble/gmm.jl")

include("result/convert.jl")
include("result/copy.jl")
include("result/counts.jl")
include("result/equal.jl")
include("result/isbetter.jl")
include("result/random_swap.jl")
include("result/reset_objective.jl")
include("result/sort.jl")

include("assignments.jl")
include("concatenate.jl")
include("distances.jl")
include("linear_algebra.jl")
include("markov.jl")
include("print.jl")
include("seed.jl")
include("random.jl")
include("evaluation/silhouette_score.jl")

@setup_workload begin
    n, d, k = 100, 2, 2

    Random.seed!(1)
    data = rand(n, d)
    distances = pairwise(SqEuclidean(), data, dims = 1)

    @compile_workload begin
        kmeans = Kmeans(rng = MersenneTwister(1))
        fit(kmeans, data, k)

        balanced_kmeans = BalancedKmeans(rng = MersenneTwister(1))
        fit(balanced_kmeans, data, k)

        kmeans_pp = KmeansPlusPlus(rng = MersenneTwister(1))
        fit(kmeans_pp, data, k)

        ksegmentation = Ksegmentation()
        fit(ksegmentation, data, k)

        kmedoids = Kmedoids(rng = MersenneTwister(1))
        fit(kmedoids, distances, k)

        balanced_kmedoids = BalancedKmedoids(rng = MersenneTwister(1))
        fit(balanced_kmedoids, distances, k)

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
