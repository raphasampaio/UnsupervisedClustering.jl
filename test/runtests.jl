using Clustering
using DelimitedFiles
using Distances
using LinearAlgebra
using Printf
using Random
using RegularizedCovarianceMatrices
using ScikitLearn
using StableRNGs
using Test
using TimerOutputs
using UnsupervisedClustering

@sk_import mixture:GaussianMixture
@sk_import cluster:KMeans

include("kmeans.jl")
include("gmm.jl")
include("gmmsk.jl")

function get_data(filename::String)
    open(joinpath("data", "$filename.csv")) do file
        table = readdlm(file, ',')

        n = size(table, 1)
        d = size(table, 2) - 1

        clusters = Set{Int}()
        expected = zeros(Int, n)

        for i in 1:n
            expected[i] = Int(table[i, 1])
            push!(clusters, expected[i])
        end
        k = length(clusters)

        return table[:, 2:size(table, 2)], k, expected
    end
end

function test_all()
    reset_timer!()

    for k in [3]
        for c in [-0.26, -0.1, 0.01, 0.21]
            for d in [2, 5, 10, 20, 30, 40]
                filename = "$(k)_$(d)_$(c)_1"
                @info filename

                data, k, expected = get_data(filename)
                n, d = size(data)
        
                # test_kmeans(data', k, expected)
                test_gmm(data, k, expected)
            end
        end
    end

        # gmm = GMM(estimator = RegularizedCovarianceMatrices.ShrunkCovariance(n, d), verbose = false)
        # Random.seed!(1)
        # result = UnsupervisedClustering.train(gmm, data, k)
        # @show result.objective
        # @show ari = Clustering.randindex(result.assignments, expected)[1]


        # gmm = GMM(estimator = RegularizedCovarianceMatrices.ShrunkCovariance(n, d))
        # for i in 1:10000
            # Random.seed!(i)
            # result = UnsupervisedClustering.train(gmm, data, k)
            # ari = Clustering.randindex(result.assignments, expected)[1]
            # objective = result.objective
            # println("$ari, $objective")

            # Random.seed!(i)
            # permutation = randperm(n)
            # # GMM - SKLEARN
            # μ_sk = zeros(k, d)
            # for i in 1:k
            #     for j in 1:d
            #         μ_sk[i, j] = data[permutation[i], j]
            #     end
            # end
            # Σ_sk = zeros(k, d, d)
            # for i in 1:k
            #     for j in 1:d
            #         Σ_sk[i, j, j] = 1.0
            #     end
            # end
            # π_sk = ones(k) ./ k

            # try
            # gmm_sk = GaussianMixture(
            #     n_components = k,
            #     covariance_type = "full",
            #     tol = 1e-3,
            #     reg_covar = 0.0,
            #     max_iter = 1000,
            #     n_init = 1,
            #     weights_init = π_sk,
            #     means_init = μ_sk,
            #     precisions_init = Σ_sk,
            #     verbose = 0,
            #     verbose_interval = 1
            # )
            # assignments = fit_predict!(gmm_sk, data)
            # ari = Clustering.randindex(assignments, expected)[1]
            # objective = score(gmm_sk, data)
    
            # println("$ari, $objective")
            # catch

            # end
        # end

        # gmm = GMM(estimator = EmpiricalCovarianceMatrix(n, d), verbose = true)
        # # gmm = GMMSK(verbose = true)
        # gmm_rs = GeneticAlgorithm(
        #     verbose = true,
        #     local_search = gmm, 
        #     max_iterations = 10, 
        #     max_iterations_without_improvement = 10,
        #     π_max = 8,
        #     π_min = 4
        # )

        # Random.seed!(1)
        # result = UnsupervisedClustering.train(gmm_rs, data, k)
        # @show ari = Clustering.randindex(result.assignments, expected)[1]
        
    #     for max_iterations in [200]# 100, 250, 500, 1000, 2500
    #         for p1 in [0.75] # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    #             max_iterations_without_improvement = round(Int, max_iterations * p1)
    #             for π_max in [10, 20, 50, 100] # 50, 100, 200
    #                 for p2 in [0.5, 0.75] # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    #                     π_min = round(Int, π_max * p2)
    #                     if π_min > 4

    #                         # for s in [123, 124, 125]

    #                         gmm_shrunk_hg = GeneticAlgorithm(
    #                             local_search = GMM(
    #                                 # verbose = true,
    #                                 rng = StableRNG(123),
    #                                 estimator = ShrunkCovarianceMatrix(n, d)
    #                             ), 
    #                             max_iterations = max_iterations, 
    #                             max_iterations_without_improvement = max_iterations_without_improvement,
    #                             π_max = π_max,
    #                             π_min = π_min
    #                         )
    #                         # gmm_shrunk_rs = RandomSwap(
    #                         #     local_search = GMM(
    #                         #         # verbose = true,
    #                         #         rng = StableRNG(123),
    #                         #         estimator = ShrunkCovarianceMatrix(n, d)
    #                         #     ), 
    #                         #     max_iterations = max_iterations, 
    #                         #     max_iterations_without_improvement = max_iterations_without_improvement
    #                         # )
    #                         Random.seed!(1)
    #                         result = UnsupervisedClustering.train(gmm_shrunk_hg, data, k)
    #                         ari1 = Clustering.randindex(result.assignments, expected)[1]

    #                         # Random.seed!(1)
    #                         # result = UnsupervisedClustering.train(gmm_shrunk_rs, data, k)
    #                         # ari2 = Clustering.randindex(result.assignments, expected)[1]

    #                         # @printf("%s,%.2f,%.2f\n", filename, ari1, ari2)
    #                         # end
                            
    #                         # @printf("%s,%d,%.2f,%d,%.2f,%.2f,%.2f\n", filename, max_iterations, p1, π_max, p2, ari1, ari2)

    #                         @printf("%s,%d,%.2f,%d,%.2f,%.2f\n", filename, max_iterations, p1, π_max, p2, ari1)
    #                     end
    #                 end
    #             end
    #         end
    #     end

    #     end
    #     end
    #     end
    #     end

    #     # gmm_hg = GeneticAlgorithm(local_search = gmm, verbose = true)
    #     # Random.seed!(1)
    #     # result = UnsupervisedClustering.train(gmm_hg, data, k)
    #     # @show result.objective
    #     # @show ari = Clustering.randindex(result.assignments, expected)[1]

    #     # @info "Testing $filename"
    #     # test_kmeans(data', k, expected)

    #     # for seed in 3:3
    #     #     test_gmm(data, k, expected, seed)
    #     # end
    # # end

    print_timer(sortby = :firstexec)

    return
end

test_all()
