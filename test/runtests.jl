using Clustering
using DelimitedFiles
using Distances
using LinearAlgebra
using Printf
using Random
using RegularizedCovarianceMatrices
using ScikitLearn
using Test
using TimerOutputs
using UnsupervisedClustering

@sk_import mixture:GaussianMixture
@sk_import cluster:KMeans

include("kmeans.jl")
include("gmm.jl")
include("gmmsk.jl")

ENV["OMP_NUM_THREADS"] = 1

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
        
                test_kmeans(data, k, expected, 123)
                test_gmm(data, k, expected, 123)
            end
        end
    end

    print_timer(sortby = :firstexec)

    return
end

test_all()
