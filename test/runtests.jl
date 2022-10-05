using Clustering
using DelimitedFiles
using GaussianMixtures
using LinearAlgebra
using Random
using ScikitLearn
using Test
using UnsupervisedClustering

@sk_import mixture:GaussianMixture
@sk_import cluster:KMeans

include("kmeans.jl")
include("gmm.jl")

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
    for filename in ["3_10_-0.26_1", "3_10_-0.1_1", "3_10_0.01_1", "3_10_0.21_1"]
        data, k, expected = get_data(filename)

        test_kmeans(data, k, expected)
        test_gmm(data, k, expected)
    end
end

test_all()
