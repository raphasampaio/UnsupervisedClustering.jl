module TestNEqualsTo0

using Distances
using RegularizedCovarianceMatrices
using StableRNGs
using UnsupervisedClustering
using Test

@testset "n = 0" begin
    n, d, k = 0, 2, 3
    data = zeros(n, d)

    @testset "KMeans" begin
        algorithm = Kmeans(rng = StableRNG(1))
        result = fit(algorithm, data, k)
        @test length(result.assignments) == 0

        result = fit(algorithm, data, Vector{Int}())
        @test length(result.assignments) == 0
    end

    @testset "Balanced KMeans" begin
        algorithm = BalancedKmeans(rng = StableRNG(1))
        result = fit(algorithm, data, k)
        @test length(result.assignments) == 0

        result = fit(algorithm, data, Vector{Int}())
        @test length(result.assignments) == 0
    end

    @testset "KMeans++" begin
        algorithm = KmeansPlusPlus(rng = StableRNG(1))
        result = fit(algorithm, data, k)
        @test length(result.assignments) == 0

        result = fit(algorithm, data, Vector{Int}())
        @test length(result.assignments) == 0
    end    

    @testset "ksegmentation" begin
        algorithm = Ksegmentation()
        result = fit(algorithm, data, k)
        @test length(result.assignments) == 0
    end

    @testset "KMedoids" begin
        algorithm = Kmedoids(rng = StableRNG(1))
        distances = pairwise(SqEuclidean(), data, dims = 1)
        result = fit(algorithm, distances, k)
        @test length(result.assignments) == 0

        result = fit(algorithm, data, Vector{Int}())
        @test length(result.assignments) == 0
    end

    @testset "GMM" begin
        algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
        result = fit(algorithm, data, k)
        @test length(result.assignments) == 0

        result = fit(algorithm, data, Vector{Int}())
        @test length(result.assignments) == 0
    end
end

end
