module TestNEqualsTo0

using StableRNGs
using UnsupervisedClustering
using Test

@testset "n = 0" begin
    n, d, k = 0, 2, 3
    data = zeros(n, d)

    @testset "kmeans" begin
        algorithm = Kmeans(rng = StableRNG(1))
        result = fit(algorithm, data, k)
        @test length(result.assignments) == 0

        result = fit(algorithm, data, Vector{Int}())
        @test length(result.assignments) == 0
    end

    @testset "balanced kmeans" begin
        algorithm = BalancedKmeans(rng = StableRNG(1))
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

    @testset "kmedoids" begin
        algorithm = Kmedoids(rng = StableRNG(1))
        distances = pairwise(SqEuclidean(), data, dims = 1)
        result = fit(algorithm, distances, k)
        @test length(result.assignments) == 0

        result = fit(algorithm, data, Vector{Int}())
        @test length(result.assignments) == 0
    end

    @testset "gmm" begin
        algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
        result = fit(algorithm, data, k)
        @test length(result.assignments) == 0

        result = fit(algorithm, data, Vector{Int}())
        @test length(result.assignments) == 0
    end
end

end
