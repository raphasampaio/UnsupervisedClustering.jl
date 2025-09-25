module TestKLargerThanN

using StableRNGs
using UnsupervisedClustering
using Test

@testset "k > n" begin
    n, d, k = 2, 2, 3
    data = zeros(n, d)

    @testset "kmeans" begin
        algorithm = Kmeans(rng = StableRNG(1))
        @test_throws AssertionError result = fit(algorithm, data, k)
    end

    @testset "balanced kmeans" begin
        algorithm = BalancedKmeans(rng = StableRNG(1))
        @test_throws AssertionError result = fit(algorithm, data, k)
    end

    @testset "ksegmentation" begin
        algorithm = Ksegmentation()
        @test_throws AssertionError result = fit(algorithm, data, k)
    end

    @testset "kmedoids" begin
        algorithm = Kmedoids(rng = StableRNG(1))
        distances = pairwise(SqEuclidean(), data, dims = 1)
        @test_throws AssertionError result = fit(algorithm, distances, k)
    end

    @testset "gmm" begin
        algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
        @test_throws AssertionError result = fit(algorithm, data, k)
    end
end
end
