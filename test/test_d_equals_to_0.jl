module TestDEqualsTo0

using Distances
using RegularizedCovarianceMatrices
using StableRNGs
using UnsupervisedClustering
using Test

@testset "d = 0" begin
    n, d, k = 3, 0, 3
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

    @testset "gmm" begin
        algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
        @test_throws AssertionError result = fit(algorithm, data, k)
    end
end

end
