module TestNEqualsToK

using Distances
using RegularizedCovarianceMatrices
using StableRNGs
using UnsupervisedClustering
using Test

@testset "n = k" begin
    n, d, k = 3, 2, 3
    data = rand(StableRNG(1), n, d)

    @testset "kmeans" begin
        algorithm = Kmeans(rng = StableRNG(1))
        result = fit(algorithm, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end

    @testset "balanced kmeans" begin
        algorithm = BalancedKmeans(rng = StableRNG(1))
        result = fit(algorithm, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end

    @testset "ksegmentation" begin
        algorithm = Ksegmentation()
        result = fit(algorithm, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end

    @testset "kmedoids" begin
        algorithm = Kmedoids(rng = StableRNG(1))
        distances = pairwise(SqEuclidean(), data, dims = 1)
        result = fit(algorithm, distances, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end

    @testset "gmm" begin
        algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
        result = fit(algorithm, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end
end

@testset "n = k (zeros)" begin
    n, d, k = 3, 2, 3
    data = zeros(n, d)

    @testset "kmeans" begin
        algorithm = Kmeans(rng = StableRNG(1))
        result = fit(algorithm, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end

    @testset "balanced kmeans" begin
        algorithm = BalancedKmeans(rng = StableRNG(1))
        result = fit(algorithm, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end

    @testset "kmedoids" begin
        algorithm = Kmedoids(rng = StableRNG(1))
        distances = pairwise(SqEuclidean(), data, dims = 1)
        result = fit(algorithm, distances, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end

    @testset "gmm" begin
        algorithm = GMM(rng = StableRNG(1), estimator = EmpiricalCovarianceMatrix(n, d))
        result = fit(algorithm, data, k)
        @test sort(result.assignments) == [i for i in 1:k]
    end
end

end
