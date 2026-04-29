module TestMinSize

using Random
using StableRNGs
using Test
using UnsupervisedClustering

@testset "MinSizeKmeans" begin
    @testset "basic floor enforcement" begin
        n, d, k = 30, 2, 3
        min_size = 5
        data = rand(StableRNG(1), n, d)

        algorithm = MinSizeKmeans(min_size = min_size, rng = StableRNG(1))
        result = fit(algorithm, data, k)

        @test minimum(counts(result)) >= min_size
        @test sum(counts(result)) == n
        @test sort(unique(result.assignments)) == 1:k
    end

    @testset "n == min_size * k forces exact balance" begin
        n, d, k = 12, 2, 3
        min_size = 4
        data = rand(StableRNG(1), n, d)

        algorithm = MinSizeKmeans(min_size = min_size, rng = StableRNG(1))
        result = fit(algorithm, data, k)

        @test all(counts(result) .== min_size)
    end

    @testset "min_size = 1 yields nonempty clusters" begin
        n, d, k = 50, 2, 5
        data = rand(StableRNG(1), n, d)

        algorithm = MinSizeKmeans(min_size = 1, rng = StableRNG(1))
        result = fit(algorithm, data, k)

        @test minimum(counts(result)) >= 1
    end

    @testset "infeasible n < min_size * k throws" begin
        n, d, k = 5, 2, 3
        min_size = 3
        data = rand(StableRNG(1), n, d)

        algorithm = MinSizeKmeans(min_size = min_size, rng = StableRNG(1))
        @test_throws AssertionError fit(algorithm, data, k)
    end

    @testset "wrapped in RandomSwap" begin
        n, d, k = 40, 2, 4
        min_size = 3
        data = rand(StableRNG(1), n, d)

        local_search = MinSizeKmeans(min_size = min_size, rng = StableRNG(1))
        algorithm = RandomSwap(
            local_search = local_search,
            max_iterations = 10,
            max_iterations_without_improvement = 5,
        )
        seed!(algorithm, 1)
        result = fit(algorithm, data, k)

        @test minimum(counts(result)) >= min_size
    end

    @testset "wrapped in MultiStart" begin
        n, d, k = 40, 2, 4
        min_size = 3
        data = rand(StableRNG(1), n, d)

        local_search = MinSizeKmeans(min_size = min_size, rng = StableRNG(1))
        algorithm = MultiStart(local_search = local_search, max_iterations = 5)
        seed!(algorithm, 1)
        result = fit(algorithm, data, k)

        @test minimum(counts(result)) >= min_size
    end
end

end
