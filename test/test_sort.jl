module TestSort

using Distances
using RegularizedCovarianceMatrices
using StableRNGs
using UnsupervisedClustering
using Test

@testset "Sort" begin
    @testset "KMeans" begin
        result = UnsupervisedClustering.KmeansResult(
            [1, 2, 3, 3, 2, 1],
            [3.0 1.0 2.0; 3.0 1.0 2.0],
            6.0,
            [3.0, 1.0, 2.0],
            1,
            1.0,
            true,
        )
        sort!(result)

        @test result.k == 3
        @test result.assignments == [3, 1, 2, 2, 1, 3]
        @test result.clusters ≈ [1.0 2.0 3.0; 1.0 2.0 3.0]
        @test result.objective ≈ 6.0
        @test result.objective_per_cluster ≈ [1.0, 2.0, 3.0]
        @test result.iterations == 1
        @test result.elapsed ≈ 1.0
        @test result.converged == true
    end

    @testset "KMedoids" begin
        result =
            UnsupervisedClustering.KmedoidsResult([1, 2, 3, 3, 2, 1], [3, 1, 2], 6.0, [3.0, 1.0, 2.0], 1, 1.0, true)
        sort!(result)

        @test result.k == 3
        @test result.assignments == [3, 1, 2, 2, 1, 3]
        @test result.clusters ≈ [1, 2, 3]
        @test result.objective ≈ 6.0
        @test result.objective_per_cluster ≈ [1.0, 2.0, 3.0]
        @test result.iterations == 1
        @test result.elapsed ≈ 1.0
        @test result.converged == true
    end
end

end
