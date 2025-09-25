module TestConcatenate

using Distances
using RegularizedCovarianceMatrices
using StableRNGs
using UnsupervisedClustering
using Test

@testset "concatenate" begin
    @test_throws MethodError concatenate()

    @testset "kmeans result" begin
        result = concatenate(
            UnsupervisedClustering.KmeansResult([1, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0], 1.0, [0.5, 0.5], 1, 1.0, true),
            UnsupervisedClustering.KmeansResult([1, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0], 2.0, [1.0, 1.0], 2, 2.0, true),
            UnsupervisedClustering.KmeansResult(
                [1, 2, 2],
                [1.0 2.0; 1.0 2.0; 1.0 2.0],
                3.0,
                [1.5, 1.5],
                3,
                3.0,
                true,
            ),
        )

        @test result.k == 6
        @test result.assignments == [1, 2, 3, 4, 5, 6, 6]
        @test result.clusters ≈ [1.0 2.0 1.0 2.0 1.0 2.0; 1.0 2.0 1.0 2.0 1.0 2.0; 1.0 2.0 1.0 2.0 1.0 2.0]
        @test result.objective ≈ 6.0
        @test result.objective_per_cluster ≈ [0.5, 0.5, 1.0, 1.0, 1.5, 1.5]
        @test result.iterations == 6
        @test result.elapsed ≈ 6.0
        @test result.converged == true
    end

    @testset "ksegmentation result" begin
        result = concatenate(
            UnsupervisedClustering.KsegmentationResult(
                [1, 2],
                [1.0 2.0; 1.0 2.0; 1.0 2.0],
                1.0,
                [0.5, 0.5],
                1,
                1.0,
                true,
            ),
            UnsupervisedClustering.KsegmentationResult(
                [1, 2],
                [1.0 2.0; 1.0 2.0; 1.0 2.0],
                2.0,
                [1.0, 1.0],
                2,
                2.0,
                true,
            ),
            UnsupervisedClustering.KsegmentationResult(
                [1, 2, 2],
                [1.0 2.0; 1.0 2.0; 1.0 2.0],
                3.0,
                [1.5, 1.5],
                3,
                3.0,
                true,
            ),
        )

        @test result.k == 6
        @test result.assignments == [1, 2, 3, 4, 5, 6, 6]
        @test result.clusters ≈ [1.0 2.0 1.0 2.0 1.0 2.0; 1.0 2.0 1.0 2.0 1.0 2.0; 1.0 2.0 1.0 2.0 1.0 2.0]
        @test result.objective ≈ 6.0
        @test result.objective_per_cluster ≈ [0.5, 0.5, 1.0, 1.0, 1.5, 1.5]
        @test result.iterations == 6
        @test result.elapsed ≈ 6.0
        @test result.converged == true
    end

    @testset "kmedoids result" begin
        result = concatenate(
            UnsupervisedClustering.KmedoidsResult([1, 2], [1, 2], 1.0, [0.5, 0.5], 1, 1.0, true),
            UnsupervisedClustering.KmedoidsResult([1, 2], [1, 2], 2.0, [1.0, 1.0], 2, 2.0, true),
            UnsupervisedClustering.KmedoidsResult([1, 2, 2], [1, 2], 3.0, [1.5, 1.5], 3, 3.0, true),
        )

        @test result.k == 6
        @test result.assignments ≈ [1, 2, 3, 4, 5, 6, 6]
        @test result.clusters == [1, 2, 3, 4, 5, 6]
        @test result.objective ≈ 6.0
        @test result.objective_per_cluster ≈ [0.5, 0.5, 1.0, 1.0, 1.5, 1.5]
        @test result.iterations == 6
        @test result.elapsed ≈ 6.0
        @test result.converged == true
    end
end

end
