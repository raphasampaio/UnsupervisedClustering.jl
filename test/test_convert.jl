module TestConvert

using Distances
using LinearAlgebra
using RegularizedCovarianceMatrices
using StableRNGs
using UnsupervisedClustering
using Test

@testset "Convert" begin
    kmeans_result = UnsupervisedClustering.KmeansResult([1, 2, 2], [1.0 2.0; 1.0 2.0; 1.0 2.0])
    gmm_result = UnsupervisedClustering.GMMResult(
        [1, 2, 2],
        [0.5, 0.5],
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        [Symmetric(Matrix{Float64}(I, 3, 3)) for _ in 1:2],
    )

    @test kmeans_result == convert(UnsupervisedClustering.KmeansResult, gmm_result)
    @test gmm_result == convert(UnsupervisedClustering.GMMResult, kmeans_result)
end
end
