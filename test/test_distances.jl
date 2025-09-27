module TestDistances

using Distances
using Random
using StableRNGs
using UnsupervisedClustering
using Test

@testset "Distances" begin
    rng = StableRNG(1)
    n, d, k = 10, 3, 2

    data = rand(rng, n, d)
    clusters = rand(rng, d, k)
    metric = SqEuclidean()

    @test UnsupervisedClustering.compute_distances(metric; clusters, data) ≈ pairwise(metric, clusters, data', dims = 2)
end

end
