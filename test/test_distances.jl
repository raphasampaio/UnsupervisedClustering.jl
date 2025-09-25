module TestDistances

using Distances
using Random
using StableRNGs
using UnsupervisedClustering
using Test

@testset "Distances" begin
    rng = StableRNG(1)
    n, d, k = 3, 2, 3
    data = rand(rng, n, d)

    kmeans = Kmeans(rng = rng)
    result = fit(kmeans, data, k)

    @test UnsupervisedClustering.pairwise_distances(kmeans; result, data) ≈
          pairwise(kmeans.metric, result.clusters, data', dims = 2)

    return nothing
end

end
