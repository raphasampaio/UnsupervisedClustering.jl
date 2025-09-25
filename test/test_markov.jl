module TestMarkov

using Distances
using RegularizedCovarianceMatrices
using StableRNGs
using UnsupervisedClustering
using Test

@testset "Markov" begin
    result1 = UnsupervisedClustering.KmeansResult([1, 2, 3, 3, 2, 1], zeros(0, 3), 0.0, zeros(0), 0, 0.0, false)
    result2 = UnsupervisedClustering.KmeansResult([2, 3, 1, 3, 1, 2], zeros(0, 3), 0.0, zeros(0), 0, 0.0, false)

    @test stochastic_matrix(result1, result2) ≈ [0.0 1.0 0.0; 0.5 0.0 0.5; 0.5 0.0 0.5]
end

end
